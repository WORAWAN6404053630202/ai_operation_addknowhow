"""
Tests for slot-fill filter correctness.

Covers bugs found during slot flow testing (2026-05):
  1. display_options not copied to pending_slot → user sees raw value, not display label
  2. Chroma $and element with 2 keys → Dense search fails with "Expected one operator"
  3. Location filter dropped after entity_type slot fill → wrong-location docs returned
  4. area_size free-text number not mapped to option ("9.2 ตรม" → "น้อยกว่า 200 ตารางเมตร")
  5. Academic _render_slot_message shows raw "กรุงเทพฯ" instead of "กรุงเทพฯ และปริมณฑล"
"""

from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Chroma filter structure validator
# ─────────────────────────────────────────────────────────────────────────────

def _is_valid_chroma_filter(f: Any, inside_and: bool = False) -> bool:
    """
    Return True if `f` is a valid Chroma where-clause.

    Chroma rules:
    - Each element inside a $and / $or list must have EXACTLY one key.
    - Nested $and / $or are allowed at the top level or inside other $and / $or.
    - A simple equality filter {"field": value} is valid at any level.
    """
    if not isinstance(f, dict):
        return False

    if "$and" in f:
        parts = f["$and"]
        if not isinstance(parts, list) or not parts:
            return False
        return all(_is_valid_chroma_filter(p, inside_and=True) for p in parts)

    if "$or" in f:
        parts = f["$or"]
        if not isinstance(parts, list) or not parts:
            return False
        return all(_is_valid_chroma_filter(p, inside_and=True) for p in parts)

    # Leaf node — when nested inside $and/$or it must have exactly 1 key
    if inside_and and len(f) != 1:
        return False

    return True


def _collect_location_values(f: dict) -> List[str]:
    """Recursively collect all location values from a Chroma filter dict."""
    found: List[str] = []
    if not isinstance(f, dict):
        return found

    for key, val in f.items():
        if key == "location":
            if isinstance(val, dict) and "$in" in val:
                found.extend(val["$in"])
            elif isinstance(val, str):
                found.append(val)
        elif key in ("$and", "$or") and isinstance(val, list):
            for part in val:
                found.extend(_collect_location_values(part))
        elif isinstance(val, dict):
            found.extend(_collect_location_values(val))
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Bug 1 — display_options not copied to pending_slot
# ─────────────────────────────────────────────────────────────────────────────

class TestDisplayOptionsPropagation:
    """display_options must be preserved when popping a slot from the queue."""

    def _build_pending_slot_entry(self, next_slot: dict) -> dict:
        """Simulates practical.py slot-queue pop logic (lines ~4952-4965)."""
        slot_opts = next_slot.get("options", [])
        entry: Dict[str, Any] = {
            "key": next_slot.get("key", ""),
            "options": list(slot_opts),
            "allow_multi": False,
        }
        if next_slot.get("context_only"):
            entry["context_only"] = True
        # The fix: copy display_options if length matches
        _disp = next_slot.get("display_options")
        if isinstance(_disp, list) and len(_disp) == len(slot_opts):
            entry["display_options"] = _disp
        return entry

    def test_display_options_copied_for_location_slot(self):
        """กรุงเทพฯ และปริมณฑล must appear in pending_slot, not raw 'กรุงเทพฯ'."""
        next_slot = {
            "key": "location",
            "options": ["กรุงเทพฯ", "ต่างจังหวัด"],
            "display_options": ["กรุงเทพฯ และปริมณฑล", "ต่างจังหวัด"],
            "question": "ร้านของคุณตั้งอยู่ในพื้นที่ใดครับ",
        }
        entry = self._build_pending_slot_entry(next_slot)

        assert "display_options" in entry, "display_options must be copied to pending_slot"
        assert entry["display_options"] == ["กรุงเทพฯ และปริมณฑล", "ต่างจังหวัด"]
        assert entry["options"] == ["กรุงเทพฯ", "ต่างจังหวัด"], "raw options (filter values) preserved"

    def test_display_options_not_set_when_lengths_differ(self):
        """display_options with wrong length must NOT be used (would break index mapping)."""
        next_slot = {
            "key": "location",
            "options": ["กรุงเทพฯ", "ต่างจังหวัด"],
            "display_options": ["กรุงเทพฯ และปริมณฑล"],  # mismatched length
        }
        entry = self._build_pending_slot_entry(next_slot)
        assert "display_options" not in entry

    def test_slot_without_display_options_unaffected(self):
        """Slots with no display_options still work normally."""
        next_slot = {
            "key": "entity_type",
            "options": ["นิติบุคคล", "บุคคลธรรมดา"],
            "question": "ธุรกิจของคุณเป็นรูปแบบใดครับ",
        }
        entry = self._build_pending_slot_entry(next_slot)
        assert entry["options"] == ["นิติบุคคล", "บุคคลธรรมดา"]
        assert "display_options" not in entry

    def test_render_uses_display_options_when_present(self):
        """Renderer should pick display_options over raw options for numbered menu."""
        options = ["กรุงเทพฯ", "ต่างจังหวัด"]
        display_options = ["กรุงเทพฯ และปริมณฑล", "ต่างจังหวัด"]

        pending = {"options": options, "display_options": display_options}
        _disp = pending.get("display_options")
        render_opts = _disp if (isinstance(_disp, list) and len(_disp) == len(options)) else options

        assert render_opts == ["กรุงเทพฯ และปริมณฑล", "ต่างจังหวัด"]
        assert render_opts != options  # must not fall back to raw values


# ─────────────────────────────────────────────────────────────────────────────
# Bug 2 — Chroma $and element with multiple keys
# ─────────────────────────────────────────────────────────────────────────────

class TestChromaFilterStructure:
    """Chroma requires each $and/$or element to have exactly ONE key."""

    def test_invalid_multikey_and_element_detected(self):
        """Flat dict with 2 keys inside $and is invalid."""
        bad = {"$and": [
            {"entity_type_normalized": "นิติบุคคล", "license_type": "ใบอนุญาต"},
            {"location": {"$in": ["กรุงเทพฯ", ""]}},
        ]}
        assert not _is_valid_chroma_filter(bad), "multi-key $and element must be detected as invalid"

    def test_valid_expanded_filter(self):
        """Each key as its own $and element is valid."""
        good = {"$and": [
            {"entity_type_normalized": "นิติบุคคล"},
            {"license_type": "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร"},
            {"location": {"$in": ["กรุงเทพฯ", ""]}},
        ]}
        assert _is_valid_chroma_filter(good)

    def test_flat_dict_at_top_level_is_valid(self):
        """Flat multi-key dict at TOP level (not inside $and) is valid for Chroma."""
        top_level = {
            "entity_type_normalized": "นิติบุคคล",
            "license_type": "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร",
        }
        assert _is_valid_chroma_filter(top_level)

    def test_expand_flat_dict_before_and_wrap(self):
        """Simulates the fixed code: expand flat dict → each key is a separate $and element."""
        entity_val = "นิติบุคคล"
        license_type = "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร"
        location = "กรุงเทพฯ"

        _meta_et = {"entity_type_normalized": entity_val, "license_type": license_type}
        _loc_flt = {"location": {"$in": [location, ""]}}

        # Buggy code (before fix): wraps flat dict as single element
        broken = {"$and": [_meta_et, _loc_flt]}
        assert not _is_valid_chroma_filter(broken), "buggy filter must fail validation"

        # Fixed code: expand each key → separate element
        _parts = [{k: v} for k, v in _meta_et.items()]
        _parts.append(_loc_flt)
        fixed = {"$and": _parts}
        assert _is_valid_chroma_filter(fixed), "fixed filter must pass validation"
        assert len(fixed["$and"]) == 3

    def test_nested_or_inside_and_is_valid(self):
        """$or nested inside $and is a valid Chroma pattern."""
        f = {"$and": [
            {"$or": [
                {"entity_type_normalized": "นิติบุคคล"},
                {"entity_type_normalized": ""},
            ]},
            {"department": "สำนักงานเขต/เทศบาล"},
            {"location": {"$in": ["กรุงเทพฯ", ""]}},
        ]}
        assert _is_valid_chroma_filter(f)

    def test_registration_type_filter_with_location(self):
        """rt + entity + license + location filter pattern must be valid."""
        f = {"$and": [
            {"entity_type_normalized": "นิติบุคคล"},
            {"license_type": "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร"},
            {"registration_type": "ห้างหุ้นส่วนจำกัด"},
            {"location": {"$in": ["กรุงเทพฯ", ""]}},
        ]}
        assert _is_valid_chroma_filter(f)


# ─────────────────────────────────────────────────────────────────────────────
# Bug 3 — Location filter dropped after entity_type slot fill
# ─────────────────────────────────────────────────────────────────────────────

class TestLocationFilterPreservation:
    """Location collected in an earlier slot must stay in subsequent retrieval filters."""

    def _build_entity_filter_with_dept(
        self,
        entity_val: str,
        known_dept: Optional[str],
        known_loc: Optional[str],
    ) -> dict:
        """Simulates entity-enriched retrieval filter building (supervisor.py ~8979-8990)."""
        if known_dept:
            _ef_parts: List[Dict] = [
                {"$or": [
                    {"entity_type_normalized": entity_val},
                    {"entity_type_normalized": ""},
                ]},
                {"department": known_dept},
            ]
            if known_loc:
                _ef_parts.append({"location": {"$in": [known_loc, ""]}})
            return {"$and": _ef_parts}
        else:
            if known_loc:
                return {"$and": [
                    {"entity_type_normalized": entity_val},
                    {"location": {"$in": [known_loc, ""]}},
                ]}
            return {"entity_type_normalized": entity_val}

    def test_location_included_in_entity_filter_with_dept(self):
        """When dept and location both known, both must appear in entity filter."""
        f = self._build_entity_filter_with_dept(
            entity_val="นิติบุคคล",
            known_dept="สำนักงานเขต/เทศบาล",
            known_loc="กรุงเทพฯ",
        )
        loc_values = _collect_location_values(f)
        assert "กรุงเทพฯ" in loc_values, "location must be present in entity filter"
        assert _is_valid_chroma_filter(f)

    def test_location_included_without_dept(self):
        """When only location is known (no dept), it must still be in entity filter."""
        f = self._build_entity_filter_with_dept(
            entity_val="บุคคลธรรมดา",
            known_dept=None,
            known_loc="ต่างจังหวัด",
        )
        loc_values = _collect_location_values(f)
        assert "ต่างจังหวัด" in loc_values
        assert _is_valid_chroma_filter(f)

    def test_no_location_slot_gives_no_location_filter(self):
        """Without location slot, filter must NOT include location (avoid false exclusions)."""
        f = self._build_entity_filter_with_dept(
            entity_val="นิติบุคคล",
            known_dept="สำนักงานเขต/เทศบาล",
            known_loc=None,
        )
        loc_values = _collect_location_values(f)
        assert not loc_values, "no location should appear when slot not collected"

    def _build_op_group_filter(
        self,
        entity_val: str,
        license_type: str,
        known_loc: Optional[str],
        known_rt: Optional[str] = None,
    ) -> dict:
        """Simulates op_group re-retrieval filter building (supervisor.py ~9297-9334).

        Uses $in:[entity, ""] explicitly so generic docs (entity_type_normalized="")
        are included regardless of whether the filter is flat or wrapped in $and.
        (_scored_search()'s auto-transform only fires on flat dicts and is bypassed
        once location wraps the filter in $and.)
        """
        # Use $in explicitly — do NOT rely on _scored_search auto-transform
        _meta_et: dict = {"entity_type_normalized": {"$in": [entity_val, ""]}}
        if license_type:
            _meta_et["license_type"] = license_type

        _BIZTYPE = ("บริษัท", "ห้างหุ้นส่วน", "บุคคลธรรมดา", "จำกัด", "สามัญ")

        if known_rt and any(kw in known_rt for kw in _BIZTYPE):
            _parts: List[dict] = [{"entity_type_normalized": {"$in": [entity_val, ""]}}]
            if license_type:
                _parts.append({"license_type": license_type})
            _parts.append({"registration_type": known_rt})
            if known_loc:
                _parts.append({"location": {"$in": [known_loc, ""]}})
            return {"$and": _parts}
        elif known_loc:
            _loc_flt = {"location": {"$in": [known_loc, ""]}}
            # Expand flat dict into separate $and elements (1 key per element for Chroma)
            _parts2 = [{k: v} for k, v in _meta_et.items()]
            _parts2.append(_loc_flt)
            return {"$and": _parts2}
        else:
            return _meta_et

    def test_op_group_filter_includes_location(self):
        """op_group re-retrieval filter must include location when known."""
        f = self._build_op_group_filter(
            entity_val="นิติบุคคล",
            license_type="ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร",
            known_loc="กรุงเทพฯ",
        )
        loc_values = _collect_location_values(f)
        assert "กรุงเทพฯ" in loc_values
        assert _is_valid_chroma_filter(f), f"filter invalid: {f}"

    def test_op_group_filter_with_rt_and_location(self):
        """op_group re-retrieval with registration_type + location both included."""
        f = self._build_op_group_filter(
            entity_val="นิติบุคคล",
            license_type="ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร",
            known_loc="กรุงเทพฯ",
            known_rt="บริษัทจำกัด",
        )
        loc_values = _collect_location_values(f)
        assert "กรุงเทพฯ" in loc_values
        assert _is_valid_chroma_filter(f)

    def test_op_group_filter_no_location_when_not_collected(self):
        """op_group filter has no location when slot was not collected."""
        f = self._build_op_group_filter(
            entity_val="นิติบุคคล",
            license_type="ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร",
            known_loc=None,
        )
        loc_values = _collect_location_values(f)
        assert not loc_values

    def test_location_generic_docs_included_via_empty_string(self):
        """location $in must include '' so generic docs without location are not excluded."""
        f = self._build_op_group_filter(
            entity_val="นิติบุคคล",
            license_type="ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร",
            known_loc="กรุงเทพฯ",
        )
        loc_values = _collect_location_values(f)
        assert "" in loc_values, "empty string must be in location $in to keep generic docs"

    def test_op_group_entity_uses_dollar_in_not_equality(self):
        """op_group filter must use $in:[entity, ''] not plain equality.

        _scored_search() auto-transform only fires on flat dicts.
        Once location wraps the filter in $and, the transform is bypassed —
        plain equality would exclude generic docs (entity_type_normalized='')
        and cause too few docs to be retrieved.
        """
        f = self._build_op_group_filter(
            entity_val="นิติบุคคล",
            license_type="ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร",
            known_loc="กรุงเทพฯ",
        )
        # Entity filter must use $in (not plain string equality)
        def _find_entity_filters(node: Any) -> List[Any]:
            found = []
            if isinstance(node, dict):
                if "entity_type_normalized" in node:
                    found.append(node["entity_type_normalized"])
                for v in node.values():
                    if isinstance(v, (dict, list)):
                        found.extend(_find_entity_filters(v))
                return found
            if isinstance(node, list):
                for item in node:
                    found.extend(_find_entity_filters(item))
            return found

        entity_vals = _find_entity_filters(f)
        assert entity_vals, "entity_type_normalized must appear in filter"
        for ev in entity_vals:
            assert isinstance(ev, dict) and "$in" in ev, (
                f"entity filter must use $in (got {ev!r}). "
                "Plain equality skips generic docs when filter is wrapped in $and."
            )
        # Generic docs (entity="") must be in the $in list
        for ev in entity_vals:
            assert "" in ev.get("$in", []), "empty string must be in entity $in to include generic docs"

    def test_op_group_no_location_entity_still_uses_dollar_in(self):
        """Without location slot, entity filter must still use $in (for generic doc inclusion)."""
        f = self._build_op_group_filter(
            entity_val="บุคคลธรรมดา",
            license_type="ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร",
            known_loc=None,
        )
        # f is a flat dict here — but entity value must still be $in not plain string
        et_val = f.get("entity_type_normalized")
        assert isinstance(et_val, dict) and "$in" in et_val, (
            "entity_type_normalized must be $in even in flat filter"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Bug 5 — area_size numeric input not mapped to slot option
# ─────────────────────────────────────────────────────────────────────────────

class TestAreaSizeNumericMapping:
    """
    _save_slots_best_effort must map numeric free-text ("9.2 ตรม", "150 ตร.ม.")
    to the correct area_size option ("น้อยกว่า 200 ตารางเมตร" / "มากกว่า 200 ตารางเมตร").

    Root cause: "ตรม" matches BOTH options as a substring → ambiguous token path →
    slot not saved → auto-fill picks the only remaining doc value (wrong option).
    """

    _OPTIONS_200 = ["น้อยกว่า 200 ตารางเมตร", "มากกว่า 200 ตารางเมตร"]

    def _numeric_area_match(self, raw: str, options: list) -> Optional[str]:
        """
        Mirrors the numeric-threshold fallback added to _save_slots_best_effort.
        Returns matched option or None.
        """
        import re
        nums = re.findall(r'\d+(?:\.\d+)?', raw)
        if not nums:
            return None
        user_area = float(nums[0])
        less_opt = next((v for v in options if 'น้อยกว่า' in v or 'ไม่เกิน' in v), None)
        more_opt = next((v for v in options if 'มากกว่า' in v or 'เกิน' in v or 'ขึ้นไป' in v), None)
        if not (less_opt and more_opt):
            return None
        thresh_m = __import__('re').search(r'(\d+(?:\.\d+)?)', less_opt)
        if not thresh_m:
            return None
        thresh = float(thresh_m.group(1))
        return less_opt if user_area < thresh else more_opt

    def test_small_area_maps_to_less_than(self):
        """9.2 ตรม < 200 → 'น้อยกว่า 200 ตารางเมตร'"""
        result = self._numeric_area_match("ประมาณ 9.2 ตรม", self._OPTIONS_200)
        assert result == "น้อยกว่า 200 ตารางเมตร"

    def test_large_area_maps_to_more_than(self):
        """350 ตรม >= 200 → 'มากกว่า 200 ตารางเมตร'"""
        result = self._numeric_area_match("350 ตารางเมตร", self._OPTIONS_200)
        assert result == "มากกว่า 200 ตารางเมตร"

    def test_exact_threshold_maps_to_more_than(self):
        """200 ตรม is NOT less than 200 → 'มากกว่า 200 ตารางเมตร'"""
        result = self._numeric_area_match("200 ตรม", self._OPTIONS_200)
        assert result == "มากกว่า 200 ตารางเมตร"

    def test_50_sqm_maps_to_less_than(self):
        """50 ตร.ม. → 'น้อยกว่า 200 ตารางเมตร'"""
        result = self._numeric_area_match("ร้านขนาด 50 ตร.ม.", self._OPTIONS_200)
        assert result == "น้อยกว่า 200 ตารางเมตร"

    def test_no_number_returns_none(self):
        """Free text without number → no numeric match → None"""
        result = self._numeric_area_match("ร้านเล็กมาก", self._OPTIONS_200)
        assert result is None

    def test_label_answer_unaffected(self):
        """Numeric match only triggers when no label matched; 'ก'/'ข' handled upstream."""
        # Simulate: 'ก' maps to option in choice_map upstream; numeric fallback returns None
        # because raw="ก" has no digits
        result = self._numeric_area_match("ก", self._OPTIONS_200)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Bug 6 — Academic _render_slot_message shows raw "กรุงเทพฯ" not display label
# ─────────────────────────────────────────────────────────────────────────────

class TestAcademicRenderDisplayLabels:
    """
    Academic _render_slot_message must show "กรุงเทพฯ และปริมณฑล" in the UI
    but keep "กรุงเทพฯ" as the stored choice-map value for Chroma filtering.
    """

    _DISPLAY_MAP = {"กรุงเทพฯ": "กรุงเทพฯ และปริมณฑล"}

    def _render_choices(self, choices: list) -> tuple:
        """
        Mirrors academic._render_slot_message choice rendering:
        returns (lines_rendered, choice_map) where choice_map stores raw values.
        """
        _LABELS = ["ก", "ข", "ค", "ง"]
        lines = []
        per_slot: Dict[str, str] = {}
        for j, c in enumerate(choices[:4]):
            lbl = _LABELS[j]
            display_c = self._DISPLAY_MAP.get(c, c)
            lines.append(f"   {lbl}) {display_c}")
            per_slot[lbl] = c
            per_slot[lbl.lower()] = c
        return lines, per_slot

    def test_bangkok_shows_display_label(self):
        """'กรุงเทพฯ' choice must render as 'กรุงเทพฯ และปริมณฑล'."""
        lines, _ = self._render_choices(["กรุงเทพฯ", "ต่างจังหวัด"])
        assert "กรุงเทพฯ และปริมณฑล" in lines[0], f"display label missing: {lines[0]}"
        assert "กรุงเทพฯ" not in lines[0].split(") ", 1)[-1].replace("กรุงเทพฯ และปริมณฑล", ""), \
            "raw 'กรุงเทพฯ' must not appear standalone in rendered line"

    def test_choice_map_stores_raw_value(self):
        """choice_map must store raw 'กรุงเทพฯ' (not display label) so Chroma filter is correct."""
        _, per_slot = self._render_choices(["กรุงเทพฯ", "ต่างจังหวัด"])
        assert per_slot["ก"] == "กรุงเทพฯ", (
            "choice map must hold raw Chroma value, not display label"
        )

    def test_provincial_unaffected(self):
        """'ต่างจังหวัด' has no override — renders and stores as-is."""
        lines, per_slot = self._render_choices(["กรุงเทพฯ", "ต่างจังหวัด"])
        assert "ต่างจังหวัด" in lines[1]
        assert per_slot["ข"] == "ต่างจังหวัด"

    def test_non_location_choices_unaffected(self):
        """Entity choices have no display override — rendered verbatim."""
        lines, per_slot = self._render_choices(["นิติบุคคล", "บุคคลธรรมดา"])
        assert per_slot["ก"] == "นิติบุคคล"
        assert "นิติบุคคล" in lines[0]
