# ResBiz Assistant — Complete Flowchart (Single Diagram)

All 9 subgraphs merged into one `flowchart TD`. Cross-subgraph arrows are listed
after the closing `end` of every subgraph.

Node-ID adaptations from the merge request:
- `D41` does not exist → used `D27` (Route to Practical or Academic persona)
- `E28` (Keep doc, intermediate) → used `E31` (Return filtered docs list, true RAG output)
- `H18` (decision node mid-trim) → used `H30` (Continue normally, session mgmt done)

```mermaid
flowchart TD

  subgraph API["API Layer"]
    A1[POST /api/v1/chat or /chat/stream]
    A1 --> A2{Services initialized?}
    A2 -->|No| A3[503 Service Unavailable]
    A2 -->|Yes| A4{Message body empty?}
    A4 -->|Yes| A5[400 Bad Request]
    A4 -->|No| A6{Rate limit: is_allowed?}
    A6 -->|Blocked| A7[429 Too Many Requests]
    A6 -->|Allowed| A8[Load or create session state]
    A8 --> A9{Session token budget exceeded?}
    A9 -->|Yes - warn only| A10[Log warning, continue]
    A9 -->|No| A10
    A10 --> A11{Token rate window exceeded?}
    A11 -->|Yes| A12[429 Token Rate Limit]
    A11 -->|No| A13{pending_slot or slot-sensitive keys in collected_slots?}
    A13 -->|Yes - skip cache| A15[run_in_executor: supervisor.handle]
    A13 -->|No - check cache| A14{Cache HIT? key=SHA256 session+question+persona+collected_slots}
    A14 -->|Yes| A16[Return cached response]
    A14 -->|No| A15
    A15 --> A17[Save state, record tokens, cache result]
    A17 --> A18[Return HandleSuccess response]
  end

  subgraph PREPROC["Pre-processing"]
    B1[supervisor.handle called]
    B1 --> B1a[trim_messages keep_last=12 and trim topic_pool to 30]
    B1a --> B2{len lt 6 or _FORMAL_RE matches?}
    B2 -->|Yes - skip rewrite| B5[Use original query]
    B2 -->|No| B6{Query rewriter cache hit?}
    B6 -->|Yes| B5
    B6 -->|No| B7[LLM rewrite: informal Thai to formal regulatory terms]
    B7 --> B8{LLM output valid? 3 lt len le 150 and different from original?}
    B8 -->|Yes| B9[Append formal expansion to original query]
    B8 -->|No| B5
    B9 --> B13[_handle_inner: supervisor routing]
    B5 --> B13
  end

  subgraph SUP["Supervisor Routing — _handle_inner priority order lines 11321-12200"]
    B13 --> S0[trim_messages keep_last=10 proactive line 11343]
    S0 --> S0a{MAX_ROUNDS exceeded? cur_round ge max_rounds gt 0 line 11349}
    S0a -->|Yes| S0b[Return limit message]
    S0a -->|No| S1{2.1: _is_academic_intake_active? stage=awaiting_topic/slots/sections line 11364}
    S1 -->|Yes| S2[academic.handle then _post_route_academic_auto_return]
    S1 -->|No| S3[2.2: Clear legacy awaiting_persona_confirmation key line 11374]
    S3 --> S4{2.2b: Academic resume — topic_changed guard then pending_opts/topic_catalog detection line 11390}
    S4 -->|_ACADEMIC_STOP_RE or LLM stop| S5[Force return to Practical]
    S4 -->|_ACADEMIC_RESUME_RE or LLM Path-A resume classifier| S6[Re-enter Academic silently]
    S4 -->|LLM Path-B: fallback_intent=elaborate conf ge 0.75 + 8-char topic overlap| S6
    S4 -->|None match| S7{2.2c: Off-topic guardrail — 6 conditions: no LEGAL_SIGNAL no selection no number no QUESTION_MARKERS no greeting no DEPTH_DETAIL line 11537}
    S7 -->|Any condition fails| S8[Bypass guardrail - continue routing]
    S7 -->|All 6 pass: check Chroma similarity_search k=1| S8b{Relevance score ge 0.72?}
    S8b -->|Yes - in domain| S8
    S8b -->|No - out of domain| S9[_handle_deflect: off-topic polite reply]
    S8 --> S10{2.3: Style request? _infer_user_style_request_hybrid + _is_short_depth_followup + LLM conf ge 0.80 line 11585}
    S10 -->|wants_long| S6
    S10 -->|wants_short| S5
    S10 -->|No style| S11{2.4: Explicit switch? verbs + target marker line 11631}
    S11 -->|Yes target=academic| S6
    S11 -->|Yes target=practical| S5
    S11 -->|No| S12{2.5: Switch-without-target? regex + LLM line 11638}
    S12 -->|Yes| S13[Toggle to academic if not already]
    S12 -->|No| S14{2.5.5: Pure number input AND no pending_slot? line 11646}
    S14 -->|Yes| S15[Restore topic from last_topic_menu by numeric index]
    S14 -->|No| S16{2.5.6: pending_dynamic_clarification in context? line 11661}
    S16 -->|Yes| S17[_resolve_dynamic_clarification]
    S16 -->|No| S18{2.5.7: Slot correction? LLM detect line 11667}
    S18 -->|Yes| S19[Correct stored slot + re-retrieve]
    S18 -->|No| S20{2.6: pending_slot in context? line 11676}
    S20 -->|Yes| S21[_route_pending_slot_to_persona]
    S20 -->|No| S22{2.6b: Likely typo? len le 8 chars: _STANDALONE_DIACRITIC or consonant mash or all-punct then LLM conf ge 0.75 line 11684}
    S22 -->|Yes| S23[_handle_typo_prompt: ask user to rephrase]
    S22 -->|No| S24{2.7: Greeting/thanks/smalltalk/noise? _looks_like_greeting + _SMALLTALK_RE + LLM + _is_noise line 11719}
    S24 -->|Yes| S24a{2.7a: last_user_legal_query exists AND len le 10 AND not THANKS_RE? line 11730}
    S24a -->|Yes: classify| S24b{_classify_yes_no_hybrid conf ge 0.78?}
    S24b -->|yes + pending_slot active| S21
    S24b -->|yes + no pending| S24c[Elaborate last_q: _ensure_practical_retrieval then practical.handle]
    S24b -->|no conf ge 0.78| S24e[_handle_greeting show_menu=True]
    S24b -->|no match| S24f[_handle_greeting show_menu=False]
    S24a -->|No active legal context| S24f
    S24 -->|No| S25{2.8: Mode status query? line 11765}
    S25 -->|Yes| S26[Show current persona: i ตอนนี้เป็นโหมด X]
    S25 -->|No| S27{2.9: Legal question? _looks_like_legal_question line 11773}
    S27 -->|Yes| C28[Legal routing sub-path]
    S27 -->|No| S28{3.0: Thai interjection? _TH_INTERJECTION_RE line 11890}
    S28 -->|Yes| S24f
    S28 -->|No| S29{3.1: New topic request? _NEW_TOPIC_RE + LLM line 11895}
    S29 -->|Yes| S24e
    S29 -->|No| S30{3.2: Elaborate? _ELABORATE_RE + LLM line 11908}
    S30 -->|Yes + new legal remainder ge 5 chars| C28
    S30 -->|Yes + same topic| S31[Re-retrieve last_q + dimension-gap check + practical.handle elaborate]
    S30 -->|No| S32{3.3: Contextual follow-up? _FOLLOWUP_CONTEXTUAL_RE + LLM line 11990}
    S32 -->|Yes| S31
    S32 -->|No| S33{3.4: Link request + active context? line 12008}
    S33 -->|Yes| S31
    S33 -->|No| S34[4: LLM fallback intent — reuse _cached_fallback_intent or call Haiku line 12025]
    S34 --> S35{fallback_intent?}
    S35 -->|new_topic + active ctx + no explicit menu req| C28
    S35 -->|new_topic + no ctx or explicit menu| S24e
    S35 -->|elaborate| S31
    S35 -->|legal_question| C28
    S35 -->|greeting| S24f
    S35 -->|unknown| S36[Deflect with context-aware follow-up]
  end

  subgraph LEGAL["Legal Routing — _ensure_practical_retrieval_for_legal + _maybe_build_slot_queue_from_docs"]
    C28 --> L0{Pending slot interrupt? check key in INTERRUPT_EXEMPT set line 11788}
    L0 -->|Non-exempt key + legal input| L0a[Clear pending_slot + topic_slot_queue]
    L0 -->|Exempt key + option text in input| L0b[Route back to _route_pending_slot_to_persona]
    L0 -->|Exempt + bypass: INFO_Q or link req or generic followup or no option match| L0a
    L0 -->|No pending slot| L1{Persona = academic?}
    L1 -->|Yes| L1a[academic.handle + _post_route_academic_auto_return]
    L1 -->|No - practical| L2{Step 0: op_topic exact match ge 8 chars in query? line 2831}
    L2 -->|Yes| L3[Chroma.get where op_topic - set current_docs - practical _internal=True]
    L2 -->|No| L4{Step 0b: OBD exact match ge 8 chars in query?}
    L4 -->|Yes| L5[Chroma.get where OBD - set current_docs - practical _internal=True]
    L4 -->|No| L6[Generic follow-up anchoring: anchor on last_user_legal_query if context]
    L6 --> L7[Op-type follow-up enrichment + _force_fresh_retrieval if op changed line 3236]
    L7 --> L8[Channel-preference anchoring if user named service channel line 3276]
    L8 --> L9{_apply_slot_change_if_detected? entity or location switch in user input}
    L9 -->|Yes| L10[Re-retrieve with combined Chroma filter - fallback chain: combined then entity-only then location-only]
    L9 -->|No| L11{_should_retrieve_new? Jaccard lt 0.22 or dimension switch}
    L11 -->|Yes| L12[Fresh retrieval]
    L11 -->|No| L13[Reuse state.current_docs]
    L10 --> L14{_BROAD_Q_RE + _SPECIFIC_LICENSE_INDICATOR_RE override + LLM: broad question? line 3675}
    L12 --> L14
    L13 --> L14
    L14 -->|Yes - broad| L15[Pass 1: semantic search on user query line 3695]
    L15 --> L16[Pass 2: targeted regulatory query prepend detected license names line 3723]
    L16 --> L17[Pass 3: alcohol-specific สุรา retrieval if สุรา in query line 3765]
    L17 --> L18[Coverage Sweep: fetch 1 doc per license_type not yet represented line 3785]
    L18 --> L19[Merge + dedup by content hash]
    L14 -->|No - specific| L20{Multi-topic: ge 2 license_types detected in query?}
    L20 -->|Yes| L21[Per-license retrieval top-3 each - merge]
    L20 -->|No| L22{main_topic substring match ge 5 chars in query? line 3511}
    L22 -->|Yes| L23[Chroma.get where main_topic - cap 60 docs - set current_docs]
    L22 -->|No| L24{sub_topic or operation_topic substring match in query? line 3567}
    L24 -->|Yes| L25[Chroma.get where sub_topic - set current_docs]
    L24 -->|No| L26[Round 1 semantic + Round 2 anchor-enriched retrieval line 3695]
    L19 --> L27[_maybe_build_slot_queue_from_docs line 4011]
    L21 --> L27
    L23 --> L27
    L25 --> L27
    L26 --> L27
    L27 --> L28{Guards: _direct_topic_match OR _broad_question OR non-reg docs dominant? line 4028}
    L28 -->|Skip - no slot queue| L29[topic_slot_queue = empty]
    L28 -->|Build queue| L30[_discover_slots_for_license: scan Chroma for slot dimensions]
    L30 --> L30a[Auto-infer entity_type/location/dept/area_type from query text]
    L30a --> L30b[Cross-topic slot memory: skip already-collected slots]
    L30b --> L31[_prefill_slots_from_message: scan user_input for slot answers - remove from queue line 3909]
    L31 --> L32[Cap queue at 2 slots max line 9033]
    L29 --> L33
    L32 --> L33{topic_slot_queue has pending slot?}
    L33 -->|Yes| L34[Set pending_slot = queue.pop - ask slot question]
    L33 -->|No| L35{Link request? skip dynamic clarification}
    L35 -->|No link req| L36{_maybe_dynamic_clarification: topic still ambiguous? line 11877}
    L36 -->|Question generated| L37[Return clarification question to user]
    L36 -->|No question| L38[practical.handle _internal=False - return answer]
    L35 -->|Link req| L38
  end

  subgraph SLOT["Slot Collection — _route_pending_slot_to_persona lines 8041-9100"]
    D1[Slot handler entry: pending_slot active]
    D1 --> D1a{Slot-skip? _NOT_SLOT_SKIP_RE then _SLOT_SKIP_RE then LLM fallback}
    D1a -->|Skip detected| D2a{More slots in topic_slot_queue?}
    D2a -->|Yes| D2b[Pop next slot from queue - ask it]
    D2a -->|No| D2c[Retrieve last_q and answer via Practical]
    D1a -->|Not skip| D3{New-topic escape? none of pending options in input AND legal question AND len ge 6 line 8098}
    D3 -->|Yes - new topic| D3a[Clear pending_slot + queue - recurse _handle_inner as new query]
    D3 -->|No - treat as slot reply| D4[_map_pending_slot_reply: exact match then fuzzy then LLM slot_mapper conf ge 0.60 line 5473]
    D4 --> D5{Mapped successfully?}
    D5 -->|No| D6[Re-ask: กรุณาตอบเป็นตัวเลขตามตัวเลือก]
    D5 -->|Yes: mapped value| D7{pending.key = topic? line 8128}
    D7 -->|Yes| D8[Save last_topic + last_user_legal_query + last_topic_menu]
    D8 --> D8a[Multi-topic or chapter or Step1+Step2 retrieval then practical _internal=True]
    D7 -->|No - non-topic slot| D9[Save cross-persona slot: state.save_collected_slot line 8155]
    D9 --> D10{key = entity_type? line 8143}
    D10 -->|Yes| D11[DataLoader._normalize_entity_type: บริษัทจำกัด to นิติบุคคล line 8146]
    D11 --> D12[Save entity_type = normalized canonical value]
    D12 --> D13{Sub-type differs from canonical? e.g. บริษัทจำกัด ne นิติบุคคล line 8180}
    D13 -->|Yes| D14[Auto-fill registration_type = original sub-type value]
    D13 -->|No| D15[No auto-fill needed]
    D14 --> D16
    D15 --> D16[Chroma filter: entity_type_normalized = นิติบุคคล or บุคคลธรรมดา line 8708]
    D10 -->|No| D17{key = operation_group? line 9072}
    D17 -->|Yes| D18[Build enriched_q from raw_op_map prefix + retrieve with license_type filter]
    D17 -->|No| D19{key = registration_type or other non-entity? line 9047}
    D19 -->|Yes: _raw ne _entity_val| D20[Include _raw sub-type in enriched_q alongside entity_val line 9051]
    D20 --> D21[Retrieve with registration_type in enriched_q + entity_type_normalized filter]
    D19 -->|Other slot key| D22[Save only - no metadata filter needed]
    D16 --> D23[_practical._retrieve_docs with metadata_filter line 8713]
    D18 --> D23
    D21 --> D23
    D22 --> D23
    D23 --> D24[Clear _broad_question flag from context line 8200]
    D24 --> D25{More slots in topic_slot_queue?}
    D25 -->|Yes| D26[Ask next slot question]
    D25 -->|No| D27[Route to Practical or Academic persona]
  end

  subgraph RAG["RAG Pipeline — _retrieve_docs + hybrid_retriever lines 2200-2400"]
    E1[Retrieval: query + optional metadata_filter]
    E1 --> E2[expand_query: synonym enrichment via SYNONYM_PATTERNS]
    E2 --> E3{metadata_filter present?}
    E3 -->|Yes| E4[_scored_search: query top_k with filter - Round 1 filtered]
    E4 --> E5{docs ge top_k divided by 2?}
    E5 -->|No - partial results| E6[Partial retry: _scored_search top_k x2 same filter - add non-dup docs line 2260]
    E5 -->|Yes - enough docs| E7[Proceed to Round 2 anchor check]
    E6 --> E8{Still 0 docs total? line 2268}
    E8 -->|No - partial found| E7
    E8 -->|Yes: Step 2 fallback| E9{Filter contains location or area_size field?}
    E9 -->|Yes: strip location/area_size - keep license_type only line 2272| E10[_scored_search query top_k with lt_only_filter]
    E10 --> E11{Still 0 docs?}
    E11 -->|No| E7
    E11 -->|Yes: Step 3 fallback| E12[Full unfiltered fallback: _scored_search expanded_query top_k no filter line 2291]
    E9 -->|No other filter type| E12
    E12 --> E7
    E3 -->|No filter| E13[_scored_search expanded_query top_k unfiltered]
    E13 --> E7
    E7 --> E14{Not filtered AND docs exist? Anchor-enriched Round 2 line 2315}
    E14 -->|Yes| E15[Majority vote top-5 docs: license_type or op_topic or main_topic]
    E15 --> E16{ge 2 votes for anchor AND anchor not already in query?}
    E16 -->|Yes| E17[Round 2: _scored_search query+anchor top_k - merge non-dup docs]
    E16 -->|No| E18[Skip Round 2]
    E14 -->|No| E18
    E17 --> E19[HYBRID: BM25+Dense+RRF fusion - or Dense-only if disabled]
    E18 --> E19
    E19 --> E20{RERANKER_ENABLED?}
    E20 -->|Yes| E21[CrossEncoder mmarco-mMiniLMv2-L12: predict score for query vs doc_content first 1500 chars]
    E20 -->|No| E22[Keep RRF-fused order]
    E21 --> E23[Metadata boost: +0.25 if keyword matches doc metadata fields]
    E22 --> E23
    E23 --> E24{doc._bm25_hit = True?}
    E24 -->|Yes - BM25-only doc| E25[Skip sim threshold filter]
    E24 -->|No| E26{doc._sim lt RETRIEVAL_MIN_SIMILARITY?}
    E26 -->|Yes| E27[Drop doc]
    E26 -->|No| E28[Keep doc]
    E25 --> E29{Remaining docs lt 2 after filter?}
    E27 --> E29
    E28 --> E29
    E29 -->|Yes| E30[Fallback: use top-N regardless of sim score]
    E29 -->|No| E31[Return filtered docs list]
    E30 --> E31
  end

  subgraph PRACT["Practical Persona"]
    F1[practical.handle entry]
    F1 --> F2{_internal mode or supervisor_owns_menu?}
    F2 -->|Internal - skip menu| F3[Skip greeting and menu logic]
    F2 -->|Normal| F4{Is greeting or thanks? EN and TH regex}
    F4 -->|Yes| F5[Show topic menu with greet_prefix]
    F4 -->|No| F6{Entity switch in user_text? _entity_switch_done flag?}
    F6 -->|Switch detected, not flagged| F7[Re-retrieve with entity_type Chroma filter]
    F6 -->|No switch| F8{_RARE_LT_PATTERNS match?}
    F8 -->|Yes| F9[Direct Chroma filter retrieval for rare license type]
    F8 -->|No| F10{_UNIVERSAL_FACT_RE? skip entity slots}
    F10 -->|Yes| F11[Retrieve without entity filter]
    F10 -->|No| F12{_slot_change_retrieval_done flag?}
    F12 -->|Yes| F13[Skip fresh retrieval, use existing docs]
    F12 -->|No| F14{_should_retrieve_new? Jaccard overlap or dimension switch}
    F14 -->|Yes| F15[Fresh RAG retrieval with metadata filter]
    F14 -->|No| F16[Reuse state.current_docs]
    F7 --> F17[Build docs_json from retrieved docs]
    F9 --> F17
    F11 --> F17
    F13 --> F17
    F15 --> F17
    F16 --> F17
    F3 --> F17
    F17 --> F18{Multi-topic detected?}
    F18 -->|Yes, count le 3| F19[Answer all topics in single response]
    F18 -->|Yes, count gt 3| F20[Show topic summary menu]
    F18 -->|No - single topic| F21{_detect_divergence: diverging entity or reg field in docs?}
    F21 -->|Yes| F22[Add clarification slot question to queue]
    F21 -->|No| F23[LLM practical action decision]
    F23 --> F24{action = ?}
    F24 -->|ask| F25{LQS license quality score = 0?}
    F25 -->|Yes| F26[LLM license detect fallback]
    F25 -->|No| F27[Format single slot question via _fallback_single_question]
    F24 -->|retrieve| F28[Re-retrieve docs with enriched query]
    F24 -->|answer| F29[Build answer with _fallback_practical_answer]
    F27 --> F30[_apply_practical_lint: analyze_practical_text]
    F29 --> F30
    F30 --> F31{Policy issues? forbidden phrase/preface/multi-question/too-long}
    F31 -->|Yes| F32{Rewrite attempts le 2?}
    F32 -->|Yes| F33[LLM rewrite with build_rewrite_prompt]
    F33 --> F30
    F32 -->|No| F34[Deterministic fallback: hard trim or minimal_question]
    F31 -->|No| F35[_fallback_practical_answer: URL dedup then link classification]
    F35 --> F36{Link type classification?}
    F36 -->|Known portal URL| F37[Label: registration]
    F36 -->|Guide keywords| F38[Label: guide]
    F36 -->|Form keywords| F39[Label: form]
    F36 -->|PDF URL| F40[Sub-classify: guide or form or ref]
    F36 -->|Other| F41[Label: ref]
    F35 --> F42[Return HandleSuccess to route_v1]
  end

  subgraph ACAD["Academic Persona"]
    G1[academic.handle entry]
    G1 --> G2{academic_flow exists in context?}
    G2 -->|No - new intake| G3[_start_intake_with_retrieval]
    G2 -->|Yes - FSM active| G4{FSM stage?}
    G4 -->|awaiting_topic| G5[_bind_choice_if_any: bind topic selection]
    G4 -->|awaiting_slots| G6[Collect slot answer - advance to next slot or sections]
    G4 -->|awaiting_sections| G6a{Entity-type change in input? นิติบุคคล or บุคคลธรรมดา without section number line 4619}
    G6a -->|Yes: new entity| G6b[Update academic_slots entity - clear flow + section catalog - recurse handle with base_q]
    G6a -->|No| G6c[_bind_choice_if_any: parse section numbers line 4664]
    G6c --> G6d{bound=False AND greeting or noise?}
    G6d -->|Yes| G6e[Re-ask section menu: return pending_question line 4668]
    G6d -->|No - bound| G7[_save_selected_sections then finalize answer]
    G4 -->|done| G8[Set auto_return_to_practical = True]
    G3 --> G9{Greeting or noise? regex + LLM}
    G9 -->|Yes| G10[Re-ask current stage question]
    G9 -->|No| G11{_META_REQUEST_RE match? vague detail request}
    G11 -->|Yes| G12{LLM: is_meta + has_embedded_topic?}
    G12 -->|Embedded topic| G13[Use extracted topic as base query]
    G12 -->|No embedded topic| G14[Use last_user_legal_query as base]
    G11 -->|No| G15[Use user input as query]
    G13 --> G16[Entity enrichment from collected_slots]
    G14 --> G16
    G15 --> G16
    G16 --> G17{Broad-docs fast-path? entity_type matches?}
    G17 -->|Yes| G18[Use pre-retrieved supervisor docs]
    G17 -->|No| G19{needs_fresh? less than 2 docs or query changed over 30 percent}
    G19 -->|Yes| G20[Fresh retrieval with entity_type Chroma dollar-or filter]
    G19 -->|No| G18
    G20 --> G21{Safety net C: multi topic_group in docs?}
    G21 -->|Multi-group| G22[Filter docs by topic_group dollar-in filter]
    G21 -->|Single group| G23[Use all retrieved docs]
    G22 --> G24[_ask_topic_selection]
    G23 --> G24
    G18 --> G24
    G24 --> G25{ge 2 distinct license_type or sub_topic?}
    G25 -->|Yes| G26{Doc-count dominance ge 3x or unique keyword?}
    G26 -->|Yes| G27[Auto-select dominant topic, skip menu]
    G26 -->|No| G28[Show numbered topic menu: stage = awaiting_topic]
    G25 -->|No - single topic| G27
    G27 --> G29[_compute_dynamic_slots: scan docs for slot diversity]
    G29 --> G30{Any slots needed? entity/location/area/reg/op?}
    G30 -->|Yes| G31[Ask first slot: stage = awaiting_slots]
    G30 -->|No| G32[_ask_sections: scan docs for section diversity]
    G32 --> G33{Distinct sections ge 1?}
    G33 -->|Yes| G34[Show numbered section menu: stage = awaiting_sections]
    G33 -->|No| G35[Generate final answer directly]
    G34 --> G35
    G35 --> G36{Has data?}
    G36 -->|Full data| G37[Structured evidence-based answer with all sections]
    G36 -->|Partial data| G38[Best-effort partial answer with caveat]
    G36 -->|No data| G39[Deflect: ไม่พบข้อมูล]
    G37 --> G40{Answer length gt 800 chars?}
    G38 --> G40
    G40 -->|Yes| G41[LLM compress for history: summarize_for_history]
    G40 -->|No| G42[Store full answer in history]
    G41 --> G43[Mark stage = done, set auto_return_to_practical = True]
    G42 --> G43
  end

  subgraph SESSION["Session Management"]
    H1[StateManager.load]
    H1 --> H2{State file exists?}
    H2 -->|No| H3[Return None: create new ConversationState]
    H2 -->|Yes| H4[Load JSON and parse Pydantic model]
    H4 --> H5{pending_slot is a dict?}
    H5 -->|No - malformed| H6[Remove pending_slot from context]
    H5 -->|Yes| H7[Sanitize: sync display_messages if empty]
    H6 --> H8[Return loaded state]
    H7 --> H8
    H8 --> H9[StateManager.save: acquire lock via O_CREAT O_EXCL atomic]
    H9 --> H10{FileExistsError? Lock held?}
    H10 -->|No| H14[Lock acquired]
    H10 -->|Yes| H11{Lock file age gt 15s stale?}
    H11 -->|Yes| H12[Unlink stale lock and retry]
    H11 -->|No| H13{Deadline 5s exceeded?}
    H13 -->|Yes| H15[Raise TimeoutError]
    H13 -->|No| H16[Poll: sleep 50ms and retry]
    H14 --> H17[_trim_state_for_save]
    H17 --> H18{messages gt max_recent 18?}
    H18 -->|Yes| H19[Trim to last 18 messages]
    H18 -->|No| H20[Keep all messages]
    H19 --> H21[Strip ephemeral keys: _broad_retrieval_docs, _multi_topic_retrieval, etc]
    H20 --> H21
    H21 --> H22[Atomic write: temp file then os.rename]
    H22 --> H23[Release lock file]
    H23 --> H24{Session total_tokens gt 80000? _TOKEN_WARN_THRESHOLD line 98}
    H24 -->|No| H30[Continue normally]
    H24 -->|Yes| H25{Academic FSM stage active? _academic_flow.stage in awaiting_slots or awaiting_sections or awaiting_topic line 128}
    H25 -->|Yes - _skip_summarize=True| H26[Trim only: state.trim_messages keep_last=4 line 148]
    H25 -->|No| H27{auto_summarize_if_needed: non_system_messages ge threshold 6? line 137}
    H27 -->|Yes| H28[LLM Haiku: summarize old messages - keep_recent=4 line 139]
    H27 -->|No| H26
    H28 --> H29{Summary returned not None?}
    H29 -->|Yes| H31[state.summarize_old_messages: replace old with summary + keep 4 recent]
    H29 -->|No - summarize failed| H26
    H26 --> H26a[Exception in token handler: also trim_messages keep_last=4 line 151]
    H30 --> H32[StateManager.purge_older_than_days]
    H31 --> H32
    H32 --> H33{Session mtime lt 7-day cutoff?}
    H33 -->|Yes - old session| H34[Delete state file]
    H33 -->|No| H35[Keep session file]
  end

  %% ── Cross-subgraph connections ──────────────────────────────────────────────
  %% API → Pre-processing
  A15 --> B1

  %% Supervisor → Slot Collection
  S21 --> D1

  %% Legal → Slot Collection (when a slot question is set)
  L34 --> D1

  %% Legal → Practical (when no slot needed, answer directly)
  L38 --> F1

  %% Slot Collection done → Practical or Academic
  D27 --> F1
  D27 --> G1

  %% RAG output → Practical docs_json build
  E31 --> F17

  %% Practical / Academic → Session save path
  F42 --> H8
  G43 --> H8

  %% Session management done → API response
  H30 --> A18
```
