"""
Shared auth dependency for admin/debug endpoints (code/router/admin.py and
the debug/* routes in code/router/monitoring.py). Those routes expose raw
session content, logs, and destructive actions (metrics reset) and must not
be reachable without the X-Admin-Key header matching conf.ADMIN_API_KEY.

ADMIN_AUTH_DISABLED (2026-09) — an explicit, opt-in escape hatch for the
internal-team PDF test instance (public IP, but team-only conversations, no
real end users): that instance wanted the admin dashboard reachable with no
key at all, for quickly eyeballing what people are chatting about. Defaults
to False (the secure, fail-closed behavior below is unchanged) — must be
deliberately set per-deployment, never a project-wide default. Removes
protection from ALL routes behind this dependency, not just the dashboard
(also the debug/* routes in router/monitoring.py, including metrics reset)
— acceptable for a low-stakes internal test instance, NOT for anything a
real user's data could reach.
"""

import secrets

from fastapi import Header, HTTPException

import conf as _conf


async def require_admin_key(x_admin_key: str = Header(default="")) -> None:
    if getattr(_conf, "ADMIN_AUTH_DISABLED", False):
        return
    expected = getattr(_conf, "ADMIN_API_KEY", "") or ""
    # Fail closed: an unset key must lock the endpoint down, not open it up.
    if not expected or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Key")
