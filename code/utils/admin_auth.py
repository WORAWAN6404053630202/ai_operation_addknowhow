"""
Shared auth dependency for admin/debug endpoints (code/router/admin.py and
the debug/* routes in code/router/monitoring.py). Those routes expose raw
session content, logs, and destructive actions (metrics reset) and must not
be reachable without the X-Admin-Key header matching conf.ADMIN_API_KEY.
"""

import secrets

from fastapi import Header, HTTPException

import conf as _conf


async def require_admin_key(x_admin_key: str = Header(default="")) -> None:
    expected = getattr(_conf, "ADMIN_API_KEY", "") or ""
    # Fail closed: an unset key must lock the endpoint down, not open it up.
    if not expected or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Key")
