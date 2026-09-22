"""API key authentication for the write-side routes (/predict, /predict/batch).

Set WAFER_API_KEY to require a matching `X-API-Key` header on those routes.
Leaving it unset disables auth, for local development only — the app logs a
warning at startup so this is never silently insecure.
"""
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_header)):
    expected = os.environ.get("WAFER_API_KEY")
    if expected is None:
        return  # auth disabled — local dev only, see module docstring
    if key != expected:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
