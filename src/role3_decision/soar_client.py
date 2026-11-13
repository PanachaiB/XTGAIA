"""
HTTP clients for Shuffle (SOAR webhooks) and TheHive 5 (case management).

All endpoints and credentials come from environment variables (loaded from
.env by docker compose `env_file`), read at call time, never hardcoded:

  SHUFFLE_BASE_URL               e.g. http://shuffle-backend:5001 inside Docker
  SHUFFLE_EXECUTION_WEBHOOK_URL  full URL, '/api/v1/hooks/webhook_<id>' or just 'webhook_<id>'
  SHUFFLE_ROLLBACK_WEBHOOK_URL   same formats, for the reverse playbook
  THEHIVE_URL                    e.g. http://thehive:9000 inside Docker
  THEHIVE_API_KEY                TheHive API key (sent as Bearer token, never logged)

Shuffle webhooks are served by the backend (port 5001) at
/api/v1/hooks/webhook_<generated-id>; the UI on 3001 only proxies /api.
If a variable is missing the call is skipped with a warning (fail closed).
"""

import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger("XTGAIA-SOAR-Client")
TIMEOUT = float(os.getenv("XTGAIA_HTTP_TIMEOUT", "15"))   # TheHive case creation can exceed 5 s under load
_IN_CONTAINER = os.path.exists("/.dockerenv")


def _env(name: str) -> Optional[str]:
    val = os.getenv(name)
    if val is None:
        return None
    val = val.strip().strip('"').strip("'").strip()
    return val or None


def _warn_localhost(url: str, name: str) -> None:
    if _IN_CONTAINER and urlparse(url).hostname in ("localhost", "127.0.0.1"):
        logger.warning("%s points at %s from inside a container; use the compose service name instead.", name, url)


def shuffle_webhook_url(kind: str) -> Optional[str]:
    """Resolve the Shuffle webhook URL for kind='execution' or 'rollback'."""
    names = {"execution": ["SHUFFLE_EXECUTION_WEBHOOK_URL", "SHUFFLE_WEBHOOK_URL"],
             "rollback": ["SHUFFLE_ROLLBACK_WEBHOOK_URL"]}[kind]
    value = next((v for v in (_env(n) for n in names) if v), None)
    if value is None:
        return None
    if not value.startswith(("http://", "https://")):
        base = _env("SHUFFLE_BASE_URL")
        if base is None:
            logger.warning("Relative Shuffle webhook %r given but SHUFFLE_BASE_URL is not set", value)
            return None
        path = value if value.startswith("/") else f"/api/v1/hooks/{value}"
        value = base.rstrip("/") + path
    _warn_localhost(value, names[0])
    return value


def post_shuffle(kind: str, payload: Dict[str, Any]) -> bool:
    url = shuffle_webhook_url(kind)
    if url is None:
        logger.warning("Shuffle %s webhook not configured; skipping dispatch.", kind)
        return False
    try:
        res = requests.post(url, json=payload, timeout=TIMEOUT)
    except requests.RequestException as exc:
        logger.error("Shuffle %s webhook unreachable: %s", kind, exc)
        return False
    if res.status_code in (200, 201, 202):
        return True
    logger.warning("Shuffle %s webhook returned HTTP %s", kind, res.status_code)
    return False


def create_thehive_case(title: str, description: str, severity: int, tags) -> Optional[str]:
    """Create a TheHive 5 case; returns its id, or None if not configured/failed."""
    base, key = _env("THEHIVE_URL"), _env("THEHIVE_API_KEY")
    if base is None or key is None:
        logger.warning("THEHIVE_URL / THEHIVE_API_KEY not configured; skipping case creation.")
        return None
    url = base.rstrip("/") + "/api/v1/case"
    _warn_localhost(url, "THEHIVE_URL")
    body = {"title": title, "description": description, "severity": int(severity), "flag": True, "tags": list(tags)}
    try:
        res = requests.post(url, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT)
    except requests.RequestException as exc:
        logger.error("TheHive unreachable: %s", exc)
        return None
    if res.status_code in (200, 201):
        try:
            return str(res.json().get("_id") or res.json().get("id"))
        except ValueError:
            return None
    logger.warning("TheHive case creation returned HTTP %s", res.status_code)
    return None
