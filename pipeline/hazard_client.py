"""HTTP client for POST /alerts/hazard-scores on CryoHealth-api — unlike Observations
(direct psycopg insert), hazard scores go through the API because tier-transition and
alert/notification logic lives there (CryoHealth-api's ARCHITECTURE.md: "Tier policy is
code + humans, never silent ML"). Auth is the GEO_SERVICE_API_KEY service key
(CryoHealth-api#12) sent as x-api-key — this service has no user to log in as.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import httpx


def _base_url() -> str:
    return os.environ.get("CRYOHEALTH_API_URL", "http://localhost:3000")


def _api_key() -> str:
    key = os.environ.get("CRYOHEALTH_API_KEY")
    if not key:
        raise RuntimeError("CRYOHEALTH_API_KEY must be set to report hazard scores")
    return key


def report_hazard_score(
    lake_id: str,
    run_id: str,
    score: float,
    tier: str,
    components: dict,
    computed_at: datetime | None = None,
) -> dict:
    """POSTs a hazard score to CryoHealth-api. Raises on any non-2xx response — a failed
    report must be visible (surfaced as a per-lake error by the caller), never silently
    swallowed."""
    payload = {
        "lakeId": lake_id,
        "runId": run_id,
        "score": score,
        "tier": tier,
        "components": components,
        "computedAt": (computed_at or datetime.now(UTC)).isoformat(),
    }
    resp = httpx.post(
        f"{_base_url()}/alerts/hazard-scores",
        json=payload,
        headers={"x-api-key": _api_key()},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()
