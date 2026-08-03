"""hazard_client.py tests run against a mocked httpx.post — no real CryoHealth-api
instance. The real endpoint contract (auth header, response shape) was verified live
against a running CryoHealth-api instance before this file existed (see
docs/ai/HANDOFF.md); these tests guard the request shape going forward."""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pytest

from pipeline.hazard_client import report_hazard_score


def test_report_hazard_score_sends_the_api_key_header_and_full_payload(monkeypatch):
    monkeypatch.setenv("CRYOHEALTH_API_KEY", "test-key-value")
    monkeypatch.setenv("CRYOHEALTH_API_URL", "http://api.example.test")
    fake_resp = MagicMock(status_code=201)
    fake_resp.json.return_value = {"alert": None, "deduped": False}
    fake_resp.raise_for_status.return_value = None

    with patch("pipeline.hazard_client.httpx.post", return_value=fake_resp) as mock_post:
        result = report_hazard_score(
            lake_id="lake-1",
            run_id="run-1",
            score=0.5,
            tier="watch",
            components={"a": 1},
            computed_at=datetime(2026, 7, 20, tzinfo=UTC),
        )

    assert result == {"alert": None, "deduped": False}
    args, kwargs = mock_post.call_args
    assert args[0] == "http://api.example.test/alerts/hazard-scores"
    assert kwargs["headers"]["x-api-key"] == "test-key-value"
    assert kwargs["json"] == {
        "lakeId": "lake-1",
        "runId": "run-1",
        "score": 0.5,
        "tier": "watch",
        "components": {"a": 1},
        "computedAt": "2026-07-20T00:00:00+00:00",
    }


def test_report_hazard_score_raises_without_an_api_key(monkeypatch):
    monkeypatch.delenv("CRYOHEALTH_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CRYOHEALTH_API_KEY"):
        report_hazard_score("lake-1", "run-1", 0.5, "watch", {})


def test_report_hazard_score_raises_on_a_non_2xx_response(monkeypatch):
    import httpx

    monkeypatch.setenv("CRYOHEALTH_API_KEY", "test-key-value")
    fake_resp = MagicMock(status_code=403)
    fake_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Forbidden", request=MagicMock(), response=fake_resp
    )

    with patch("pipeline.hazard_client.httpx.post", return_value=fake_resp):
        with pytest.raises(httpx.HTTPStatusError):
            report_hazard_score("lake-1", "run-1", 0.5, "watch", {})


def test_report_hazard_score_defaults_computed_at_to_now(monkeypatch):
    monkeypatch.setenv("CRYOHEALTH_API_KEY", "test-key-value")
    fake_resp = MagicMock(status_code=201)
    fake_resp.json.return_value = {}
    fake_resp.raise_for_status.return_value = None

    with patch("pipeline.hazard_client.httpx.post", return_value=fake_resp) as mock_post:
        report_hazard_score("lake-1", "run-1", 0.5, "watch", {})

    computed_at = mock_post.call_args.kwargs["json"]["computedAt"]
    assert computed_at.startswith(date.today().isoformat())
