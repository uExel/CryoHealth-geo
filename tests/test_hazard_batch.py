"""hazard_batch.py orchestrates db reads + dem.py + exposure.py + hazard.py +
hazard_client.py — all mocked/patched here so these tests exercise only its own control
flow, no real DB/network/DEM/WorldPop involved."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from unittest.mock import patch

from pipeline.hazard import LakeStaticInputs
from pipeline.hazard_batch import HazardRunResult, run_hazard_pass


@contextmanager
def _fake_connect():
    yield object()


_UNSET = object()


def _patched(
    lake_id_map,
    static=_UNSET,
    observations=_UNSET,
    slope=15.0,
    exposure=_UNSET,
    report_response=_UNSET,
    report_side_effect=None,
):
    static = static if static is not _UNSET else LakeStaticInputs("moraine", True, False)
    observations = observations if observations is not _UNSET else [(date(2026, 7, 1), 1.0)]
    exposure = exposure if exposure is not _UNSET else {"population_within_buffer": 100.0}
    report_response = report_response if report_response is not _UNSET else {"alert": None}

    report_kwargs = (
        {"side_effect": report_side_effect} if report_side_effect else {"return_value": report_response}
    )

    return (
        patch("pipeline.hazard_batch.connect", _fake_connect),
        patch(
            "pipeline.hazard_batch.lake_id_for_slug",
            side_effect=lambda conn, slug: lake_id_map.get(slug),
        ),
        patch("pipeline.hazard_batch._lake_static_inputs", return_value=static),
        patch("pipeline.hazard_batch._observations", return_value=observations),
        patch("pipeline.hazard_batch.mean_slope_degrees", return_value=slope),
        patch("pipeline.hazard_batch.population_within_buffer", return_value=exposure),
        patch("pipeline.hazard_batch.report_hazard_score", **report_kwargs),
    )


def test_run_hazard_pass_reports_error_when_lake_not_seeded():
    patchers = _patched({})
    with patchers[0], patchers[1]:
        results = run_hazard_pass(lake_slugs=["shishper"])

    assert results == [HazardRunResult("shishper", error="lake not seeded in DB")]


def test_run_hazard_pass_reports_error_when_static_inputs_missing():
    patchers = _patched({"shishper": "lake-1"}, static=None)
    with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5]:
        results = run_hazard_pass(lake_slugs=["shishper"])

    assert results[0].error == "lake not seeded in DB"


def test_run_hazard_pass_computes_and_reports_a_score(monkeypatch):
    monkeypatch.setenv("CRYOHEALTH_API_KEY", "test-key")
    patchers = _patched(
        {"shishper": "lake-1"},
        report_response={"alert": {"id": "alert-1"}},
    )
    p0, p1, p2, p3, p4, p5, p6 = patchers
    with p0, p1, p2, p3, p4, p5, p6 as mock_report:
        results = run_hazard_pass(lake_slugs=["shishper"], as_of=date(2026, 7, 1))

    assert len(results) == 1
    result = results[0]
    assert result.slug == "shishper"
    assert result.error is None
    assert result.score is not None
    assert result.tier is not None
    assert result.alert_created is True
    mock_report.assert_called_once()
    call_kwargs = mock_report.call_args
    assert call_kwargs.args[0] == "lake-1"  # lake_id


def test_run_hazard_pass_alert_created_is_false_when_no_alert_in_response():
    patchers = _patched({"shishper": "lake-1"}, report_response={"alert": None})
    with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5], patchers[6]:
        results = run_hazard_pass(lake_slugs=["shishper"])

    assert results[0].alert_created is False


def test_run_hazard_pass_captures_exceptions_per_lake_without_sinking_the_pass():
    patchers = _patched({"shishper": "lake-1"}, report_side_effect=RuntimeError("API unreachable"))
    with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5], patchers[6]:
        results = run_hazard_pass(lake_slugs=["shishper"])

    assert results[0].error == "API unreachable"
    assert results[0].score is None
