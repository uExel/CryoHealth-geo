"""hazard_batch.py orchestrates db reads + dem.py + exposure.py + hazard.py +
hazard_client.py — all mocked/patched here so these tests exercise only its own control
flow, no real DB/network/DEM/WorldPop involved."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import date
from unittest.mock import patch

from pipeline.hazard import LakeStaticInputs
from pipeline.hazard_batch import run_hazard_pass
from pipeline.meteo import MeteoInputs, MeteoUnavailableError


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
    meteo=_UNSET,
    meteo_side_effect=None,
    report_response=_UNSET,
    report_side_effect=None,
):
    static = static if static is not _UNSET else LakeStaticInputs("moraine", True, False)
    observations = observations if observations is not _UNSET else [(date(2026, 7, 1), 1.0)]
    exposure = exposure if exposure is not _UNSET else {"population_within_buffer": 100.0}
    meteo = (
        meteo
        if meteo is not _UNSET
        else MeteoInputs(
            max_temp_anomaly_3d_c=1.5,
            precip_14d_mm=10.0,
            freezing_level_m=4200.0,
            source="test",
            fetched_at="2026-09-03T00:00:00Z",
        )
    )
    report_response = report_response if report_response is not _UNSET else {"alert": None}

    meteo_kwargs = (
        {"side_effect": meteo_side_effect} if meteo_side_effect else {"return_value": meteo}
    )
    report_kwargs = (
        {"side_effect": report_side_effect} if report_side_effect else {"return_value": report_response}
    )

    # Return a plain list — callers use _run_with_patchers() rather than
    # indexing into this list directly, so order changes here are safe.
    # report_hazard_score must be last so mocks[-1] is always mock_report.
    return [
        patch("pipeline.hazard_batch.connect", _fake_connect),
        patch(
            "pipeline.hazard_batch.lake_id_for_slug",
            side_effect=lambda conn, slug: lake_id_map.get(slug),
        ),
        patch("pipeline.hazard_batch._lake_static_inputs", return_value=static),
        patch("pipeline.hazard_batch._observations", return_value=observations),
        patch("pipeline.hazard_batch.mean_slope_degrees", return_value=slope),
        patch("pipeline.hazard_batch.population_within_buffer", return_value=exposure),
        patch("pipeline.hazard_batch.fetch_meteo_inputs", **meteo_kwargs),
        patch("pipeline.hazard_batch.report_hazard_score", **report_kwargs),
    ]


def _run_with_patchers(patchers, **kwargs):
    """Enter all patchers via ExitStack, run run_hazard_pass, return (results, mocks).

    Using ExitStack means adding or removing a patcher in _patched() never
    requires updating every test's 'with' line — a silent off-by-one with
    the old tuple-index approach caused the static-inputs test to skip
    patcher[6] entirely.
    """
    with ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in patchers]
        results = run_hazard_pass(**kwargs)
    return results, mocks


def test_run_hazard_pass_reports_error_when_lake_not_seeded():
    patchers = _patched({})
    results, _ = _run_with_patchers(patchers, lake_slugs=["shishper"])

    assert len(results) == 1
    assert results[0].slug == "shishper"
    assert results[0].error == "lake not seeded in DB"
    assert results[0].computed_at == ""


def test_run_hazard_pass_reports_error_when_static_inputs_missing():
    patchers = _patched({"shishper": "lake-1"}, static=None)
    results, _ = _run_with_patchers(patchers, lake_slugs=["shishper"])

    assert len(results) == 1
    assert "static fields" in results[0].error
    assert "NULL" in results[0].error
    assert results[0].computed_at == ""


def test_run_hazard_pass_computes_and_reports_a_score(monkeypatch):
    monkeypatch.setenv("CRYOHEALTH_API_KEY", "test-key")
    patchers = _patched(
        {"shishper": "lake-1"},
        report_response={"alert": {"id": "alert-1"}},
    )
    results, mocks = _run_with_patchers(
        patchers, lake_slugs=["shishper"], as_of=date(2026, 7, 1)
    )
    mock_report = mocks[-1]  # report_hazard_score is always the last patcher

    assert len(results) == 1
    result = results[0]
    assert result.slug == "shishper"
    assert result.error is None
    assert result.score is not None
    assert result.tier is not None
    assert result.alert_created is True
    assert result.computed_at != ""  # timestamp populated on success
    mock_report.assert_called_once()
    call_kwargs = mock_report.call_args
    assert call_kwargs.args[0] == "lake-1"  # lake_id


def test_run_hazard_pass_alert_created_is_false_when_no_alert_in_response():
    patchers = _patched({"shishper": "lake-1"}, report_response={"alert": None})
    results, _ = _run_with_patchers(patchers, lake_slugs=["shishper"])

    assert results[0].alert_created is False
    assert results[0].computed_at != ""  # still a successful run


def test_run_hazard_pass_captures_exceptions_per_lake_without_sinking_the_pass():
    patchers = _patched({"shishper": "lake-1"}, report_side_effect=RuntimeError("API unreachable"))
    results, _ = _run_with_patchers(patchers, lake_slugs=["shishper"])

    assert results[0].error == "API unreachable"
    assert results[0].score is None
    assert results[0].computed_at == ""  # error run — no timestamp


def test_run_hazard_pass_includes_meteo_in_reported_components():
    """When weather is available, meteo raw inputs are included in reported components."""
    custom_meteo = MeteoInputs(
        max_temp_anomaly_3d_c=3.2,
        precip_14d_mm=45.0,
        freezing_level_m=4900.0,
        source="open-meteo-forecast",
        fetched_at="2026-09-03T00:00:00Z",
    )
    patchers = _patched({"shishper": "lake-1"}, meteo=custom_meteo)
    results, mocks = _run_with_patchers(patchers, lake_slugs=["shishper"])
    mock_report = mocks[-1]

    assert results[0].error is None
    call_kwargs = mock_report.call_args
    reported_components = call_kwargs.args[4]
    assert reported_components["raw_inputs"]["meteo"]["max_temp_anomaly_3d_c"] == 3.2
    assert reported_components["normalized_risks"]["thermal"] > 0.0


def test_run_hazard_pass_gracefully_degrades_when_meteo_unavailable():
    """When Open-Meteo raises MeteoUnavailableError, pass succeeds with thermal=0."""
    patchers = _patched(
        {"shishper": "lake-1"},
        meteo_side_effect=MeteoUnavailableError("Open-Meteo API down"),
    )
    results, mocks = _run_with_patchers(patchers, lake_slugs=["shishper"])
    mock_report = mocks[-1]

    assert results[0].error is None
    assert results[0].score is not None
    call_kwargs = mock_report.call_args
    reported_components = call_kwargs.args[4]
    assert reported_components["raw_inputs"]["meteo"] is None
    assert reported_components["normalized_risks"]["thermal"] == 0.0

