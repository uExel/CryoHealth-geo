"""CdseSource tests run against mocked HTTP — no real network, no real credentials.
Everything here was checked against the real CDSE API by hand before this file existed
(see docs/ai/PLAN.md and pipeline/cdse_source.py's module docstring for what was
verified live and how); these tests guard the request/response shape going forward."""

from __future__ import annotations

import io
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds

from pipeline.cdse_source import CdseAuthError, CdseSource, _read_multiband_tiff
from pipeline.stac_source import SceneRef


def test_missing_credentials_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("CDSE_CLIENT_SECRET", raising=False)
    with pytest.raises(CdseAuthError, match="CDSE_CLIENT_ID"):
        CdseSource()


@patch("pipeline.cdse_source.Client.open")
def test_token_request_failure_surfaces_the_real_api_error(mock_stac_open):
    source = CdseSource(client_id="id", client_secret="bad-secret")
    fake_resp = MagicMock(status_code=401, text='{"error":"invalid_client"}')
    with patch("pipeline.cdse_source.requests.post", return_value=fake_resp):
        with pytest.raises(CdseAuthError, match="invalid_client"):
            source._access_token()


@patch("pipeline.cdse_source.Client.open")
def test_token_is_cached_across_calls_within_its_lifetime(mock_stac_open):
    source = CdseSource(client_id="id", client_secret="secret")
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"access_token": "tok-1", "expires_in": 1800}
    with patch("pipeline.cdse_source.requests.post", return_value=fake_resp) as mock_post:
        first = source._access_token()
        second = source._access_token()
    assert first == second == "tok-1"
    mock_post.assert_called_once()


def _fake_tiff_bytes(bands: dict[str, np.ndarray]) -> bytes:
    arrays = list(bands.values())
    height, width = arrays[0].shape
    transform = transform_from_bounds(74.6, 36.39, 74.62, 36.41, width, height)
    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(arrays),
        dtype=arrays[0].dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(arr, i)
    return buf.getvalue()


def test_read_multiband_tiff_maps_bytes_back_to_named_bands():
    b03 = np.full((10, 12), 500, dtype=np.uint16)
    b08 = np.full((10, 12), 300, dtype=np.uint16)
    scl = np.full((10, 12), 4, dtype=np.uint16)
    tiff = _fake_tiff_bytes({"B03": b03, "B08": b08, "SCL": scl})

    result = _read_multiband_tiff(tiff, ["B03", "B08", "SCL"])

    assert set(result) == {"B03", "B08", "SCL"}
    assert np.array_equal(result["B03"], b03)
    assert np.array_equal(result["B08"], b08)
    assert np.array_equal(result["SCL"], scl)
    # The whole point of the Process API path: every band lands on one shared grid.
    assert result["B03"].shape == result["B08"].shape == result["SCL"].shape


@patch("pipeline.cdse_source.Client.open")
def test_find_recent_scenes_builds_scene_refs_from_stac_items(mock_stac_open):
    mock_item = MagicMock()
    mock_item.id = "S2A_TEST_SCENE"
    mock_item.datetime.date.return_value = date(2026, 7, 20)
    mock_item.properties = {"eo:cloud_cover": 5.0}
    mock_search = MagicMock()
    mock_search.items.return_value = [mock_item]
    mock_stac_open.return_value.search.return_value = mock_search

    source = CdseSource(client_id="id", client_secret="secret")
    scenes = source.find_recent_scenes((74.6, 36.39, 74.62, 36.41), limit=12)

    assert scenes == [
        SceneRef(scene_id="S2A_TEST_SCENE", captured_at=date(2026, 7, 20), cloud_cover_pct=5.0, assets={})
    ]
