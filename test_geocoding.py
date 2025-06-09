import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from geocoding import create_location

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "latitude": [51.5074, 40.7128],
        "longitude": [-0.1278, -74.0060],
        "location": [pd.NA, pd.NA]
    })

@pytest.fixture
def fake_geocoder():
    mock = MagicMock()
    mock.reverse_geocode.side_effect = [
        [{"components": {"country": "United Kingdom"}}],
        [{"components": {"country": "United States"}}]
    ]
    return [mock]

def test_create_location_with_mocked_geocoder(tmp_path, sample_df, fake_geocoder):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("{}")

    with patch("geocoding.OpenCageGeocode", side_effect=lambda key: fake_geocoder[0]):
        result_df = create_location(
            df=sample_df,
            api_keys=["dummy_key"],
            cache_file=str(cache_file),
            limit=2
        )

    assert "location" in result_df.columns
    assert result_df.loc[0, "location"] == "United Kingdom"
    assert result_df.loc[1, "location"] == "United States"

def test_create_location_with_existing_cache(tmp_path, sample_df):
    cache = {str((51.5074, -0.1278)): "Cached UK"}
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(str(cache).replace("'", '"'))

    with patch("geocoding.OpenCageGeocode"):
        result_df = create_location(
            df=sample_df,
            api_keys=["dummy_key"],
            cache_file=str(cache_file),
            limit=1
        )

    assert result_df.loc[0, "location"] == "Cached UK"

def test_create_location_with_nan_coordinates(tmp_path):
    df = pd.DataFrame({
        "latitude": [None, 10.0],
        "longitude": [105.0, None],
        "location": [pd.NA, pd.NA]
    })

    cache_file = tmp_path / "cache.json"
    cache_file.write_text("{}")

    with patch("geocoding.OpenCageGeocode"):
        result_df = create_location(
            df=df,
            api_keys=["dummy_key"],
            cache_file=str(cache_file),
            limit=2
        )

    assert result_df.loc[0, "location"] == "Unknown"
    assert result_df.loc[1, "location"] == "Unknown"
