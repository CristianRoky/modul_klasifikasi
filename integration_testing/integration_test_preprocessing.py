import sys
from pathlib import Path

# Set path ke parent folder
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import unittest
import pandas as pd

from module.source_counter import count_sources
from module.geocoding import create_location
from module.encoding import encode

class TestPreprocessingIntegration(unittest.TestCase):

    def setUp(self):
        # Data dummy untuk tes
        self.df = pd.DataFrame({
            "sources": ["abc,def", "ghi", "abc"],
            "latitude": [1.23, 4.56, 7.89],
            "longitude": [101.1, 102.2, 103.3],
            "magType": ["mb", "ml", "mb"],
            "net": ["us", "ci", "us"]
        })

    def test_integration_source_geocode_encode(self):
        # Step 1: Source Counter
        df_counted = count_sources(self.df.copy(), column_name="sources")
        self.assertIn("source_counter", df_counted.columns)

        # Step 2: Geocoding with cache only (tidak request ke API)
        df_geo = create_location(
            df_counted.copy(),
            api_keys=["DUMMY_KEY"],
            lat_col="latitude",
            lon_col="longitude",
            loc_col="location",
            limit=100,
            save_as=None,
            cache_file="it_geocode_cache.json"
        )
        self.assertIn("location", df_geo.columns)
        self.assertTrue(df_geo["location"].notnull().all())

        # Step 3: Encoding
        X_train_encoded, _ = encode(
            df_geo.copy(),
            columns=["magType", "net", "location"],
            X_test=None,
            method="ordinal",
            prefix="testenc",
            save_path="./saved_encoders",
            save_encoded=False
        )
        self.assertTrue(all(col in X_train_encoded.columns for col in ["magType", "net", "location"]))
        self.assertFalse(X_train_encoded.isnull().any().any())

if __name__ == "__main__":
    unittest.main()