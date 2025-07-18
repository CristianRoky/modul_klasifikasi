import sys
from pathlib import Path

# Tambah path ke folder parent agar modul bisa diimpor
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import unittest
import pandas as pd

from module.source_counter import count_sources
from module.geocoding import create_location
from module.encoding import encode
from module.feature_selection import feature_selection_based_on_feature_importance
from module.build_model import build_model


class TestPreprocessingIntegration(unittest.TestCase):

    def setUp(self):
        # Data dummy aman untuk semua pengujian
        self.df = pd.DataFrame({
            "sources": ["abc,def", "ghi", "abc", "def", "abc,ghi", "ghi", "def,abc", "abc"],
            "latitude": [1.23, 4.56, 7.89, 2.34, 5.67, 3.21, 6.78, 2.22],
            "longitude": [101.1, 102.2, 103.3, 104.4, 105.5, 106.6, 107.7, 108.8],
            "magType": ["mb", "ml", "mb", "ml", "mb", "mb", "ml", "mb"],
            "net": ["us", "ci", "us", "ci", "ci", "us", "ci", "us"],
            "label": [0, 1, 0, 1, 0, 0, 1, 0]
        })

    def test_full_preprocessing_pipeline(self):
        """
        Skenario 1: Melibatkan seluruh modul preprocessing:
        source_counter.py, geocoding.py, encoding.py, feature_selection.py
        """
        df = self.df.copy()

        # Source Counter
        df = count_sources(df, column_name="sources")
        self.assertIn("source_counter", df.columns)

        # Geocoding (gunakan cache dummy agar tidak request API)
        df = create_location(
            df,
            api_keys=["DUMMY_KEY"],
            lat_col="latitude",
            lon_col="longitude",
            loc_col="location",
            limit=10,
            save_as=None,
            cache_file="it_geocode_cache.json"  # pastikan file ada atau isi {}
        )
        self.assertIn("location", df.columns)
        self.assertTrue(df["location"].notnull().all())

        # Encoding
        X_encoded, _ = encode(
            df.copy(),
            columns=["magType", "net", "location"],
            X_test=None,
            method="ordinal",
            prefix="testenc",
            save_path="./saved_encoders",
            save_encoded=False
        )
        self.assertIsInstance(X_encoded, pd.DataFrame)
        self.assertFalse(X_encoded.isnull().any().any())

        # Feature Selection
        model, X_selected = feature_selection_based_on_feature_importance(X_encoded, df["label"],top_n=2)
        self.assertIsInstance(X_selected, pd.DataFrame)

        # Gunakan fitur penting dari X_selected.columns
        selected_features = X_selected.columns.tolist()
        self.assertGreater(len(selected_features), 0)
        self.assertEqual(X_selected.shape[1], len(selected_features))

    def test_source_counter_geocoding_only(self):
        """
        Skenario 2: Hanya melibatkan source_counter.py dan geocoding.py
        """
        df = self.df.copy()

        df = count_sources(df, column_name="sources")
        self.assertIn("source_counter", df.columns)

        df = create_location(
            df,
            api_keys=["DUMMY_KEY"],
            lat_col="latitude",
            lon_col="longitude",
            loc_col="location",
            limit=100,
            save_as=None,
            cache_file="it_geocode_cache.json"
        )
        self.assertIn("location", df.columns)
        self.assertTrue(df["location"].notnull().all())

    def test_encoding_and_build_model_only(self):
        """
        Skenario 3: Hanya melibatkan encoding.py dan langsung ke build_model.py
        Tanpa modul source_counter, geocoding, atau feature_selection.
        """
        df = self.df.copy()
        X_raw = df.drop(columns=["label"])

        # Buang kolom yang tidak digunakan dalam model
        X_raw = X_raw.drop(columns=["sources"])  # FIX untuk ValueError

        y = df["label"]

        # Encoding kolom kategorikal yang diperlukan model
        X_encoded, _ = encode(
            X_raw.copy(),
            columns=["magType", "net"],  # Kolom kategorikal yang digunakan
            X_test=None,
            method="ordinal",
            prefix="testenc3",
            save_path="./saved_encoders",
            save_encoded=False
        )

        # Fitur numerik dan kategorikal (setelah encoding)
        numeric_features = ["latitude", "longitude"]
        category_cols = ["magType", "net"]

        model_pipeline = build_model(
            X_train=X_encoded,
            y_train=y,
            numeric_features=numeric_features,
            category_cols=category_cols,
            model_type="xgb",
            imputer_strategy="median",
            scaler_type="robust",
            resample_type=None,
            model_params={"max_depth": 2},
            save_model=False
        )

        self.assertIsNotNone(model_pipeline)
        self.assertTrue(hasattr(model_pipeline, "predict"))

if __name__ == "__main__":
    unittest.main()