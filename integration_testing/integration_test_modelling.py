import sys
from pathlib import Path

# Set path ke parent folder
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import os

from module.hyperparameter_tuning import tune_model
from module.validation import validate_model
from module.build_model import build_model

class TestModelPipelineIntegration(unittest.TestCase):

    def setUp(self):
        # Buat data dummy klasifikasi
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5,
                                   n_redundant=2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            pd.DataFrame(X), pd.Series(y), test_size=0.2, random_state=42, stratify=y
        )

    def test_tuning_validation_build(self):
        # Step 1: Tuning
        best_params = tune_model(
            X_train=self.X_train,
            y_train=self.y_train,
            numeric_features=None,
            algo="xgb",
            resampler=None,
            scaler=None,
            imputer_strategy=None,
            cv_splits=3,
            random_state=42,
            init_points=2,
            n_iter=2,
            verbose=0,
            return_scores=False,
            save_best_param_path="test_best_params.pkl"
        )
        self.assertIsInstance(best_params, dict)
        self.assertTrue("max_depth" in best_params or len(best_params) > 0)

        # Step 2: Validation
        val_result = validate_model(
            X=self.X_train,
            y=self.y_train,
            numeric_features=None,
            algo="xgb",
            best_param_path="test_best_params.pkl",
            cv_splits=3,
            save_plot_path="output_it"
        )
       
        self.assertIn("F1_0", val_result)
        self.assertGreaterEqual(val_result["F1_0"].iloc[0], 0)

        # Step 3: Build Final Model
        model = build_model(
            X_train=self.X_train,
            y_train=self.y_train,
            numeric_features=None,
            model_type="xgb",
            best_param_path="test_best_params.pkl",
            imputer_strategy=None,
            resample_type=None,
            save_model=False  # True jika ingin menyimpan
        )
        self.assertTrue(hasattr(model, "fit"))

    def tearDown(self):
        # Bersihkan file parameter jika dibuat
        if os.path.exists("test_best_params.pkl"):
            os.remove("test_best_params.pkl")

if __name__ == "__main__":
    unittest.main()