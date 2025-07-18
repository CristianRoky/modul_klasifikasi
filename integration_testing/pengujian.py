import sys
from pathlib import Path

# Tambah path agar bisa mengimpor modul dari folder parent
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import unittest
from module.build_model import build_model
from module.test import evaluate_model
from module.validation import validate_model  # Untuk skenario pertama
import pandas as pd
from sklearn.model_selection import train_test_split

class TestBuildModelAndEvaluation(unittest.TestCase):

    def setUp(self):
        # Dataset dummy sederhana untuk klasifikasi
        self.data = pd.DataFrame({
            "feature1": list(range(1, 21)),
            "feature2": [10 * i for i in range(1, 21)],
            "target":   [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        })

        self.X = self.data[["feature1", "feature2"]]
        self.y = self.data["target"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )

    def test_model_build_validation_and_test(self):
        """
        Pengujian ini bertujuan untuk memastikan model dapat dibangun,
        divalidasi, dan diuji menggunakan data pengujian.
        """
        # Validasi sebelum evaluasi
        val_result = validate_model(
            X=self.X_train,
            y=self.y_train,
            numeric_features=None,
            algo="xgb",
            best_param_path=None,
            cv_splits=3,
            save_plot_path=None
        )
        self.assertIn("F1_0", val_result)

        # Build model setelah validasi
        model = build_model(
            X_train=self.X_train,
            y_train=self.y_train,
            model_type="xgb"
        )

        # Evaluasi pada data uji
        y_pred, acc, report, cm, roc_train, roc_test = evaluate_model(
            model_or_path=model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test
        )

        self.assertIsNotNone(y_pred)
        self.assertGreaterEqual(acc, 0)
        self.assertIn("precision", report.lower())
        self.assertEqual(len(cm.shape), 2)
        self.assertGreater(roc_train, 0)
        self.assertGreater(roc_test, 0)

    def test_model_build_and_test_without_validation(self):
        """
        Pengujian ini bertujuan untuk memastikan model dapat dibangun
        dan diuji, tanpa melibatkan proses validasi.
        """
        # Bangun model langsung tanpa validasi
        model = build_model(
            X_train=self.X_train,
            y_train=self.y_train,
            model_type="xgb",
            model_params={"max_depth": 2}
        )

        # Evaluasi
        y_pred, acc, report, cm, roc_train, roc_test = evaluate_model(
            model_or_path=model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test
        )

        self.assertIsNotNone(y_pred)
        self.assertGreaterEqual(acc, 0)
        self.assertIn("precision", report.lower())
        self.assertEqual(len(cm.shape), 2)
        self.assertGreater(roc_train, 0)
        self.assertGreater(roc_test, 0)

if __name__ == '__main__':
    unittest.main()
