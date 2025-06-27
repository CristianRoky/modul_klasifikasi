import sys
from pathlib import Path

# Set path ke parent folder
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import unittest
from module.build_model import build_model
from module.test import evaluate_model
import pandas as pd
from sklearn.model_selection import train_test_split

class TestBuildModelAndEvaluation(unittest.TestCase):
    def setUp(self):
        # Dummy dataset: sesuaikan dengan kebutuhan model Anda
        self.data = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
            "feature2": [10, 20, 30, 40, 50, 60,70,80,90,100,100,100,130,140,150,150,170,180,190,200],
            "target":   [0, 1, 0, 1, 0, 1,0,0,0,0,0,0,0,0,1,1,1,1,1,1]
        })
        self.X = self.data[["feature1", "feature2"]]
        self.y = self.data["target"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

    def test_build_model_and_evaluation(self):
        # Build model
        model = build_model(self.X_train, self.y_train, model_type="xgb")

        # Evaluate
        y_pred, acc, report, cm, roc_train, roc_test = evaluate_model(
            model_or_path=model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test
        )

        # Check evaluation outputs
        self.assertIsNotNone(y_pred)
        self.assertGreaterEqual(acc, 0)
        self.assertIn("precision", report.lower())
        self.assertEqual(len(cm.shape), 2)
        self.assertGreater(roc_train, 0)
        self.assertGreater(roc_test, 0)

if __name__ == '__main__':
    unittest.main()
