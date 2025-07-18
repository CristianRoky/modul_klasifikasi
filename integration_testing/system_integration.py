import sys
from pathlib import Path

# Set path ke folder parent
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import unittest
from integration_testing.main_pipeline import run_pipeline  # Fungsi ini harus memanggil ke-9 modul, dan return model + data
from module.test import evaluate_model  # Asumsinya evaluasi terpisah di modul test/

class TestFullIntegration(unittest.TestCase):
    
    def test_full_pipeline_and_evaluation(self):
        # Jalankan semua modul dan dapatkan model + dataset
        model, X_train, y_train, X_test, y_test = run_pipeline("Data_Tsunami.xlsx")
        
        # Evaluasi model
        y_pred, acc, report, cm, roc_train, roc_test = evaluate_model(
            model_or_path=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        # Cek metrik evaluasi
        self.assertIsNotNone(y_pred)
        self.assertGreaterEqual(acc, 0)
        self.assertIn("precision", report.lower())
        self.assertEqual(len(cm.shape), 2)
        self.assertGreater(roc_train, 0)
        self.assertGreater(roc_test, 0)

if __name__ == '__main__':
    unittest.main()
