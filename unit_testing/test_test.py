import sys
from pathlib import Path

# Set path ke parent folder
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
import pytest
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from module.test import evaluate_model  

@pytest.fixture
def trained_model(tmp_path):
    # Siapkan data dummy (binary classification)
    data = load_iris()
    X, y = data.data, (data.target == 0).astype(int)  # ubah ke biner
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model dan simpan ke file sementara
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return str(model_path), X_train, y_train, X_test, y_test

def test_evaluate_model_from_file(trained_model):
    model_path, X_train, y_train, X_test, y_test = trained_model
    
    y_pred, acc, report, cm, roc_auc_train, roc_auc_test = evaluate_model(
        model_path, X_train, y_train, X_test, y_test,
        save_report=False, save_confusion_matrix=False, save_roc_curve=False
    )

    assert isinstance(acc, float)
    assert len(y_pred) == len(y_test)
    assert "precision" in report
    assert cm.shape == (2, 2)