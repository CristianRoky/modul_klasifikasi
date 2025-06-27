import sys
from pathlib import Path

# Set path ke parent folder
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
import pytest
import pandas as pd
from unittest.mock import patch
from sklearn.datasets import make_classification
from module.hyperparameter_tuning import tune_model

# Buat data dummy
@pytest.fixture
def dummy_data():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    y = pd.Series(y, name="target")
    return X, y

# Mock BayesianOptimization.maximize agar tidak benar-benar melakukan BO
@patch("module.hyperparameter_tuning.BayesianOptimization.maximize")
@patch("module.hyperparameter_tuning.BayesianOptimization.__init__", return_value=None)
@patch("module.hyperparameter_tuning.BayesianOptimization.max", new_callable=lambda: {"params": {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1}})
def test_tune_model_rf(mock_bo_max, mock_bo_init, mock_bo_maximize, dummy_data):
    X, y = dummy_data

    # Panggil fungsi dengan parameter minimum
    best_params = tune_model(
        X_train=X,
        y_train=y,
        numeric_features=["feat_0", "feat_1", "feat_2"],
        algo="rf",
        scaler="robust",
        imputer_strategy="median",
        resampler="smoteenn",
        init_points=1,
        n_iter=1,
        verbose=0,
        save_best_param_path="dummy_params.json"
    )

    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert best_params["n_estimators"] == 100

@patch("module.hyperparameter_tuning.BayesianOptimization.maximize")
@patch("module.hyperparameter_tuning.BayesianOptimization.__init__", return_value=None)
@patch("module.hyperparameter_tuning.BayesianOptimization.max", new_callable=lambda: {"params": {"learning_rate": 0.23911943547917341, "max_depth": 5, "min_child_weight": 3, "n_estimators": 810}})

def test_tune_model_xgb(mock_bo_max, mock_bo_init, mock_bo_maximize, dummy_data):
    X, y = dummy_data

    # Panggil fungsi dengan parameter minimum
    best_params_xgb = tune_model(
        X_train=X,
        y_train=y,
        numeric_features=["feat_0", "feat_1", "feat_2"],
        algo="xgb",
        scaler="robust",
        imputer_strategy="median",
        resampler="smoteenn",
        init_points=1,
        n_iter=1,
        verbose=0,
        save_best_param_path="dummy_xgb_params.json"
    )

    assert isinstance(best_params_xgb, dict)
    assert "n_estimators" in best_params_xgb
    assert best_params_xgb["n_estimators"] == 810
