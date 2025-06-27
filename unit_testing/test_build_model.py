import sys
from pathlib import Path

# Set path ke parent folder
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
import pytest
import pandas as pd
from sklearn.datasets import make_classification
from module.build_model import build_model  # Ganti 'model_module' dengan nama file Python yang memuat fungsi build_model

# --- Fixture: Dummy dataset ---
@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )
    df_X = pd.DataFrame(X, columns=['num1', 'num2', 'num3', 'num4'])
    df_y = pd.Series(y)
    return df_X, df_y

# --- Test: Basic model building ---
def test_build_model_default(sample_data):
    X, y = sample_data
    model = build_model(
        X_train=X,
        y_train=y,
        numeric_features=['num1', 'num2', 'num3', 'num4'],
        model_type='xgb',
        imputer_strategy=None,
        scaler_type=None,
        resample_type=None
    )

    # Assert pipeline has fit and predict
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')

    # Make prediction
    preds = model.predict(X)
    assert len(preds) == len(X)

# --- Test: Model with SMOTE ---
def test_build_model_with_smoteenn(sample_data):
    X, y = sample_data
    model = build_model(
        X_train=X,
        y_train=y,
        numeric_features=['num1', 'num2', 'num3', 'num4'],
        model_type='xgb',
        imputer_strategy='median',
        scaler_type='robust',
        resample_type='smoteenn'
    )

    # Make prediction
    preds = model.predict(X)
    assert len(preds) == len(X)

def test_invalid_model_type(sample_data):
    X, y = sample_data
    with pytest.raises(ValueError):
        build_model(X, y, model_type='invalid')

# --- Test: Invalid resample_type ---
def test_invalid_resample_type(sample_data):
    X, y = sample_data
    with pytest.raises(ValueError):
        build_model(
            X_train=X,
            y_train=y,
            numeric_features=['num1'],
            resample_type='invalid'
        )
