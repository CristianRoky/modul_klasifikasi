import sys
from pathlib import Path

# Set path ke parent folder
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import pandas as pd
from sklearn.datasets import make_classification
from module.validation import validate_model

def test_validate_model_runs():
    # Buat dummy data
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X = pd.DataFrame(X, columns=["feat1", "feat2", "feat3", "feat4"])
    y = pd.Series(y)

    df_result = validate_model(
        X=X,
        y=y,
        numeric_features=["feat1", "feat2", "feat3", "feat4"],
        algo="rf",
        scaler_type="robust",
        imputer_strategy="mean",
        resampler_type=None,
        cv_splits=3,
        best_param_path="dummy_params.json",
        save_plot_path="validation.png"
    )

    assert isinstance(df_result, pd.DataFrame)
    assert 'macro_f1' in df_result.columns
