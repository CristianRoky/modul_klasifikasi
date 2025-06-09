import os
import pickle
import pytest
import pandas as pd
from feature_selection import feature_selection_based_on_feature_importance

# Data dummy untuk training
@pytest.fixture
def dummy_data():
    X = pd.DataFrame({
        'f1': [1, 2, 3, 4, 5],
        'f2': [5, 4, 3, 2, 1],
        'f3': [2, 3, 4, 5, 6],
        'f4': [6, 5, 4, 3, 2]
    })
    y = pd.Series([1, 0, 1, 0, 1])
    return X, y

def test_feature_selection_xgb_and_rf(dummy_data):
    X, y = dummy_data

    # Test algo 'xgb'
    model_xgb, selected_xgb = feature_selection_based_on_feature_importance(X, y, algo='xgb', top_n=2)
    assert hasattr(model_xgb, 'feature_importances_')
    assert isinstance(selected_xgb, pd.DataFrame)
    assert selected_xgb.shape[1] == 2

    # Test algo 'rf'
    model_rf, selected_rf = feature_selection_based_on_feature_importance(X, y, algo='rf', importance_threshold=0)
    assert hasattr(model_rf, 'feature_importances_')
    assert isinstance(selected_rf, pd.DataFrame)
    assert selected_rf.shape[1] <= X.shape[1]

def test_feature_selection_invalid_algo(dummy_data):
    X, y = dummy_data
    with pytest.raises(ValueError):
        feature_selection_based_on_feature_importance(X, y, algo='invalid_algo')

def test_feature_selection_save_plot_and_df(tmp_path, dummy_data):
    X, y = dummy_data
    plot_path = tmp_path / "feature_importance.png"
    df_path = tmp_path / "selected_features.pkl"

    model, selected = feature_selection_based_on_feature_importance(
        X, y,
        algo='xgb',
        top_n=2,
        save_path=str(plot_path),
        save_df_path=str(df_path)
    )

    # Cek file plot dibuat
    assert plot_path.exists()
    # Cek file pickle dibuat
    assert df_path.exists()

    # Load pickle dan cek isinya sama dengan selected
    with open(df_path, 'rb') as f:
        loaded_df = pickle.load(f)
    pd.testing.assert_frame_equal(selected, loaded_df)

def test_feature_selection_importance_threshold(dummy_data):
    X, y = dummy_data
    # Gunakan threshold yang cukup tinggi supaya tidak semua fitur terpilih
    _, selected = feature_selection_based_on_feature_importance(X, y, algo='xgb', importance_threshold=0.5)
    # Pastikan subset (bisa kosong atau <= original fitur)
    assert isinstance(selected, pd.DataFrame)
    assert selected.shape[1] <= X.shape[1]