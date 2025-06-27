import sys
from pathlib import Path

# Set path ke parent folder
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import pandas as pd
import numpy as np
import os
import shutil
import pytest
from module.exploratory_data_analysis import eda_summary  # sesuaikan nama file modulmu

@pytest.fixture
def sample_df():
    # Buat DataFrame dummy dengan missing, outlier, dan label
    data = {
        'feature1': [1, 2, 3, np.nan, 5, 100],  # 100 = outlier
        'feature2': [1.0, 2.5, 3.0, 4.0, np.nan, 6.0],
        'label': ['A', 'B', 'A', 'B', 'A', 'A']
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="function")
def clean_eda_output():
    dir_path = "eda_output"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    yield
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

def test_eda_summary_basic(sample_df, clean_eda_output):
    results = eda_summary(sample_df, label_column='label',y_train=sample_df['label'], save_dir="eda_output")

    # 1. Missing values
    assert isinstance(results['missing'], dict) or results['missing'] is None

    # 2. Heatmap file
    if results['missing']:
        assert os.path.exists(results['missing_heatmap'])

    # 3. Outlier detection
    assert isinstance(results['outlier_counts'], dict)
    for col, count in results['outlier_counts'].items():
        assert isinstance(count, int)

    # 4. Boxplot file
    assert os.path.exists(results['outliers_boxplot'])

    # 5. Class distribution
    assert isinstance(results['class_counts'], dict)
    assert results['class_counts']['A'] == 4
    assert os.path.exists(results['distribusi_kelas'])

    # 6. Correlation matrix
    if results['correlation_matrix'] is not None:
        assert 'feature1' in results['correlation_matrix'].columns

    assert os.path.exists(results['correlation_matrix_path'])

    # 7. Duplicates
    assert results['duplicates'] == 0