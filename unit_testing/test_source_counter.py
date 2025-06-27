import sys
from pathlib import Path

# Set path ke parent folder
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import pytest
import pandas as pd
import numpy as np
from module.source_counter import count_sources  # Ganti dengan nama modul Anda

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'sources': [
            "source1, source2",
            "source3",
            "",
            np.nan,
            "source1, source2, source3"
        ]
    })

def test_column_not_found(sample_df):
    with pytest.raises(ValueError, match="Kolom 'invalid_column' tidak ditemukan di DataFrame."):
        count_sources(sample_df, 'invalid_column')

def test_counting_sources(sample_df):
    result = count_sources(sample_df, 'sources')
    assert 'source_counter' in result.columns
    assert result['source_counter'].tolist() == [2, 1, 0, 0, 3]
