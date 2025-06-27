import sys
from pathlib import Path

# Set path ke parent folder
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
import os
import shutil
import pandas as pd
import pytest
from module.encoding import encode

@pytest.fixture
def sample_data():
    train = pd.DataFrame({
        'color': ['red', 'blue', 'green', None],
        'size': ['S', 'M', 'L', 'S']
    })
    test = pd.DataFrame({
        'color': ['blue', 'yellow'],
        'size': ['L', 'M']
    })
    return train, test

def test_ordinal_encoding(sample_data):
    train, test = sample_data
    encoded_train, encoded_test = encode(
        X_train=train,
        columns=['color'],
        X_test=test,
        method='ordinal'
    )

    assert 'color' in encoded_train.columns
    assert encoded_train['color'].dtype.kind in 'iufc'
    assert not encoded_train.isnull().values.any()
    assert encoded_test is not None

def test_save_encoded_files(tmp_path, sample_data):
    train, test = sample_data
    save_dir = tmp_path / "output"

    encode(
        X_train=train,
        columns=['color'],
        X_test=test,
        method='ordinal',
        save_encoded=True,
        save_path=save_dir,
        prefix='testdata'
    )

    train_file = save_dir / "testdata_X_train_encoded.csv"
    test_file = save_dir / "testdata_X_test_encoded.csv"

    assert train_file.exists()
    assert test_file.exists()

    # cek isi file
    df_train_loaded = pd.read_csv(train_file)
    assert not df_train_loaded.empty

