import pickle
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

def encode(
    X_train: pd.DataFrame,
    columns: list,
    X_test: pd.DataFrame = None,
    method: str = "ordinal",  
    prefix: str = "encoder",
    save_path: str = ".",
    save_encoded: bool = False,
    save_encoder: bool = True
):
    """
    Parameters:
    - X_train: DataFrame training
    - columns: list nama kolom yang ingin di-encode
    - X_test: DataFrame testing (opsional)
    - method: "ordinal"
    - prefix: prefix nama file encoder yang akan disimpan
    - save_path: path untuk menyimpan file encoder dan hasil encoded
    - save_encoded: jika True, simpan hasil encoded ke file CSV
    - save_encoder: jika True, encoder disimpan dalam bentuk pickle

    Returns:
    - X_train_encoded: DataFrame hasil encoding
    - X_test_encoded: DataFrame hasil encoding atau None jika X_test tidak diberikan
    """

    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy() if X_test is not None else None

    os.makedirs(save_path, exist_ok=True)

    for col in columns:
        if method == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train_encoded[[col]] = encoder.fit_transform(X_train_encoded[[col]])
            if X_test_encoded is not None:
                X_test_encoded[[col]] = encoder.transform(X_test_encoded[[col]])
        else:
            raise ValueError("Metode encoding hanya 'ordinal'")

        # Save encoder
        if save_encoder:
            with open(os.path.join(save_path, f"{prefix}_{col}_encoder.pkl"), 'wb') as f:
                pickle.dump(encoder, f)

    # Save encoded datasets if requested
    if save_encoded:
        X_train_encoded.to_csv(os.path.join(save_path, f"{prefix}_X_train_encoded.csv"), index=False)
        if X_test_encoded is not None:
            X_test_encoded.to_csv(os.path.join(save_path, f"{prefix}_X_test_encoded.csv"), index=False)

    return X_train_encoded, X_test_encoded