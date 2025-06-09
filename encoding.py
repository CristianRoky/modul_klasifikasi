import pickle
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

def encode(
    X_train: pd.DataFrame,
    columns: list,
    X_test: pd.DataFrame = None,
    method: str = "ordinal",  # pilihan: "ordinal" atau "onehot"
    prefix: str = "encoder",
    save_path: str = ".",
    save_encoded: bool = False
):
    """
    Encode kolom kategorikal dengan metode yang dipilih, handle missing, dan simpan encoder.

    Parameters:
    - X_train: DataFrame training
    - columns: list nama kolom yang ingin di-encode
    - X_test: DataFrame testing (opsional)
    - method: "ordinal" atau "onehot"
    - prefix: prefix nama file encoder yang akan disimpan
    - save_path: path untuk menyimpan file encoder dan hasil encoded
    - save_encoded: jika True, simpan hasil encoded ke file CSV

    Returns:
    - X_train_encoded: DataFrame hasil encoding
    - X_test_encoded: DataFrame hasil encoding atau None jika X_test tidak diberikan
    """

    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy() if X_test is not None else None

    os.makedirs(save_path, exist_ok=True)

    for col in columns:
        # Imputer untuk missing values (isi missing dengan string 'missing')
        
        if method == "ordinal":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train_encoded[[col]] = encoder.fit_transform(X_train_encoded[[col]])
            if X_test_encoded is not None:
                X_test_encoded[[col]] = encoder.transform(X_test_encoded[[col]])
        elif method == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            # Fit transform untuk train
            encoded_train = encoder.fit_transform(X_train_encoded[[col]])
            encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]

            encoded_train_df = pd.DataFrame(encoded_train, columns=encoded_cols, index=X_train_encoded.index)
            X_train_encoded = X_train_encoded.drop(columns=[col]).join(encoded_train_df)

            if X_test_encoded is not None:
                encoded_test = encoder.transform(X_test_encoded[[col]])
                encoded_test_df = pd.DataFrame(encoded_test, columns=encoded_cols, index=X_test_encoded.index)
                X_test_encoded = X_test_encoded.drop(columns=[col]).join(encoded_test_df)
        else:
            raise ValueError("Metode encoding hanya 'ordinal' atau 'onehot'")

    if save_encoded:
        X_train_encoded.to_csv(os.path.join(save_path, f"{prefix}_X_train_encoded.csv"), index=False)
        if X_test_encoded is not None:
            X_test_encoded.to_csv(os.path.join(save_path, f"{prefix}_X_test_encoded.csv"), index=False)

    return X_train_encoded, X_test_encoded