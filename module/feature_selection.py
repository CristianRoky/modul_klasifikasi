import matplotlib.pyplot as plt
import pandas as pd
import pickle

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def feature_selection_based_on_feature_importance(
    X_train, y_train,
    algo='xgb',
    top_n=None,
    importance_threshold=0,
    save_path=None,
    save_df_path=None,
    figsize=(10, 6),
    color='royalblue'
):
    """
    Train model (XGB or RF), plot feature importance, seleksi fitur, dan simpan hasil.

    Parameters:
    - X_train, y_train: data training
    - algo: 'xgb' atau 'rf'
    - top_n: jumlah fitur teratas yang ingin dipilih (prioritas ke top_n jika ada)
    - importance_threshold: jika top_n None, pilih fitur dengan importance > threshold
    - save_path: path file PNG untuk simpan plot, None artinya tidak simpan
    - save_df_path: path file pickle untuk simpan DataFrame fitur terpilih, None artinya tidak simpan
    - figsize, color: parameter plot

    Returns:
    - model terlatih
    - DataFrame fitur terpilih (subset dari X_train, inplace)
    """

    # Buat dan train model sesuai algo
    if algo == 'xgb':
        model = XGBClassifier()
    elif algo == 'rf':
        model = RandomForestClassifier()
    else:
        raise ValueError("algo harus 'xgb' atau 'rf'")

    model.fit(X_train, y_train)

    # Ambil feature importance
    importances = model.feature_importances_
    feature_names = X_train.columns

    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Pilih fitur berdasarkan top_n atau importance_threshold
    if top_n is not None:
        selected_features = fi_df.head(top_n)['Feature'].tolist()
    else:
        selected_features = fi_df[fi_df['Importance'] > importance_threshold]['Feature'].tolist()

    # Plot
    plot_df = fi_df[fi_df['Feature'].isin(selected_features)]
    plt.figure(figsize=figsize)
    plt.barh(plot_df['Feature'], plot_df['Importance'], color=color)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature Name")
    plt.title(f"Feature Importances from {algo.upper()}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()

    # Seleksi fitur pada X_train (inplace)
    X_train_selected = X_train[selected_features]

    # Simpan dataframe jika perlu
    if save_df_path:
        with open(save_df_path, 'wb') as f:
            pickle.dump(X_train_selected, f)

    return model, X_train_selected