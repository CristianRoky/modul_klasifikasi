import pandas as pd
from storage.tuning import tune_xgb_bayesian
from storage.validation import validate_model

# --- 1. Muat data pengguna sendiri (ubah sesuai nama file/format) ---
# Contoh: data CSV dengan fitur dan target
df = pd.read_excel('Data_Tsunami.xlsx')  # Ganti dengan nama file kamu

# --- 2. Pisahkan fitur dan target ---
from sklearn.model_selection import train_test_split
X = df.drop(['tsunami', 'place', 'time', 'updated', 'tz', 'url', 'detail',
             'alert', 'status', 'code', 'ids', 'types', 'type', 'title',
             'geometry_type', 'id' ], axis=1)
y = df['tsunami']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

from source_counter import count_sources

# Proses fitur sumber
X_train = count_sources(X_train, column_name='sources')
X_test = count_sources(X_test, column_name='sources')

import pandas as pd
from geocoding import create_location

X_train = create_location(
    X_train,
    api_keys=["c495403464674c99a186f112b0950775"],
    lat_col="lat",
    lon_col="lon",
    loc_col="location",
    limit=2000,
    cache_file="geocode_cache.json"
)


from encoding import encode

# Lakukan encoding dan simpan encoder-nya
X_train_enc, X_test_enc = encode(
    X_train,
    columns=['magType', 'net', 'location'],
    X_test=X_test,
    method="ordinal",      # bisa ganti "onehot"
    prefix="myencoder",
    save_path="./saved_encoders",
    save_encoded=True
)

from feature_selection import feature_selection_based_on_feature_importance

model, X_train = feature_selection_based_on_feature_importance(
    X_train, y_train,
    algo='xgb',
    top_n=5,
    save_path='fi_plot.png',
    save_df_path='X_train_selected.pkl'
)

# --- 3. Panggil fungsi tuning ---
best_params, _ = tune_xgb_bayesian(X_train, y_train, init_points=1, n_iter=2)
# --- 4. Tampilkan hasil terbaik ---

print("Best Parameters:")
print(best_params)

#label harus numerik,
#target di setting binary, tidak bisa multiklass

validate_model(X_train,y_train, best_params=best_params)

from storage.testing import evaluate_final_model

# best_params adalah hasil dari tuning
y_pred_test = evaluate_final_model(X_train, y_train, X_test, y_test, best_params)
