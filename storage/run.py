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

# Panggil fungsi tuning
from Hyperparameter_Tuning import tune_model  # ganti sesuai nama file Python-mu

best_params = tune_model(
    X_train=X_train,
    y_train=y_train,
    numeric_features=['felt'],
    algo="xgb",                        # atau "rf"
    scoring=None,                     # default: f1, precision, recall
    resampler="smoteenn",            # atau "smotetomek" atau None
    scaler="robust",                 # atau "standard"
    imputer_strategy="median",       # atau "mean", "most_frequent"
    cv_splits=5,
    random_state=42,
    init_points=10,
    n_iter=30,
    verbose=2,
    return_scores=False,
    save_best_param_path="xgb_best_params.pkl"
)

print("Best Hyperparameters:", best_params)

numeric_features = ['felt', 'cdi', 'mag', 'nst', 'longitude', 'latitude',
                    'sig', 'depth', 'rms', 'mmi', 'dmin', 'gap']

from Build_Model import build_model
pipeline = build_model(
    X_train,
    y_train,
    numeric_features=numeric_features,
    model_type='xgb',
    model_params={'n_estimators': 100, 'max_depth': 4},
    use_imputer=True,
    use_scaler=True,
    use_resample=True,
    resample_type='smote',
    save_model=True,
    model_path='xgb_model.pkl'
)

from Test import evaluate_model

evaluate_model(
    pipeline,
    X_test,
    y_test,
    save_report=True,
    save_confusion_matrix=True,
    output_dir='results',
    report_filename='report_xgb.txt',
    cm_filename='cm_xgb.png'
)