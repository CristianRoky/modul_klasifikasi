def run_pipeline(filepath):
    import pandas as pd

    from module.hyperparameter_tuning  import tune_model
    from module.validation import validate_model
    # --- 1. Muat data pengguna sendiri (ubah sesuai nama file/format) ---
    # Contoh: data CSV dengan fitur dan target
    
    df = pd.read_excel(filepath)  # Ganti dengan nama file 

    # --- 2. Pisahkan fitur dan target ---
    from sklearn.model_selection import train_test_split
    X = df.drop(['tsunami', 'place', 'time', 'updated', 'tz', 'url', 'detail',
                'alert', 'status', 'code', 'ids', 'types', 'type', 'title',
                'geometry_type', 'id' ], axis=1)
    y = df['tsunami']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    # numeric_features = ['felt', 'cdi', 'mag', 'nst', 'longitude', 'latitude',
    #                     'sig', 'depth', 'rms', 'mmi', 'dmin', 'gap']
    from module.source_counter import count_sources

    # Proses fitur sumber
    X_train = count_sources(X_train, column_name='sources')
    X_test = count_sources(X_test, column_name='sources')

    import pandas as pd
    from module.geocoding import create_location

    X_train = create_location(
        X_train,
        api_keys=["c495403464674c99a186f112b0950775"],
        lat_col="latitude",
        lon_col="longitude",
        loc_col="location",
        limit=4000,
        save_as="s.csv",
        cache_file="geocode_cache.json"
    )
    X_test = create_location(
        X_test,
        api_keys=["c495403464674c99a186f112b0950775"],
        lat_col="latitude",
        lon_col="longitude",
        loc_col="location",
        limit=2000,
        save_as="ss.csv",
        cache_file="geocode_cache.json"
    )

    from module.encoding import encode

    X_train, X_test = encode(
        X_train,
        columns=['magType', 'net', 'location'],
        X_test=X_test,
        method="ordinal",      # bisa ganti "onehot"
        prefix="myencoder",
        save_path="./saved_encoders",
        save_encoded=False
    ) 

    from module.feature_selection import feature_selection_based_on_feature_importance

    model, X_train = feature_selection_based_on_feature_importance(
        X_train=X_train, y_train=y_train,
        algo='xgb',
        top_n=None,
        save_path='fi_plot.png',
        save_df_path='X_train_selected.pkl'
    )

    from module.hyperparameter_tuning import tune_model  # ganti sesuai nama file Python-mu

    best_params = tune_model(
        X_train=X_train,
        y_train=y_train,
        numeric_features=None,
        algo="xgb",                       
        resampler=None,            
        scaler=None,               
        imputer_strategy=None,     
        cv_splits=5,
        random_state=42,
        init_points=2,
        n_iter=2,
        verbose=2,
        return_scores=False,
        save_best_param_path="xgb_best_params.pkl"
    )

    print("Best Hyperparameters:", best_params)

    from module.validation import validate_model

    # Pastikan X dan y adalah pandas DataFrame dan Series
    hasil = validate_model(
        X=X_train,
        y=y_train,
        numeric_features=None,
        algo='xgb',
        best_param_path='xgb_best_params.pkl',
        cv_splits=10,
        save_plot_path="val.png"
    )

    print(hasil)

    from module.build_model import build_model
    model = build_model(
        X_train=X_train, y_train=y_train,
        numeric_features=None,
        model_type='xgb',
        best_param_path='xgb_best_params.pkl',
        imputer_strategy=None,
        resample_type=None,
        save_model=True,
        model_path='final_model.pkl'
    )

    from module.test import evaluate_model
    # 2) Kalau model disimpan di file 'model.pkl'
    y_pred, acc, report, cm, roc_train, roc_test = evaluate_model(
        model_or_path=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    return model, X_train, y_train, X_test, y_test