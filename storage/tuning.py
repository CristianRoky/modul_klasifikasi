import xgboost as xgb
from sklearn.metrics import make_scorer, f1_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd


def tune_xgb_bayesian(X_train, y_train, numeric_features=None, init_points=5, n_iter=25, random_state=42):
    """
    Fungsi untuk melakukan hyperparameter tuning pada XGBoostClassifier
    menggunakan Bayesian Optimization dan pipeline resampling (SMOTEENN).

    Parameters
    ----------
    X_train : pd.DataFrame
        Data latih (fitur) yang ingin digunakan untuk training dan tuning.
    y_train : pd.Series atau array-like
        Label target dari data latih.
    numeric_features : list of str, optional
        Daftar nama kolom numerik yang akan dilakukan scaling.
        Jika None, akan otomatis dipilih dari X_train bertipe numerik.
    init_points : int, default=5
        Jumlah titik awal acak untuk optimasi Bayesian.
    n_iter : int, default=25
        Jumlah iterasi optimasi Bayesian setelah titik awal.
    random_state : int, default=42
        Seed random untuk reproducibility.

    Returns
    -------
    best_params : dict
        Parameter terbaik hasil tuning.
    optimizer : BayesianOptimization object
        Objek optimizer berisi semua log tuning.
    """

    if numeric_features is None:
        numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

    # Fungsi evaluasi untuk Bayesian Optimization
    def xgb_evaluate(learning_rate, max_depth, min_child_weight, n_estimators):
        params = {
            'learning_rate': learning_rate,
            'max_depth': int(max_depth),
            'min_child_weight': int(min_child_weight),
            'n_estimators': int(n_estimators),
            'eval_metric': 'logloss',
            'random_state': random_state
        }

        # Pipeline: scaling -> resample -> model
        pipeline = Pipeline(steps=[
            ('model', xgb.XGBClassifier(**params))
        ])

        # 5-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        # Gunakan f1-score sebagai metrik
        scores = cross_val_score(pipeline, X_train, y_train,
                                 cv=cv,
                                 scoring=make_scorer(f1_score))
        return scores.mean()

    # Definisikan ruang parameter untuk optimasi
    optimizer = BayesianOptimization(
        f=xgb_evaluate,
        pbounds={
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
            'min_child_weight': (1, 10),
            'n_estimators': (50, 1000)
        },
        random_state=random_state,
        verbose=2
    )

    # Mulai proses tuning
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # Ambil parameter terbaik
    best_params = optimizer.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    best_params['n_estimators'] = int(best_params['n_estimators'])

    return best_params, optimizer
