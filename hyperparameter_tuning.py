import pickle
import streamlit as st
import json
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN, SMOTETomek
import pandas as pd

def build_numeric_transformer(scaler=None, imputer_strategy=None):
    steps = []
    if imputer_strategy is not None:
        steps.append(('imputer', SimpleImputer(strategy=imputer_strategy)))
    if scaler == "robust":
        steps.append(('scaler', RobustScaler()))
    elif scaler == "standard":
        steps.append(('scaler', StandardScaler()))
    return Pipeline(steps=steps) if steps else 'passthrough'

def tune_model(
    X_train,
    y_train,
    numeric_features=None,
    algo="xgb",
    resampler=None,
    scaler=None,
    imputer_strategy=None,
    cv_splits=5,
    random_state=42,
    init_points=10,
    n_iter=20,
    verbose=2,
    return_scores=False,
    save_best_param_path="best_params.json",
    bounds=None
):
    # Build preprocessing
    transformers = []

    if numeric_features:
        numeric_transformer = build_numeric_transformer(scaler,imputer_strategy)
        transformers.append(('num', numeric_transformer, numeric_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough', force_int_remainder_cols=False)

    if resampler == "smoteenn":
        resampler_obj = SMOTEENN(random_state=random_state)
    elif resampler == "smotetomek":
        resampler_obj = SMOTETomek(random_state=random_state)
    else:
        resampler_obj = None

    # Model selector
    def get_model(params):
        if algo == "rf":
            return RandomForestClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                min_samples_split=int(params['min_samples_split']),
                min_samples_leaf=int(params['min_samples_leaf']),
                random_state=random_state,
                verbose=0
            )
        elif algo == "xgb":
            return XGBClassifier(
                learning_rate=params['learning_rate'],
                max_depth=int(params['max_depth']),
                min_child_weight=int(params['min_child_weight']),
                n_estimators=int(params['n_estimators']),
                eval_metric='logloss',
                random_state=random_state,
                verbosity=0,
                use_label_encoder=False
            )
                
    # Fungsi evaluasi BO
    def evaluate_model(**kwargs):
        model = get_model(kwargs)
        steps = [('preprocessing', preprocessor)]
        if resampler_obj:
            steps.append(('resample', resampler_obj))
        steps.append(('model', model))

        pipeline = Pipeline(steps)
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        
        result = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=make_scorer(f1_score, average='macro')
        )
        # st.write("Evaluating with params:", kwargs)
        return result.mean()

    # Bounds default
    if bounds is None:
        if algo == "rf":
            bounds = {
                'n_estimators': (50, 1000),
                'max_depth': (3, 10),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10)
            }
        elif algo == "xgb":
            bounds = {
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'min_child_weight': (1, 10),
                'n_estimators': (50, 1000)
            }

    optimizer = BayesianOptimization(
        f=evaluate_model,
        pbounds=bounds,
        random_state=random_state,
        verbose=verbose
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    # Ambil parameter terbaik
    best_params = optimizer.max['params']
    for k in best_params:
        if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_child_weight', 'min_samples_leaf']:
            best_params[k] = int(best_params[k])
    # Simpan hasil
    with open(save_best_param_path, "w") as f:
        json.dump(best_params, f)

    if return_scores:
        return best_params, optimizer
    else:
        return best_params
