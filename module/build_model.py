import pickle
import json
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
import os

def build_model(
    X_train,
    y_train,
    numeric_features=None,
    category_cols=None,
    model_type='xgb',               # 'rf' or 'xgb'
    model_params=None,
    best_param_path=None,
    imputer_strategy=None,         
    scaler_type=None,              
    resample_type=None,            
    save_model=False,
    model_path='model.pkl'
):
    # Load best parameters if provided
    best_params = {}
    if best_param_path and os.path.exists(best_param_path):
        with open(best_param_path, 'r') as f:
            best_params = json.load(f)
    if model_params is None:
        model_params = {}
    model_params = {**best_params, **model_params}
    print(f"Best params loaded from {model_params}")

    # Define numeric transformer steps
    num_steps = []
    if imputer_strategy in ['median']:
        num_steps.append(('imputer', SimpleImputer(strategy=imputer_strategy)))
    if scaler_type in ['robust']:
        scaler = RobustScaler()
        num_steps.append(('scaler', scaler))

    numeric_transformer = Pipeline(steps=num_steps) if num_steps else 'passthrough'

    # Column transformer
    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if category_cols:
        cat_transformer = SimpleImputer(strategy='most_frequent')
        transformers.append(('cat', cat_transformer, category_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough',
        force_int_remainder_cols=False
    )

    # Define the model
    if model_type == 'rf':
        model = RandomForestClassifier(**model_params, random_state=42)
    elif model_type == 'xgb':
        model = XGBClassifier(eval_metric='logloss', **model_params, random_state=42)
    else:
        raise ValueError("model_type must be 'rf' or 'xgb'")

    # Pipeline steps
    steps = [('preprocessing', preprocessor)]

    if resample_type:
        if resample_type == 'smoteenn':
            resampler = SMOTEENN(random_state=42)
        else:
            raise ValueError("resample_type must be 'smoteenn'")
        steps.append(('resample', resampler))

    steps.append(('model', model))
    pipeline_class = ImbPipeline 
    pipeline = pipeline_class(steps=steps)

    pipeline.fit(X_train, y_train)

    # Save model
    if save_model:
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"Model saved to {model_path}")

    return pipeline