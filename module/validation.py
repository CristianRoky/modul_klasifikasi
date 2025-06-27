import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as Pipeline
from imblearn.combine import SMOTEENN
from sklearn.metrics import f1_score, classification_report
import numpy as np
import json

def validate_model(
    X,
    y,
    numeric_features=None,
    algo="xgb",
    resampler_type=None,
    scaler_type=None,
    imputer_strategy=None,
    cv_splits=10,
    random_state=42,
    save_plot_path="validation_results.png",
    best_param_path=None,
    class_names=None  # Misalnya: ['0', '1'] atau ['Tidak Tsunami', 'Tsunami']
):
    # Load best params if available
    best_params = {}
    if best_param_path is not None:
        with open(best_param_path, 'r') as f:
            best_params = json.load(f)

    # --- Preprocessing (hanya jika semua komponen diberikan) ---
    if numeric_features and imputer_strategy and scaler_type:
        scaler_cls = RobustScaler
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=imputer_strategy)),
            ('scaler', scaler_cls())
        ])
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features)
        ], remainder='passthrough')
    else:
        preprocessor = 'passthrough'

    # --- Resampler (jika diberikan) ---
    if resampler_type == "smoteenn":
        resampler_obj = SMOTEENN(random_state=random_state)
    else:
        resampler_obj = None

    # --- Model ---
    if algo == "rf":
        model = RandomForestClassifier(random_state=random_state, **best_params)
    elif algo == "xgb":
        model = XGBClassifier(
            random_state=random_state,
            verbosity=0,
            eval_metric='logloss',
            **best_params
        )
    else:
        raise ValueError("Algorithm must be 'rf' or 'xgb'")

    # --- Pipeline ---
    steps = [('preprocessing', preprocessor)]
    if resampler_obj:
        steps.append(('resample', resampler_obj))
    steps.append(('model', model))
    pipeline = Pipeline(steps)

    # --- Cross-validation ---
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    f1_per_class_all_folds = []
    fold_num = 1

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Validasi metrik F1 Macro
        macro_f1 = f1_score(y_test, y_pred, average='macro')

        # Detail per kelas
        report = classification_report(y_test, y_pred, output_dict=True)
        f1_scores = {
            'fold': fold_num,
            'macro_f1': macro_f1
        }

        if class_names is None:
            class_labels = sorted(np.unique(y))
            class_names = [str(c) for c in class_labels]

        for label in class_names:
            if label in report:
                f1_scores[f'F1_{label}'] = report[label]['f1-score']
            else:
                f1_scores[f'F1_{label}'] = 0.0  # Jika label tidak muncul di prediksi

        f1_per_class_all_folds.append(f1_scores)
        fold_num += 1

    df_f1 = pd.DataFrame(f1_per_class_all_folds)

    # --- Plot F1 per kelas ---
    df_plot = df_f1.set_index('fold')
    plt.figure(figsize=(10, 5))

    # Pilih hanya kolom F1 kelas
    class_cols = [col for col in df_plot.columns if col.startswith('F1_')]

    colors = ['green', 'blue', 'red', 'purple', 'orange']
    for i, col in enumerate(class_cols):
        plt.plot(df_plot.index, df_plot[col], marker='o', label=col.replace('F1_', 'Kelas '), color=colors[i % len(colors)])

    plt.title('F1 Score per Kelas untuk Tiap Fold')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Kelas')
    plt.xticks(df_plot.index)
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300)
    
    return df_f1