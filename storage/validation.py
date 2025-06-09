import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
import xgboost as xgb

def validate_model(X, y, best_params):
    """
    Validasi 10-Fold dan visualisasi F1-score per kelas untuk setiap fold.
    
    Args:
        X (DataFrame): Data fitur.
        y (Series): Label target.
        best_params (dict): Hyperparameter terbaik hasil tuning.
    """
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    fold_f1_scores = []
    fold = 1

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = Pipeline(steps=[
            ('model', xgb.XGBClassifier(**best_params, eval_metric='logloss', random_state=42))
        ])

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)

        report = classification_report(y_val, y_pred, output_dict=True)

        # Ambil F1-score per kelas
        f1_per_class = {label: report[label]['f1-score'] for label in report if label.isdigit()}
        fold_f1_scores.append(f1_per_class)

        print(f"\n===== Fold {fold} =====")
        print(f"Akurasi: {acc:.4f}")
        print(classification_report(y_val, y_pred))
        fold += 1

    print("\n===== Validasi Akhir dengan 10-Fold CV =====")
    print(f"Rata-rata Akurasi: {np.mean(accuracies):.4f}")
    print(f"Standar Deviasi: {np.std(accuracies):.4f}")

    # === Visualisasi F1 per kelas setiap fold ===
    classes = sorted(fold_f1_scores[0].keys(), key=int)  # Urutkan kelas secara numerik
    folds = range(1, len(fold_f1_scores) + 1)

    # Siapkan data untuk plot
    f1_data = {cls: [fold_f1_scores[i].get(cls, 0) for i in range(10)] for cls in classes}

    bar_width = 0.2
    x = np.arange(len(folds))

    plt.figure(figsize=(10, 5))
    for i, cls in enumerate(classes):
        plt.bar(x + i * bar_width, f1_data[cls], width=bar_width, label=f'Kelas {cls}')

    plt.title('F1-score per Kelas di Tiap Fold (10-Fold CV)')
    plt.xlabel('Fold')
    plt.ylabel('F1-score')
    plt.xticks(x + bar_width, [f'Fold {i}' for i in folds])
    plt.ylim(0, 1)
    plt.legend(title='Kelas')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
