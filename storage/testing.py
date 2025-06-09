import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

def evaluate_final_model(X_train, y_train, X_test, y_test, best_params, show_confusion_matrix=True):
    """
    Latih dan evaluasi model akhir dengan XGBoost + best_params.

    Parameters:
    - X_train, y_train: Data latih
    - X_test, y_test: Data uji
    - best_params: dict hasil tuning
    - show_confusion_matrix: bool, jika True tampilkan heatmap confusion matrix

    Returns:
    - y_pred_test: hasil prediksi terhadap X_test
    """
    print("\n===== Evaluasi Akhir pada Test Set =====")

    # Bangun pipeline
    final_pipeline = Pipeline(steps=[
        ('model', xgb.XGBClassifier(**best_params, eval_metric='logloss', random_state=42))
    ])

    # Fit dan prediksi
    final_pipeline.fit(X_train, y_train)
    y_pred_test = final_pipeline.predict(X_test)

    # Evaluasi
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Classification Report:\n", classification_report(y_test, y_pred_test))

    # Confusion Matrix
    if show_confusion_matrix:
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Test Set')
        plt.tight_layout()
        plt.show()

    return y_pred_test
