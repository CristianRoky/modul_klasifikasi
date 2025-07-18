import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import os
import pickle

def evaluate_model(
    model_or_path,
    X_train,
    y_train,
    X_test,
    y_test,
    save_report=True,
    save_confusion_matrix=True,
    save_roc_curve=True,
    output_dir='output',
    report_filename='classification_report.txt',
    cm_filename='confusion_matrix.png',
    roc_filename='roc_curve.png'
):
    # ======= Load model jika input berupa path =======
    if isinstance(model_or_path, str):
        with open(model_or_path, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"Model loaded from {model_or_path}")
    else:
        pipeline = model_or_path

    # ======= Predict dan evaluasi =======
    y_pred = pipeline.predict(X_test)

    print("\n===== Evaluasi Akhir pada Test Set =====")
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print("Accuracy:", acc)
    print("Classification Report:\n", report)

    # ======= Simpan classification report (optional) =======
    if save_report:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, report_filename), 'w') as f:
            f.write("Accuracy: {:.4f}\n\n".format(acc))
            f.write("Classification Report:\n")
            f.write(report)
        print(f"Classification report saved to {os.path.join(output_dir, report_filename)}")

    # ======= Confusion Matrix =======
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Set')

    if save_confusion_matrix:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, cm_filename))
        print(f"Confusion matrix image saved to {os.path.join(output_dir, cm_filename)}")
    

    # ======= ROC Curve & AUC untuk Train dan Test =======
    roc_auc_train = None
    roc_auc_test = None

    if hasattr(pipeline, "predict_proba"):
        # Probabilitas train dan test
        y_proba_train = pipeline.predict_proba(X_train)
        y_proba_test = pipeline.predict_proba(X_test)

        # Asumsi binary classification: ambil probabilitas kelas positif
        if y_proba_train.shape[1] == 2 and y_proba_test.shape[1] == 2:
            y_scores_train = y_proba_train[:, 1]
            y_scores_test = y_proba_test[:, 1]

            fpr_train, tpr_train, _ = roc_curve(y_train, y_scores_train)
            roc_auc_train = auc(fpr_train, tpr_train)
            print(f"ROC AUC (Train): {roc_auc_train:.4f}")

            fpr_test, tpr_test, _ = roc_curve(y_test, y_scores_test)
            roc_auc_test = auc(fpr_test, tpr_test)
            print(f"ROC AUC (Test): {roc_auc_test:.4f}")

            plt.figure(figsize=(6, 4))
            plt.plot(fpr_train, tpr_train, label=f'Train ROC curve (area = {roc_auc_train:.4f})')
            plt.plot(fpr_test, tpr_test, label=f'Test ROC curve (area = {roc_auc_test:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")

            if save_roc_curve:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, roc_filename))
                print(f"ROC curve image saved to {os.path.join(output_dir, roc_filename)}")
            
        else:
            print("Multiclass ROC tidak didukung dalam fungsi ini.")
    else:
        print("Model tidak mendukung predict_proba, ROC AUC tidak dihitung.")

    return y_pred, acc, report, cm, roc_auc_train, roc_auc_test
