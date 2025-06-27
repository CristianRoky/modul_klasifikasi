import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def eda_summary(eda_data, label_column=None, y_train=None, save_dir="eda_output"):
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    # 1. Missing Values
    missing = eda_data.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        results['missing'] = None
    else:
        results['missing'] = missing.to_dict()

        plt.figure(figsize=(10, 6))
        sns.heatmap(eda_data.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        heatmap_path = os.path.join(save_dir, "missing_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        results['missing_heatmap'] = heatmap_path

    # 2. Outlier
    numeric_cols = eda_data.select_dtypes(include=['int64', 'float64']).columns
    outlier_counts = {}

    for col in numeric_cols:
        Q1 = eda_data[col].quantile(0.25)
        Q3 = eda_data[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = eda_data[(eda_data[col] < Q1 - 1.5 * IQR) | (eda_data[col] > Q3 + 1.5 * IQR)]
        outlier_counts[col] = len(outliers)

    results['outlier_counts'] = outlier_counts

    plt.figure(figsize=(len(numeric_cols) * 2, 6))
    sns.boxplot(data=eda_data[numeric_cols], orient='v')
    plt.title("Boxplot Outlier Semua Kolom Numerik")
    plt.xticks(rotation=45)
    outlier_path = os.path.join(save_dir, "outliers_boxplot.png")
    plt.savefig(outlier_path, dpi=300, bbox_inches='tight')
    plt.close()
    results['outliers_boxplot'] = outlier_path

    # 3. Distribusi Kelas
    if y_train is not None:
        class_counts = y_train.value_counts()
        results['class_counts'] = class_counts.to_dict()
        plt.figure(figsize=(6, 4))
        sns.countplot(x=y_train)
        plt.title("Distribusi Kelas")
        dist_path = os.path.join(save_dir, "distribusi_kelas.png")
        plt.savefig(dist_path, dpi=300)
        plt.close()
        results['distribusi_kelas'] = dist_path
    else:
        results['class_counts'] = None
        results['distribusi_kelas'] = None
        

    # 4. Korelasi Numerik
    if len(numeric_cols) >= 2:
        corr = eda_data[numeric_cols].corr()
        results['correlation_matrix'] = corr

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        corr_path = os.path.join(save_dir, "correlation_matrix.png")
        plt.savefig(corr_path, dpi=300)
        plt.close()
        results['correlation_matrix_path'] = corr_path
    else:
        results['correlation_matrix'] = None
        results['correlation_matrix_path'] = None

    # 5. Duplikat
    duplicates = eda_data.duplicated().sum()
    results['duplicates'] = duplicates

    return results