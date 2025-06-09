import pandas as pd
from storage.tuning import tune_xgb_bayesian
from storage.validation import validate_model



# --- 1. Muat data pengguna sendiri (ubah sesuai nama file/format) ---
# Contoh: data CSV dengan fitur dan target
data = pd.read_csv('iris.csv')  # Ganti dengan nama file kamu

# --- 2. Pisahkan fitur dan target ---
X_train = data.drop(columns=['Species'])  # Ganti 'target' sesuai kolom target kamu
y_train = data['Species']

# --- 3. Panggil fungsi tuning ---
best_params, _ = tune_xgb_bayesian(X_train, y_train, init_points=1, n_iter=2)
# --- 4. Tampilkan hasil terbaik ---

print("Best Parameters:")
print(best_params)

#label harus numerik,
#target di setting binary, tidak bisa multiklass

validate_model(X_train,y_train, best_params=best_params)