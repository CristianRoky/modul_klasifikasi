import pickle

# Buka file dan load isi dictionary
with open("selected_features (1).pkl", "rb") as f:
    data = pickle.load(f)

# Ambil masing-masing data
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# Tampilkan contoh data
# print(y_train)

import pickle

# Load the model from the .pkl file
with open('trained_model.pkl', 'rb') as f:  # 'rb' = read binary mode
    model = pickle.load(f)
import numpy as np
X_test = np.array([
    [5.1, 3.5,4,4]
])
# Use the model for predictions
predictions = model.predict(X_test)  # Replace X_test with your data1
print(predictions)