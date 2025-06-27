import streamlit as st
import pickle
import numpy as np
import pandas as pd
from utils import get_location, safe_float_input

with open("ordinal_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
    
# Load model XGBoost
@st.cache_resource
def load_model():
    with open('XGB3_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# UI input
st.title("Prediksi Potensi Tsunami")
st.write("Masukkan parameter gempa untuk prediksi:")

# Input fitur
mag = safe_float_input("Magnitude", "5.0")
nst = safe_float_input("NST", "30")
longitude = safe_float_input("Longitude", "100")
latitude = safe_float_input("Latitude", "0")
depth = safe_float_input("Depth", "10")
sig = safe_float_input("Significance", "100")
cdi = safe_float_input("CDI", "3")
mmi = safe_float_input("MMI", "3")
rms = safe_float_input("RMS", "0.5")
gap = safe_float_input("Gap", "1")
dmin = safe_float_input("DMIN", "1")
felt = safe_float_input("DMIN", "200")
magType = st.text_input("tipe magnitudo")
location = st.text_input("Lokasi")

if not location:
    location = get_location(latitude,longitude,0)
input_cats = [[magType, location]]
encoded_cats = encoder.transform(input_cats)

if (depth<0):
    depth = 18.4599990844727 #berdasarkan nilai median pada x_train
    
sources_input = st.text_input("Sources (pisahkan dengan koma, contoh: ax,us,tx)", value="ax,us")
if sources_input.strip() == "":
    source_counter = np.nan
else:
    sources_list = [s.strip() for s in sources_input.split(',') if s.strip()]
    source_counter = len(sources_list)
    
# Prediksi jika tombol ditekan

# Prepare input for prediction
input_dict = {
    'mag': mag,
    'source_counter': source_counter,
    'nst': nst,
    'location': location,
    'longitude': longitude,
    'dmin': dmin,
    'mmi': mmi,
    'latitude': latitude,
    'magType': magType,
    'depth': depth,
    'sig': sig,
    'cdi': cdi,
    'rms': rms,
    'felt': felt,
    'gap': gap
}


# Create a DataFrame for prediction and apply encoded values
encoded_input = pd.DataFrame([input_dict])
encoded_input[['magType', 'net', 'location']] = encoded_cats[:, [0, 2]]  # Use encoded magType and location

# Exclude `net` from prediction by not including it in the input DataFrame for model prediction
encoded_input = encoded_input.drop(columns=['net'])

# Predict when the button is pressed
if st.button("Prediksi"):
    pred = model.predict(encoded_input)
    st.write("Hasil Prediksi:", int(pred[0]))