from opencage.geocoder import OpenCageGeocode
import time
import pandas as pd
import streamlit as st
import numpy as np

    
API_KEYS = ["c495403464674c99a186f112b0950775", "6b798f57f6ca4bc19866b5b9b23c23bd"]
geocoders = [OpenCageGeocode(key) for key in API_KEYS]

def make_cache_key(lat, lon, precision=4):
    return (round(float(lat), precision), round(float(lon), precision))

def get_location(lat, lon, index):
    key = make_cache_key(lat, lon)  # Rounded key
    api_index = index % len(API_KEYS)
    geocoder = geocoders[api_index]

    for attempt in range(3):
        try:
            result = geocoder.reverse_geocode(lat, lon)
            if result and len(result) > 0:
                components = result[0].get("components", {})
                if "country" in components:
                    location = components["country"]
                elif "body_of_water" in components:
                    location = components["body_of_water"]
                else:
                    location = "Unknown"
                break
            time.sleep(1)
        except Exception as e:
            print(f"Attempt {attempt + 1}: Geocoding error for {key}: {e}")
    return location

#fungsi null able
def safe_float_input(label, default=""):
    user_input = st.text_input(label, value=default)
    try:
        return float(user_input)
    except ValueError:
        return np.nan