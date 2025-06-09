import time
import json
import os
import pandas as pd
from opencage.geocoder import OpenCageGeocode
from tqdm import tqdm

# ====== Cache Setup ======
CACHE_FILE = "geocode_cache.json"

def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
            return {eval(k): v for k, v in cache.items()}
    return {}

def save_cache(cache, cache_file):
    with open(cache_file, "w") as f:
        json.dump({str(k): v for k, v in cache.items()}, f, indent=2)

def make_cache_key(lat, lon, precision=4):
    return (round(float(lat), precision), round(float(lon), precision))

def format_eta(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# ====== Geocoding One Point ======
def get_location(lat, lon, index, cache, geocoders):
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown"

    key = make_cache_key(lat, lon)
    if key in cache:
        return cache[key]

    geocoder = geocoders[index % len(geocoders)]
    location = "Unknown"

    for attempt in range(3):
        try:
            result = geocoder.reverse_geocode(lat, lon)
            if result:
                components = result[0].get("components", {})
                location = components.get("country") or components.get("body_of_water") or "Unknown"
                break
            time.sleep(1)
        except Exception as e:
            print(f"Attempt {attempt + 1}: Geocoding error for {key}: {e}")

    cache[key] = location
    return location

# ====== Main Processor ======
def create_location(
    df: pd.DataFrame,
    api_keys: list,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    loc_col: str = "location",
    limit: int = 5000,
    start_index: int = 0,
    save_as: str = None,
    cache_file: str = "geocode_cache.json"    
) -> pd.DataFrame:
    
    df = df.copy()
    if loc_col not in df.columns:
        df[loc_col] = pd.NA
    cache = load_cache(cache_file)
    geocoders = [OpenCageGeocode(key) for key in api_keys]
    
    count = 0
    total_rows = len(df)
    start_time = time.time()
    progress_bar = tqdm(total=min(len(df), limit), desc="Processing", unit="row", dynamic_ncols=True)

    for i, row in df.iterrows():
        
        if count >= limit:
            print("API limit reached, stopping...")
            break

        if pd.isna(row.get(loc_col)):
            lat, lon = row.get(lat_col), row.get(lon_col)
            df.at[i, loc_col] = get_location(lat, lon, start_index + i, cache, geocoders)
            count += 1

        if count % 50 == 0:
            save_cache(cache, cache_file)

        elapsed = time.time() - start_time
        remaining_rows = limit - count
        remaining_time = (elapsed / max(count, 1)) * max(remaining_rows, 1)
        progress_bar.set_postfix({"ETA": format_eta(remaining_time), "Processed": count})
        progress_bar.update(1)

    progress_bar.close()
    save_cache(cache, cache_file)

    # ====== Save to file if needed ======
    if save_as:
        if save_as.endswith(".csv"):
            df.to_csv(save_as, index=False)
        elif save_as.endswith(".json"):
            df.to_json(save_as, orient="records", lines=True)
        elif save_as.endswith(".pkl"):
            df.to_pickle(save_as)
        elif save_as.endswith(".xlsx"):
            df.to_excel(save_as, index=False)
        elif save_as.endswith(".parquet"):
            df.to_parquet(save_as, index=False)
        else:
            raise ValueError("Format file tidak didukung. Gunakan .csv, .json, .pkl, .xlsx, atau .parquet.")

    return df