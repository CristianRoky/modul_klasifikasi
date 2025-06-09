import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd

from source_counter import count_sources
from geocoding import create_location
from encoding import encode
from feature_selection import feature_selection_based_on_feature_importance
from hyperparameter_tuning import tune_model
from build_model import build_model
from validation import validate_model
from test import evaluate_model

import pydeck as pdk
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from best_model.preprocessing import get_location, safe_float_input

# Load encoder for all features (magType, location)
with open("best_model\ordinal_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load model XGBoost
@st.cache_resource
def load_model():
    with open('best_model\XGB3_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# UI input

halaman = st.sidebar.radio("Navigasi", ["Pengembangan Model","Beranda", "Panduan"])
if halaman == "Pengembangan Model":
    st.title("Aplikasi Analisis & Prediksi Modular")

    # ========================== STEP 1: Upload File ==========================
    import pickle
    import pandas as pd
    from sklearn.model_selection import train_test_split

    uploaded_file = st.file_uploader("Upload file data (.pkl / .csv / .xlsx)", type=["pkl", "csv", "xlsx"])

    if uploaded_file:
        # Jika file pickle
        if uploaded_file.name.endswith(".pkl"):
            X_train, X_test, y_train, y_test = pickle.load(uploaded_file)

            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test

            st.success("✅ Data berhasil dimuat ke session_state.")
            st.dataframe(st.session_state['X_train'].head())

        # Jika CSV atau Excel
        else:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            # Bersihkan kolom object
            for col in data.select_dtypes(include='object').columns:
                data[col] = data[col].astype(str)

            st.dataframe(data.head())

            st.session_state['raw_data'] = data

            # STEP: Pilih fitur dan target
            columns = data.columns.tolist()
            feature_cols = st.multiselect("Pilih kolom fitur (X)", columns)
            target_col = st.selectbox("Pilih kolom target (y)", [col for col in columns if col not in feature_cols])

            if feature_cols and target_col:
                X = data[feature_cols]
                y = data[target_col]

                test_size = st.slider("Test size (%)", 10, 50, 20)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42,  stratify=y)

                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['target_col'] = target_col

                st.success("Data berhasil dipisahkan menjadi X_train, X_test, y_train, y_test.")

    # Modul lanjutan akan berjalan hanya jika data sudah tersedia
    if 'X_train' in st.session_state and 'X_test' in st.session_state:
        st.subheader("Modul lanjutan")
    if st.checkbox("Modul Exploratory Data Analysis"):
        from exploratory_data_analysis import eda_summary
        import os
        import pandas as pd

        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        target_col = st.session_state['target_col']
        combined_data = pd.concat([X_train, X_test], axis=0)
        save_dir = "eda_output"

        if st.button("Jalankan EDA"):
            with st.spinner("Menjalankan EDA..."):
                results = eda_summary(combined_data, label_column=target_col, save_dir=save_dir)

                st.subheader("📥 Unduh Gambar Visualisasi")

                # List file gambar yang mungkin ada berdasarkan keys dari results
                image_files = [
                    results.get('missing_heatmap'),
                    results.get('outliers_boxplot'),
                    results.get('distribusi_kelas'),
                    results.get('correlation_matrix_path')
                ]

                for img_path in image_files:
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=os.path.basename(img_path))
                        with open(img_path, "rb") as f:
                            st.download_button(label=f"Unduh {os.path.basename(img_path)}", data=f, file_name=os.path.basename(img_path), mime="image/png")
                    else:
                        st.info(f"Gambar `{os.path.basename(img_path) if img_path else 'N/A'}` belum tersedia atau belum dibuat.")

                st.subheader("📊 Ringkasan EDA")
                # Tampilkan ringkasan missing values
                if results['missing'] is None:
                    st.write("Tidak ada missing values.")
                else:
                    st.write("Missing values per kolom:")
                    st.json(results['missing'])

                # Tampilkan jumlah outlier
                st.write("Jumlah outlier per kolom:")
                st.json(results['outlier_counts'])

                # Tampilkan distribusi kelas jika ada
                if results['class_counts']:
                    st.write("Distribusi kelas:")
                    st.json(results['class_counts'])
                else:
                    st.write("Kolom label tidak ditemukan atau tidak diberikan.")

                # Tampilkan korelasi numerik jika ada
                if results['correlation_matrix'] is not None:
                    st.write("Matriks korelasi numerik:")
                    st.dataframe(results['correlation_matrix'])
                else:
                    st.write("Tidak cukup kolom numerik untuk menghitung korelasi.")

                # Tampilkan jumlah duplikat
                st.write(f"Jumlah baris duplikat: {results['duplicates']}")

                st.warning("📎 Catatan: Semua visualisasi dihasilkan oleh fungsi `eda_summary()` dan disimpan di direktori `eda_output`.")

    if st.checkbox("Modul source_counter"):

        kolom = st.selectbox(
            "Pilih kolom dalam X_train untuk dihitung frekuensinya", 
            X_train.columns,
            key="source_counter_kolom_selectbox"
        )

        if st.button("Jalankan count_sources"):
            st.session_state['source_counter_done'] = True
            st.session_state['source_counter_kolom'] = kolom

        if st.session_state.get('source_counter_done', False):
            kolom_terpilih = st.session_state.get('source_counter_kolom')
            if kolom_terpilih in X_train.columns:
                X_train = count_sources(X_train, kolom_terpilih)
                X_test = count_sources(X_test, kolom_terpilih)

                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test

                st.success(f"Frekuensi dari '{kolom_terpilih}' berhasil ditambahkan.")
                st.dataframe(X_train.head())

                # Gabungkan semua dalam satu dictionary
                data_pickle = {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": st.session_state['y_train'],
                    "y_test": st.session_state['y_test']
                }

                # Serialisasi semua ke satu file pickle
                combined_pkl = pickle.dumps(data_pickle)

                # Tombol untuk download satu file .pkl
                st.download_button(
                    label="Download Data (.pkl)",
                    data=combined_pkl,
                    file_name="processed_sources_count_data.pkl"
                )
            else:
                st.error("Kolom tidak ditemukan dalam X_train.")

    if st.checkbox("Gunakan Geocoding"):
        from geocoding import create_location
        st.markdown("### Konfigurasi Geocoding")
        
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']

        # Parameter dari pengguna
        lat_col = st.selectbox("Pilih kolom latitude", X_train.columns, index=X_train.columns.get_loc("latitude") if "latitude" in X_train.columns else 0)
        lon_col = st.selectbox("Pilih kolom longitude", X_train.columns, index=X_train.columns.get_loc("longitude") if "longitude" in X_train.columns else 0)
        loc_col = st.text_input("Nama kolom lokasi baru", "location")
        api_keys_input = st.text_area("Masukkan API keys (pisahkan dengan koma)", placeholder="API_KEY_1,API_KEY_key2,API_KEY_3")
        api_keys = [key.strip() for key in api_keys_input.split(",") if key.strip()]

        limit = st.number_input("Batas jumlah baris yang diproses (limit)", min_value=1, value=5000, step=100)
        start_index = st.number_input("Index awal (start_index)", min_value=0, value=0)
        cache_file = st.text_input("Nama file cache", "geocode_cache.json")
        
        uploaded_cache = st.file_uploader("Upload file cache geocoding (opsional)", type=["json"])
        
        if uploaded_cache is not None:
            import json
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                tmp.write(uploaded_cache.read())
                cache_file = tmp.name
                st.success("Cache berhasil diupload dan akan digunakan.")
        if st.button("Jalankan Geocoding"):
            st.session_state['geocoding_done'] = True
        if st.session_state.get('geocoding_done', False):
            if not api_keys:
                st.error("Harap masukkan setidaknya satu API key.")
            else:
                with st.spinner("Sedang menjalankan geocoding pada X_train..."):
                    X_train = create_location(
                        X_train, api_keys,
                        lat_col=lat_col,
                        lon_col=lon_col,
                        loc_col=loc_col,
                        limit=limit,
                        start_index=start_index,
                        cache_file=cache_file
                    )
                with st.spinner("Sedang menjalankan geocoding pada X_test..."):
                    X_test = create_location(
                        X_test, api_keys,
                        lat_col=lat_col,
                        lon_col=lon_col,
                        loc_col=loc_col,
                        limit=limit,
                        start_index=start_index,
                        cache_file=cache_file
                    )
                # Simpan kembali ke session_state
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test

                st.success("Geocoding selesai!")
                st.dataframe(X_train)
                st.dataframe(X_test)
                
                data_pickle = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": st.session_state['y_train'],
                "y_test": st.session_state['y_test']
                }

                # Serialisasi semua ke satu file pickle
                combined_pkl = pickle.dumps(data_pickle)

                # Tombol untuk download satu file .pkl
                st.download_button(
                    label="Download Data (.pkl)",
                    data=combined_pkl,
                    file_name="processed_geocoding_data.pkl"
                )

    # Fitur Encoding
    if st.checkbox("Gunakan Encoding"):
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        import pickle

        X_train = st.session_state.get('X_train')
        X_test = st.session_state.get('X_test')
        y_train = st.session_state.get('y_train')
        y_test = st.session_state.get('y_test')

        if X_train is None or X_test is None:
            st.warning("Data X_train dan X_test belum tersedia.")
        else:
            if 'original_X_train' not in st.session_state:
                st.session_state['original_X_train'] = X_train.copy()
                st.session_state['original_X_test'] = X_test.copy()

            all_columns = st.session_state['original_X_train'].select_dtypes(include=['object', 'string']).columns.tolist()
            selected_columns = st.multiselect("Pilih kolom untuk encoding", all_columns)
            method = st.selectbox("Pilih metode encoding", ["ordinal", "label"])

            if(st.button("Jalankan encoding")):
                st.session_state['encoding_done'] = True
            if st.session_state.get('encoding_done',False):
                if not selected_columns:
                    st.warning("Pilih setidaknya satu kolom.")
                else:
                    X_train_enc = X_train.copy()
                    X_test_enc = X_test.copy()
                    encoders = {}

                    for col in selected_columns:
                        if method == "label":
                            le = LabelEncoder()
                            X_train_enc[col] = le.fit_transform(X_train_enc[col].astype(str))
                            X_test_enc[col] = le.transform(X_test_enc[col].astype(str))
                            encoders[col] = le

                        elif method == "ordinal":
                            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                            X_train_enc[[col]] = oe.fit_transform(X_train_enc[[col]].astype(str))
                            X_test_enc[[col]] = oe.transform(X_test_enc[[col]].astype(str))
                            encoders[col] = oe

                        # Pastikan hasilnya numerik
                        X_train_enc[col] = X_train_enc[col].astype(float)
                        X_test_enc[col] = X_test_enc[col].astype(float)

                    # Simpan hasil ke session_state
                    st.session_state['X_train'] = X_train_enc
                    st.session_state['X_test'] = X_test_enc
                    st.session_state['encoding_done'] = True
                    st.session_state['encoding_columns'] = selected_columns
                    st.session_state['encoding_method'] = method

                    st.success("Encoding berhasil diterapkan.")
                    st.dataframe(X_train_enc.head())
                    st.dataframe(X_test_enc.head())

                    # Unduh encoder per kolom
                    for col, encoder in encoders.items():
                        encoder_bytes = pickle.dumps(encoder)
                        st.download_button(
                            label=f"Unduh encoder untuk kolom: {col}",
                            data=encoder_bytes,
                            file_name=f"{col}_{method}_encoder.pkl"
                        )

                    # Unduh seluruh data hasil encode
                    encoded_data = {
                        'X_train': X_train_enc,
                        'X_test': X_test_enc,
                        'y_train': y_train,
                        'y_test': y_test
                    }

                    encoded_pickle = pickle.dumps(encoded_data)

                    st.download_button(
                        label="Unduh Data Hasil Encoding (.pkl)",
                        data=encoded_pickle,
                        file_name="encoded_data.pkl"
                    )

            if st.session_state.get('encoding_done', False):
                st.info(f"Encoding sudah diterapkan pada kolom: {st.session_state.get('encoding_columns', [])}")

    if st.checkbox("Gunakan seleksi fitur"):
        from feature_selection import feature_selection_based_on_feature_importance
        import tempfile
        import pickle
        import os

        X_train_fs = st.session_state.get('X_train')
        y_train_fs = st.session_state.get('y_train')
        X_test_fs = st.session_state.get('X_test')
        y_test_fs = st.session_state.get('y_test')
        categorical_cols = X_train_fs.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        if X_train_fs is None or y_train_fs is None:
            st.warning("X_train atau y_train belum tersedia.")
        else:
            st.write(X_train_fs.dtypes)
            if categorical_cols and not st.session_state.get('encoding_done', False):
                st.warning("Terdapat kolom kategori yang belum diencoding: {categorical_cols}. Harap jalankan encoding terlebih dahulu.")
            else:
                st.markdown("### Konfigurasi Seleksi Fitur Berdasarkan Feature Importance")
                algo = st.selectbox("Pilih algoritma", ["xgb"])
                top_n = st.number_input("Jumlah fitur teratas (opsional, 0 = semua)", min_value=0, value=0, step=1)
                threshold = st.slider("Threshold importance minimum", min_value=0.0, max_value=1.0, value=0.01, step=0.01)

                if st.button("Jalankan Seleksi Fitur"):
                    st.session_state['fs_done'] = True
                if st.session_state.get('fs_done', False):
                    with st.spinner("Sedang menjalankan seleksi fitur..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_png, \
                                tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_pkl:

                            save_path = tmp_png.name
                            save_df_path = tmp_pkl.name

                        model, X_selected = feature_selection_based_on_feature_importance(
                            X_train=X_train_fs,
                            y_train=y_train_fs,
                            algo=algo,
                            top_n=top_n if top_n > 0 else None,
                            importance_threshold=threshold,
                            save_path=save_path,
                            save_df_path=save_df_path,
                        )
                        # Update session_state hasil seleksi
                        st.session_state['X_train'] = X_selected
                        st.session_state['y_train'] = y_train_fs
                        st.session_state['X_test'] = X_test_fs
                        st.session_state['y_test'] = y_test_fs
                        
                        selected_data = {
                            'X_train': X_selected,
                            'X_test': X_test_fs,
                            'y_train': y_train_fs,
                            'y_test': y_test_fs
                        }
                        selected_data_bytes = pickle.dumps(selected_data)
                        # Tandai sudah selesai fs
                        st.session_state['fs_done'] = True

                        st.success("Seleksi fitur selesai!")
                        st.write("Data setelah seleksi:")
                        st.dataframe(X_selected.head())

                        st.image(save_path, caption="Feature Importance")
                        with open(save_path, "rb") as f_img:
                            st.download_button(
                                label="Download Gambar Feature Importance (.png)",
                                data=f_img,
                                file_name="feature_importance.png"
                            )

                        st.download_button(
                            label="Download Data Hasil Seleksi (.pkl)",
                            data=selected_data_bytes,
                            file_name="selected_features.pkl"
                        )

                        os.unlink(save_path)
                        os.unlink(save_df_path)

    if st.checkbox("Gunakan tuning model"):
        from hyperparameter_tuning import tune_model
        import json
        X_train = st.session_state.get('X_train')
        y_train = st.session_state.get('y_train')
        st.write(X_train)
                    
        all_col = X_train.columns.tolist()
        numeric_features = st.multiselect("Pilih kolom numerik untuk preprocessing", all_col)
        
        algo = st.selectbox("Pilih algoritma", options=["xgb", "rf"])
        # --- Imputer strategy ---
        imputer_strategy = st.selectbox(
            "Strategi imputasi untuk fitur numerik",
            options=["none", "mean", "median"],
            index=0
        )
        if imputer_strategy == "none":
            imputer_strategy = None

       # --- Resampler ---
        resampler = st.selectbox("Resampling method", options=["none", "smoteenn", "smotetomek"])
        if resampler == "none":
            resampler = None

        # --- Scaler ---
        scaler = st.selectbox("Scaler method", options=["none", "robust", "standard"])
        if scaler == "none":
            scaler = None

        # --- Other params ---
        cv_splits = st.number_input("CV splits (StratifiedKFold)", min_value=2, max_value=10, value=5, step=1)
        init_points = st.number_input("Bayesian Optimization init points", min_value=1, max_value=50, value=20, step=1)
        n_iter = st.number_input("Bayesian Optimization iterations", min_value=1, max_value=200, value=100, step=1)

        # --- Button untuk mulai tuning ---
        if st.button("Mulai Tuning"):
            if not numeric_features:
                st.warning("Tidak ada fitur numerik yang dipilih. Tuning akan dilakukan tanpa preprocessing numerik.")
            # Menggunakan seluruh kolom di X_train untuk mencari hyperparameter
            # X_train_selected = X_train.copy()
            # y_train_selected = y_train

            with st.spinner("Sedang melakukan tuning, mohon tunggu..."):
                best_params = tune_model(
                    X_train=X_train,
                    y_train=y_train,
                    numeric_features=numeric_features,
                    algo=algo,
                    resampler=resampler,
                    scaler=scaler,
                    imputer_strategy=imputer_strategy,
                    cv_splits=cv_splits,
                    init_points=init_points,
                    n_iter=n_iter,
                    verbose=3,
                    random_state=42,
                    return_scores=False
                )
            st.session_state['best_parameter_tuning'] = best_params
            st.success("Tuning selesai!")
            st.write("Best parameters:")
            st.json(best_params)
            json_str = json.dumps(best_params, indent=4)
            # Download button
            st.download_button(
                label="Download Best Parameters",
                data=json_str,
                file_name="best_params.json",
                mime="application/json"
            )

    import io
    import json
    import os
    import tempfile

    if st.checkbox("Validasi model"):
        
        X = st.session_state.get('X_train')
        y = st.session_state.get('y_train')
        
        # Pilihan parameter dari user
        algo = st.selectbox("Pilih algoritma", ["xgb", "rf"],key="algovalidasi")
        imputer_strategy = st.selectbox("Strategi imputasi untuk fitur numerik ", options=["none", "mean", "median"], index=0)
        if imputer_strategy == "none":
            imputer_strategy = None
    
        # --- Metode Resampling ---
        resample_type = st.selectbox(
            "Metode Resampling",
            options=["none", "smote", "smoteenn", "smotetomek"],
            index=0
        )
        if resample_type == "none":
            resample_type = None

        # --- Skaler ---
        scaler_type = st.selectbox(
            "Scaler Method ",
            options=["none", "robust", "standard"],
            index=0
        )
        if scaler_type == "none":
            scaler_type = None

        cv_splits = st.number_input(
            "CV splits (StratifiedKFold)", 
            min_value=2, max_value=10, 
            value=5, step=1, 
            key="split_validasi"
        )
        uploaded_best_params = st.file_uploader("Upload file Best Params (.json, opsional)", type=["json"])

        best_params_path = None
        if uploaded_best_params is not None:
            try:
                st.success("Berhasil upload dan Best Params akan digunakan.")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

        if st.button("Mulai Validasi Model"):
            st.write("🔄 Sedang melakukan validasi...")

            df_f1 = validate_model(
                X=X,
                y=y,
                numeric_features=numeric_features,
                algo=algo,
                resampler_type=resample_type,
                scaler_type=scaler_type,
                imputer_strategy=imputer_strategy,
                best_param_path=best_params_path,
                cv_splits=cv_splits
            )

            st.write("Hasil F1 Score per Fold:")
            st.dataframe(df_f1)

            # Tampilkan gambar jika ada
            if os.path.exists("validation_results.png"):
                with open("validation_results.png", "rb") as f:
                    img_bytes = f.read()
                st.image(img_bytes, caption="F1 Score per Kelas untuk Tiap Fold", use_container_width=True)

                # Tombol download gambar
                st.download_button(
                    label="📥 Download Gambar (F1 Score per Fold)",
                    data=img_bytes,
                    file_name="f1_score_plot.png",
                    mime="image/png"
                )

            # Tombol download class report dalam format .txt
            txt_buffer = io.StringIO()
            df_f1.to_string(buf=txt_buffer, index=False)
            st.download_button(
                label="📥 Download Class Report (.txt)",
                data=txt_buffer.getvalue(),
                file_name="class_report.txt",
                mime="text/plain"
            )
            
    if st.checkbox("Bangun model"):
        from build_model import build_model
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train

        categorical_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        if categorical_cols and not st.session_state.get('encoding_done', False):
            st.warning(f"Terdapat kolom kategori yang belum diencoding: {categorical_cols}. Harap jalankan encoding terlebih dahulu.")

        else:
            # Pilih fitur numerik & kategorikal
            numeric_features = st.multiselect("Pilih kolom numerik untuk preprocessing", X_train.columns.tolist(), key='num_feats build model')

            # Pilih model & konfigurasi
            model_type = st.selectbox("Pilih Algoritma", ['xgb', 'rf'], key ="algo build model")
            imputer_strategy = st.selectbox("Strategi imputasi untuk fitur numerik ", options=["none", "mean", "median"], index=0, key="imputer build model")
            if imputer_strategy == "none":
                imputer_strategy = None
            
           # --- Metode Resampling ---
            resample_type = st.selectbox(
                "Metode Resampling",
                options=["none", "smote", "smoteenn", "smotetomek"],
                index=0, key="resample build model"
            )
            if resample_type == "none":
                resample_type = None

            # --- Skaler ---
            scaler_type = st.selectbox(
                "Scaler Method ",
                options=["none", "robust", "standard"],
                index=0, key="scaler build model"
            )
            if scaler_type == "none":
                scaler_type = None

            save_model = st.checkbox("Simpan Model sebagai File .pkl?")
            model_path = st.text_input("Nama File Model", value="trained_model.pkl") if save_model else None

            uploaded_best_params = st.file_uploader("Upload file Best Params (JSON, opsional)", type=["json"])
            
            if uploaded_best_params is not None:
                try:
                    best_params = json.load(uploaded_best_params)
                    st.write(best_params)
                    st.success("Berhasil upload dan Best Params digunakan.")
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")
            else:
                best_params = None
            # Tombol Train
                
            if st.button("🚀 Train Model"):
                st.session_state['train_done'] = True
            if st.session_state.get('train_done',False):
                with st.spinner("Training model..."):

                    # Build the model
                    model = build_model(
                        X_train=X_train,
                        y_train=y_train,
                        numeric_features=numeric_features,
                        category_cols=None,
                        best_param_path=None,
                        model_params=best_params,
                        model_type=model_type,
                        imputer_strategy=imputer_strategy,
                        scaler_type=scaler_type,
                        resample_type=resample_type,
                        save_model=save_model,
                        model_path=model_path
                    )

                    # Simpan ke session state
                    st.session_state['trained_model'] = model
                    st.success("✅ Model berhasil dilatih dan disimpan di session state.")

                    if save_model:
                        st.success(f"Model juga disimpan ke file: `{model_path}`")

                    # Tombol download
                    if save_model:
                        with open(model_path, "rb") as f:
                            model_bytes = f.read()
                        st.download_button("⬇️ Download Model", model_bytes, file_name=model_path, mime="application/octet-stream")

    if st.checkbox("Testing"):
        from test import evaluate_model  
        import pickle
        import os
        import io
        X_train = st.session_state.get('X_train')
        y_train = st.session_state.get('y_train')
        X_test = st.session_state.get('X_test')
        y_test = st.session_state.get('y_test')
        model = st.session_state.get('trained_model')  # Lebih aman dari KeyError
        if model is None:
            uploaded_model = st.file_uploader("Upload file model (.pkl)", type=["pkl"])
            if uploaded_model is not None:
                try:
                    model = pickle.load(uploaded_model)
                    st.session_state['trained_model'] = model  # Simpan ke session jika berhasil
                    st.success("Model berhasil dimuat.")
                except Exception as e:
                    st.error(f"Gagal memuat model: {e}")
            else:
                st.warning("Silakan upload file model terlebih dahulu.")
        
        # Evaluasi model
        if st.button("Mulai Pengujian Model"):
            with st.spinner("⏳ Mengevaluasi model..."):
                y_pred,acc, report, cm, roc_auc_train, roc_auc_test = evaluate_model(
                    model_or_path=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    output_dir="output"
                )

            st.subheader("🎯 Akurasi")
            st.metric(label="Akurasi", value=f"{acc:.5f}")

            # Tombol download classification report
            report_txt = f"Accuracy: {acc:.4f}\n\n{report}"
            from sklearn.metrics import classification_report
            import pandas as pd

            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.write(report_df)

            st.download_button(
                label="📥 Download Classification Report (.txt)",
                data=report_txt,
                file_name="classification_report.txt",
                mime="text/plain"
            )

            # Tampilkan confusion matrix
            cm_path = os.path.join("output", "confusion_matrix.png")
            if os.path.exists(cm_path):
                with open(cm_path, "rb") as img:
                    st.image(img.read(), caption="Confusion Matrix", use_container_width=True)
                    st.download_button(
                        label="📥 Download Confusion Matrix (.png)",
                        data=img,
                        file_name="confusion_matrix.png",
                        mime="image/png"
                    )

            # Tampilkan ROC Curve jika ada
            roc_path = os.path.join("output", "roc_curve.png")
            if os.path.exists(roc_path):
                with open(roc_path, "rb") as img:
                    st.image(img.read(), caption="ROC Curve", use_container_width=True)
                    st.download_button(
                        label="📥 Download ROC Curve (.png)",
                        data=img,
                        file_name="roc_curve.png",
                        mime="image/png"
                    )

    if st.button("session"):
        for key, value in st.session_state.items():
            st.write(f"{key}: {value}")
            if(key=="trained_model"):
                st.write(st.session_state["trained_model"].named_steps['model'].get_params())
        import sklearn
        import imblearn
        import numpy as np

        st.write("scikit-learn:", sklearn.__version__)
        st.write("imb:", imblearn.__version__)
        st.write("np:", np.__version__)


                



elif halaman == "Beranda":
    st.title("Prediksi Potensi Tsunami dengan algoritma XGBoost")
    st.write("Masukkan parameter gempa bumi untuk prediksi:")

    # Input features with safe_float_input
    mag = safe_float_input("Magnitudo", "5.0")
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
    felt = safe_float_input("Felt", "200")
    magType = st.text_input("Tipe Magnitudo", "mw")
    location = st.text_input("Lokasi", "Indonesia")

    # If location is empty, use reverse geocoding to fill it
    if not location:
        location = get_location(latitude, longitude, 0)
    # Prepare input_cats with all features (magType,  location)
    input_cats = pd.DataFrame([[magType, location]], columns=['magType', 'location'])
    encoded_cats = encoder.transform(input_cats)

    # Handle negative depth
    if depth < 0:
        depth = 18.4599990844727  # Median value based on X_train

    # Process the sources input
    sources_input = st.text_input("Sources (pisahkan dengan koma, contoh: ax,us,tx)", value="ax,us")
    if sources_input.strip() == "":
        source_counter = np.nan
    else:
        sources_list = [s.strip() for s in sources_input.split(',') if s.strip()]
        source_counter = len(sources_list)

    # Prepare input for prediction (including encoded magType and location)
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

    encoded_input = pd.DataFrame([input_dict])
    # Assign encoded values to magType and location
    encoded_input[['magType', 'location']] = encoded_cats[:, [0, 1]]  # Use encoded magType and location


    # Predict when the button is pressed
    if st.button("Prediksi"):
        pred = model.predict(encoded_input)
        st.write("Hasil Prediksi:", int(pred[0]))
        if(int(pred[0])==1):
            st.write("Gempa berpotensi menyebabkan Tsunami")
        else:
            st.write("Gempa tidak berpotensi Tsunami")
        
        map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
        st.subheader("Lokasi Gempa di Peta")
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[lon, lat]',
            get_color='[255, 0, 0, 160]',
            get_radius=10000,
        )

        # State tampilan awal peta
        view_state = pdk.ViewState(
            latitude=latitude,
            longitude=longitude,
            zoom=6,
            pitch=0,
        )

        # Tampilan satelit
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/satellite-v9",  
            initial_view_state=view_state,
            layers=[layer]
        ))

    # Show data to user
    st.subheader("\nData yang Dikirim ke Model:")
    st.write(encoded_input)
elif halaman == "Panduan":
    st.title("Panduan Input Parameter Gempa Bumi")
    st.write("Berikut adalah deskripsi dari masing-masing parameter yang digunakan untuk memproses data gempa:")

    st.markdown("### Daftar Parameter:")
    st.markdown("""
    - **magnitudo**: Magnitudo gempa yang menunjukkan kekuatan gempa (contoh: `5.6`).
    - **nst**: Jumlah stasiun seismik yang digunakan untuk menghitung lokasi gempa.
    - **longitude**: Koordinat geografis garis bujur tempat gempa terjadi.
    - **latitude**: Koordinat geografis garis lintang tempat gempa terjadi.
    - **depth**: Kedalaman pusat gempa dalam kilometer.
    - **Significance**: Signifikansi dari gempa (nilai semakin tinggi, semakin signifikan).
    - **cdi**: Community Internet Intensity – rata-rata dari laporan masyarakat.
    - **mmi**: Indikator intensitas gempa yang dirasakan (Modified Mercalli Intensity).
    - **rms**: Root mean square dari sinyal gempa, menggambarkan kekuatan getaran.
    - **gap**: Sudut maksimum antara stasiun yang mencatat gempa terhadap pusat gempa.
    - **dmin**: Jarak minimum dari lokasi kejadian ke stasiun pencatat dalam derajat.
    - **felt**: Jumlah laporan dari orang yang merasakan gempa.
    - **Tipe Magnitudo**: Jenis skala magnitudo yang digunakan, seperti `mb`, `ml`, `mw`.
    - **lokasi**: Deskripsi lokasi terjadinya gempa (misalnya nama negara atau laut, contoh : Indonesia).
    - **sources**: sumber yang mencatat kejadian gempa (contoh: ax,tx).
    """)

    # Penjelasan tambahan jika ada format khusus
    st.markdown("### 🔍 Catatan Format:")
    st.write("- Parameter seperti `magnitudo`, `nst`, dan lainnya bisa disediakan dalam bentuk numerik (float atau int), sesuai dengan konteks input data.")
    st.write("- Parameter `location`, `Tipe Magnitudo`, dan `sources` harus berupa teks/string.")
    st.write("Kategori yang telah dipelajari oleh encoder:")
    for i, cats in enumerate(encoder.categories_):
        st.write(f"Kolom {i}: {cats}")
    st.write("\n- Harap menghubungi surel D1041211020@student.untan.ac.id bila terdapat kritik dan saran.")
    
