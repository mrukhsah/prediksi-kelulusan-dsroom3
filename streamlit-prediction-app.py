"""
🎓 Aplikasi Prediksi Kelulusan Mahasiswa
Streamlit App untuk prediksi IPK menggunakan berbagai algoritma Machine Learning
Author: [Your Name]
GitHub: [Your GitHub URL]
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Setup
warnings.filterwarnings('ignore')
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #2563EB;
    }
    .success-box {
        background-color: #D1FAE5;
        border: 1px solid #10B981;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .feature-importance-bar {
        height: 8px;
        background: linear-gradient(90deg, #3B82F6, #10B981);
        border-radius: 4px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk logging
def log_activity(activity, details):
    """Log aktivitas aplikasi"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {activity}: {details}"
    logging.info(log_entry)
    print(log_entry)

# Fungsi untuk menghitung semua metrik evaluasi
def calculate_metrics(y_true, y_pred):
    """Menghitung berbagai metrik evaluasi model"""
    try:
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE dengan handling division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, np.nan))) * 100
            mape = np.nan_to_num(mape, nan=0.0)
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': round(mae, 4),
            'MAPE': round(mape, 2),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4)
        }
    except Exception as e:
        log_activity("ERROR", f"Error calculating metrics: {str(e)}")
        return {
            'MAE': np.nan,
            'MAPE': np.nan,
            'MSE': np.nan,
            'RMSE': np.nan,
            'R2': np.nan
        }

# Fungsi validasi dataset
def validate_dataset(df):
    """Validasi struktur dan data dari dataset"""
    required_columns = ['nama', 'jenis_kelamin', 'umur', 'status_menikah', 
                       'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 
                       'aktivitas_elearning', 'ipk']
    
    # Cek kolom yang dibutuhkan
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Kolom berikut tidak ditemukan: {', '.join(missing_cols)}"
    
    # Validasi tipe data
    numeric_cols = ['umur', 'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning', 'ipk']
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            return False, f"Kolom '{col}' mengandung nilai non-numerik"
    
    # Validasi range
    if (df['kehadiran'] < 0).any() or (df['kehadiran'] > 100).any():
        return False, "Kehadiran harus berada di range 0-100%"
    
    if (df['ipk'] < 0).any() or (df['ipk'] > 4).any():
        return False, "IPK harus berada di range 0-4"
    
    # Cek missing values
    missing_values = df[required_columns].isnull().sum().sum()
    if missing_values > 0:
        return False, f"Terdapat {missing_values} nilai kosong (missing values) dalam dataset"
    
    return True, "Dataset valid"

# Fungsi preprocessing data
def preprocess_data(df):
    """Preprocessing data untuk training"""
    df_processed = df.copy()
    
    # Encode categorical features
    le_gender = LabelEncoder()
    le_married = LabelEncoder()
    
    df_processed['jenis_kelamin_encoded'] = le_gender.fit_transform(df_processed['jenis_kelamin'])
    df_processed['status_menikah_encoded'] = le_married.fit_transform(df_processed['status_menikah'])
    
    return df_processed, le_gender, le_married

# Fungsi untuk training model
def train_model(df, model_type='random_forest', hyperparams=None):
    """Training model dengan cross-validation"""
    if hyperparams is None:
        hyperparams = {}
    
    # Preprocessing
    df_processed, le_gender, le_married = preprocess_data(df)
    
    # Features dan Target
    features = ['jenis_kelamin_encoded', 'umur', 'status_menikah_encoded', 
                'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning']
    
    X = df_processed[features]
    y = df_processed['ipk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling untuk KNN
    scaler = StandardScaler()
    if model_type == 'knn':
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Pilih model dengan hyperparameters
    if model_type == 'random_forest':
        n_estimators = hyperparams.get('n_estimators', 100)
        max_depth = hyperparams.get('max_depth', 10)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    elif model_type == 'gradient_boosting':
        n_estimators = hyperparams.get('n_estimators', 100)
        max_depth = hyperparams.get('max_depth', 5)
        learning_rate = hyperparams.get('learning_rate', 0.1)
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
    elif model_type == 'knn':
        n_neighbors = hyperparams.get('n_neighbors', 5)
        weights = hyperparams.get('weights', 'distance')
        model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights
        )
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    if model_type == 'knn':
        X_scaled = scaler.fit_transform(X)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
    else:
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Training model final
    model.fit(X_train_scaled, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test_scaled)
    
    # Hitung metrik
    metrics = calculate_metrics(y_test, y_pred)
    
    # Log training
    log_activity("TRAINING", f"Model {model_type} trained successfully. R2: {metrics['R2']}")
    
    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'metrics': metrics,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'le_gender': le_gender,
        'le_married': le_married,
        'scaler': scaler,
        'features': features,
        'hyperparams': hyperparams
    }

# Fungsi untuk menyimpan model
def save_model(model_data, model_name):
    """Menyimpan model ke disk"""
    try:
        os.makedirs('saved_models', exist_ok=True)
        
        # Simpan model
        model_path = f"saved_models/{model_name}_model.pkl"
        joblib.dump(model_data['model'], model_path)
        
        # Simpan preprocessing objects
        preprocess_path = f"saved_models/{model_name}_preprocess.pkl"
        joblib.dump({
            'le_gender': model_data['le_gender'],
            'le_married': model_data['le_married'],
            'scaler': model_data['scaler'],
            'features': model_data['features']
        }, preprocess_path)
        
        log_activity("SAVE", f"Model {model_name} saved successfully")
        return True, "Model berhasil disimpan"
    except Exception as e:
        log_activity("ERROR", f"Error saving model: {str(e)}")
        return False, f"Error: {str(e)}"

# Fungsi untuk memuat model
def load_model(model_name):
    """Memuat model dari disk"""
    try:
        model_path = f"saved_models/{model_name}_model.pkl"
        preprocess_path = f"saved_models/{model_name}_preprocess.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(preprocess_path):
            return False, "File model tidak ditemukan"
        
        model = joblib.load(model_path)
        preprocess = joblib.load(preprocess_path)
        
        return True, {
            'model': model,
            'le_gender': preprocess['le_gender'],
            'le_married': preprocess['le_married'],
            'scaler': preprocess['scaler'],
            'features': preprocess['features']
        }
    except Exception as e:
        log_activity("ERROR", f"Error loading model: {str(e)}")
        return False, f"Error: {str(e)}"

# Header utama
st.markdown('<h1 class="main-header">🎓 Aplikasi Prediksi Kelulusan Mahasiswa</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.1rem; color: #6B7280;">
        Prediksi IPK mahasiswa menggunakan Machine Learning dengan berbagai algoritma
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
    st.markdown("### 📋 Menu Navigasi")
    
    menu = st.radio(
        "Pilih menu:",
        ["🏠 Dashboard", "📤 Upload & Training", "🔮 Prediksi Individual", 
         "📊 Prediksi Batch", "📈 Visualisasi", "⚖️ Perbandingan Model", 
         "💾 Model Management"]
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Pengaturan")
    
    # Theme toggle
    theme = st.selectbox("Tema", ["Light", "Dark"])
    
    # Data sample
    if st.button("📋 Contoh Dataset"):
        sample_data = {
            'nama': ['Andi', 'Budi', 'Citra', 'Dewi', 'Eko'],
            'jenis_kelamin': ['Laki-laki', 'Laki-laki', 'Perempuan', 'Perempuan', 'Laki-laki'],
            'umur': [21, 22, 20, 23, 21],
            'status_menikah': ['Belum Menikah', 'Belum Menikah', 'Belum Menikah', 'Menikah', 'Belum Menikah'],
            'kehadiran': [85, 90, 95, 80, 75],
            'partisipasi_diskusi': [80, 85, 90, 75, 70],
            'nilai_tugas': [85.5, 88.0, 92.5, 78.0, 82.5],
            'aktivitas_elearning': [70, 75, 85, 65, 60],
            'ipk': [3.5, 3.7, 3.9, 3.2, 3.0]
        }
        sample_df = pd.DataFrame(sample_data)
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="Download Contoh CSV",
            data=csv,
            file_name="contoh_dataset.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    st.markdown("#### 📊 Info Aplikasi")
    st.markdown("""
    **Version:** 2.0  
    **Algoritma:** 
    - Random Forest
    - Gradient Boosting
    - KNN Regressor
    
    **Metrik Evaluasi:**
    - MAE, MAPE, RMSE
    - R² Score
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Session state untuk menyimpan model dan data
if 'models' not in st.session_state:
    st.session_state.models = {}
    st.session_state.results = {}
    st.session_state.trained = False
    st.session_state.df = None
    st.session_state.saved_models = []

# MENU 1: Dashboard
if menu == "🏠 Dashboard":
    st.markdown('<h2 class="sub-header">📊 Dashboard Utama</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Tersedia", len(st.session_state.models))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.df is not None:
            st.metric("Data Points", len(st.session_state.df))
        else:
            st.metric("Data Points", 0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        best_r2 = 0
        best_model = "-"
        for model_type, result in st.session_state.results.items():
            if result['metrics']['R2'] > best_r2:
                best_r2 = result['metrics']['R2']
                best_model = result.get('name', model_type)
        st.metric("Best Model R²", f"{best_r2:.3f}")
        st.caption(f"Model: {best_model}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🚀 Quick Start Guide")
    
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    
    with guide_col1:
        st.markdown("""
        ### 1. Upload Data
        - Klik menu **Upload & Training**
        - Upload dataset CSV/Excel
        - Pastikan format sesuai
        """)
    
    with guide_col2:
        st.markdown("""
        ### 2. Training Model
        - Pilih algoritma yang diinginkan
        - Atur hyperparameters (opsional)
        - Klik **Mulai Training**
        """)
    
    with guide_col3:
        st.markdown("""
        ### 3. Prediksi & Analisis
        - **Prediksi Individual** untuk satu mahasiswa
        - **Prediksi Batch** untuk banyak data
        - **Visualisasi** untuk analisis
        """)
    
    st.markdown("### 📋 Dataset Format")
    
    with st.expander("Lihat format dataset yang dibutuhkan"):
        st.markdown("""
        Dataset harus memiliki kolom berikut:
        
        | Kolom | Tipe Data | Contoh | Keterangan |
        |-------|-----------|--------|------------|
        | nama | String | "Andi" | Nama mahasiswa |
        | jenis_kelamin | String | "Laki-laki" / "Perempuan" | Jenis kelamin |
        | umur | Integer | 21 | Umur dalam tahun |
        | status_menikah | String | "Belum Menikah" / "Menikah" | Status pernikahan |
        | kehadiran | Numeric (0-100) | 85 | Persentase kehadiran |
        | partisipasi_diskusi | Numeric (0-100) | 80 | Skor partisipasi |
        | nilai_tugas | Numeric (0-100) | 85.5 | Rata-rata nilai tugas |
        | aktivitas_elearning | Numeric (0-100) | 70 | Skor aktivitas e-learning |
        | ipk | Numeric (0-4) | 3.5 | IPK (target variable) |
        
        **Catatan:** IPK harus berada di range 0-4
        """)
    
    # Recent Activity
    if st.session_state.trained:
        st.markdown("### 📈 Recent Activity")
        
        recent_results = []
        for model_type, result in st.session_state.results.items():
            recent_results.append({
                'Model': result.get('name', model_type),
                'R²': result['metrics']['R2'],
                'RMSE': result['metrics']['RMSE'],
                'Status': '✅ Trained'
            })
        
        if recent_results:
            st.dataframe(pd.DataFrame(recent_results), use_container_width=True)

# MENU 2: Upload & Training
elif menu == "📤 Upload & Training":
    st.markdown('<h2 class="sub-header">📤 Upload Dataset dan Training Model</h2>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload file dataset (CSV atau Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Pastikan file memiliki format yang sesuai"
    )
    
    if uploaded_file is not None:
        try:
            # Load data berdasarkan tipe file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validasi dataset
            is_valid, message = validate_dataset(df)
            
            if not is_valid:
                st.error(f"❌ {message}")
            else:
                st.success(f"✅ Dataset berhasil diupload! Total {len(df)} baris data")
                
                # Tampilkan preview data
                st.markdown("### 📊 Preview Data")
                st.dataframe(df.head(), use_container_width=True)
                
                # Statistik deskriptif
                st.markdown("### 📈 Statistik Deskriptif")
                desc_col1, desc_col2, desc_col3 = st.columns(3)
                
                with desc_col1:
                    st.metric("Rata-rata IPK", f"{df['ipk'].mean():.2f}")
                    st.metric("Rata-rata Kehadiran", f"{df['kehadiran'].mean():.1f}%")
                
                with desc_col2:
                    st.metric("Rata-rata Umur", f"{df['umur'].mean():.1f} tahun")
                    st.metric("Rata-rata Nilai Tugas", f"{df['nilai_tugas'].mean():.1f}")
                
                with desc_col3:
                    gender_counts = df['jenis_kelamin'].value_counts()
                    st.metric("Jenis Kelamin", f"L: {gender_counts.get('Laki-laki', 0)} | P: {gender_counts.get('Perempuan', 0)}")
                    st.metric("Status Menikah", f"Belum: {(df['status_menikah'] == 'Belum Menikah').sum()}")
                
                # Pilih model untuk training
                st.markdown("### 🤖 Pilih Model untuk Training")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    train_rf = st.checkbox("Random Forest Regressor", value=True)
                    if train_rf:
                        with st.expander("Hyperparameters RF"):
                            rf_n_estimators = st.slider("n_estimators", 50, 200, 100, key='rf_n')
                            rf_max_depth = st.slider("max_depth", 5, 20, 10, key='rf_depth')
                
                with col2:
                    train_gb = st.checkbox("Gradient Boosting Regressor", value=True)
                    if train_gb:
                        with st.expander("Hyperparameters GB"):
                            gb_n_estimators = st.slider("n_estimators", 50, 200, 100, key='gb_n')
                            gb_max_depth = st.slider("max_depth", 3, 10, 5, key='gb_depth')
                            gb_lr = st.select_slider("learning_rate", 
                                                    options=[0.01, 0.05, 0.1, 0.2, 0.3], 
                                                    value=0.1, key='gb_lr')
                
                with col3:
                    train_knn = st.checkbox("KNN Regressor", value=False)
                    if train_knn:
                        with st.expander("Hyperparameters KNN"):
                            knn_n = st.slider("n_neighbors", 3, 15, 5, key='knn_n')
                            knn_weights = st.selectbox("weights", ['uniform', 'distance'], 
                                                      key='knn_w')
                
                # Tombol training
                if st.button("🚀 Mulai Training Model", type="primary", use_container_width=True):
                    models_to_train = []
                    hyperparams_dict = {}
                    
                    if train_rf:
                        models_to_train.append(('Random Forest', 'random_forest'))
                        hyperparams_dict['random_forest'] = {
                            'n_estimators': rf_n_estimators,
                            'max_depth': rf_max_depth
                        }
                    
                    if train_gb:
                        models_to_train.append(('Gradient Boosting', 'gradient_boosting'))
                        hyperparams_dict['gradient_boosting'] = {
                            'n_estimators': gb_n_estimators,
                            'max_depth': gb_max_depth,
                            'learning_rate': gb_lr
                        }
                    
                    if train_knn:
                        models_to_train.append(('KNN', 'knn'))
                        hyperparams_dict['knn'] = {
                            'n_neighbors': knn_n,
                            'weights': knn_weights
                        }
                    
                    if not models_to_train:
                        st.error("❌ Pilih minimal satu model untuk ditraining!")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, (model_name, model_type) in enumerate(models_to_train):
                            status_text.text(f"Training {model_name}... ({idx+1}/{len(models_to_train)})")
                            
                            try:
                                # Training model
                                result = train_model(
                                    df, 
                                    model_type=model_type,
                                    hyperparams=hyperparams_dict.get(model_type, {})
                                )
                                
                                # Simpan ke session state
                                st.session_state.models[model_type] = {
                                    'model': result['model'],
                                    'scaler': result['scaler'],
                                    'le_gender': result['le_gender'],
                                    'le_married': result['le_married'],
                                    'features': result['features'],
                                    'hyperparams': result['hyperparams']
                                }
                                
                                st.session_state.results[model_type] = {
                                    'name': model_name,
                                    'X_test': result['X_test'],
                                    'y_test': result['y_test'],
                                    'y_pred': result['y_pred'],
                                    'metrics': result['metrics'],
                                    'cv_mean': result['cv_mean'],
                                    'cv_std': result['cv_std']
                                }
                                
                                progress_bar.progress((idx + 1) / len(models_to_train))
                                
                                # Log training success
                                log_activity("TRAINING_SUCCESS", f"{model_name} trained successfully")
                                
                            except Exception as e:
                                st.error(f"❌ Error saat training {model_name}: {str(e)}")
                                log_activity("TRAINING_ERROR", f"{model_name}: {str(e)}")
                        
                        st.session_state.trained = True
                        st.session_state.df = df
                        status_text.text("✅ Training selesai!")
                        
                        # Tampilkan hasil training
                        st.markdown("### 📊 Hasil Evaluasi Model")
                        
                        # Tabel hasil
                        results_data = []
                        for model_type, result in st.session_state.results.items():
                            results_data.append({
                                'Model': result['name'],
                                'MAE': result['metrics']['MAE'],
                                'MAPE (%)': result['metrics']['MAPE'],
                                'RMSE': result['metrics']['RMSE'],
                                'R²': result['metrics']['R2'],
                                'CV R² Mean': f"{result['cv_mean']:.4f}",
                                'CV R² Std': f"{result['cv_std']:.4f}"
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df.style.highlight_min(subset=['MAE', 'MAPE (%)', 'RMSE'])
                                                  .highlight_max(subset=['R²', 'CV R² Mean']), 
                                    use_container_width=True)
                        
                        # Visualisasi perbandingan metrik
                        st.markdown("### 📈 Visualisasi Performa Model")
                        
                        metric_to_plot = st.selectbox(
                            "Pilih metrik untuk visualisasi",
                            ['MAE', 'RMSE', 'R²', 'MAPE (%)']
                        )
                        
                        fig = px.bar(
                            results_df, 
                            x='Model', 
                            y=metric_to_plot,
                            title=f'Perbandingan {metric_to_plot}',
                            color=metric_to_plot,
                            color_continuous_scale='RdYlGn' if metric_to_plot == 'R²' else 'RdYlGn_r'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tombol untuk menyimpan model
                        st.markdown("### 💾 Simpan Model")
                        
                        for model_type, result in st.session_state.results.items():
                            col_save1, col_save2 = st.columns([3, 1])
                            with col_save1:
                                st.write(f"**{result['name']}** - R²: {result['metrics']['R2']:.4f}")
                            with col_save2:
                                if st.button(f"💾 Simpan", key=f"save_{model_type}"):
                                    success, message = save_model(
                                        st.session_state.models[model_type],
                                        result['name'].replace(' ', '_')
                                    )
                                    if success:
                                        st.success(message)
                                    else:
                                        st.error(message)
                        
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")
            log_activity("FILE_ERROR", str(e))
    else:
        st.info("👆 Silakan upload dataset untuk memulai")
        
        # Display dataset format
        st.markdown("""
        ### 📋 Format Dataset yang Diperlukan
        
        Dataset harus dalam format CSV atau Excel dengan kolom berikut:
        
        ```csv
        nama,jenis_kelamin,umur,status_menikah,kehadiran,partisipasi_diskusi,nilai_tugas,aktivitas_elearning,ipk
        Andi,Laki-laki,21,Belum Menikah,85,80,85.5,70,3.5
        Budi,Laki-laki,22,Belum Menikah,90,85,88.0,75,3.7
        Citra,Perempuan,20,Belum Menikah,95,90,92.5,85,3.9
        ```
        
        **Keterangan:**
        - `kehadiran`: Persentase (0-100%)
        - `partisipasi_diskusi`: Skor (0-100)
        - `nilai_tugas`: Rata-rata nilai (0-100)
        - `aktivitas_elearning`: Skor (0-100)
        - `ipk`: IPK akhir (0-4)
        """)

# MENU 3: Prediksi Individual
elif menu == "🔮 Prediksi Individual":
    st.markdown('<h2 class="sub-header">🔮 Prediksi IPK Individual</h2>', unsafe_allow_html=True)
    
    if not st.session_state.trained:
        st.warning("""
        ⚠️ Model belum ditraining. 
        
        Silakan upload data dan training model terlebih dahulu di menu **Upload & Training**.
        """)
    else:
        st.success(f"✅ {len(st.session_state.models)} model siap digunakan!")
        
        # Pilih model untuk prediksi
        model_options = {v['name']: k for k, v in st.session_state.results.items()}
        selected_model_name = st.selectbox(
            "Pilih Model untuk Prediksi:",
            list(model_options.keys()),
            help="Pilih model yang telah ditraining"
        )
        selected_model_type = model_options[selected_model_name]
        
        # Form input data
        st.markdown("### 📝 Input Data Mahasiswa")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Identitas")
            nama = st.text_input("Nama Mahasiswa", "Mahasiswa")
            jenis_kelamin = st.selectbox(
                "Jenis Kelamin", 
                ["Laki-laki", "Perempuan"]
            )
            umur = st.number_input(
                "Umur", 
                min_value=17, 
                max_value=50, 
                value=20,
                help="Umur dalam tahun"
            )
            status_menikah = st.selectbox(
                "Status Menikah", 
                ["Belum Menikah", "Menikah"]
            )
        
        with col2:
            st.markdown("#### Aktivitas Akademik")
            kehadiran = st.slider(
                "Kehadiran (%)", 
                0, 100, 85,
                help="Persentase kehadiran dalam kelas"
            )
            partisipasi = st.slider(
                "Partisipasi Diskusi", 
                0, 100, 75,
                help="Skor partisipasi dalam diskusi"
            )
            nilai_tugas = st.slider(
                "Nilai Tugas (Rata-rata)", 
                0.0, 100.0, 80.0,
                step=0.5,
                help="Rata-rata nilai tugas"
            )
            aktivitas = st.slider(
                "Aktivitas E-Learning", 
                0, 100, 70,
                help="Skor aktivitas dalam platform e-learning"
            )
        
        # Tombol prediksi
        if st.button("🎯 Prediksi IPK", type="primary", use_container_width=True):
            model_data = st.session_state.models[selected_model_type]
            
            try:
                # Encode input
                gender_encoded = model_data['le_gender'].transform([jenis_kelamin])[0]
                married_encoded = model_data['le_married'].transform([status_menikah])[0]
                
                # Buat dataframe input
                input_data = pd.DataFrame({
                    'jenis_kelamin_encoded': [gender_encoded],
                    'umur': [umur],
                    'status_menikah_encoded': [married_encoded],
                    'kehadiran': [kehadiran],
                    'partisipasi_diskusi': [partisipasi],
                    'nilai_tugas': [nilai_tugas],
                    'aktivitas_elearning': [aktivitas]
                })
                
                # Scale jika KNN
                if selected_model_type == 'knn':
                    input_data_scaled = model_data['scaler'].transform(input_data)
                else:
                    input_data_scaled = input_data
                
                # Prediksi
                prediksi = model_data['model'].predict(input_data_scaled)[0]
                prediksi = max(0, min(prediksi, 4))  # Clamp antara 0-4
                
                # Log prediksi
                log_activity("PREDICTION", 
                           f"{selected_model_name}: {nama} - IPK Prediksi: {prediksi:.2f}")
                
                # Tampilkan hasil
                st.markdown("### 📊 Hasil Prediksi")
                
                # Determine status kelulusan
                if prediksi >= 3.5:
                    status = "Cumlaude 🏆"
                    color = "🟢"
                    description = "Prestasi luar biasa! Pertahankan!"
                elif prediksi >= 3.0:
                    status = "Sangat Memuaskan 👍"
                    color = "🔵"
                    description = "Prestasi sangat baik!"
                elif prediksi >= 2.75:
                    status = "Memuaskan ✅"
                    color = "🟡"
                    description = "Prestasi baik, masih bisa ditingkatkan"
                else:
                    status = "Perlu Peningkatan ⚠️"
                    color = "🔴"
                    description = "Perlu perbaikan dalam beberapa aspek"
                
                # Tampilkan metrik
                col_result1, col_result2, col_result3 = st.columns(3)
                
                with col_result1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Model Digunakan", selected_model_name)
                    st.metric("Nama", nama)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_result2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("IPK Diprediksi", f"{prediksi:.2f}", 
                             delta=f"{color} {status.split()[0]}")
                    st.progress(prediksi / 4.0)
                    st.caption(f"{prediksi:.2f}/4.0")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_result3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Status Kelulusan", status)
                    st.markdown(f"**Keterangan:** {description}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Detail prediksi
                with st.expander("📋 Detail Input dan Prediksi"):
                    input_details = pd.DataFrame({
                        'Fitur': ['Nama', 'Jenis Kelamin', 'Umur', 'Status Menikah', 
                                 'Kehadiran', 'Partisipasi Diskusi', 'Nilai Tugas', 'Aktivitas E-Learning'],
                        'Nilai': [nama, jenis_kelamin, umur, status_menikah, 
                                 f"{kehadiran}%", f"{partisipasi}/100", 
                                 f"{nilai_tugas:.1f}/100", f"{aktivitas}/100"]
                    })
                    st.table(input_details)
                
                # Perbandingan dengan model lain
                st.markdown("### ⚖️ Perbandingan dengan Model Lain")
                
                all_predictions = []
                for model_type, model_data_all in st.session_state.models.items():
                    if model_type == selected_model_type:
                        continue
                    
                    # Prepare input
                    if model_type == 'knn':
                        input_scaled_all = model_data_all['scaler'].transform(input_data)
                    else:
                        input_scaled_all = input_data
                    
                    pred_all = model_data_all['model'].predict(input_scaled_all)[0]
                    pred_all = max(0, min(pred_all, 4))
                    
                    all_predictions.append({
                        'Model': st.session_state.results[model_type]['name'],
                        'Prediksi IPK': f"{pred_all:.2f}",
                        'Selisih': f"{abs(prediksi - pred_all):.2f}"
                    })
                
                if all_predictions:
                    pred_df = pd.DataFrame(all_predictions)
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Visualisasi perbandingan
                    fig = px.bar(pred_df, x='Model', y='Prediksi IPK',
                               title='Perbandingan Prediksi Model Lain',
                               color='Prediksi IPK',
                               color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error dalam prediksi: {str(e)}")
                log_activity("PREDICTION_ERROR", str(e))

# MENU 4: Prediksi Batch
elif menu == "📊 Prediksi Batch":
    st.markdown('<h2 class="sub-header">📊 Prediksi IPK Batch</h2>', unsafe_allow_html=True)
    
    if not st.session_state.trained:
        st.warning("⚠️ Model belum ditraining. Silakan training model terlebih dahulu.")
    else:
        # Upload file batch
        uploaded_batch = st.file_uploader(
            "Upload file data untuk prediksi batch (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="File harus memiliki kolom yang sama dengan dataset training (kecuali IPK)"
        )
        
        if uploaded_batch:
            try:
                # Load data
                if uploaded_batch.name.endswith('.csv'):
                    batch_df = pd.read_csv(uploaded_batch)
                else:
                    batch_df = pd.read_excel(uploaded_batch)
                
                st.success(f"✅ Data berhasil diupload! Total {len(batch_df)} baris")
                
                # Tampilkan preview
                st.markdown("### 📋 Preview Data")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                # Pilih model
                model_options = {v['name']: k for k, v in st.session_state.results.items()}
                selected_model_name = st.selectbox(
                    "Pilih Model untuk Prediksi Batch:",
                    list(model_options.keys())
                )
                selected_model_type = model_options[selected_model_name]
                
                if st.button("🚀 Mulai Prediksi Batch", type="primary", use_container_width=True):
                    model_data = st.session_state.models[selected_model_type]
                    
                    # Validasi kolom
                    required_cols = ['nama', 'jenis_kelamin', 'umur', 'status_menikah',
                                   'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning']
                    
                    missing_cols = [col for col in required_cols if col not in batch_df.columns]
                    if missing_cols:
                        st.error(f"❌ Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
                    else:
                        with st.spinner(f"Memproses {len(batch_df)} data..."):
                            try:
                                # Preprocessing
                                batch_df_processed = batch_df.copy()
                                batch_df_processed['jenis_kelamin_encoded'] = model_data['le_gender']\
                                    .transform(batch_df_processed['jenis_kelamin'])
                                batch_df_processed['status_menikah_encoded'] = model_data['le_married']\
                                    .transform(batch_df_processed['status_menikah'])
                                
                                # Prepare features
                                features = model_data['features']
                                X_batch = batch_df_processed[features]
                                
                                # Scale jika KNN
                                if selected_model_type == 'knn':
                                    X_batch_scaled = model_data['scaler'].transform(X_batch)
                                else:
                                    X_batch_scaled = X_batch
                                
                                # Predict
                                predictions = model_data['model'].predict(X_batch_scaled)
                                predictions = np.clip(predictions, 0, 4)
                                
                                # Add predictions to dataframe
                                batch_df['IPK_Prediksi'] = predictions
                                
                                # Add status
                                def get_status(ipk):
                                    if ipk >= 3.5:
                                        return 'Cumlaude 🏆'
                                    elif ipk >= 3.0:
                                        return 'Sangat Memuaskan 👍'
                                    elif ipk >= 2.75:
                                        return 'Memuaskan ✅'
                                    else:
                                        return 'Perlu Peningkatan ⚠️'
                                
                                batch_df['Status'] = batch_df['IPK_Prediksi'].apply(get_status)
                                batch_df['IPK_Prediksi'] = batch_df['IPK_Prediksi'].round(2)
                                
                                # Tampilkan hasil
                                st.success(f"✅ Prediksi selesai untuk {len(batch_df)} data")
                                
                                # Statistik hasil
                                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                                
                                with col_stat1:
                                    st.metric("Rata-rata IPK Prediksi", 
                                             f"{batch_df['IPK_Prediksi'].mean():.2f}")
                                
                                with col_stat2:
                                    cumlaude_count = (batch_df['Status'] == 'Cumlaude 🏆').sum()
                                    st.metric("Jumlah Cumlaude", cumlaude_count)
                                
                                with col_stat3:
                                    perlu_improve = (batch_df['Status'] == 'Perlu Peningkatan ⚠️').sum()
                                    st.metric("Perlu Peningkatan", perlu_improve)
                                
                                with col_stat4:
                                    st.metric("Min IPK", f"{batch_df['IPK_Prediksi'].min():.2f}")
                                
                                # Tampilkan tabel hasil
                                st.markdown("### 📊 Hasil Prediksi")
                                st.dataframe(batch_df[['nama', 'IPK_Prediksi', 'Status']], 
                                           use_container_width=True)
                                
                                # Visualisasi distribusi
                                fig = px.histogram(batch_df, x='IPK_Prediksi',
                                                 title='Distribusi IPK Prediksi',
                                                 nbins=20,
                                                 color_discrete_sequence=['#3B82F6'])
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Status distribution
                                status_counts = batch_df['Status'].value_counts().reset_index()
                                status_counts.columns = ['Status', 'Jumlah']
                                
                                fig2 = px.pie(status_counts, values='Jumlah', names='Status',
                                            title='Distribusi Status Kelulusan',
                                            color_discrete_sequence=px.colors.sequential.RdBu)
                                st.plotly_chart(fig2, use_container_width=True)
                                
                                # Download hasil
                                st.markdown("### 💾 Download Hasil")
                                
                                csv = batch_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Hasil Prediksi (CSV)",
                                    data=csv,
                                    file_name=f"hasil_prediksi_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                                log_activity("BATCH_PREDICTION", 
                                           f"{selected_model_name}: {len(batch_df)} data processed")
                                
                            except Exception as e:
                                st.error(f"❌ Error dalam prediksi batch: {str(e)}")
                                log_activity("BATCH_ERROR", str(e))
                
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")

# MENU 5: Visualisasi
elif menu == "📈 Visualisasi":
    st.markdown('<h2 class="sub-header">📊 Visualisasi Data & Hasil</h2>', unsafe_allow_html=True)
    
    if not st.session_state.trained:
        st.warning("⚠️ Model belum ditraining. Silakan upload data dan training model terlebih dahulu.")
    else:
        df = st.session_state.df
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribusi Data", "🔥 Korelasi", "📊 Model Performance", "📋 Feature Importance"])
        
        with tab1:
            st.markdown("### Distribusi Variabel")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribusi IPK
                fig_ipk = px.histogram(df, x='ipk', nbins=20,
                                     title='Distribusi IPK Mahasiswa',
                                     color_discrete_sequence=['#10B981'])
                st.plotly_chart(fig_ipk, use_container_width=True)
            
            with col2:
                # Distribusi Kehadiran
                fig_attendance = px.box(df, y='kehadiran',
                                      title='Distribusi Kehadiran',
                                      color_discrete_sequence=['#3B82F6'])
                st.plotly_chart(fig_attendance, use_container_width=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Distribusi berdasarkan gender
                fig_gender = px.histogram(df, x='ipk', color='jenis_kelamin',
                                        title='Distribusi IPK berdasarkan Jenis Kelamin',
                                        barmode='overlay',
                                        opacity=0.7)
                st.plotly_chart(fig_gender, use_container_width=True)
            
            with col4:
                # Distribusi berdasarkan status menikah
                fig_married = px.box(df, x='status_menikah', y='ipk',
                                   title='Distribusi IPK berdasarkan Status Menikah',
                                   color='status_menikah')
                st.plotly_chart(fig_married, use_container_width=True)
        
        with tab2:
            st.markdown("### Korelasi Antar Variabel")
            
            # Select numeric columns
            numeric_cols = ['umur', 'kehadiran', 'partisipasi_diskusi', 
                          'nilai_tugas', 'aktivitas_elearning', 'ipk']
            
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Heatmap
            fig_corr = px.imshow(corr_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               title='Heatmap Korelasi Antar Variabel',
                               color_continuous_scale='RdBu_r',
                               labels=dict(color="Korelasi"))
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Scatter plot IPK vs variabel lain
            st.markdown("### Hubungan IPK dengan Variabel Lain")
            
            x_var = st.selectbox("Pilih variabel X:", 
                               ['kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning'])
            
            fig_scatter = px.scatter(df, x=x_var, y='ipk',
                                   trendline="ols",
                                   title=f'Hubungan {x_var} dengan IPK',
                                   color_discrete_sequence=['#EF4444'])
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab3:
            st.markdown("### Performa Model")
            
            # Pilih model
            model_options = {v['name']: k for k, v in st.session_state.results.items()}
            selected_model_name = st.selectbox("Pilih Model:", list(model_options.keys()))
            selected_model_type = model_options[selected_model_name]
            
            result = st.session_state.results[selected_model_type]
            
            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
            
            with col_metric1:
                st.metric("R² Score", f"{result['metrics']['R2']:.4f}")
            with col_metric2:
                st.metric("MAE", f"{result['metrics']['MAE']:.4f}")
            with col_metric3:
                st.metric("RMSE", f"{result['metrics']['RMSE']:.4f}")
            with col_metric4:
                st.metric("MAPE", f"{result['metrics']['MAPE']:.2f}%")
            
            # Scatter plot actual vs predicted
            fig_scatter = px.scatter(
                x=result['y_test'], 
                y=result['y_pred'],
                labels={'x': 'IPK Aktual', 'y': 'IPK Prediksi'},
                title=f'Prediksi vs Aktual - {selected_model_name}',
                trendline="ols"
            )
            
            # Add perfect prediction line
            min_val = min(result['y_test'].min(), result['y_pred'].min())
            max_val = max(result['y_test'].max(), result['y_pred'].max())
            
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val], 
                y=[min_val, max_val],
                mode='lines', 
                name='Perfect Prediction',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Residual plot
            residuals = result['y_test'] - result['y_pred']
            
            fig_residual = px.scatter(x=result['y_pred'], y=residuals,
                                    labels={'x': 'Prediksi', 'y': 'Residual (Aktual - Prediksi)'},
                                    title=f'Residual Plot - {selected_model_name}')
            fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig_residual, use_container_width=True)
            
            # Distribution of residuals
            fig_resid_dist = px.histogram(x=residuals, 
                                        title='Distribusi Residual',
                                        labels={'x': 'Residual'})
            fig_resid_dist.add_vline(x=0, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig_resid_dist, use_container_width=True)
        
        with tab4:
            st.markdown("### Feature Importance")
            
            # Pilih model (hanya untuk RF dan GB)
            available_models = []
            for model_type, model_data in st.session_state.models.items():
                if model_type in ['random_forest', 'gradient_boosting']:
                    available_models.append(model_type)
            
            if available_models:
                selected_fi_model = st.selectbox(
                    "Pilih Model untuk Feature Importance:",
                    available_models,
                    format_func=lambda x: st.session_state.results[x]['name']
                )
                
                model_obj = st.session_state.models[selected_fi_model]['model']
                
                # Get feature importance
                importance = model_obj.feature_importances_
                feature_names = ['Jenis Kelamin', 'Umur', 'Status Menikah', 
                               'Kehadiran', 'Partisipasi Diskusi', 'Nilai Tugas', 'Aktivitas E-Learning']
                
                # Create dataframe
                fi_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=True)
                
                # Plot
                fig_fi = px.bar(fi_df, x='Importance', y='Feature',
                              title=f'Feature Importance - {st.session_state.results[selected_fi_model]["name"]}',
                              orientation='h',
                              color='Importance',
                              color_continuous_scale='Viridis')
                
                st.plotly_chart(fig_fi, use_container_width=True)
                
                # Tampilkan dalam bentuk persentase
                fi_df['Percentage'] = (fi_df['Importance'] * 100).round(2)
                
                st.markdown("#### 📋 Feature Importance dalam Persentase")
                st.dataframe(fi_df[['Feature', 'Percentage']].sort_values('Percentage', ascending=False),
                           use_container_width=True)
                
            else:
                st.info("ℹ️ Feature Importance hanya tersedia untuk Random Forest dan Gradient Boosting")

# MENU 6: Perbandingan Model
elif menu == "⚖️ Perbandingan Model":
    st.markdown('<h2 class="sub-header">⚖️ Perbandingan Performa Model</h2>', unsafe_allow_html=True)
    
    if not st.session_state.trained:
        st.warning("⚠️ Model belum ditraining. Silakan training model terlebih dahulu.")
    else:
        # Tabel perbandingan
        st.markdown("### 📊 Tabel Perbandingan Metrik")
        
        results_data = []
        for model_type, result in st.session_state.results.items():
            results_data.append({
                'Model': result['name'],
                'R²': result['metrics']['R2'],
                'MAE': result['metrics']['MAE'],
                'RMSE': result['metrics']['RMSE'],
                'MAPE (%)': result['metrics']['MAPE'],
                'CV R² Mean': result['cv_mean'],
                'CV R² Std': result['cv_std']
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Style dataframe
        def highlight_best(row):
            styles = [''] * len(row)
            if row.name == results_df['R²'].idxmax():
                styles[results_df.columns.get_loc('R²')] = 'background-color: #D1FAE5'
            if row.name == results_df['MAE'].idxmin():
                styles[results_df.columns.get_loc('MAE')] = 'background-color: #D1FAE5'
            if row.name == results_df['RMSE'].idxmin():
                styles[results_df.columns.get_loc('RMSE')] = 'background-color: #D1FAE5'
            if row.name == results_df['MAPE (%)'].idxmin():
                styles[results_df.columns.get_loc('MAPE (%)')] = 'background-color: #D1FAE5'
            return styles
        
        styled_df = results_df.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Visualisasi perbandingan
        st.markdown("### 📈 Visualisasi Perbandingan")
        
        metric_choice = st.selectbox(
            "Pilih metrik untuk perbandingan:",
            ['R²', 'MAE', 'RMSE', 'MAPE (%)', 'CV R² Mean']
        )
        
        fig_comparison = px.bar(results_df, x='Model', y=metric_choice,
                              title=f'Perbandingan {metric_choice} Antar Model',
                              color=metric_choice,
                              color_continuous_scale='RdYlGn' if metric_choice in ['R²', 'CV R² Mean'] else 'RdYlGn_r')
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Radar chart untuk perbandingan multi-metrik
        st.markdown("### 📊 Radar Chart Perbandingan")
        
        # Normalize metrics for radar chart
        metrics_for_radar = ['R²', 'MAE', 'RMSE', 'MAPE (%)']
        
        # Prepare data for radar chart
        radar_data = []
        for _, row in results_df.iterrows():
            radar_data.append(go.Scatterpolar(
                r=[row['R²'], 
                   (1 - row['MAE'] / results_df['MAE'].max()) if results_df['MAE'].max() > 0 else 0,
                   (1 - row['RMSE'] / results_df['RMSE'].max()) if results_df['RMSE'].max() > 0 else 0,
                   (1 - row['MAPE (%)'] / results_df['MAPE (%)'].max()) if results_df['MAPE (%)'].max() > 0 else 0],
                theta=['R²', 'MAE', 'RMSE', 'MAPE (%)'],
                fill='toself',
                name=row['Model']
            ))
        
        fig_radar = go.Figure(data=radar_data)
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Radar Chart Perbandingan Model (Normalized)"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Prediksi vs Aktual semua model
        st.markdown("### 📈 Perbandingan Prediksi vs Aktual")
        
        fig_all = go.Figure()
        
        colors = px.colors.qualitative.Set3
        for idx, (model_type, result) in enumerate(st.session_state.results.items()):
            fig_all.add_trace(go.Scatter(
                x=result['y_test'],
                y=result['y_pred'],
                mode='markers',
                name=result['name'],
                marker=dict(size=8, opacity=0.6, color=colors[idx % len(colors)]),
                hovertemplate='Aktual: %{x:.2f}<br>Prediksi: %{y:.2f}<br>Model: ' + result['name']
            ))
        
        # Perfect prediction line
        y_min = min([r['y_test'].min() for r in st.session_state.results.values()])
        y_max = max([r['y_test'].max() for r in st.session_state.results.values()])
        
        fig_all.add_trace(go.Scatter(
            x=[y_min, y_max],
            y=[y_min, y_max],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig_all.update_layout(
            title='Prediksi vs Aktual Semua Model',
            xaxis_title='IPK Aktual',
            yaxis_title='IPK Prediksi',
            hovermode='closest'
        )
        
        st.plotly_chart(fig_all, use_container_width=True)
        
        # Rekomendasi model terbaik
        st.markdown("### 🏆 Rekomendasi Model")
        
        best_r2_idx = results_df['R²'].idxmax()
        best_mae_idx = results_df['MAE'].idxmin()
        best_rmse_idx = results_df['RMSE'].idxmin()
        
        col_best1, col_best2, col_best3 = st.columns(3)
        
        with col_best1:
            st.success(f"**Model dengan R² Terbaik**")
            st.markdown(f"**{results_df.loc[best_r2_idx, 'Model']}**")
            st.markdown(f"R² = {results_df.loc[best_r2_idx, 'R²']:.4f}")
            st.markdown(f"MAE = {results_df.loc[best_r2_idx, 'MAE']:.4f}")
        
        with col_best2:
            st.success(f"**Model dengan MAE Terbaik**")
            st.markdown(f"**{results_df.loc[best_mae_idx, 'Model']}**")
            st.markdown(f"MAE = {results_df.loc[best_mae_idx, 'MAE']:.4f}")
            st.markdown(f"R² = {results_df.loc[best_mae_idx, 'R²']:.4f}")
        
        with col_best3:
            st.success(f"**Model dengan RMSE Terbaik**")
            st.markdown(f"**{results_df.loc[best_rmse_idx, 'Model']}**")
            st.markdown(f"RMSE = {results_df.loc[best_rmse_idx, 'RMSE']:.4f}")
            st.markdown(f"R² = {results_df.loc[best_rmse_idx, 'R²']:.4f}")

# MENU 7: Model Management
elif menu == "💾 Model Management":
    st.markdown('<h2 class="sub-header">💾 Manajemen Model</h2>', unsafe_allow_html=True)
    
    tab_manage1, tab_manage2, tab_manage3 = st.tabs(["💾 Simpan Model", "📂 Load Model", "🗑️ Hapus Model"])
    
    with tab_manage1:
        st.markdown("### Simpan Model ke Disk")
        
        if not st.session_state.trained:
            st.warning("⚠️ Tidak ada model yang bisa disimpan. Silakan training model terlebih dahulu.")
        else:
            st.info("Model yang tersedia untuk disimpan:")
            
            for model_type, result in st.session_state.results.items():
                col_save1, col_save2, col_save3 = st.columns([3, 2, 1])
                
                with col_save1:
                    st.write(f"**{result['name']}**")
                    st.caption(f"R²: {result['metrics']['R2']:.4f} | MAE: {result['metrics']['MAE']:.4f}")
                
                with col_save2:
                    model_name = st.text_input(
                        "Nama file",
                        value=result['name'].replace(' ', '_'),
                        key=f"save_name_{model_type}",
                        label_visibility="collapsed"
                    )
                
                with col_save3:
                    if st.button("💾 Simpan", key=f"btn_save_{model_type}"):
                        success, message = save_model(
                            st.session_state.models[model_type],
                            model_name
                        )
                        if success:
                            st.success(f"✅ {message}")
                            log_activity("MODEL_SAVED", f"{model_name} saved")
                        else:
                            st.error(f"❌ {message}")
    
    with tab_manage2:
        st.markdown("### Load Model dari Disk")
        
        # Check saved models
        saved_models = []
        if os.path.exists('saved_models'):
            model_files = [f for f in os.listdir('saved_models') 
                          if f.endswith('_model.pkl') and not f.endswith('_preprocess.pkl')]
            saved_models = [f.replace('_model.pkl', '') for f in model_files]
        
        if not saved_models:
            st.warning("⚠️ Tidak ada model yang tersimpan di disk.")
        else:
            st.success(f"✅ Ditemukan {len(saved_models)} model tersimpan")
            
            selected_model = st.selectbox(
                "Pilih model untuk di-load:",
                saved_models
            )
            
            col_load1, col_load2 = st.columns([3, 1])
            
            with col_load1:
                st.write(f"**Model:** {selected_model}")
            
            with col_load2:
                if st.button("📂 Load Model", use_container_width=True):
                    success, result = load_model(selected_model)
                    
                    if success:
                        # Simpan ke session state
                        model_type = f"loaded_{selected_model}"
                        st.session_state.models[model_type] = result
                        
                        # Untuk loaded model, kita tidak punya hasil evaluasi
                        # Tapi kita tetap buat entry di results untuk konsistensi
                        st.session_state.results[model_type] = {
                            'name': selected_model,
                            'metrics': {'R2': None, 'MAE': None, 'RMSE': None, 'MAPE': None}
                        }
                        
                        st.session_state.trained = True
                        
                        st.success(f"✅ Model '{selected_model}' berhasil di-load!")
                        log_activity("MODEL_LOADED", f"{selected_model} loaded")
                    else:
                        st.error(f"❌ {result}")
    
    with tab_manage3:
        st.markdown("### Hapus Model dari Disk")
        
        # List saved models
        saved_models = []
        if os.path.exists('saved_models'):
            model_files = [f for f in os.listdir('saved_models') 
                          if f.endswith('_model.pkl') and not f.endswith('_preprocess.pkl')]
            saved_models = [f.replace('_model.pkl', '') for f in model_files]
        
        if not saved_models:
            st.info("ℹ️ Tidak ada model yang tersimpan di disk.")
        else:
            model_to_delete = st.selectbox(
                "Pilih model untuk dihapus:",
                saved_models,
                key="delete_select"
            )
            
            st.warning(f"⚠️ Anda akan menghapus model: **{model_to_delete}**")
            st.warning("Tindakan ini tidak dapat dibatalkan!")
            
            if st.button("🗑️ Hapus Model", type="secondary", use_container_width=True):
                try:
                    # Delete model files
                    model_path = f"saved_models/{model_to_delete}_model.pkl"
                    preprocess_path = f"saved_models/{model_to_delete}_preprocess.pkl"
                    
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    if os.path.exists(preprocess_path):
                        os.remove(preprocess_path)
                    
                    st.success(f"✅ Model '{model_to_delete}' berhasil dihapus!")
                    log_activity("MODEL_DELETED", f"{model_to_delete} deleted")
                    
                    # Refresh page
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error menghapus model: {str(e)}")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 3, 2])

with footer_col1:
    st.markdown("**🎓 Prediksi Kelulusan**")
    st.caption("v2.0 | ML-Based")

with footer_col2:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.caption("Dibuat dengan ❤️ menggunakan Streamlit, Scikit-learn, dan Plotly")
    st.markdown("</div>", unsafe_allow_html=True)

with footer_col3:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
    st.caption(f"© {datetime.now().year} | All rights reserved")
    st.markdown("</div>", unsafe_allow_html=True)
