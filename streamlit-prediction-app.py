import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Fungsi untuk menghitung semua metrik evaluasi
def calculate_metrics(y_true, y_pred):
    """Menghitung metrik evaluasi dengan error handling"""
    try:
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAPE dengan handling zero values
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.inf
        
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return mae, mape, rmse, r2
    except Exception as e:
        st.error(f"Error dalam perhitungan metrik: {str(e)}")
        return 0, 0, 0, 0

# Fungsi validasi dataset
def validate_dataset(df):
    """Validasi kolom dan kualitas dataset"""
    required_columns = ['jenis_kelamin', 'umur', 'status_menikah', 
                       'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 
                       'aktivitas_elearning', 'ipk']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        return False, f"Kolom yang hilang: {', '.join(missing_cols)}"
    
    # Cek nilai missing
    if df[required_columns].isnull().any().any():
        return False, "Dataset mengandung nilai kosong (NaN)"
    
    # Cek range IPK
    if (df['ipk'] < 0).any() or (df['ipk'] > 4).any():
        return False, "IPK harus dalam range 0-4"
    
    # Cek range persentase
    if (df['kehadiran'] < 0).any() or (df['kehadiran'] > 100).any():
        return False, "Kehadiran harus dalam range 0-100"
    
    return True, "Dataset valid"

# Fungsi preprocessing data dengan feature engineering
def preprocess_data(df):
    """Preprocessing dengan feature engineering yang lebih baik"""
    df_processed = df.copy()
    
    # Encode categorical features
    le_gender = LabelEncoder()
    le_married = LabelEncoder()
    
    df_processed['jenis_kelamin_encoded'] = le_gender.fit_transform(df_processed['jenis_kelamin'])
    df_processed['status_menikah_encoded'] = le_married.fit_transform(df_processed['status_menikah'])
    
    # Feature Engineering: Buat fitur gabungan yang lebih bermakna
    df_processed['rata_rata_akademik'] = (
        df_processed['nilai_tugas'] + 
        df_processed['partisipasi_diskusi'] + 
        df_processed['aktivitas_elearning']
    ) / 3
    
    df_processed['engagement_score'] = (
        df_processed['kehadiran'] * 0.4 + 
        df_processed['partisipasi_diskusi'] * 0.3 + 
        df_processed['aktivitas_elearning'] * 0.3
    )
    
    # Interaksi antara kehadiran dan nilai tugas
    df_processed['kehadiran_x_tugas'] = (
        df_processed['kehadiran'] / 100 * df_processed['nilai_tugas']
    )
    
    return df_processed, le_gender, le_married

# Fungsi untuk training model dengan cross-validation
def train_model(df, model_type='random_forest', test_size=0.2):
    """Training model dengan hyperparameter yang lebih optimal"""
    try:
        # Preprocessing
        df_processed, le_gender, le_married = preprocess_data(df)
        
        # Features yang diperluas
        features = [
            'jenis_kelamin_encoded', 'umur', 'status_menikah_encoded', 
            'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 
            'aktivitas_elearning', 'rata_rata_akademik', 
            'engagement_score', 'kehadiran_x_tugas'
        ]
        
        X = df_processed[features]
        y = df_processed['ipk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        # Scaling
        scaler = StandardScaler()
        if model_type == 'knn':
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Pilih model dengan hyperparameter yang lebih baik
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            )
        elif model_type == 'knn':
            model = KNeighborsRegressor(
                n_neighbors=7,
                weights='distance',
                metric='euclidean'
            )
        
        # Training model
        model.fit(X_train_scaled, y_train)
        
        # Prediksi
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Hitung metrik untuk training dan testing
        train_mae, train_mape, train_rmse, train_r2 = calculate_metrics(y_train, y_pred_train)
        test_mae, test_mape, test_rmse, test_r2 = calculate_metrics(y_test, y_pred_test)
        
        # Cross-validation score (5-fold)
        if model_type != 'knn':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                       scoring='r2', n_jobs=-1)
            cv_r2_mean = cv_scores.mean()
            cv_r2_std = cv_scores.std()
        else:
            cv_r2_mean = test_r2
            cv_r2_std = 0
        
        return {
            'model': model,
            'scaler': scaler,
            'le_gender': le_gender,
            'le_married': le_married,
            'features': features,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'train_metrics': {
                'mae': train_mae,
                'mape': train_mape,
                'rmse': train_rmse,
                'r2': train_r2
            },
            'test_metrics': {
                'mae': test_mae,
                'mape': test_mape,
                'rmse': test_rmse,
                'r2': test_r2
            },
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std
        }
        
    except Exception as e:
        st.error(f"❌ Error saat training model: {str(e)}")
        return None

# Header
st.title("🎓 Aplikasi Prediksi IPK Mahasiswa (Improved)")
st.markdown("Aplikasi prediksi IPK dengan **feature engineering** dan **validasi data** yang lebih baik")

# Sidebar
st.sidebar.header("📋 Menu")
menu = st.sidebar.radio("Pilih Menu:", 
                        ["Upload & Training", "Prediksi Individual", 
                         "Visualisasi", "Analisis Model"])

# Session state
if 'models' not in st.session_state:
    st.session_state.models = {}
    st.session_state.results = {}
    st.session_state.trained = False
    st.session_state.df = None

# MENU 1: Upload & Training
if menu == "Upload & Training":
    st.header("📤 Upload Dataset dan Training Model")
    
    uploaded_file = st.file_uploader("Upload file CSV/Excel/TSV", 
                                     type=['csv', 'xlsx', 'tsv', 'txt'])
    
    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:  # TSV or TXT
                df = pd.read_csv(uploaded_file, sep='\t')
                
            st.success(f"✅ Data berhasil diupload! Total: {len(df)} baris")
            
        except Exception as e:
            st.error(f"❌ Error saat membaca file: {str(e)}")
            st.stop()
        
        # Validasi dataset
        is_valid, message = validate_dataset(df)
        
        if not is_valid:
            st.error(f"❌ {message}")
            st.info("Pastikan dataset memiliki kolom yang benar dan tidak ada nilai kosong")
            st.stop()
        else:
            st.success(f"✅ {message}")
        
        # Tampilkan data
        st.subheader("📊 Preview Data")
        st.dataframe(df.head(10))
        
        # Statistik deskriptif
        st.subheader("📈 Statistik Data")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Data", len(df))
        with col2:
            st.metric("Rata-rata IPK", f"{df['ipk'].mean():.2f}")
        with col3:
            st.metric("Std Dev IPK", f"{df['ipk'].std():.2f}")
        with col4:
            st.metric("IPK Min", f"{df['ipk'].min():.2f}")
        with col5:
            st.metric("IPK Max", f"{df['ipk'].max():.2f}")
        
        # Warning jika dataset kecil
        if len(df) < 300:
            st.warning("⚠️ Dataset Anda relatif kecil (< 300 data). Model mungkin tidak optimal.")
        
        # Analisis kualitas data
        with st.expander("🔍 Analisis Kualitas Data"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Distribusi IPK:**")
                fig = px.histogram(df, x='ipk', nbins=30, 
                                 title='Distribusi IPK')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Korelasi dengan IPK:**")
                numeric_cols = ['umur', 'kehadiran', 'partisipasi_diskusi', 
                              'nilai_tugas', 'aktivitas_elearning', 'ipk']
                corr_with_ipk = df[numeric_cols].corr()['ipk'].sort_values(ascending=False)
                
                fig = px.bar(x=corr_with_ipk.index[1:], y=corr_with_ipk.values[1:],
                           title='Korelasi Fitur dengan IPK',
                           labels={'x': 'Fitur', 'y': 'Korelasi'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Pilih model dan parameter
        st.subheader("🤖 Konfigurasi Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Pilih Model:**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                train_rf = st.checkbox("Random Forest", value=True)
            with col_b:
                train_gb = st.checkbox("Gradient Boosting", value=True)
            with col_c:
                train_knn = st.checkbox("KNN", value=False)
        
        with col2:
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            st.caption(f"Training: {int((1-test_size)*100)}% | Testing: {int(test_size*100)}%")
        
        # Tombol training
        if st.button("🚀 Mulai Training Model", type="primary"):
            models_to_train = []
            if train_rf:
                models_to_train.append(('Random Forest', 'random_forest'))
            if train_gb:
                models_to_train.append(('Gradient Boosting', 'gradient_boosting'))
            if train_knn:
                models_to_train.append(('KNN', 'knn'))
            
            if not models_to_train:
                st.error("❌ Pilih minimal satu model!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                st.session_state.models = {}
                st.session_state.results = {}
                
                for idx, (model_name, model_type) in enumerate(models_to_train):
                    status_text.text(f"⏳ Training {model_name}... ({idx+1}/{len(models_to_train)})")
                    
                    result = train_model(df, model_type, test_size)
                    
                    if result:
                        st.session_state.models[model_type] = result
                        st.session_state.results[model_type] = {
                            'name': model_name,
                            'train_metrics': result['train_metrics'],
                            'test_metrics': result['test_metrics'],
                            'cv_r2_mean': result['cv_r2_mean'],
                            'cv_r2_std': result['cv_r2_std']
                        }
                    
                    progress_bar.progress((idx + 1) / len(models_to_train))
                
                st.session_state.trained = True
                st.session_state.df = df
                status_text.text("✅ Training selesai!")
                
                # Tampilkan hasil
                st.subheader("📊 Hasil Training")
                
                # Tabel hasil
                results_data = []
                for model_type, result in st.session_state.results.items():
                    results_data.append({
                        'Model': result['name'],
                        'Train R²': f"{result['train_metrics']['r2']:.4f}",
                        'Test R²': f"{result['test_metrics']['r2']:.4f}",
                        'CV R² (mean±std)': f"{result['cv_r2_mean']:.4f}±{result['cv_r2_std']:.4f}",
                        'Test MAE': f"{result['test_metrics']['mae']:.4f}",
                        'Test RMSE': f"{result['test_metrics']['rmse']:.4f}"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Warning jika overfitting
                for model_type, result in st.session_state.results.items():
                    train_r2 = result['train_metrics']['r2']
                    test_r2 = result['test_metrics']['r2']
                    
                    if train_r2 - test_r2 > 0.2:
                        st.warning(f"⚠️ {result['name']}: Kemungkinan overfitting (Train R²={train_r2:.3f}, Test R²={test_r2:.3f})")
                    
                    if test_r2 < 0:
                        st.error(f"❌ {result['name']}: R² negatif! Model tidak dapat memprediksi dengan baik. Coba periksa kualitas data Anda.")
                
                # Visualisasi perbandingan
                st.subheader("📈 Visualisasi Performa")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    r2_data = []
                    for result in st.session_state.results.values():
                        r2_data.append({
                            'Model': result['name'],
                            'Train R²': result['train_metrics']['r2'],
                            'Test R²': result['test_metrics']['r2'],
                            'CV R²': result['cv_r2_mean']
                        })
                    
                    r2_df = pd.DataFrame(r2_data)
                    r2_melted = r2_df.melt(id_vars=['Model'], 
                                          var_name='Metric', 
                                          value_name='R²')
                    
                    fig = px.bar(r2_melted, x='Model', y='R²', color='Metric',
                               barmode='group', title='Perbandingan R² Score')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    mae_data = [r['test_metrics']['mae'] for r in st.session_state.results.values()]
                    rmse_data = [r['test_metrics']['rmse'] for r in st.session_state.results.values()]
                    model_names = [r['name'] for r in st.session_state.results.values()]
                    
                    fig = go.Figure(data=[
                        go.Bar(name='MAE', x=model_names, y=mae_data),
                        go.Bar(name='RMSE', x=model_names, y=rmse_data)
                    ])
                    fig.update_layout(title='Perbandingan MAE & RMSE',
                                    barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Silakan upload dataset terlebih dahulu")
        
        with st.expander("📖 Panduan Dataset"):
            st.markdown("""
            **Format dataset yang dibutuhkan:**
            
            | Kolom | Tipe | Deskripsi |
            |-------|------|-----------|
            | nama | Text | Nama mahasiswa (opsional) |
            | jenis_kelamin | Text | "Laki-laki" atau "Perempuan" |
            | umur | Integer | Umur mahasiswa |
            | status_menikah | Text | "Menikah" atau "Belum Menikah" |
            | kehadiran | Float | Persentase kehadiran (0-100) |
            | partisipasi_diskusi | Float | Skor partisipasi (0-100) |
            | nilai_tugas | Float | Rata-rata nilai tugas (0-100) |
            | aktivitas_elearning | Float | Skor aktivitas e-learning (0-100) |
            | ipk | Float | IPK target prediksi (0-4) |
            | status_akademik | Text | "Lulus" atau "Tidak" (opsional) |
            
            **Format file yang didukung:** CSV, TSV, Excel (.xlsx)
            
            **Tips untuk hasil terbaik:**
            - Minimal 300 data untuk hasil optimal
            - Pastikan tidak ada nilai kosong (NaN)
            - Data harus konsisten dan logis
            """)

# MENU 2: Prediksi Individual
elif menu == "Prediksi Individual":
    st.header("🔮 Prediksi IPK Individual")
    
    if not st.session_state.trained:
        st.warning("⚠️ Model belum ditraining. Silakan training model terlebih dahulu!")
    else:
        st.success(f"✅ {len(st.session_state.models)} model siap digunakan!")
        
        # Pilih model
        model_options = {v['name']: k for k, v in st.session_state.results.items()}
        selected_model_name = st.selectbox("Pilih Model:", list(model_options.keys()))
        selected_model_type = model_options[selected_model_name]
        
        # Tampilkan metrik model
        result = st.session_state.results[selected_model_type]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test R²", f"{result['test_metrics']['r2']:.4f}")
        with col2:
            st.metric("Test MAE", f"{result['test_metrics']['mae']:.4f}")
        with col3:
            st.metric("Test RMSE", f"{result['test_metrics']['rmse']:.4f}")
        
        st.markdown("---")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            umur = st.number_input("Umur", min_value=17, max_value=50, value=20)
            status_menikah = st.selectbox("Status Menikah", ["Belum Menikah", "Menikah"])
            kehadiran = st.slider("Kehadiran (%)", 0, 100, 80)
        
        with col2:
            partisipasi = st.number_input("Partisipasi Diskusi (0-100)", 
                                         min_value=0, max_value=100, value=75)
            nilai_tugas = st.number_input("Nilai Tugas (0-100)", 
                                         min_value=0.0, max_value=100.0, value=80.0)
            aktivitas = st.number_input("Aktivitas E-Learning (0-100)", 
                                       min_value=0, max_value=100, value=70)
        
        if st.button("🎯 Prediksi IPK", type="primary"):
            model_data = st.session_state.models[selected_model_type]
            
            # Encode input
            gender_encoded = model_data['le_gender'].transform([jenis_kelamin])[0]
            married_encoded = model_data['le_married'].transform([status_menikah])[0]
            
            # Hitung feature engineering
            rata_rata_akademik = (nilai_tugas + partisipasi + aktivitas) / 3
            engagement_score = (kehadiran * 0.4 + partisipasi * 0.3 + aktivitas * 0.3)
            kehadiran_x_tugas = (kehadiran / 100 * nilai_tugas)
            
            # Buat input
            input_data = pd.DataFrame({
                'jenis_kelamin_encoded': [gender_encoded],
                'umur': [umur],
                'status_menikah_encoded': [married_encoded],
                'kehadiran': [kehadiran],
                'partisipasi_diskusi': [partisipasi],
                'nilai_tugas': [nilai_tugas],
                'aktivitas_elearning': [aktivitas],
                'rata_rata_akademik': [rata_rata_akademik],
                'engagement_score': [engagement_score],
                'kehadiran_x_tugas': [kehadiran_x_tugas]
            })
            
            # Scale jika KNN
            if selected_model_type == 'knn':
                input_scaled = model_data['scaler'].transform(input_data)
            else:
                input_scaled = input_data
            
            # Prediksi
            prediksi = model_data['model'].predict(input_scaled)[0]
            prediksi = max(0, min(4, prediksi))  # Clip ke range 0-4
            
            # Hasil
            st.subheader("📊 Hasil Prediksi")
            
            if prediksi >= 3.5:
                status = "Cumlaude"
                color = "green"
                emoji = "🏆"
            elif prediksi >= 3.0:
                status = "Sangat Memuaskan"
                color = "blue"
                emoji = "⭐"
            elif prediksi >= 2.75:
                status = "Memuaskan"
                color = "orange"
                emoji = "👍"
            else:
                status = "Perlu Peningkatan"
                color = "red"
                emoji = "📚"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("IPK Diprediksi", f"{prediksi:.2f}")
            with col2:
                st.markdown(f"### {emoji} :{color}[{status}]")
            with col3:
                confidence = result['test_metrics']['r2']
                st.metric("Confidence (R²)", f"{confidence:.2%}")
            
            st.progress(min(prediksi / 4.0, 1.0))
            
            # Prediksi dari semua model
            if len(st.session_state.models) > 1:
                if st.checkbox("📊 Lihat prediksi dari semua model"):
                    all_preds = []
                    for mtype, mdata in st.session_state.models.items():
                        if mtype == 'knn':
                            inp = mdata['scaler'].transform(input_data)
                        else:
                            inp = input_data
                        pred = mdata['model'].predict(inp)[0]
                        pred = max(0, min(4, pred))
                        
                        all_preds.append({
                            'Model': st.session_state.results[mtype]['name'],
                            'Prediksi IPK': f"{pred:.2f}",
                            'R² Score': f"{st.session_state.results[mtype]['test_metrics']['r2']:.4f}"
                        })
                    
                    st.dataframe(pd.DataFrame(all_preds), use_container_width=True)

# MENU 3: Visualisasi
elif menu == "Visualisasi":
    st.header("📊 Visualisasi Data & Model")
    
    if not st.session_state.trained:
        st.warning("⚠️ Silakan training model terlebih dahulu")
    else:
        df = st.session_state.df
        
        tab1, tab2, tab3 = st.tabs(["📈 Distribusi Data", "🔗 Korelasi", "🎯 Prediksi Model"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='ipk', nbins=25, 
                                 title='Distribusi IPK',
                                 color_discrete_sequence=['#3b82f6'])
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.box(df, y='kehadiran', title='Distribusi Kehadiran')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(df, x='nilai_tugas', y='ipk',
                               title='Nilai Tugas vs IPK',
                               trendline='ols')
                st.plotly_chart(fig, use_container_width=True)
                
                fig = px.scatter(df, x='kehadiran', y='ipk',
                               title='Kehadiran vs IPK',
                               trendline='ols')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            numeric_cols = ['umur', 'kehadiran', 'partisipasi_diskusi', 
                          'nilai_tugas', 'aktivitas_elearning', 'ipk']
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                           title='Heatmap Korelasi',
                           color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            model_options = {v['name']: k for k, v in st.session_state.results.items()}
            selected = st.selectbox("Pilih Model:", list(model_options.keys()))
            model_type = model_options[selected]
            
            model_data = st.session_state.models[model_type]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot prediksi vs aktual
                fig = px.scatter(
                    x=model_data['y_test'], 
                    y=model_data['y_pred_test'],
                    labels={'x': 'IPK Aktual', 'y': 'IPK Prediksi'},
                    title=f'Prediksi vs Aktual - {selected}'
                )
                fig.add_trace(go.Scatter(
                    x=[model_data['y_test'].min(), model_data['y_test'].max()],
                    y=[model_data['y_test'].min(), model_data['y_test'].max()],
                    mode='lines', name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Residual plot
                residuals = model_data['y_test'] - model_data['y_pred_test']
                fig = px.scatter(x=model_data['y_pred_test'], y=residuals,
                               labels={'x': 'Prediksi', 'y': 'Residual'},
                               title='Residual Plot')
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if model_type in ['random_forest', 'gradient_boosting']:
                st.subheader("🎯 Feature Importance")
                
                feature_names = [
                    'Jenis Kelamin', 'Umur', 'Status Menikah', 
                    'Kehadiran', 'Partisipasi', 'Nilai Tugas', 
                    'Aktivitas E-Learning', 'Rata² Akademik',
                    'Engagement Score', 'Kehadiran×Tugas'
                ]
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model_data['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df, x='Importance', y='Feature',
                           orientation='h',
                           title=f'Feature Importance - {selected}')
                st.plotly_chart(fig, use_container_width=True)

# MENU 4: Analisis Model
elif menu == "Analisis Model":
    st.header("⚖️ Analisis & Perbandingan Model")
    
    if not st.session_state.trained:
        st.warning("⚠️ Silakan training model terlebih dahulu")
    else:
        st.subheader("📊 Ringkasan Performa Model")
        
        # Tabel perbandingan lengkap
        comparison_data = []
        for model_type, result in st.session_state.results.items():
            comparison_data.append({
                'Model': result['name'],
                'Train R²': result['train_metrics']['r2'],
                'Test R²': result['test_metrics']['r2'],
                'CV R² (mean)': result['cv_r2_mean'],
                'CV R² (std)': result['cv_r2_std'],
                'Test MAE': result['test_metrics']['mae'],
                'Test RMSE': result['test_metrics']['rmse'],
                'Test MAPE (%)': result['test_metrics']['mape'],
                'Overfitting Gap': result['train_metrics']['r2'] - result['test_metrics']['r2']
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Format tabel
        st.dataframe(
            comp_df.style.format({
                'Train R²': '{:.4f}',
                'Test R²': '{:.4f}',
                'CV R² (mean)': '{:.4f}',
                'CV R² (std)': '{:.4f}',
                'Test MAE': '{:.4f}',
                'Test RMSE': '{:.4f}',
                'Test MAPE (%)': '{:.2f}',
                'Overfitting Gap': '{:.4f}'
            }).background_gradient(subset=['Test R²'], cmap='RdYlGn')
              .background_gradient(subset=['Test MAE', 'Test RMSE'], cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        # Rekomendasi model terbaik
        best_model_idx = comp_df['Test R²'].idxmax()
        best_model = comp_df.iloc[best_model_idx]
        
        st.success(f"""
        🏆 **Model Terbaik: {best_model['Model']}**
        - Test R² Score: {best_model['Test R²']:.4f}
        - Test MAE: {best_model['Test MAE']:.4f}
        - CV R² (mean±std): {best_model['CV R² (mean)']:.4f}±{best_model['CV R² (std)']:.4f}
        """)
        
        # Warning untuk model dengan performa buruk
        for _, row in comp_df.iterrows():
            if row['Test R²'] < 0:
                st.error(f"❌ **{row['Model']}**: R² negatif! Model gagal memprediksi.")
            elif row['Overfitting Gap'] > 0.2:
                st.warning(f"⚠️ **{row['Model']}**: Overfitting terdeteksi (gap={row['Overfitting Gap']:.3f})")
        
        # Visualisasi perbandingan
        st.subheader("📈 Visualisasi Perbandingan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(comp_df, x='Model', y=['Train R²', 'Test R²', 'CV R² (mean)'],
                        barmode='group', title='Perbandingan R² Score')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(comp_df, x='Test MAE', y='Test R²',
                           text='Model', size='Test RMSE',
                           title='MAE vs R² Score (size=RMSE)')
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
        
        # Tips interpretasi
        with st.expander("💡 Tips Interpretasi Metrik"):
            st.markdown("""
            **R² Score (Coefficient of Determination):**
            - **1.0**: Prediksi sempurna
            - **0.7-0.9**: Model sangat bagus
            - **0.5-0.7**: Model cukup bagus
            - **< 0.5**: Model kurang bagus
            - **< 0**: Model lebih buruk dari mean sederhana
            
            **MAE (Mean Absolute Error):**
            - Rata-rata kesalahan prediksi
            - Semakin kecil semakin baik
            - Dalam satuan IPK (misal: MAE=0.3 artinya rata-rata error ±0.3 poin IPK)
            
            **RMSE (Root Mean Squared Error):**
            - Mirip MAE tapi lebih sensitif terhadap error besar
            - Semakin kecil semakin baik
            
            **Cross-Validation R²:**
            - Lebih reliable daripada single train-test split
            - Std yang kecil menunjukkan model stabil
            
            **Overfitting Gap:**
            - Selisih Train R² dan Test R²
            - Gap > 0.2 menunjukkan overfitting
            - Model menghafal data training, tidak generalisasi
            """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🎓 Aplikasi Prediksi IPK v3.0 (Improved)")
st.sidebar.caption("✨ Dengan Feature Engineering & Cross-Validation")

# Tips di sidebar
with st.sidebar.expander("💡 Tips Penggunaan"):
    st.markdown("""
    **Untuk hasil terbaik:**
    1. Pastikan dataset minimal 300+ baris
    2. Tidak ada nilai kosong (NaN)
    3. Data konsisten dan logis
    4. Pilih model dengan Test R² tertinggi
    5. Hindari model dengan overfitting gap > 0.2
    
    **Jika R² negatif:**
    - Cek kualitas data
    - Mungkin fitur tidak relevan dengan target
    - Coba tambah lebih banyak data
    - Hubungan IPK dengan fitur mungkin non-linear
    """)
