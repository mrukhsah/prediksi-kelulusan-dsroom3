import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Fungsi untuk menghitung semua metrik evaluasi
def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mae, mape, rmse, r2

# Fungsi preprocessing data
def preprocess_data(df):
    df_processed = df.copy()
    
    # Encode categorical features
    le_gender = LabelEncoder()
    le_married = LabelEncoder()
    
    df_processed['jenis_kelamin_encoded'] = le_gender.fit_transform(df_processed['jenis_kelamin'])
    df_processed['status_menikah_encoded'] = le_married.fit_transform(df_processed['status_menikah'])
    
    # Handle missing values
    df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
    
    return df_processed, le_gender, le_married

# Fungsi untuk training model
def train_model(df, model_type='random_forest'):
    # Preprocessing
    df_processed, le_gender, le_married = preprocess_data(df)
    
    # Features dan Target
    features = ['jenis_kelamin_encoded', 'umur', 'status_menikah_encoded', 
                'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning']
    
    X = df_processed[features]
    y = df_processed['ipk']
    
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    
    # Scaling untuk semua model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Pilih model dengan hyperparameter yang optimal
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=7,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8
        )
    elif model_type == 'knn':
        model = KNeighborsRegressor(
            n_neighbors=10,
            weights='distance',
            metric='minkowski',
            p=2
        )
    
    # Training model
    model.fit(X_train_scaled, y_train)
    
    # Prediksi pada data training
    y_train_pred = model.predict(X_train_scaled)
    mae_train, mape_train, rmse_train, r2_train = calculate_metrics(y_train, y_train_pred)
    
    # Prediksi pada data testing
    y_test_pred = model.predict(X_test_scaled)
    mae_test, mape_test, rmse_test, r2_test = calculate_metrics(y_test, y_test_pred)
    
    return (model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred,
            mae_train, mape_train, rmse_train, r2_train,
            mae_test, mape_test, rmse_test, r2_test,
            le_gender, le_married, features, scaler)

# Header
st.title("🎓 Aplikasi Prediksi Kelulusan Mahasiswa")
st.markdown("Aplikasi ini memprediksi IPK mahasiswa menggunakan berbagai algoritma Machine Learning")
st.info("📊 **Split Data:** 80% Training | 20% Testing")

# Sidebar
st.sidebar.header("📋 Menu")
menu = st.sidebar.radio("Pilih Menu:", ["Upload & Training", "Prediksi Individual", "Visualisasi", "Perbandingan Model"])

# Session state untuk menyimpan model
if 'models' not in st.session_state:
    st.session_state.models = {}
    st.session_state.results = {}
    st.session_state.trained = False

# MENU 1: Upload & Training
if menu == "Upload & Training":
    st.header("📤 Upload Dataset dan Training Model")
    
    uploaded_file = st.file_uploader("Upload file CSV/Excel", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Data berhasil diupload! Total: {len(df)} baris")
        
        # Tampilkan data
        st.subheader("📊 Preview Data")
        st.dataframe(df.head(10))
        
        # Info data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Data", len(df))
        with col2:
            st.metric("Data Training (80%)", int(len(df) * 0.8))
        with col3:
            st.metric("Data Testing (20%)", int(len(df) * 0.2))
        with col4:
            st.metric("Rata-rata IPK", f"{df['ipk'].mean():.2f}")
        
        # Statistik tambahan
        st.subheader("📈 Statistik Dataset")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min IPK", f"{df['ipk'].min():.2f}")
        with col2:
            st.metric("Max IPK", f"{df['ipk'].max():.2f}")
        with col3:
            st.metric("Std Dev IPK", f"{df['ipk'].std():.2f}")
        with col4:
            missing_data = df.isnull().sum().sum()
            st.metric("Missing Values", missing_data)
        
        # Cek kualitas data
        if df['ipk'].std() < 0.3:
            st.warning("⚠️ Variasi IPK cukup kecil, model mungkin kurang optimal.")
        else:
            st.success(f"✅ Variasi IPK baik (std: {df['ipk'].std():.2f})")
        
        # Pilih model
        st.subheader("🤖 Pilih Model untuk Training")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_rf = st.checkbox("Random Forest Regressor", value=True)
        with col2:
            train_gb = st.checkbox("Gradient Boosting Regressor", value=False)
        with col3:
            train_knn = st.checkbox("KNN Regressor", value=False)
        
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
                st.error("❌ Pilih minimal satu model untuk ditraining!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, (model_name, model_type) in enumerate(models_to_train):
                    status_text.text(f"Training {model_name}... ({idx+1}/{len(models_to_train)})")
                    
                    try:
                        result = train_model(df, model_type)
                        (model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred,
                         mae_train, mape_train, rmse_train, r2_train,
                         mae_test, mape_test, rmse_test, r2_test,
                         le_gender, le_married, features, scaler) = result
                        
                        # Simpan ke session state
                        st.session_state.models[model_type] = {
                            'model': model,
                            'scaler': scaler,
                            'le_gender': le_gender,
                            'le_married': le_married,
                            'features': features
                        }
                        
                        st.session_state.results[model_type] = {
                            'name': model_name,
                            'X_train': X_train,
                            'X_test': X_test,
                            'y_train': y_train,
                            'y_test': y_test,
                            'y_train_pred': y_train_pred,
                            'y_test_pred': y_test_pred,
                            'mae_train': mae_train,
                            'mape_train': mape_train,
                            'rmse_train': rmse_train,
                            'r2_train': r2_train,
                            'mae_test': mae_test,
                            'mape_test': mape_test,
                            'rmse_test': rmse_test,
                            'r2_test': r2_test
                        }
                        
                        progress_bar.progress((idx + 1) / len(models_to_train))
                        
                    except Exception as e:
                        st.error(f"❌ Error saat training {model_name}: {str(e)}")
                
                st.session_state.trained = True
                st.session_state.df = df
                status_text.text("✅ Training selesai!")
                
                # Tampilkan hasil semua model
                st.subheader("📈 Hasil Evaluasi Semua Model")
                
                # Tabel untuk Training Set
                st.markdown("### 🔵 Performa pada Data Training (80%)")
                train_data = []
                for model_type, result in st.session_state.results.items():
                    train_data.append({
                        'Model': result['name'],
                        'MAE': f"{result['mae_train']:.4f}",
                        'MAPE': f"{result['mape_train']:.2f}%",
                        'RMSE': f"{result['rmse_train']:.4f}",
                        'R²': f"{result['r2_train']:.4f}"
                    })
                
                train_df = pd.DataFrame(train_data)
                st.dataframe(train_df, use_container_width=True)
                
                # Tabel untuk Testing Set
                st.markdown("### 🟢 Performa pada Data Testing (20%)")
                test_data = []
                for model_type, result in st.session_state.results.items():
                    test_data.append({
                        'Model': result['name'],
                        'MAE': f"{result['mae_test']:.4f}",
                        'MAPE': f"{result['mape_test']:.2f}%",
                        'RMSE': f"{result['rmse_test']:.4f}",
                        'R²': f"{result['r2_test']:.4f}"
                    })
                
                test_df = pd.DataFrame(test_data)
                st.dataframe(test_df, use_container_width=True)
                
                # Visualisasi perbandingan Training vs Testing
                st.subheader("📊 Perbandingan Training vs Testing")
                
                for model_type, result in st.session_state.results.items():
                    with st.expander(f"📊 Detail {result['name']}"):
                        # Metrik Training vs Testing
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🔵 Training Set")
                            subcol1, subcol2 = st.columns(2)
                            with subcol1:
                                st.metric("MAE", f"{result['mae_train']:.4f}")
                                st.metric("RMSE", f"{result['rmse_train']:.4f}")
                            with subcol2:
                                st.metric("MAPE", f"{result['mape_train']:.2f}%")
                                st.metric("R²", f"{result['r2_train']:.4f}")
                        
                        with col2:
                            st.markdown("#### 🟢 Testing Set")
                            subcol1, subcol2 = st.columns(2)
                            with subcol1:
                                st.metric("MAE", f"{result['mae_test']:.4f}")
                                st.metric("RMSE", f"{result['rmse_test']:.4f}")
                            with subcol2:
                                st.metric("MAPE", f"{result['mape_test']:.2f}%")
                                st.metric("R²", f"{result['r2_test']:.4f}")
                        
                        # Scatter plot Training
                        fig = px.scatter(
                            x=result['y_train'], y=result['y_train_pred'],
                            labels={'x': 'IPK Aktual', 'y': 'IPK Prediksi'},
                            title=f"Training Set - Prediksi vs Aktual - {result['name']}"
                        )
                        fig.add_trace(go.Scatter(
                            x=[result['y_train'].min(), result['y_train'].max()], 
                            y=[result['y_train'].min(), result['y_train'].max()],
                            mode='lines', name='Perfect Prediction',
                            line=dict(color='red', dash='dash')
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Scatter plot Testing
                        fig = px.scatter(
                            x=result['y_test'], y=result['y_test_pred'],
                            labels={'x': 'IPK Aktual', 'y': 'IPK Prediksi'},
                            title=f"Testing Set - Prediksi vs Aktual - {result['name']}"
                        )
                        fig.add_trace(go.Scatter(
                            x=[result['y_test'].min(), result['y_test'].max()], 
                            y=[result['y_test'].min(), result['y_test'].max()],
                            mode='lines', name='Perfect Prediction',
                            line=dict(color='red', dash='dash')
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature importance (untuk RF dan GB)
                        if model_type in ['random_forest', 'gradient_boosting']:
                            model_obj = st.session_state.models[model_type]['model']
                            importance_df = pd.DataFrame({
                                'Feature': ['Jenis Kelamin', 'Umur', 'Status Menikah', 
                                          'Kehadiran', 'Partisipasi Diskusi', 'Nilai Tugas', 'Aktivitas E-Learning'],
                                'Importance': model_obj.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig2 = px.bar(importance_df, x='Importance', y='Feature', 
                                        title=f'Feature Importance - {result["name"]}',
                                        orientation='h')
                            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("👆 Silakan upload dataset terlebih dahulu")
        st.markdown("""
        **Format dataset yang dibutuhkan (500 baris):**
        - **Nama** - Nama mahasiswa
        - **jenis_kelamin** - Laki-laki/Perempuan
        - **umur** - Umur mahasiswa
        - **status_menikah** - Menikah/Belum Menikah
        - **kehadiran** - Skor 1-100
        - **partisipasi_diskusi** - Skor 1-100
        - **nilai_tugas** - Skor 1-100
        - **aktivitas_elearning** - Skor 1-100
        - **ipk** - IPK 1.5-4.0 (target prediksi)
        - **status_akademik** - Lulus/Tidak Lulus
        
        **Split Data:**
        - 80% (400 baris) untuk Training
        - 20% (100 baris) untuk Testing
        """)

# MENU 2: Prediksi Individual
elif menu == "Prediksi Individual":
    st.header("🔮 Prediksi IPK Individual")
    
    if not st.session_state.trained:
        st.warning("⚠️ Model belum ditraining. Silakan upload data dan training model terlebih dahulu di menu 'Upload & Training'")
    else:
        st.success(f"✅ {len(st.session_state.models)} model siap digunakan!")
        
        # Pilih model untuk prediksi
        model_options = {v['name']: k for k, v in st.session_state.results.items()}
        selected_model_name = st.selectbox("Pilih Model:", list(model_options.keys()))
        selected_model_type = model_options[selected_model_name]
        
        # Tampilkan performa model yang dipilih
        result = st.session_state.results[selected_model_type]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE (Test)", f"{result['mae_test']:.4f}")
        with col2:
            st.metric("RMSE (Test)", f"{result['rmse_test']:.4f}")
        with col3:
            st.metric("MAPE (Test)", f"{result['mape_test']:.2f}%")
        with col4:
            st.metric("R² (Test)", f"{result['r2_test']:.4f}")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            umur = st.number_input("Umur", min_value=17, max_value=50, value=20)
            status_menikah = st.selectbox("Status Menikah", ["Belum Menikah", "Menikah"])
            kehadiran = st.slider("Kehadiran (1-100)", 1, 100, 80)
        
        with col2:
            partisipasi = st.slider("Partisipasi Diskusi (1-100)", 1, 100, 75)
            nilai_tugas = st.slider("Nilai Tugas (1-100)", 1, 100, 80)
            aktivitas = st.slider("Aktivitas E-Learning (1-100)", 1, 100, 70)
        
        if st.button("🎯 Prediksi IPK", type="primary"):
            model_data = st.session_state.models[selected_model_type]
            
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
            
            # Scale
            input_data_scaled = model_data['scaler'].transform(input_data)
            
            # Prediksi
            prediksi = model_data['model'].predict(input_data_scaled)[0]
            
            # Tampilkan hasil
            st.subheader("📊 Hasil Prediksi")
            
            # Determine status
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
                st.metric("Model Digunakan", selected_model_name)
            with col2:
                st.metric("IPK Diprediksi", f"{prediksi:.2f}")
            with col3:
                st.markdown(f"### {emoji} :{color}[{status}]")
            
            # Progress bar
            st.progress(min(prediksi / 4.0, 1.0))
            
            # Status akademik
            status_akademik = "LULUS" if prediksi >= 2.75 else "TIDAK LULUS"
            if status_akademik == "LULUS":
                st.success(f"✅ Prediksi Status Akademik: **{status_akademik}**")
            else:
                st.error(f"❌ Prediksi Status Akademik: **{status_akademik}**")
            
            # Prediksi dari semua model (opsional)
            if st.checkbox("Lihat prediksi dari semua model"):
                st.subheader("Perbandingan Prediksi Semua Model")
                all_predictions = []
                
                for model_type, model_data in st.session_state.models.items():
                    input_scaled = model_data['scaler'].transform(input_data)
                    pred = model_data['model'].predict(input_scaled)[0]
                    
                    all_predictions.append({
                        'Model': st.session_state.results[model_type]['name'],
                        'Prediksi IPK': f"{pred:.2f}",
                        'Status': "Lulus" if pred >= 2.75 else "Tidak Lulus"
                    })
                
                pred_df = pd.DataFrame(all_predictions)
                st.dataframe(pred_df, use_container_width=True)

# MENU 3: Visualisasi
elif menu == "Visualisasi":
    st.header("📊 Visualisasi Data & Hasil")
    
    if not st.session_state.trained:
        st.warning("⚠️ Model belum ditraining. Silakan upload data dan training model terlebih dahulu.")
    else:
        df = st.session_state.df
        
        tab1, tab2, tab3 = st.tabs(["Distribusi Data", "Korelasi", "Model Performance"])
        
        with tab1:
            st.subheader("Distribusi IPK")
            fig = px.histogram(df, x='ipk', nbins=30, title='Distribusi IPK Mahasiswa')
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(df, y='kehadiran', title='Distribusi Kehadiran')
                st.plotly_chart(fig, use_container_width=True)
                fig = px.box(df, y='nilai_tugas', title='Distribusi Nilai Tugas')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(df, y='partisipasi_diskusi', title='Distribusi Partisipasi Diskusi')
                st.plotly_chart(fig, use_container_width=True)
                fig = px.box(df, y='aktivitas_elearning', title='Distribusi Aktivitas E-Learning')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Korelasi Antar Variabel")
            numeric_cols = ['umur', 'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning', 'ipk']
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                           title='Heatmap Korelasi',
                           color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plots untuk melihat hubungan dengan IPK
            st.subheader("Hubungan Fitur dengan IPK")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(df, x='kehadiran', y='ipk', title='Kehadiran vs IPK',
                               trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(df, x='nilai_tugas', y='ipk', title='Nilai Tugas vs IPK',
                               trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Model Performance")
            
            # Pilih model untuk visualisasi
            model_options = {v['name']: k for k, v in st.session_state.results.items()}
            selected_model_name = st.selectbox("Pilih Model:", list(model_options.keys()))
            selected_model_type = model_options[selected_model_name]
            
            result = st.session_state.results[selected_model_type]
            
            # Pilih dataset
            dataset_choice = st.radio("Pilih Dataset:", ["Training (80%)", "Testing (20%)"])
            
            if dataset_choice == "Training (80%)":
                y_actual = result['y_train']
                y_pred = result['y_train_pred']
                mae = result['mae_train']
                rmse = result['rmse_train']
                r2 = result['r2_train']
            else:
                y_actual = result['y_test']
                y_pred = result['y_test_pred']
                mae = result['mae_test']
                rmse = result['rmse_test']
                r2 = result['r2_test']
            
            # Metrik
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"{mae:.4f}")
            with col2:
                st.metric("RMSE", f"{rmse:.4f}")
            with col3:
                st.metric("MAPE", f"{result['mape_train' if dataset_choice == 'Training (80%)' else 'mape_test']:.2f}%")
            with col4:
                st.metric("R²", f"{r2:.4f}")
            
            # Scatter plot
            fig = px.scatter(
                x=y_actual, 
                y=y_pred,
                labels={'x': 'IPK Aktual', 'y': 'IPK Prediksi'},
                title=f'{dataset_choice} - Prediksi vs Aktual - {selected_model_name}'
            )
            fig.add_trace(go.Scatter(
                x=[y_actual.min(), y_actual.max()], 
                y=[y_actual.min(), y_actual.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual plot
            residuals = y_actual - y_pred
            fig = px.scatter(x=y_pred, y=residuals,
                           labels={'x': 'Prediksi', 'y': 'Residual'},
                           title=f'Residual Plot - {selected_model_name} - {dataset_choice}')
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

# MENU 4: Perbandingan Model
elif menu == "Perbandingan Model":
    st.header("⚖️ Perbandingan Model")
    
    if not st.session_state.trained:
        st.warning("⚠️ Model belum ditraining. Silakan upload data dan training model terlebih dahulu.")
    else:
        # Pilih dataset untuk perbandingan
        dataset_compare = st.radio("Bandingkan pada:", ["Testing Set (20%)", "Training Set (80%)"])
        
        st.subheader(f"📊 Metrik Evaluasi - {dataset_compare}")
        
        # Tabel perbandingan
        results_data = []
        for model_type, result in st.session_state.results.items():
            if dataset_compare == "Testing Set (20%)":
                results_data.append({
                    'Model': result['name'],
                    'MAE': result['mae_test'],
                    'MAPE (%)': result['mape_test'],
                    'RMSE': result['rmse_test'],
                    'R²': result['r2_test']
                })
            else:
                results_data.append({
                    'Model': result['name'],
                    'MAE': result['mae_train'],
                    'MAPE (%)': result['mape_train'],
                    'RMSE': result['rmse_train'],
                    'R²': result['r2_train']
                })
        
        results_df = pd.DataFrame(results_data)
        
        # Highlight best scores
        styled_df = results_df.style.highlight_min(subset=['MAE', 'MAPE (%)', 'RMSE'], color='lightgreen')
        styled_df = styled_df.highlight_max(subset=['R²'], color='lightgreen')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Visualisasi perbandingan
        st.subheader("📈 Visualisasi Perbandingan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(results_df, x='Model', y='MAE', 
                        title='Perbandingan MAE (Lower is Better)',
                        color='MAE',
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(results_df, x='Model', y='RMSE',
                        title='Perbandingan RMSE (Lower is Better)',
                        color='RMSE',
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig = px.bar(results_df, x='Model', y='MAPE (%)',
                        title='Perbandingan MAPE (Lower is Better)',
                        color='MAPE (%)',
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            fig = px.bar(results_df, x='Model', y='R²',
                        title='Perbandingan R² Score (Higher is Better)',
                        color='R²',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot semua model
        st.subheader("📈 Perbandingan Prediksi vs Aktual Semua Model")
        
        fig = go.Figure()
        
        colors = ['blue', 'green', 'orange', 'purple', 'red']
        for idx, (model_type, result) in enumerate(st.session_state.results.items()):
            if dataset_compare == "Testing Set (20%)":
                y_actual = result['y_test']
                y_pred = result['y_test_pred']
            else:
                y_actual = result['y_train']
                y_pred = result['y_train_pred']
            
            fig.add_trace(go.Scatter(
                x=y_actual,
                y=y_pred,
                mode='markers',
                name=result['name'],
                marker=dict(color=colors[idx % len(colors)], size=8, opacity=0.6)
            ))
        
        # Perfect prediction line
        all_y = []
        for result in st.session_state.results.values():
            if dataset_compare == "Testing Set (20%)":
                all_y.extend(result['y_test'].values)
            else:
                all_y.extend(result['y_train'].values)
        
        y_min = min(all_y)
        y_max = max(all_y)
        
        fig.add_trace(go.Scatter(
            x=[y_min, y_max],
            y=[y_min, y_max],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=f'Perbandingan Semua Model - {dataset_compare}',
            xaxis_title='IPK Aktual',
            yaxis_title='IPK Prediksi',
            hovermode='closest',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rekomendasi model terbaik
        st.subheader("🏆 Rekomendasi Model Terbaik")
        
        best_model_mae = results_df.loc[results_df['MAE'].idxmin()]
        best_model_r2 = results_df.loc[results_df['R²'].idxmax()]
        best_model_rmse = results_df.loc[results_df['RMSE'].idxmin()]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"🥇 **Model dengan MAE Terbaik**")
            st.write(f"**{best_model_mae['Model']}**")
            st.caption(f"MAE = {best_model_mae['MAE']:.4f}")
        with col2:
            st.success(f"🥇 **Model dengan RMSE Terbaik**")
            st.write(f"**{best_model_rmse['Model']}**")
            st.caption(f"RMSE = {best_model_rmse['RMSE']:.4f}")
        with col3:
            st.success(f"🥇 **Model dengan R² Terbaik**")
            st.write(f"**{best_model_r2['Model']}**")
            st.caption(f"R² = {best_model_r2['R²']:.4f}")
        
        # Analisis Overfitting/Underfitting
        st.subheader("🔍 Analisis Overfitting/Underfitting")
        
        for model_type, result in st.session_state.results.items():
            with st.expander(f"Analisis {result['name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Training Set")
                    st.metric("R²", f"{result['r2_train']:.4f}")
                    st.metric("MAE", f"{result['mae_train']:.4f}")
                
                with col2:
                    st.markdown("#### Testing Set")
                    st.metric("R²", f"{result['r2_test']:.4f}")
                    st.metric("MAE", f"{result['mae_test']:.4f}")
                
                # Analisis
                r2_diff = abs(result['r2_train'] - result['r2_test'])
                mae_diff = abs(result['mae_train'] - result['mae_test'])
                
                if r2_diff > 0.15 or mae_diff > 0.2:
                    st.warning("⚠️ **Kemungkinan Overfitting**: Performa training jauh lebih baik dari testing")
                elif result['r2_test'] < 0.5:
                    st.warning("⚠️ **Kemungkinan Underfitting**: R² testing masih rendah")
                else:
                    st.success("✅ **Model Baik**: Performa training dan testing seimbang")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🎓 Aplikasi Prediksi Kelulusan v2.0")
st.sidebar.caption("Dibuat dengan Streamlit & Multiple ML Models")
st.sidebar.markdown("**Split Data:** 80% Training | 20% Testing")
