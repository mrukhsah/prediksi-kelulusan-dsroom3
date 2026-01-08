import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
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
    numeric_cols = ['umur', 'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning', 'ipk']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed, le_gender, le_married

# Fungsi untuk training model dengan dataset yang sudah di-split
def train_model_with_split(X_train, X_test, y_train, y_test, model_type='random_forest'):
    # Scaling untuk semua model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Pilih model dengan hyperparameter yang optimal
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8
        )
    elif model_type == 'knn':
        model = KNeighborsRegressor(
            n_neighbors=7,
            weights='distance',
            metric='minkowski',
            p=2
        )
    
    # Training model
    model.fit(X_train_scaled, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test_scaled)
    
    # Hitung metrik
    mae, mape, rmse, r2 = calculate_metrics(y_test, y_pred)
    
    return model, y_pred, mae, mape, rmse, r2, scaler

# Header
st.title("🎓 Aplikasi Prediksi Kelulusan Mahasiswa")
st.markdown("Aplikasi ini memprediksi IPK mahasiswa menggunakan berbagai algoritma Machine Learning")

# Sidebar
st.sidebar.header("📋 Menu")
menu = st.sidebar.radio("Pilih Menu:", ["Upload & Training", "Prediksi Individual", "Visualisasi", "Perbandingan Model"])

# Session state untuk menyimpan model
if 'models' not in st.session_state:
    st.session_state.models = {}
    st.session_state.results = {}
    st.session_state.trained = False
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None

# MENU 1: Upload & Training
if menu == "Upload & Training":
    st.header("📤 Upload Dataset dan Training Model")
    
    # Pilih metode upload
    upload_method = st.radio(
        "Pilih Metode Upload:",
        ["Auto Split 80:20", "Upload Training & Testing Terpisah"]
    )
    
    if upload_method == "Auto Split 80:20":
        st.info("📌 Upload 1 file dataset, sistem akan otomatis split 80% training dan 20% testing")
        
        uploaded_file = st.file_uploader("Upload file CSV/Excel (Full Dataset)", type=['csv', 'xlsx'], key='full')
        
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
            
            # Preprocessing
            df_processed, le_gender, le_married = preprocess_data(df)
            
            # Features dan Target
            features = ['jenis_kelamin_encoded', 'umur', 'status_menikah_encoded', 
                        'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning']
            
            X = df_processed[features]
            y = df_processed['ipk']
            
            # Split data 80:20
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
            
            # Info split
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Data", len(df))
            with col2:
                st.metric("Training Data", f"{len(X_train)} (80%)")
            with col3:
                st.metric("Testing Data", f"{len(X_test)} (20%)")
            with col4:
                st.metric("Rata-rata IPK", f"{df['ipk'].mean():.2f}")
            
            # Simpan ke session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.le_gender = le_gender
            st.session_state.le_married = le_married
            st.session_state.features = features
            st.session_state.df = df
            
            st.success("✅ Data berhasil di-split dan siap untuk training!")
    
    else:  # Upload Training & Testing Terpisah
        st.info("📌 Upload 2 file terpisah: 1 untuk training dan 1 untuk testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 File Training")
            train_file = st.file_uploader("Upload Training Dataset (CSV/Excel)", type=['csv', 'xlsx'], key='train')
            
            if train_file is not None:
                if train_file.name.endswith('.csv'):
                    df_train = pd.read_csv(train_file)
                else:
                    df_train = pd.read_excel(train_file)
                
                st.success(f"✅ Training data: {len(df_train)} baris")
                st.dataframe(df_train.head(5))
        
        with col2:
            st.subheader("📁 File Testing")
            test_file = st.file_uploader("Upload Testing Dataset (CSV/Excel)", type=['csv', 'xlsx'], key='test')
            
            if test_file is not None:
                if test_file.name.endswith('.csv'):
                    df_test = pd.read_csv(test_file)
                else:
                    df_test = pd.read_excel(test_file)
                
                st.success(f"✅ Testing data: {len(df_test)} baris")
                st.dataframe(df_test.head(5))
        
        # Jika kedua file sudah diupload
        if train_file is not None and test_file is not None:
            # Preprocessing training data
            df_train_processed, le_gender, le_married = preprocess_data(df_train)
            
            # Preprocessing testing data dengan encoder yang sama
            df_test_processed = df_test.copy()
            df_test_processed['jenis_kelamin_encoded'] = le_gender.transform(df_test_processed['jenis_kelamin'])
            df_test_processed['status_menikah_encoded'] = le_married.transform(df_test_processed['status_menikah'])
            
            # Handle missing values di test data
            numeric_cols = ['umur', 'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning', 'ipk']
            for col in numeric_cols:
                if col in df_test_processed.columns:
                    df_test_processed[col] = df_test_processed[col].fillna(df_test_processed[col].median())
            
            # Features dan Target
            features = ['jenis_kelamin_encoded', 'umur', 'status_menikah_encoded', 
                        'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning']
            
            X_train = df_train_processed[features]
            y_train = df_train_processed['ipk']
            X_test = df_test_processed[features]
            y_test = df_test_processed['ipk']
            
            # Info split
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Data", len(df_train) + len(df_test))
            with col2:
                st.metric("Training Data", len(X_train))
            with col3:
                st.metric("Testing Data", len(X_test))
            with col4:
                percentage = (len(X_test) / (len(X_train) + len(X_test))) * 100
                st.metric("Split Ratio", f"{100-percentage:.0f}:{percentage:.0f}")
            
            # Simpan ke session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.le_gender = le_gender
            st.session_state.le_married = le_married
            st.session_state.features = features
            st.session_state.df = pd.concat([df_train, df_test], ignore_index=True)
            
            st.success("✅ Data training dan testing berhasil diupload dan siap untuk training!")
    
    # Training Section (muncul setelah data siap)
    if st.session_state.X_train is not None:
        st.markdown("---")
        st.subheader("🤖 Training Model")
        
        # Pilih model
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
                        model, y_pred, mae, mape, rmse, r2, scaler = train_model_with_split(
                            st.session_state.X_train,
                            st.session_state.X_test,
                            st.session_state.y_train,
                            st.session_state.y_test,
                            model_type
                        )
                        
                        # Simpan ke session state
                        st.session_state.models[model_type] = {
                            'model': model,
                            'scaler': scaler,
                            'le_gender': st.session_state.le_gender,
                            'le_married': st.session_state.le_married,
                            'features': st.session_state.features
                        }
                        
                        st.session_state.results[model_type] = {
                            'name': model_name,
                            'y_test': st.session_state.y_test,
                            'y_pred': y_pred,
                            'mae': mae,
                            'mape': mape,
                            'rmse': rmse,
                            'r2': r2
                        }
                        
                        progress_bar.progress((idx + 1) / len(models_to_train))
                        
                    except Exception as e:
                        st.error(f"❌ Error saat training {model_name}: {str(e)}")
                
                st.session_state.trained = True
                status_text.text("✅ Training selesai!")
                
                # Tampilkan hasil semua model
                st.subheader("📈 Hasil Evaluasi Semua Model")
                
                results_data = []
                for model_type, result in st.session_state.results.items():
                    results_data.append({
                        'Model': result['name'],
                        'MAE': f"{result['mae']:.4f}",
                        'MAPE': f"{result['mape']:.2f}%",
                        'RMSE': f"{result['rmse']:.4f}",
                        'R²': f"{result['r2']:.4f}"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Visualisasi perbandingan
                col1, col2 = st.columns(2)
                
                with col1:
                    mae_data = [result['mae'] for result in st.session_state.results.values()]
                    model_names = [result['name'] for result in st.session_state.results.values()]
                    
                    fig = px.bar(x=model_names, y=mae_data, 
                                title='Perbandingan MAE',
                                labels={'x': 'Model', 'y': 'MAE'})
                    st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot semua model
        st.subheader("📈 Perbandingan Prediksi vs Aktual")
        
        fig = go.Figure()
        
        colors = ['blue', 'green', 'orange']
        for idx, (model_type, result) in enumerate(st.session_state.results.items()):
            fig.add_trace(go.Scatter(
                x=result['y_test'],
                y=result['y_pred'],
                mode='markers',
                name=result['name'],
                marker=dict(color=colors[idx % len(colors)], size=8, opacity=0.6)
            ))
        
        # Perfect prediction line
        y_min = min([r['y_test'].min() for r in st.session_state.results.values()])
        y_max = max([r['y_test'].max() for r in st.session_state.results.values()])
        
        fig.add_trace(go.Scatter(
            x=[y_min, y_max],
            y=[y_min, y_max],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title='Perbandingan Semua Model',
            xaxis_title='IPK Aktual',
            yaxis_title='IPK Prediksi',
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rekomendasi model terbaik
        best_model_mae = results_df.loc[results_df['MAE'].idxmin()]
        best_model_r2 = results_df.loc[results_df['R²'].idxmax()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🏆 **Model dengan MAE Terbaik:** {best_model_mae['Model']}")
            st.caption(f"MAE = {best_model_mae['MAE']:.4f}, RMSE = {best_model_mae['RMSE']:.4f}")
        with col2:
            st.success(f"🏆 **Model dengan R² Terbaik:** {best_model_r2['Model']}")
            st.caption(f"R² = {best_model_r2['R²']:.4f}, MAPE = {best_model_r2['MAPE (%)']:.2f}%")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🎓 Aplikasi Prediksi Kelulusan v3.0")
st.sidebar.caption("Dibuat dengan Streamlit & Multiple ML Models")
st.sidebar.markdown("**Fitur:**")
st.sidebar.markdown("✅ Auto Split 80:20")
st.sidebar.markdown("✅ Upload Training & Testing Terpisah")
st.sidebar.markdown("✅ 3 Model ML (RF, GB, KNN)")
st.sidebar.markdown("✅ 4 Metrik Evaluasi (MAE, MAPE, RMSE, R²)")
                
                with col2:
                    r2_data = [result['r2'] for result in st.session_state.results.values()]
                    
                    fig = px.bar(x=model_names, y=r2_data,
                                title='Perbandingan R² Score',
                                labels={'x': 'Model', 'y': 'R²'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detail untuk setiap model
                for model_type, result in st.session_state.results.items():
                    with st.expander(f"📊 Detail {result['name']}"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MAE", f"{result['mae']:.4f}")
                        with col2:
                            st.metric("MAPE", f"{result['mape']:.2f}%")
                        with col3:
                            st.metric("RMSE", f"{result['rmse']:.4f}")
                        with col4:
                            st.metric("R² Score", f"{result['r2']:.4f}")
                        
                        # Scatter plot
                        fig = px.scatter(
                            x=result['y_test'], y=result['y_pred'],
                            labels={'x': 'IPK Aktual', 'y': 'IPK Prediksi'},
                            title=f"Prediksi vs Aktual - {result['name']}"
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            umur = st.number_input("Umur", min_value=17, max_value=50, value=20)
            status_menikah = st.selectbox("Status Menikah", ["Belum Menikah", "Menikah"])
            kehadiran = st.slider("Kehadiran (%)", 0, 100, 80)
        
        with col2:
            partisipasi = st.number_input("Partisipasi Diskusi (skor)", min_value=0, max_value=100, value=75)
            nilai_tugas = st.number_input("Nilai Tugas (rata-rata)", min_value=0.0, max_value=100.0, value=80.0)
            aktivitas = st.number_input("Aktivitas E-Learning (skor)", min_value=0, max_value=100, value=70)
        
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
            elif prediksi >= 3.0:
                status = "Sangat Memuaskan"
                color = "blue"
            elif prediksi >= 2.75:
                status = "Memuaskan"
                color = "orange"
            else:
                status = "Cukup"
                color = "red"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Digunakan", selected_model_name)
            with col2:
                st.metric("IPK Diprediksi", f"{prediksi:.2f}")
            with col3:
                st.markdown(f"**Status:** :{color}[{status}]")
            
            # Progress bar
            st.progress(min(prediksi / 4.0, 1.0))
            
            # Prediksi dari semua model (opsional)
            if st.checkbox("Lihat prediksi dari semua model"):
                st.subheader("Perbandingan Prediksi Semua Model")
                all_predictions = []
                
                for model_type, model_data in st.session_state.models.items():
                    input_scaled = model_data['scaler'].transform(input_data)
                    pred = model_data['model'].predict(input_scaled)[0]
                    
                    all_predictions.append({
                        'Model': st.session_state.results[model_type]['name'],
                        'Prediksi IPK': f"{pred:.2f}"
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
            fig = px.histogram(df, x='ipk', nbins=20, title='Distribusi IPK Mahasiswa')
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(df, y='kehadiran', title='Distribusi Kehadiran')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(df, y='nilai_tugas', title='Distribusi Nilai Tugas')
                st.plotly_chart(fig, use_container_width=True)
            
            # Distribusi status akademik
            if 'status_akademik' in df.columns:
                st.subheader("Distribusi Status Akademik")
                status_count = df['status_akademik'].value_counts()
                fig = px.pie(values=status_count.values, names=status_count.index, 
                            title='Distribusi Status Akademik')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Korelasi Antar Variabel")
            numeric_cols = ['umur', 'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning', 'ipk']
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title='Heatmap Korelasi',
                           color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Model Performance")
            
            # Pilih model untuk visualisasi
            model_options = {v['name']: k for k, v in st.session_state.results.items()}
            selected_model_name = st.selectbox("Pilih Model:", list(model_options.keys()))
            selected_model_type = model_options[selected_model_name]
            
            result = st.session_state.results[selected_model_type]
            
            # Scatter plot
            fig = px.scatter(
                x=result['y_test'], 
                y=result['y_pred'],
                labels={'x': 'IPK Aktual', 'y': 'IPK Prediksi'},
                title=f'Prediksi vs Aktual - {selected_model_name}'
            )
            fig.add_trace(go.Scatter(
                x=[result['y_test'].min(), result['y_test'].max()], 
                y=[result['y_test'].min(), result['y_test'].max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual plot
            residuals = result['y_test'] - result['y_pred']
            fig = px.scatter(x=result['y_pred'], y=residuals,
                           labels={'x': 'Prediksi', 'y': 'Residual'},
                           title=f'Residual Plot - {selected_model_name}')
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

# MENU 4: Perbandingan Model
elif menu == "Perbandingan Model":
    st.header("⚖️ Perbandingan Model")
    
    if not st.session_state.trained:
        st.warning("⚠️ Model belum ditraining. Silakan upload data dan training model terlebih dahulu.")
    else:
        st.subheader("📊 Metrik Evaluasi Semua Model")
        
        # Tabel perbandingan
        results_data = []
        for model_type, result in st.session_state.results.items():
            results_data.append({
                'Model': result['name'],
                'MAE': result['mae'],
                'MAPE (%)': result['mape'],
                'RMSE': result['rmse'],
                'R²': result['r2']
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Highlight best scores
        styled_df = results_df.style.highlight_min(subset=['MAE', 'MAPE (%)', 'RMSE'], color='lightgreen')
        styled_df = styled_df.highlight_max(subset=['R²'], color='lightgreen')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Visualisasi perbandingan
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
