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

# Fungsi untuk menghitung MAE dan MAPE
def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, mape

# Fungsi preprocessing data
def preprocess_data(df):
    df_processed = df.copy()
    
    # Encode categorical features
    le_gender = LabelEncoder()
    le_married = LabelEncoder()
    
    df_processed['jenis_kelamin_encoded'] = le_gender.fit_transform(df_processed['jenis_kelamin'])
    df_processed['status_menikah_encoded'] = le_married.fit_transform(df_processed['status_menikah'])
    
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
    
    # Pilih model
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1)
    elif model_type == 'knn':
        model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    
    # Training model
    model.fit(X_train_scaled, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test_scaled)
    
    # Hitung metrik
    mae, mape = calculate_metrics(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, mae, mape, le_gender, le_married, features, scaler

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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", len(df))
        with col2:
            st.metric("Rata-rata IPK", f"{df['ipk'].mean():.2f}")
        with col3:
            st.metric("Rata-rata Kehadiran", f"{df['kehadiran'].mean():.1f}%")
        
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
                        model, X_test, y_test, y_pred, mae, mape, le_gender, le_married, features, scaler = train_model(df, model_type)
                        
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
                            'X_test': X_test,
                            'y_test': y_test,
                            'y_pred': y_pred,
                            'mae': mae,
                            'mape': mape
                        }
                        
                        progress_bar.progress((idx + 1) / len(models_to_train))
                        
                    except Exception as e:
                        st.error(f"❌ Error saat training {model_name}: {str(e)}")
                
                st.session_state.trained = True
                st.session_state.df = df
                status_text.text("✅ Training selesai!")
                
                # Tampilkan hasil semua model
                st.subheader("📈 Hasil Evaluasi Semua Model")
                
                results_data = []
                for model_type, result in st.session_state.results.items():
                    results_data.append({
                        'Model': result['name'],
                        'MAE': f"{result['mae']:.4f}",
                        'MAPE': f"{result['mape']:.2f}%"
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
                
                with col2:
                    mape_data = [result['mape'] for result in st.session_state.results.values()]
                    
                    fig = px.bar(x=model_names, y=mape_data,
                                title='Perbandingan MAPE (%)',
                                labels={'x': 'Model', 'y': 'MAPE (%)'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detail untuk setiap model
                for model_type, result in st.session_state.results.items():
                    with st.expander(f"📊 Detail {result['name']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("MAE", f"{result['mae']:.4f}")
                        with col2:
                            st.metric("MAPE", f"{result['mape']:.2f}%")
                        
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
    else:
        st.info("👆 Silakan upload dataset terlebih dahulu")
        st.markdown("""
        **Format dataset yang dibutuhkan:**
        - Nama
        - jenis_kelamin (Laki-laki/Perempuan)
        - umur
        - status_menikah (Menikah/Belum Menikah)
        - kehadiran (dalam %, contoh: 85)
        - partisipasi_diskusi (skor)
        - nilai_tugas (rata-rata)
        - aktivitas_elearning (skor)
        - ipk (target prediksi)
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
            
            # Scale jika KNN
            if selected_model_type == 'knn':
                input_data_scaled = model_data['scaler'].transform(input_data)
            else:
                input_data_scaled = input_data
            
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
                status = "Perlu Peningkatan"
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
                    if model_type == 'knn':
                        input_scaled = model_data['scaler'].transform(input_data)
                    else:
                        input_scaled = input_data
                    
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
                'MAPE (%)': result['mape']
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df.style.highlight_min(subset=['MAE', 'MAPE (%)'], color='lightgreen'), 
                    use_container_width=True)
        
        # Visualisasi perbandingan
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(results_df, x='Model', y='MAE', 
                        title='Perbandingan MAE (Lower is Better)',
                        color='MAE',
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(results_df, x='Model', y='MAPE (%)',
                        title='Perbandingan MAPE (Lower is Better)',
                        color='MAPE (%)',
                        color_continuous_scale='RdYlGn_r')
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
        best_model = results_df.loc[results_df['MAE'].idxmin()]
        st.success(f"🏆 **Model Terbaik:** {best_model['Model']} dengan MAE = {best_model['MAE']:.4f} dan MAPE = {best_model['MAPE (%)']:.2f}%")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🎓 Aplikasi Prediksi Kelulusan v2.0")
st.sidebar.caption("Dibuat dengan Streamlit & Multiple ML Models")
