import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
def train_model(df):
    # Preprocessing
    df_processed, le_gender, le_married = preprocess_data(df)
    
    # Features dan Target
    features = ['jenis_kelamin_encoded', 'umur', 'status_menikah_encoded', 
                'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning']
    
    X = df_processed[features]
    y = df_processed['ipk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung metrik
    mae, mape = calculate_metrics(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, mae, mape, le_gender, le_married, features

# Header
st.title("🎓 Aplikasi Prediksi Kelulusan Mahasiswa")
st.markdown("Aplikasi ini memprediksi IPK mahasiswa menggunakan **Random Forest Regressor**")

# Sidebar
st.sidebar.header("📋 Menu")
menu = st.sidebar.radio("Pilih Menu:", ["Upload & Training", "Prediksi Individual", "Visualisasi"])

# Session state untuk menyimpan model
if 'model' not in st.session_state:
    st.session_state.model = None
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
        
        # Tombol training
        if st.button("🚀 Mulai Training Model", type="primary"):
            with st.spinner("Training model... Mohon tunggu"):
                try:
                    model, X_test, y_test, y_pred, mae, mape, le_gender, le_married, features = train_model(df)
                    
                    # Simpan ke session state
                    st.session_state.model = model
                    st.session_state.le_gender = le_gender
                    st.session_state.le_married = le_married
                    st.session_state.features = features
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    st.session_state.trained = True
                    st.session_state.df = df
                    
                    st.success("✅ Model berhasil ditraining!")
                    
                    # Tampilkan metrik
                    st.subheader("📈 Hasil Evaluasi Model")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("MAE (Mean Absolute Error)", f"{mae:.4f}")
                        st.caption("Rata-rata kesalahan prediksi IPK")
                    with col2:
                        st.metric("MAPE (Mean Absolute % Error)", f"{mape:.2f}%")
                        st.caption("Persentase kesalahan prediksi")
                    
                    # Scatter plot
                    fig = px.scatter(
                        x=y_test, y=y_pred,
                        labels={'x': 'IPK Aktual', 'y': 'IPK Prediksi'},
                        title='Perbandingan IPK Aktual vs Prediksi'
                    )
                    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                            y=[y_test.min(), y_test.max()],
                                            mode='lines', name='Perfect Prediction',
                                            line=dict(color='red', dash='dash')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance
                    importance_df = pd.DataFrame({
                        'Feature': ['Jenis Kelamin', 'Umur', 'Status Menikah', 
                                   'Kehadiran', 'Partisipasi Diskusi', 'Nilai Tugas', 'Aktivitas E-Learning'],
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig2 = px.bar(importance_df, x='Importance', y='Feature', 
                                 title='Feature Importance - Faktor Paling Berpengaruh',
                                 orientation='h')
                    st.plotly_chart(fig2, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error saat training: {str(e)}")
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
        st.success("✅ Model siap digunakan!")
        
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
            # Encode input
            gender_encoded = st.session_state.le_gender.transform([jenis_kelamin])[0]
            married_encoded = st.session_state.le_married.transform([status_menikah])[0]
            
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
            
            # Prediksi
            prediksi = st.session_state.model.predict(input_data)[0]
            
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
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("IPK Diprediksi", f"{prediksi:.2f}")
            with col2:
                st.markdown(f"**Status:** :{color}[{status}]")
            
            # Progress bar
            st.progress(prediksi / 4.0)

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
            
            # Scatter plot
            fig = px.scatter(
                x=st.session_state.y_test, 
                y=st.session_state.y_pred,
                labels={'x': 'IPK Aktual', 'y': 'IPK Prediksi'},
                title='Prediksi vs Aktual'
            )
            fig.add_trace(go.Scatter(
                x=[st.session_state.y_test.min(), st.session_state.y_test.max()], 
                y=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual plot
            residuals = st.session_state.y_test - st.session_state.y_pred
            fig = px.scatter(x=st.session_state.y_pred, y=residuals,
                           labels={'x': 'Prediksi', 'y': 'Residual'},
                           title='Residual Plot')
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🎓 Aplikasi Prediksi Kelulusan v1.0")
st.sidebar.caption("Dibuat dengan Streamlit & Random Forest")