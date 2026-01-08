import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Fungsi untuk preprocessing data
def preprocess_data(df, is_training=True, encoders=None, scaler=None):
    df_processed = df.copy()
    
    # Encode categorical variables
    categorical_cols = ['jenis_kelamin', 'status_menikah']
    
    if is_training:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            encoders[col] = le
        
        # Encode target variable
        if 'status_akademik' in df_processed.columns:
            le_target = LabelEncoder()
            df_processed['status_akademik'] = le_target.fit_transform(df_processed['status_akademik'])
            encoders['status_akademik'] = le_target
    else:
        for col in categorical_cols:
            if col in df_processed.columns:
                df_processed[col] = encoders[col].transform(df_processed[col])
    
    # Features untuk training
    feature_cols = ['jenis_kelamin', 'umur', 'status_menikah', 'kehadiran', 
                    'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning', 'ipk']
    
    X = df_processed[feature_cols]
    
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y = df_processed['status_akademik'] if 'status_akademik' in df_processed.columns else None
        return X_scaled, y, encoders, scaler, feature_cols
    else:
        X_scaled = scaler.transform(X)
        return X_scaled, encoders, scaler, feature_cols

# Fungsi untuk training model
def train_model(df):
    st.info("🔄 Memulai proses training model...")
    
    # Preprocessing
    X, y, encoders, scaler, feature_cols = preprocess_data(df, is_training=True)
    
    # Split data 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    st.write(f"📊 Jumlah data training: {len(X_train)} | Jumlah data testing: {len(X_test)}")
    
    # Training beberapa model
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    st.success(f"✅ Model terbaik: **{best_name}** dengan akurasi **{best_score:.2%}**")
    
    # Simpan model dan preprocessing objects
    model_data = {
        'model': best_model,
        'encoders': encoders,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': best_name,
        'accuracy': best_score
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/saved_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data, results

# Fungsi untuk load model
def load_model():
    try:
        with open('models/saved_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Fungsi prediksi
def predict(model_data, input_data):
    X_scaled, _, _, _ = preprocess_data(input_data, is_training=False, 
                                        encoders=model_data['encoders'], 
                                        scaler=model_data['scaler'])
    
    prediction = model_data['model'].predict(X_scaled)
    probability = model_data['model'].predict_proba(X_scaled)
    
    # Decode prediction
    prediction_label = model_data['encoders']['status_akademik'].inverse_transform(prediction)
    
    return prediction_label, probability

# Main App
def main():
    st.title("🎓 Aplikasi Prediksi Kelulusan Mahasiswa")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Menu Navigasi")
    menu = st.sidebar.radio("Pilih Menu:", 
                            ["🏠 Dashboard", 
                             "🤖 Training Model", 
                             "👤 Prediksi Individual", 
                             "📁 Prediksi Batch",
                             "📊 Visualisasi Data"])
    
    # Load model jika ada
    model_data = load_model()
    if model_data:
        st.sidebar.success(f"✅ Model aktif: {model_data['model_name']}")
        st.sidebar.info(f"📈 Akurasi: {model_data['accuracy']:.2%}")
    else:
        st.sidebar.warning("⚠️ Belum ada model terlatih")
    
    # Dashboard
    if menu == "🏠 Dashboard":
        st.header("Dashboard Prediksi Kelulusan")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Dataset", "500 baris")
        with col2:
            st.metric("Data Training", "400 (80%)")
        with col3:
            st.metric("Data Testing", "100 (20%)")
        
        st.markdown("### 📋 Fitur Aplikasi")
        st.write("""
        - **Training Model**: Latih model prediksi dengan dataset Anda
        - **Prediksi Individual**: Input data mahasiswa secara manual untuk prediksi
        - **Prediksi Batch**: Upload file CSV untuk prediksi massal
        - **Visualisasi Data**: Lihat analisis dan performa model
        """)
        
        if model_data:
            st.markdown("### 🎯 Model Performance")
            st.write(f"**Model:** {model_data['model_name']}")
            st.write(f"**Akurasi:** {model_data['accuracy']:.2%}")

# Training Model
    elif menu == "🤖 Training Model":
        st.header("Training Model Prediksi")
        
        uploaded_file = st.file_uploader("Upload Dataset CSV", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.write("### Preview Dataset")
            st.dataframe(df.head())
            
            st.write(f"**Jumlah data:** {len(df)} baris, {len(df.columns)} kolom")
            
            if st.button("🚀 Mulai Training"):
                with st.spinner("Training model sedang berjalan..."):
                    model_data, results = train_model(df)
                    
                    st.markdown("### 📊 Hasil Training")
                    for name, result in results.items():
                        st.write(f"**{name}:** {result['accuracy']:.2%}")
                    
                    # Confusion Matrix
                    best_result = results[model_data['model_name']]
                    cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Prediksi')
                    ax.set_ylabel('Aktual')
                    ax.set_title(f'Confusion Matrix - {model_data["model_name"]}')
                    st.pyplot(fig)
                    
                    st.success("✅ Model berhasil dilatih dan disimpan!")

# Prediksi Individual
    elif menu == "👤 Prediksi Individual":
        st.header("Prediksi Kelulusan Individual")
        
        if model_data is None:
            st.warning("⚠️ Silakan latih model terlebih dahulu di menu Training Model")
            return
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                nama = st.text_input("Nama Mahasiswa")
                jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
                umur = st.number_input("Umur", min_value=17, max_value=50, value=20)
                status_menikah = st.selectbox("Status Menikah", ["Belum Menikah", "Menikah"])
            
            with col2:
                kehadiran = st.slider("Kehadiran (%)", 0, 100, 80)
                partisipasi_diskusi = st.slider("Partisipasi Diskusi (%)", 0, 100, 70)
                nilai_tugas = st.slider("Nilai Tugas", 0, 100, 75)
                aktivitas_elearning = st.slider("Aktivitas E-Learning (%)", 0, 100, 65)
            
            ipk = st.number_input("IPK", min_value=1.5, max_value=4.0, value=3.0, step=0.1)
            
            submit = st.form_submit_button("🔮 Prediksi")
            
            if submit:
                input_data = pd.DataFrame({
                    'nama': [nama],
                    'jenis_kelamin': [jenis_kelamin],
                    'umur': [umur],
                    'status_menikah': [status_menikah],
                    'kehadiran': [kehadiran],
                    'partisipasi_diskusi': [partisipasi_diskusi],
                    'nilai_tugas': [nilai_tugas],
                    'aktivitas_elearning': [aktivitas_elearning],
                    'ipk': [ipk]
                })
                
                prediction, probability = predict(model_data, input_data)
                
                st.markdown("---")
                st.markdown("### 🎯 Hasil Prediksi")
                
                if prediction[0] == 'lulus':
                    st.success(f"✅ **{nama}** diprediksi: **LULUS**")
                    st.info(f"Probabilitas Kelulusan: **{probability[0][1]:.2%}**")
                else:
                    st.error(f"❌ **{nama}** diprediksi: **TIDAK LULUS**")
                    st.info(f"Probabilitas Tidak Lulus: **{probability[0][0]:.2%}**")
                
                # Progress bar
                st.write("**Confidence Score:**")
                st.progress(float(max(probability[0])))

# Prediksi Batch
    elif menu == "📁 Prediksi Batch":
        st.header("Prediksi Batch (Upload CSV)")
        
        if model_data is None:
            st.warning("⚠️ Silakan latih model terlebih dahulu di menu Training Model")
            return
        
        st.write("Upload file CSV dengan kolom yang sama seperti dataset training")
        
        uploaded_file = st.file_uploader("Upload CSV untuk Prediksi", type=['csv'])
        
        if uploaded_file is not None:
            df_predict = pd.read_csv(uploaded_file)
            st.write("### Preview Data")
            st.dataframe(df_predict.head())
            
            if st.button("🔮 Prediksi Semua"):
                predictions, probabilities = predict(model_data, df_predict)
                
                df_result = df_predict.copy()
                df_result['Prediksi'] = predictions
                df_result['Probabilitas'] = [f"{max(prob):.2%}" for prob in probabilities]
                
                st.write("### 📊 Hasil Prediksi")
                st.dataframe(df_result)
                
                # Download hasil
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Prediksi",
                    data=csv,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )
                
                # Statistik
                lulus_count = (predictions == 'lulus').sum()
                tidak_lulus_count = (predictions == 'tidak lulus').sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediksi Lulus", lulus_count)
                with col2:
                    st.metric("Prediksi Tidak Lulus", tidak_lulus_count)

# Visualisasi
    elif menu == "📊 Visualisasi Data":
        st.header("Visualisasi & Analisis Data")
        
        uploaded_file = st.file_uploader("Upload Dataset untuk Visualisasi", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            tab1, tab2, tab3 = st.tabs(["📈 Distribusi", "🔗 Korelasi", "📊 Analisis"])
            
            with tab1:
                st.subheader("Distribusi Status Akademik")
                fig = px.pie(df, names='status_akademik', title='Distribusi Status Kelulusan')
                st.plotly_chart(fig)
                
                st.subheader("Distribusi berdasarkan Jenis Kelamin")
                fig2 = px.histogram(df, x='status_akademik', color='jenis_kelamin', 
                                   barmode='group', title='Status Kelulusan per Jenis Kelamin')
                st.plotly_chart(fig2)
            
            with tab2:
                st.subheader("Correlation Matrix")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                corr = df[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            
            with tab3:
                st.subheader("Analisis Fitur")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Rata-rata IPK berdasarkan Status**")
                    avg_ipk = df.groupby('status_akademik')['ipk'].mean()
                    st.bar_chart(avg_ipk)
                
                with col2:
                    st.write("**Rata-rata Kehadiran berdasarkan Status**")
                    avg_kehadiran = df.groupby('status_akademik')['kehadiran'].mean()
                    st.bar_chart(avg_kehadiran)

if __name__ == "__main__":
    main()

