import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide"
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
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin-bottom: 1rem;
    }
    .model-card {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #bae6fd;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎓 Prediksi Status Kelulusan Mahasiswa</h1>', unsafe_allow_html=True)
st.markdown("**Analisis menggunakan Random Forest, Gradient Boosting, dan KNN Regressor**")

# Sidebar
st.sidebar.header("⚙️ Konfigurasi Model")
st.sidebar.markdown("### Parameter Model")

# Parameter untuk Random Forest
rf_n_estimators = st.sidebar.slider("Random Forest: n_estimators", 50, 200, 100, 10)
rf_max_depth = st.sidebar.slider("Random Forest: max_depth", 5, 30, 10, 1)

# Parameter untuk Gradient Boosting
gb_n_estimators = st.sidebar.slider("Gradient Boosting: n_estimators", 50, 200, 100, 10)
gb_learning_rate = st.sidebar.slider("Gradient Boosting: learning_rate", 0.01, 0.3, 0.1, 0.01)

# Parameter untuk KNN
knn_n_neighbors = st.sidebar.slider("KNN: n_neighbors", 3, 15, 5, 1)

split_ratio = st.sidebar.slider("Train-Test Split Ratio", 70, 90, 80, 5)
ipk_threshold = st.sidebar.slider("Threshold IPK untuk Lulus", 2.0, 3.5, 2.75, 0.05)

# Fungsi untuk menghitung MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Menghindari division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_and_process_data():
    # Membuat dataset dari data yang diberikan
    data = pd.read_csv('csv_Dataset_Mahasiswa_Kehadiran_Aktivitas_IPK.csv')  # Ganti dengan path file Anda
    
    # Data preprocessing
    le = LabelEncoder()
    data['jenis_kelamin'] = le.fit_transform(data['jenis_kelamin'])  # Laki-laki: 1, Perempuan: 0
    data['status_menikah'] = le.fit_transform(data['status_menikah'])  # Menikah: 1, Belum: 0
    data['status_akademik'] = le.fit_transform(data['status_akademik'])  # Lulus: 1, Tidak: 0
    
    return data

# Main function
def main():
    # Tab untuk navigasi
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview Data", 
        "🤖 Training Model", 
        "🔮 Prediksi Manual",
        "📁 Prediksi Batch"
    ])
    
    # Load data
    try:
        df = load_and_process_data()
    except:
        # Jika file tidak ditemukan, buat dataset dummy berdasarkan pola data Anda
        st.warning("File tidak ditemukan. Menggunakan dataset contoh...")
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'nama': [f'Mahasiswa_{i}' for i in range(n_samples)],
            'jenis_kelamin': np.random.choice(['Laki-laki', 'Perempuan'], n_samples),
            'umur': np.random.randint(18, 25, n_samples),
            'status_menikah': np.random.choice(['Menikah', 'Belum Menikah'], n_samples),
            'kehadiran': np.random.randint(60, 101, n_samples),
            'partisipasi_diskusi': np.random.randint(50, 101, n_samples),
            'nilai_tugas': np.random.randint(60, 101, n_samples),
            'aktivitas_elearning': np.random.randint(50, 101, n_samples),
            'ipk': np.round(np.random.uniform(1.5, 4.0, n_samples), 2),
        }
        
        # Membuat status akademik berdasarkan kondisi
        df = pd.DataFrame(data)
        df['status_akademik'] = np.where(
            (df['ipk'] >= 2.75) & 
            (df['kehadiran'] >= 70) & 
            (df['nilai_tugas'] >= 65),
            'Lulus', 'Tidak'
        )
        
        # Save untuk contoh
        df.to_csv('dataset_mahasiswa_contoh.csv', index=False)
    
    # Tab 1: Overview Data
    with tab1:
        st.markdown('<h2 class="sub-header">📊 Overview Dataset</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Data", f"{len(df)} baris")
            st.metric("Jumlah Fitur", f"{df.shape[1]} kolom")
            
        with col2:
            lulus_count = df[df['status_akademik'] == 'Lulus'].shape[0] if 'status_akademik' in df.columns else df[df['status_akademik'] == 1].shape[0]
            tidak_count = len(df) - lulus_count
            st.metric("Lulus", f"{lulus_count} ({lulus_count/len(df)*100:.1f}%)")
            st.metric("Tidak Lulus", f"{tidak_count} ({tidak_count/len(df)*100:.1f}%)")
        
        # Tampilkan data
        st.subheader("Preview Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Statistik deskriptif
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Visualisasi distribusi
        st.subheader("Distribusi IPK")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['ipk'], bins=20, kde=True, ax=ax)
        ax.axvline(x=ipk_threshold, color='red', linestyle='--', label=f'Threshold: {ipk_threshold}')
        ax.set_xlabel('IPK')
        ax.set_ylabel('Frekuensi')
        ax.set_title('Distribusi IPK Mahasiswa')
        ax.legend()
        st.pyplot(fig)
    
    # Preprocessing data untuk model
    with tab2:
        st.markdown('<h2 class="sub-header">🤖 Training Model Regressor</h2>', unsafe_allow_html=True)
        
        # Encode data jika belum
        if 'status_akademik' in df.columns and df['status_akademik'].dtype == 'object':
            le = LabelEncoder()
            df['jenis_kelamin'] = le.fit_transform(df['jenis_kelamin'])
            df['status_menikah'] = le.fit_transform(df['status_menikah'])
            df['status_akademik'] = le.fit_transform(df['status_akademik'])
        
        # Pisahkan fitur dan target
        X = df[['jenis_kelamin', 'umur', 'status_menikah', 'kehadiran', 
                'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning']]
        y = df['ipk']  # Target regresi: IPK
        y_class = df['status_akademik']  # Target klasifikasi
        
        # Split data
        X_train, X_test, y_train, y_test, y_class_train, y_class_test = train_test_split(
            X, y, y_class, test_size=(100-split_ratio)/100, random_state=42
        )
        
        # Normalisasi data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Training model
        st.subheader("Training Model...")
        
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=gb_n_estimators,
                learning_rate=gb_learning_rate,
                random_state=42
            ),
            'KNN': KNeighborsRegressor(
                n_neighbors=knn_n_neighbors
            )
        }
        
        results = {}
        predictions = {}
        
        progress_bar = st.progress(0)
        for idx, (name, model) in enumerate(models.items()):
            # Training
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            predictions[name] = y_pred
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            }
            
            progress_bar.progress((idx + 1) / len(models))
        
        # Display results
        st.subheader("Hasil Evaluasi Model")
        
        # Tampilkan metrics dalam columns
        cols = st.columns(3)
        model_names = list(results.keys())
        
        for idx, col in enumerate(cols):
            if idx < len(model_names):
                model_name = model_names[idx]
                metrics = results[model_name]
                
                with col:
                    st.markdown(f'<div class="model-card">', unsafe_allow_html=True)
                    st.markdown(f"### {model_name}")
                    st.metric("MSE", f"{metrics['MSE']:.4f}")
                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                    st.metric("R² Score", f"{metrics['R2']:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualisasi perbandingan model
        st.subheader("Perbandingan Performa Model")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['MSE', 'RMSE', 'MAPE', 'R2']
        
        for idx, metric in enumerate(metrics_to_plot):
            if idx < len(axes):
                ax = axes[idx]
                metric_values = [results[model][metric] for model in model_names]
                
                bars = ax.bar(model_names, metric_values, color=['#1E3A8A', '#10B981', '#F59E0B'])
                ax.set_title(f'{metric} per Model')
                ax.set_ylabel(metric)
                
                # Tambahkan nilai di atas bar
                for bar in bars:
                    height = bar.get_height()
                    if metric == 'MAPE':
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.2f}%', ha='center', va='bottom')
                    else:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Visualisasi prediksi vs aktual
        st.subheader("Prediksi vs Aktual (Random Forest)")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, predictions['Random Forest'], alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('IPK Aktual')
        ax.set_ylabel('IPK Prediksi')
        ax.set_title('Prediksi vs Aktual IPK')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Konversi prediksi IPK ke status kelulusan
        st.subheader("Klasifikasi Status Kelulusan berdasarkan Prediksi IPK")
        
        # Threshold untuk klasifikasi
        y_pred_class = (predictions['Random Forest'] >= ipk_threshold).astype(int)
        y_test_class = (y_test >= ipk_threshold).astype(int)
        
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
        accuracy = accuracy_score(y_test_class, y_pred_class)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Threshold IPK", f"{ipk_threshold}")
        with col2:
            st.metric("Akurasi Klasifikasi", f"{accuracy:.2%}")
        with col3:
            st.metric("Split Data", f"{split_ratio}/{100-split_ratio}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test_class, y_pred_class)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    # Tab 3: Prediksi Manual
    with tab3:
        st.markdown('<h2 class="sub-header">🔮 Prediksi Manual</h2>', unsafe_allow_html=True)
        
        # Gunakan model terbaik
        best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
        best_model = models[best_model_name]
        
        st.info(f"**Model terbaik:** {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
                umur = st.slider("Umur", 18, 25, 20)
                status_menikah = st.selectbox("Status Menikah", ["Belum Menikah", "Menikah"])
                kehadiran = st.slider("Kehadiran (%)", 1, 100, 75)
            
            with col2:
                partisipasi_diskusi = st.slider("Partisipasi Diskusi (%)", 1, 100, 75)
                nilai_tugas = st.slider("Nilai Tugas (%)", 1, 100, 75)
                aktivitas_elearning = st.slider("Aktivitas E-Learning (%)", 1, 100, 75)
            
            submitted = st.form_submit_button("🔮 Prediksi IPK & Status")
        
        if submitted:
            # Encode input
            input_data = {
                'jenis_kelamin': 1 if jenis_kelamin == "Laki-laki" else 0,
                'umur': umur,
                'status_menikah': 1 if status_menikah == "Menikah" else 0,
                'kehadiran': kehadiran,
                'partisipasi_diskusi': partisipasi_diskusi,
                'nilai_tugas': nilai_tugas,
                'aktivitas_elearning': aktivitas_elearning
            }
            
            # Convert to DataFrame dan scale
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            
            # Predict
            predicted_ipk = best_model.predict(input_scaled)[0]
            predicted_status = "Lulus" if predicted_ipk >= ipk_threshold else "Tidak Lulus"
            
            # Display results
            st.success("### Hasil Prediksi")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("IPK Prediksi", f"{predicted_ipk:.2f}")
                st.metric("Threshold Lulus", f"{ipk_threshold}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                if predicted_status == "Lulus":
                    st.success(f"**Status: {predicted_status}** 🎉")
                else:
                    st.error(f"**Status: {predicted_status}** ⚠️")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature importance untuk Random Forest
            if best_model_name == 'Random Forest':
                st.subheader("Kontribusi Fitur (Feature Importance)")
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(feature_importance['feature'], feature_importance['importance'])
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance')
                plt.gca().invert_yaxis()
                st.pyplot(fig)
    
    # Tab 4: Prediksi Batch
    with tab4:
        st.markdown('<h2 class="sub-header">📁 Prediksi Batch</h2>', unsafe_allow_html=True)
        
        st.info("Upload file CSV dengan format yang sama dengan dataset training")
        
        uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Load uploaded data
                batch_df = pd.read_csv(uploaded_file)
                
                # Check required columns
                required_cols = ['jenis_kelamin', 'umur', 'status_menikah', 'kehadiran',
                               'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning']
                
                missing_cols = [col for col in required_cols if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Kolom berikut tidak ditemukan: {missing_cols}")
                else:
                    st.success(f"✅ File berhasil diupload! ({len(batch_df)} baris)")
                    
                    # Encode categorical columns
                    batch_df_encoded = batch_df.copy()
                    if batch_df['jenis_kelamin'].dtype == 'object':
                        batch_df_encoded['jenis_kelamin'] = batch_df_encoded['jenis_kelamin'].map(
                            {'Laki-laki': 1, 'Perempuan': 0}
                        )
                    if batch_df['status_menikah'].dtype == 'object':
                        batch_df_encoded['status_menikah'] = batch_df_encoded['status_menikah'].map(
                            {'Menikah': 1, 'Belum Menikah': 0}
                        )
                    
                    # Prepare features
                    X_batch = batch_df_encoded[required_cols]
                    X_batch_scaled = scaler.transform(X_batch)
                    
                    # Predict
                    predictions_batch = best_model.predict(X_batch_scaled)
                    status_batch = ['Lulus' if ipk >= ipk_threshold else 'Tidak' for ipk in predictions_batch]
                    
                    # Add predictions to dataframe
                    result_df = batch_df.copy()
                    result_df['IPK_Prediksi'] = np.round(predictions_batch, 2)
                    result_df['Status_Prediksi'] = status_batch
                    
                    # Display results
                    st.subheader("Hasil Prediksi")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("Statistik Hasil Prediksi")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_ipk = result_df['IPK_Prediksi'].mean()
                        st.metric("Rata-rata IPK Prediksi", f"{avg_ipk:.2f}")
                    with col2:
                        lulus_count = (result_df['Status_Prediksi'] == 'Lulus').sum()
                        st.metric("Diprediksi Lulus", f"{lulus_count} ({lulus_count/len(result_df)*100:.1f}%)")
                    with col3:
                        tidak_count = len(result_df) - lulus_count
                        st.metric("Diprediksi Tidak Lulus", f"{tidak_count} ({tidak_count/len(result_df)*100:.1f}%)")
                    
                    # Download button
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil Prediksi (CSV)",
                        data=csv,
                        file_name="hasil_prediksi_kelulusan.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
