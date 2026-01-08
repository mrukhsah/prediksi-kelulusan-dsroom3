import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Prediksi Kelulusan", layout="wide")

# Title
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.markdown("Menggunakan Random Forest, Gradient Boosting, dan KNN")

# Sidebar
with st.sidebar:
    st.header("⚙️ Parameter")
    split_ratio = st.slider("Train-Test Split", 70, 90, 80)
    ipk_threshold = st.slider("Threshold IPK Lulus", 2.0, 3.5, 2.75, 0.05)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data_mahasiswa.csv')
    except:
        # Create sample data
        np.random.seed(42)
        data = {
            'nama': [f'Mahasiswa_{i}' for i in range(500)],
            'jenis_kelamin': np.random.choice(['Laki-laki', 'Perempuan'], 500),
            'umur': np.random.randint(18, 25, 500),
            'status_menikah': np.random.choice(['Menikah', 'Belum Menikah'], 500),
            'kehadiran': np.random.randint(60, 101, 500),
            'partisipasi_diskusi': np.random.randint(50, 101, 500),
            'nilai_tugas': np.random.randint(60, 101, 500),
            'aktivitas_elearning': np.random.randint(50, 101, 500),
            'ipk': np.round(np.random.uniform(1.5, 4.0, 500), 2),
        }
        df = pd.DataFrame(data)
        df['status_akademik'] = np.where(
            (df['ipk'] >= 2.75) & (df['kehadiran'] >= 70),
            'Lulus', 'Tidak'
        )
    return df

# Main function
def main():
    df = load_data()
    
    tab1, tab2, tab3 = st.tabs(["📊 Data", "🤖 Model", "🔮 Prediksi"])
    
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        st.metric("Total Data", len(df))
        
        # Encode data
        le = LabelEncoder()
        df_encoded = df.copy()
        df_encoded['jenis_kelamin'] = le.fit_transform(df_encoded['jenis_kelamin'])
        df_encoded['status_menikah'] = le.fit_transform(df_encoded['status_menikah'])
        df_encoded['status_akademik'] = le.fit_transform(df_encoded['status_akademik'])
        
        # Prepare features
        X = df_encoded[['jenis_kelamin', 'umur', 'status_menikah', 'kehadiran', 
                       'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning']]
        y_ipk = df_encoded['ipk']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_ipk, test_size=(100-split_ratio)/100, random_state=42
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store in session state
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['scaler'] = scaler
        st.session_state['df'] = df
    
    with tab2:
        if 'X_train' in st.session_state:
            # Train models
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'KNN': KNeighborsRegressor(n_neighbors=5)
            }
            
            results = {}
            for name, model in models.items():
                model.fit(st.session_state['X_train'], st.session_state['y_train'])
                y_pred = model.predict(st.session_state['X_test'])
                
                mse = mean_squared_error(st.session_state['y_test'], y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(st.session_state['y_test'], y_pred)
                r2 = r2_score(st.session_state['y_test'], y_pred)
                
                results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
                
                # Store best model
                if name == 'Random Forest':
                    st.session_state['best_model'] = model
            
            # Display results
            col1, col2, col3 = st.columns(3)
            for idx, (name, metrics) in enumerate(results.items()):
                with [col1, col2, col3][idx]:
                    st.metric(f"Model: {name}", f"R²: {metrics['R2']:.4f}")
                    st.write(f"MSE: {metrics['MSE']:.4f}")
                    st.write(f"RMSE: {metrics['RMSE']:.4f}")
                    st.write(f"MAE: {metrics['MAE']:.4f}")
    
    with tab3:
        st.subheader("Prediksi Manual")
        
        col1, col2 = st.columns(2)
        with col1:
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            umur = st.number_input("Umur", 18, 25, 20)
            status_menikah = st.selectbox("Status Menikah", ["Belum Menikah", "Menikah"])
            kehadiran = st.slider("Kehadiran", 0, 100, 75)
        
        with col2:
            partisipasi = st.slider("Partisipasi Diskusi", 0, 100, 75)
            nilai_tugas = st.slider("Nilai Tugas", 0, 100, 75)
            aktivitas_elearning = st.slider("Aktivitas E-Learning", 0, 100, 75)
        
        if st.button("Prediksi", type="primary"):
            if 'best_model' in st.session_state:
                # Prepare input
                input_data = [[
                    1 if jenis_kelamin == "Laki-laki" else 0,
                    umur,
                    1 if status_menikah == "Menikah" else 0,
                    kehadiran,
                    partisipasi,
                    nilai_tugas,
                    aktivitas_elearning
                ]]
                
                # Scale and predict
                input_scaled = st.session_state['scaler'].transform(input_data)
                predicted_ipk = st.session_state['best_model'].predict(input_scaled)[0]
                
                # Display results
                st.success(f"**IPK Prediksi: {predicted_ipk:.2f}**")
                status = "LULUS" if predicted_ipk >= ipk_threshold else "TIDAK LULUS"
                st.info(f"**Status: {status}** (Threshold: {ipk_threshold})")

if __name__ == "__main__":
    main()
