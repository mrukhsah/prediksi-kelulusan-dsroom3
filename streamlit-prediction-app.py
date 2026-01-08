import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Fungsi untuk menentukan status akademik berdasarkan IPK
def tentukan_status(ipk):
    if ipk >= 3.5:
        return "Cumlaude"
    elif ipk >= 3.0:
        return "Sangat Memuaskan"
    elif ipk >= 2.75:
        return "Memuaskan"
    else:
        return "Perlu Peningkatan"

# Fungsi untuk menentukan kelulusan berdasarkan IPK
def tentukan_kelulusan(ipk):
    return "Lulus" if ipk >= 2.0 else "Tidak Lulus"

# Fungsi untuk menghitung semua metrik evaluasi
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return mae, mape, rmse, r2

# Fungsi untuk membuat fitur baru (Feature Engineering)
def create_new_features(df):
    df_new = df.copy()
    
    # 1. Rata-rata performa akademik
    df_new['rata_performa'] = (df_new['kehadiran'] + df_new['partisipasi_diskusi'] + 
                              df_new['nilai_tugas'] + df_new['aktivitas_elearning']) / 4
    
    # 2. Total aktivitas (skala 0-400)
    df_new['total_aktivitas'] = df_new['kehadiran'] + df_new['partisipasi_diskusi'] + \
                               df_new['nilai_tugas'] + df_new['aktivitas_elearning']
    
    # 3. Interaksi antara kehadiran dan nilai tugas
    df_new['kehadiran_nilai_interaksi'] = (df_new['kehadiran'] * df_new['nilai_tugas']) / 100
    
    # 4. Performa konsistensi (std dev dari 4 komponen)
    df_new['konsistensi'] = df_new[['kehadiran', 'partisipasi_diskusi', 
                                    'nilai_tugas', 'aktivitas_elearning']].std(axis=1)
    
    # 5. Kategori umur
    df_new['kategori_umur'] = pd.cut(df_new['umur'], 
                                     bins=[17, 20, 22, 25, 30], 
                                     labels=[0, 1, 2, 3])
    
    # 6. Performa tertimbang (berat lebih pada nilai tugas)
    df_new['performa_tertimbang'] = (df_new['kehadiran']*0.2 + 
                                    df_new['partisipasi_diskusi']*0.2 + 
                                    df_new['nilai_tugas']*0.4 + 
                                    df_new['aktivitas_elearning']*0.2)
    
    return df_new

# Fungsi untuk training model yang DIPERBAIKI
def train_model_improved(df, model_type='random_forest'):
    # Buat fitur baru
    df_enhanced = create_new_features(df)
    
    # Definisikan features dan target
    numerical_features = ['umur', 'kehadiran', 'partisipasi_diskusi', 
                         'nilai_tugas', 'aktivitas_elearning',
                         'rata_performa', 'total_aktivitas', 
                         'kehadiran_nilai_interaksi', 'konsistensi',
                         'performa_tertimbang']
    
    categorical_features = ['jenis_kelamin', 'status_menikah']
    
    features = numerical_features + categorical_features
    
    X = df_enhanced[features]
    y = df_enhanced['ipk']
    
    # Split data 70% training, 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Buat preprocessing pipeline
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Pilih model dengan hyperparameter yang dioptimalkan
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            subsample=0.8
        )
    elif model_type == 'knn':
        model = KNeighborsRegressor(
            n_neighbors=7,
            weights='distance',
            metric='euclidean'
        )
    
    # Buat pipeline lengkap
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Training model
    pipeline.fit(X_train, y_train)
    
    # Prediksi
    y_pred = pipeline.predict(X_test)
    
    # Hitung metrik
    mae, mape, rmse, r2 = calculate_metrics(y_test, y_pred)
    
    # Hitung baseline (mean prediction)
    baseline_pred = np.full_like(y_test, y_train.mean())
    mae_baseline, mape_baseline, rmse_baseline, r2_baseline = calculate_metrics(y_test, baseline_pred)
    
    return pipeline, X_test, y_test, y_pred, mae, mape, rmse, r2, features, mae_baseline, r2_baseline

# Header
st.title("🎓 Aplikasi Prediksi Kelulusan Mahasiswa - IMPROVED")
st.markdown("""
Aplikasi ini memprediksi IPK mahasiswa menggunakan algoritma Machine Learning dengan **feature engineering** dan **preprocessing yang ditingkatkan**.
""")

# Sidebar
st.sidebar.header("📋 Menu")
menu = st.sidebar.radio("Pilih Menu:", ["Upload & Training", "Prediksi Individual", "Visualisasi", "Perbandingan Model", "Analisis Data"])

# Session state untuk menyimpan model
if 'models' not in st.session_state:
    st.session_state.models = {}
    st.session_state.results = {}
    st.session_state.trained = False
    st.session_state.feature_importance = {}

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
        st.subheader("📈 Informasi Dataset")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Data", len(df))
        with col2:
            st.metric("Data Training (70%)", f"{int(len(df) * 0.7)}")
        with col3:
            st.metric("Data Testing (30%)", f"{int(len(df) * 0.3)}")
        with col4:
            st.metric("Rata-rata IPK", f"{df['ipk'].mean():.2f}")
        
        # Analisis korelasi sederhana - PERBAIKAN DI SINI
        st.subheader("🔍 Analisis Korelasi dengan IPK")
        numeric_cols = ['umur', 'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning', 'ipk']
        
        if all(col in df.columns for col in numeric_cols):
            corr_with_ipk = df[numeric_cols].corr()['ipk'].sort_values(ascending=False)
            corr_df = pd.DataFrame({'Fitur': corr_with_ipk.index, 'Korelasi dengan IPK': corr_with_ipk.values})
            
            fig = px.bar(corr_df, x='Fitur', y='Korelasi dengan IPK',
                        title='Korelasi Fitur dengan IPK',
                        color='Korelasi dengan IPK',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
            
            # PERBAIKAN: Ganti background_gradient dengan style sederhana
            st.dataframe(corr_df)
            
            # Atau gunakan plotly table sebagai alternatif
            fig_table = go.Figure(data=[go.Table(
                header=dict(values=list(corr_df.columns),
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[corr_df['Fitur'], corr_df['Korelasi dengan IPK']],
                          fill_color='lavender',
                          align='left'))
            ])
            fig_table.update_layout(title='Tabel Korelasi')
            st.plotly_chart(fig_table, use_container_width=True)
        
        # Pilih model
        st.subheader("🤖 Pilih Model untuk Training")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_rf = st.checkbox("Random Forest Regressor", value=True)
        with col2:
            train_gb = st.checkbox("Gradient Boosting Regressor", value=True)
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
                        pipeline, X_test, y_test, y_pred, mae, mape, rmse, r2, features, mae_baseline, r2_baseline = train_model_improved(df, model_type)
                        
                        # Simpan ke session state
                        st.session_state.models[model_type] = {
                            'pipeline': pipeline,
                            'features': features
                        }
                        
                        st.session_state.results[model_type] = {
                            'name': model_name,
                            'X_test': X_test,
                            'y_test': y_test,
                            'y_pred': y_pred,
                            'mae': mae,
                            'mape': mape,
                            'rmse': rmse,
                            'r2': r2,
                            'mae_baseline': mae_baseline,
                            'r2_baseline': r2_baseline
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
                    improvement_mae = ((result['mae_baseline'] - result['mae']) / result['mae_baseline']) * 100
                    improvement_r2 = (result['r2'] - result['r2_baseline']) * 100
                    
                    results_data.append({
                        'Model': result['name'],
                        'MAE': f"{result['mae']:.4f}",
                        'MAE Baseline': f"{result['mae_baseline']:.4f}",
                        'Improvement MAE': f"{improvement_mae:.1f}%",
                        'MAPE': f"{result['mape']:.2f}%",
                        'RMSE': f"{result['rmse']:.4f}",
                        'R²': f"{result['r2']:.4f}",
                        'R² Baseline': f"{result['r2_baseline']:.4f}",
                        'Improvement R²': f"{improvement_r2:.1f}%"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Visualisasi perbandingan
                st.subheader("📊 Visualisasi Perbandingan Model")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    r2_data = [result['r2'] for result in st.session_state.results.values()]
                    r2_baseline_data = [result['r2_baseline'] for result in st.session_state.results.values()]
                    model_names = [result['name'] for result in st.session_state.results.values()]
                    
                    fig = go.Figure(data=[
                        go.Bar(name='R² Model', x=model_names, y=r2_data),
                        go.Bar(name='R² Baseline', x=model_names, y=r2_baseline_data)
                    ])
                    fig.update_layout(
                        title='Perbandingan R² Score vs Baseline',
                        xaxis_title='Model',
                        yaxis_title='R² Score',
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    mae_data = [result['mae'] for result in st.session_state.results.values()]
                    mae_baseline_data = [result['mae_baseline'] for result in st.session_state.results.values()]
                    
                    fig = go.Figure(data=[
                        go.Bar(name='MAE Model', x=model_names, y=mae_data),
                        go.Bar(name='MAE Baseline', x=model_names, y=mae_baseline_data)
                    ])
                    fig.update_layout(
                        title='Perbandingan MAE vs Baseline',
                        xaxis_title='Model',
                        yaxis_title='MAE',
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detail untuk setiap model
                for model_type, result in st.session_state.results.items():
                    with st.expander(f"📊 Detail {result['name']}"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MAE", f"{result['mae']:.4f}", 
                                     delta=f"{(result['mae_baseline'] - result['mae']):.4f} vs baseline")
                        with col2:
                            st.metric("MAPE", f"{result['mape']:.2f}%")
                        with col3:
                            st.metric("RMSE", f"{result['rmse']:.4f}")
                        with col4:
                            improvement = (result['r2'] - result['r2_baseline']) * 100
                            st.metric("R² Score", f"{result['r2']:.4f}", 
                                     delta=f"{improvement:.1f}% vs baseline")
                        
                        # Scatter plot dengan line of best fit
                        fig = px.scatter(
                            x=result['y_test'], y=result['y_pred'],
                            labels={'x': 'IPK Aktual', 'y': 'IPK Prediksi'},
                            title=f"Prediksi vs Aktual - {result['name']}",
                            trendline="ols",
                            trendline_color_override="red"
                        )
                        fig.add_trace(go.Scatter(
                            x=[result['y_test'].min(), result['y_test'].max()], 
                            y=[result['y_test'].min(), result['y_test'].max()],
                            mode='lines', name='Perfect Prediction',
                            line=dict(color='green', dash='dash')
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Residual plot
                        residuals = result['y_test'] - result['y_pred']
                        fig = px.scatter(
                            x=result['y_pred'], y=residuals,
                            labels={'x': 'Prediksi', 'y': 'Residual'},
                            title=f'Residual Plot - {result["name"]}'
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Silakan upload dataset terlebih dahulu")
        st.markdown("""
        **Format dataset yang dibutuhkan:**
        - nama
        - jenis_kelamin (Laki-laki/Perempuan)
        - umur
        - status_menikah (Menikah/Belum Menikah)
        - kehadiran (1-100)
        - partisipasi_diskusi (1-100)
        - nilai_tugas (1-100)
        - aktivitas_elearning (1-100)
        - ipk (1.5-4.0)
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
            st.subheader("Informasi Pribadi")
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            umur = st.number_input("Umur", min_value=17, max_value=50, value=20)
            status_menikah = st.selectbox("Status Menikah", ["Belum Menikah", "Menikah"])
        
        with col2:
            st.subheader("Perform Akademik")
            kehadiran = st.slider("Kehadiran (%)", 0, 100, 80, help="Persentase kehadiran di kelas")
            partisipasi = st.slider("Partisipasi Diskusi (1-100)", 0, 100, 75, help="Tingkat partisipasi dalam diskusi")
            nilai_tugas = st.slider("Nilai Tugas (1-100)", 0, 100, 80, help="Rata-rata nilai tugas")
            aktivitas = st.slider("Aktivitas E-Learning (1-100)", 0, 100, 70, help="Aktivitas dalam e-learning")
        
        if st.button("🎯 Prediksi IPK", type="primary"):
            model_data = st.session_state.models[selected_model_type]
            
            # Hitung fitur-fitur baru
            rata_performa = (kehadiran + partisipasi + nilai_tugas + aktivitas) / 4
            total_aktivitas = kehadiran + partisipasi + nilai_tugas + aktivitas
            kehadiran_nilai_interaksi = (kehadiran * nilai_tugas) / 100
            performa_tertimbang = (kehadiran*0.2 + partisipasi*0.2 + nilai_tugas*0.4 + aktivitas*0.2)
            
            # Kategori umur
            if umur <= 20:
                kategori_umur = 0
            elif umur <= 22:
                kategori_umur = 1
            elif umur <= 25:
                kategori_umur = 2
            else:
                kategori_umur = 3
            
            # Konsistensi (simulasi sederhana)
            nilai_array = np.array([kehadiran, partisipasi, nilai_tugas, aktivitas])
            konsistensi = np.std(nilai_array)
            
            # Buat dataframe input sesuai urutan features
            input_data = pd.DataFrame({
                'umur': [umur],
                'kehadiran': [kehadiran],
                'partisipasi_diskusi': [partisipasi],
                'nilai_tugas': [nilai_tugas],
                'aktivitas_elearning': [aktivitas],
                'rata_performa': [rata_performa],
                'total_aktivitas': [total_aktivitas],
                'kehadiran_nilai_interaksi': [kehadiran_nilai_interaksi],
                'konsistensi': [konsistensi],
                'performa_tertimbang': [performa_tertimbang],
                'jenis_kelamin': [jenis_kelamin],
                'status_menikah': [status_menikah]
            })
            
            # Reorder columns sesuai urutan training
            input_data = input_data[model_data['features']]
            
            # Prediksi
            prediksi_ipk = model_data['pipeline'].predict(input_data)[0]
            
            # Pastikan prediksi dalam rentang wajar
            prediksi_ipk = max(1.5, min(4.0, prediksi_ipk))
            
            # Tentukan status dan kelulusan
            status_akademik = tentukan_status(prediksi_ipk)
            kelulusan = tentukan_kelulusan(prediksi_ipk)
            
            # Tampilkan hasil
            st.subheader("📊 Hasil Prediksi")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model Digunakan", selected_model_name)
            with col2:
                st.metric("IPK Diprediksi", f"{prediksi_ipk:.2f}")
            with col3:
                color = "green" if status_akademik == "Cumlaude" else "blue" if status_akademik == "Sangat Memuaskan" else "orange" if status_akademik == "Memuaskan" else "red"
                st.markdown(f"**Status:** :{color}[{status_akademik}]")
            with col4:
                status_color = "green" if kelulusan == "Lulus" else "red"
                st.markdown(f"**Prediksi Kelulusan:** :{status_color}[{kelulusan}]")
            
            # Progress bar IPK
            st.progress(min(prediksi_ipk / 4.0, 1.0))
            st.caption(f"IPK: {prediksi_ipk:.2f}/4.0")
            
            # Prediksi dari semua model
            st.subheader("📈 Perbandingan Prediksi Semua Model")
            all_predictions = []
            
            for model_type, model_data in st.session_state.models.items():
                pred = model_data['pipeline'].predict(input_data)[0]
                pred = max(1.5, min(4.0, pred))
                all_predictions.append({
                    'Model': st.session_state.results[model_type]['name'],
                    'Prediksi IPK': f"{pred:.2f}",
                    'Status': tentukan_status(pred),
                    'Kelulusan': tentukan_kelulusan(pred)
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
            st.subheader("Distribusi Data")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x='ipk', nbins=20, 
                                 title='Distribusi IPK Mahasiswa',
                                 color_discrete_sequence=['#2E86AB'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y='ipk', 
                           title='Box Plot IPK',
                           color_discrete_sequence=['#2E86AB'])
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Distribusi Fitur")
            feature_to_plot = st.selectbox("Pilih fitur untuk divisualisasi:", 
                                          ['kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning', 'umur'])
            
            fig = px.histogram(df, x=feature_to_plot, nbins=20,
                             title=f'Distribusi {feature_to_plot}',
                             color_discrete_sequence=['#A23B72'])
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Korelasi Antar Variabel")
            
            # Pilih fitur untuk heatmap
            all_numeric_features = ['umur', 'kehadiran', 'partisipasi_diskusi', 
                                   'nilai_tugas', 'aktivitas_elearning', 'ipk']
            
            selected_features = st.multiselect("Pilih fitur untuk heatmap:", 
                                             all_numeric_features,
                                             default=all_numeric_features)
            
            if len(selected_features) >= 2:
                corr_matrix = df[selected_features].corr()
                
                fig = px.imshow(corr_matrix, 
                              text_auto=True, 
                              aspect="auto",
                              title='Heatmap Korelasi',
                              color_continuous_scale='RdBu_r',
                              zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot interaktif
            st.subheader("Scatter Plot Interaktif")
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Sumbu X:", all_numeric_features, index=0)
            with col2:
                y_axis = st.selectbox("Sumbu Y:", all_numeric_features, index=len(all_numeric_features)-1)
            
            if x_axis != y_axis:
                fig = px.scatter(df, x=x_axis, y=y_axis,
                               trendline="ols",
                               title=f'{y_axis} vs {x_axis}',
                               color='ipk',
                               color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Model Performance")
            
            if st.session_state.results:
                # Pilih model untuk visualisasi
                model_options = {v['name']: k for k, v in st.session_state.results.items()}
                selected_model_name = st.selectbox("Pilih Model:", list(model_options.keys()))
                selected_model_type = model_options[selected_model_name]
                
                result = st.session_state.results[selected_model_type]
                
                # Actual vs Predicted
                fig = px.scatter(x=result['y_test'], y=result['y_pred'],
                               labels={'x': 'IPK Aktual', 'y': 'IPK Prediksi'},
                               title=f'Actual vs Predicted - {selected_model_name}',
                               trendline="ols")
                fig.add_trace(go.Scatter(
                    x=[result['y_test'].min(), result['y_test'].max()], 
                    y=[result['y_test'].min(), result['y_test'].max()],
                    mode='lines', name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Error distribution
                errors = result['y_test'] - result['y_pred']
                fig = px.histogram(x=errors, nbins=30,
                                 title=f'Distribusi Error - {selected_model_name}',
                                 labels={'x': 'Error (Aktual - Prediksi)'})
                fig.add_vline(x=0, line_dash="dash", line_color="red")
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
                'MAPE': result['mape'],
                'RMSE': result['rmse'],
                'R²': result['r2']
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Highlight dengan cara sederhana
        st.dataframe(results_df, use_container_width=True)
        
        # Visualisasi bar chart
        st.subheader("📈 Visualisasi Perbandingan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(results_df, x='Model', y='R²',
                        title='Perbandingan R² Score (Higher is Better)',
                        color='R²',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(results_df, x='Model', y='MAE',
                        title='Perbandingan MAE (Lower is Better)',
                        color='MAE',
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        
        # Rekomendasi model terbaik
        if not results_df.empty:
            best_model_r2 = results_df.loc[results_df['R²'].idxmax()]
            best_model_mae = results_df.loc[results_df['MAE'].idxmin()]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏆 **Model dengan R² Terbaik:** {best_model_r2['Model']}")
                st.caption(f"R² = {best_model_r2['R²']:.4f}, MAE = {best_model_r2['MAE']:.4f}")
            with col2:
                st.success(f"🏆 **Model dengan MAE Terbaik:** {best_model_mae['Model']}")
                st.caption(f"MAE = {best_model_mae['MAE']:.4f}, R² = {best_model_mae['R²']:.4f}")

# MENU 5: Analisis Data
elif menu == "Analisis Data":
    st.header("🔍 Analisis Data Mendalam")
    
    if not st.session_state.trained:
        st.warning("⚠️ Silakan upload data dan training model terlebih dahulu.")
    else:
        df = st.session_state.df
        
        st.subheader("📊 Statistik Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("🔗 Analisis Hubungan")
        
        # Pilih dua variabel untuk analisis
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Variabel 1:", 
                               ['ipk', 'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning'],
                               index=0)
        with col2:
            var2 = st.selectbox("Variabel 2:", 
                               ['ipk', 'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 'aktivitas_elearning'],
                               index=1)
        
        if var1 != var2:
            # Scatter plot dengan regresi
            fig = px.scatter(df, x=var1, y=var2,
                           trendline="ols",
                           title=f'Hubungan antara {var1} dan {var2}',
                           color='ipk' if var1 != 'ipk' and var2 != 'ipk' else None,
                           color_continuous_scale='viridis')
            
            # Hitung korelasi
            correlation = df[var1].corr(df[var2])
            fig.add_annotation(
                x=0.05, y=0.95,
                xref="paper", yref="paper",
                text=f"Korelasi: {correlation:.3f}",
                showarrow=False,
                font=dict(size=14, color="red"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📈 Distribusi berdasarkan Status")
        if 'status_akademik' in df.columns:
            # Box plot IPK berdasarkan status akademik
            fig = px.box(df, x='status_akademik', y='ipk',
                       title='Distribusi IPK berdasarkan Status Akademik',
                       color='status_akademik')
            st.plotly_chart(fig, use_container_width=True)
            
            # Violin plot
            fig = px.violin(df, x='status_akademik', y='ipk',
                          box=True, points="all",
                          title='Violin Plot IPK berdasarkan Status Akademik')
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
🎓 **Aplikasi Prediksi Kelulusan v3.0**  
Dengan **Feature Engineering** dan **Preprocessing Enhanced**  
Split Data: **70% Training - 30% Testing**
""")
st.sidebar.caption("Dibuat dengan Streamlit & Scikit-learn")
