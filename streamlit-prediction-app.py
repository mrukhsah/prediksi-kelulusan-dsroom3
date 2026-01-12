import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Fungsi untuk membuat label klasifikasi
def create_classification_labels(df, mode='binary'):
    """Buat label klasifikasi dari IPK"""
    df_copy = df.copy()
    
    if mode == 'binary':
        # Binary: Lulus (IPK >= 2.75) vs Tidak Lulus (IPK < 2.75)
        df_copy['target'] = df_copy['ipk'].apply(
            lambda x: 'Lulus' if x >= 2.75 else 'Tidak Lulus'
        )
        return df_copy, ['Tidak Lulus', 'Lulus']
    
    elif mode == 'multiclass':
        # Multi-class: 4 kategori predikat
        def get_predikat(ipk):
            if ipk >= 3.5:
                return 'Cumlaude'
            elif ipk >= 3.0:
                return 'Sangat Memuaskan'
            elif ipk >= 2.75:
                return 'Memuaskan'
            else:
                return 'Tidak Lulus'
        
        df_copy['target'] = df_copy['ipk'].apply(get_predikat)
        return df_copy, ['Tidak Lulus', 'Memuaskan', 'Sangat Memuaskan', 'Cumlaude']

# Fungsi validasi dataset
def validate_dataset(df):
    """Validasi kolom dan kualitas dataset"""
    required_columns = ['jenis_kelamin', 'umur', 'status_menikah', 
                       'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 
                       'aktivitas_elearning', 'ipk']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        return False, f"Kolom yang hilang: {', '.join(missing_cols)}"
    
    if df[required_columns].isnull().any().any():
        return False, "Dataset mengandung nilai kosong (NaN)"
    
    if (df['ipk'] < 0).any() or (df['ipk'] > 4).any():
        return False, "IPK harus dalam range 0-4"
    
    if (df['kehadiran'] < 0).any() or (df['kehadiran'] > 100).any():
        return False, "Kehadiran harus dalam range 0-100"
    
    return True, "Dataset valid"

# Fungsi preprocessing dengan feature engineering
def preprocess_data(df):
    """Preprocessing dengan feature engineering"""
    df_processed = df.copy()
    
    # Encode categorical features
    le_gender = LabelEncoder()
    le_married = LabelEncoder()
    
    df_processed['jenis_kelamin_encoded'] = le_gender.fit_transform(df_processed['jenis_kelamin'])
    df_processed['status_menikah_encoded'] = le_married.fit_transform(df_processed['status_menikah'])
    
    # Feature Engineering
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
    
    df_processed['kehadiran_x_tugas'] = (
        df_processed['kehadiran'] / 100 * df_processed['nilai_tugas']
    )
    
    return df_processed, le_gender, le_married

# Fungsi untuk training model klasifikasi
def train_classification_model(df, model_type='random_forest', classification_mode='binary', test_size=0.2):
    """Training model klasifikasi"""
    try:
        # Buat label klasifikasi
        df_labeled, class_names = create_classification_labels(df, classification_mode)
        
        # Preprocessing
        df_processed, le_gender, le_married = preprocess_data(df_labeled)
        
        # Features
        features = [
            'jenis_kelamin_encoded', 'umur', 'status_menikah_encoded', 
            'kehadiran', 'partisipasi_diskusi', 'nilai_tugas', 
            'aktivitas_elearning', 'rata_rata_akademik', 
            'engagement_score', 'kehadiran_x_tugas'
        ]
        
        X = df_processed[features]
        y = df_processed['target']
        
        # Encode target labels
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, 
            stratify=y_encoded, shuffle=True
        )
        
        # Scaling (untuk Logistic Regression)
        scaler = StandardScaler()
        if model_type == 'logistic_regression':
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Pilih model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'xgboost':
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                C=1.0
            )
        
        # Training
        model.fit(X_train_scaled, y_train)
        
        # Prediksi
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Prediksi probabilitas
        y_pred_proba_test = model.predict_proba(X_test_scaled)
        
        # Hitung metrik
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Untuk multi-class, gunakan weighted average
        avg_method = 'binary' if classification_mode == 'binary' else 'weighted'
        
        precision = precision_score(y_test, y_pred_test, average=avg_method, zero_division=0)
        recall = recall_score(y_test, y_pred_test, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average=avg_method, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                    cv=5, scoring='accuracy', n_jobs=-1)
        
        # Classification report
        target_names = [le_target.inverse_transform([i])[0] for i in range(len(class_names))]
        class_report = classification_report(y_test, y_pred_test, 
                                            target_names=target_names,
                                            output_dict=True,
                                            zero_division=0)
        
        return {
            'model': model,
            'scaler': scaler,
            'le_gender': le_gender,
            'le_married': le_married,
            'le_target': le_target,
            'features': features,
            'class_names': class_names,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_pred_proba_test': y_pred_proba_test,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'classification_report': class_report
        }
        
    except Exception as e:
        st.error(f"❌ Error saat training: {str(e)}")
        return None

# Header
st.title("🎓 Aplikasi Klasifikasi Kelulusan Mahasiswa")
st.markdown("Prediksi **kategori kelulusan** menggunakan Machine Learning Classification")

# Sidebar
st.sidebar.header("📋 Menu")
menu = st.sidebar.radio("Pilih Menu:", 
                        ["Upload & Training", "Prediksi Individual", 
                         "Visualisasi", "Perbandingan Model"])

# Session state
if 'models' not in st.session_state:
    st.session_state.models = {}
    st.session_state.results = {}
    st.session_state.trained = False
    st.session_state.df = None
    st.session_state.classification_mode = 'binary'

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
            else:
                df = pd.read_csv(uploaded_file, sep='\t')
                
            st.success(f"✅ Data berhasil diupload! Total: {len(df)} baris")
            
        except Exception as e:
            st.error(f"❌ Error saat membaca file: {str(e)}")
            st.stop()
        
        # Validasi
        is_valid, message = validate_dataset(df)
        
        if not is_valid:
            st.error(f"❌ {message}")
            st.stop()
        else:
            st.success(f"✅ {message}")
        
        # Preview data
        st.subheader("📊 Preview Data")
        st.dataframe(df.head(10))
        
        # Pilih mode klasifikasi
        st.subheader("🎯 Mode Klasifikasi")
        col1, col2 = st.columns(2)
        
        with col1:
            classification_mode = st.radio(
                "Pilih Mode:",
                ['binary', 'multiclass'],
                format_func=lambda x: 'Binary (2 kelas: Lulus/Tidak)' if x == 'binary' else 'Multi-class (4 kelas: Predikat)',
                key='class_mode'
            )
        
        with col2:
            if classification_mode == 'binary':
                st.info("""
                **Binary Classification:**
                - ✅ Lulus (IPK ≥ 2.75)
                - ❌ Tidak Lulus (IPK < 2.75)
                """)
            else:
                st.info("""
                **Multi-class Classification:**
                - 🏆 Cumlaude (IPK ≥ 3.5)
                - ⭐ Sangat Memuaskan (3.0-3.5)
                - 👍 Memuaskan (2.75-3.0)
                - ❌ Tidak Lulus (< 2.75)
                """)
        
        # Preview distribusi kelas
        df_preview, class_names = create_classification_labels(df, classification_mode)
        class_dist = df_preview['target'].value_counts()
        
        st.subheader("📊 Distribusi Kelas")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(values=class_dist.values, names=class_dist.index,
                        title='Distribusi Kelas dalam Dataset')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Jumlah per Kelas:**")
            for cls, count in class_dist.items():
                pct = count / len(df) * 100
                st.metric(cls, f"{count} ({pct:.1f}%)")
        
        # Warning untuk imbalanced data
        if class_dist.max() / class_dist.min() > 3:
            st.warning("⚠️ Dataset tidak seimbang (imbalanced)! Model sudah menggunakan class_weight='balanced'")
        
        # Konfigurasi training
        st.subheader("🤖 Konfigurasi Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Pilih Model:**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                train_rf = st.checkbox("Random Forest", value=True)
            with col_b:
                train_xgb = st.checkbox("XGBoost", value=True)
            with col_c:
                train_lr = st.checkbox("Logistic Regression", value=True)
        
        with col2:
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
        
        # Tombol training
        if st.button("🚀 Mulai Training", type="primary"):
            models_to_train = []
            if train_rf:
                models_to_train.append(('Random Forest', 'random_forest'))
            if train_xgb:
                models_to_train.append(('XGBoost', 'xgboost'))
            if train_lr:
                models_to_train.append(('Logistic Regression', 'logistic_regression'))
            
            if not models_to_train:
                st.error("❌ Pilih minimal satu model!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                st.session_state.models = {}
                st.session_state.results = {}
                st.session_state.classification_mode = classification_mode
                
                for idx, (model_name, model_type) in enumerate(models_to_train):
                    status_text.text(f"⏳ Training {model_name}... ({idx+1}/{len(models_to_train)})")
                    
                    result = train_classification_model(df, model_type, classification_mode, test_size)
                    
                    if result:
                        st.session_state.models[model_type] = result
                        st.session_state.results[model_type] = {
                            'name': model_name,
                            'metrics': result
                        }
                    
                    progress_bar.progress((idx + 1) / len(models_to_train))
                
                st.session_state.trained = True
                st.session_state.df = df
                status_text.text("✅ Training selesai!")
                
                # Tampilkan hasil
                st.subheader("📊 Hasil Training")
                
                results_data = []
                for model_type, result in st.session_state.results.items():
                    metrics = result['metrics']
                    results_data.append({
                        'Model': result['name'],
                        'Train Acc': f"{metrics['train_accuracy']:.4f}",
                        'Test Acc': f"{metrics['test_accuracy']:.4f}",
                        'CV Acc': f"{metrics['cv_accuracy_mean']:.4f}±{metrics['cv_accuracy_std']:.4f}",
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1-Score': f"{metrics['f1_score']:.4f}"
                    })
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Best model
                best_idx = results_df['Test Acc'].astype(float).idxmax()
                best_model = results_df.iloc[best_idx]
                st.success(f"🏆 Model Terbaik: **{best_model['Model']}** dengan Test Accuracy = {best_model['Test Acc']}")
                
                # Visualisasi
                st.subheader("📈 Perbandingan Metrik")
                
                metrics_df = pd.DataFrame({
                    'Model': [r['name'] for r in st.session_state.results.values()],
                    'Accuracy': [r['metrics']['test_accuracy'] for r in st.session_state.results.values()],
                    'Precision': [r['metrics']['precision'] for r in st.session_state.results.values()],
                    'Recall': [r['metrics']['recall'] for r in st.session_state.results.values()],
                    'F1-Score': [r['metrics']['f1_score'] for r in st.session_state.results.values()]
                })
                
                metrics_melted = metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
                
                fig = px.bar(metrics_melted, x='Model', y='Score', color='Metric',
                           barmode='group', title='Perbandingan Metrik Evaluasi')
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Silakan upload dataset terlebih dahulu")
        
        with st.expander("📖 Panduan Dataset"):
            st.markdown("""
            **Format dataset yang dibutuhkan:**
            - jenis_kelamin: "Laki-laki" atau "Perempuan"
            - umur: Integer
            - status_menikah: "Menikah" atau "Belum Menikah"
            - kehadiran: 0-100
            - partisipasi_diskusi: 0-100
            - nilai_tugas: 0-100
            - aktivitas_elearning: 0-100
            - ipk: 0-4
            
            **Perbedaan dengan Regresi:**
            - Regresi: Prediksi angka IPK (contoh: 3.45)
            - Klasifikasi: Prediksi kategori (contoh: "Lulus" atau "Cumlaude")
            
            **Keuntungan Klasifikasi:**
            ✅ Lebih robust terhadap data noisy
            ✅ Output lebih actionable (Lulus/Tidak)
            ✅ Metrik lebih mudah dipahami
            """)

# MENU 2: Prediksi Individual
elif menu == "Prediksi Individual":
    st.header("🔮 Prediksi Kelulusan Individual")
    
    if not st.session_state.trained:
        st.warning("⚠️ Model belum ditraining!")
    else:
        mode_text = "Binary" if st.session_state.classification_mode == 'binary' else "Multi-class"
        st.success(f"✅ Mode: **{mode_text}** | {len(st.session_state.models)} model siap!")
        
        # Pilih model
        model_options = {v['name']: k for k, v in st.session_state.results.items()}
        selected_name = st.selectbox("Pilih Model:", list(model_options.keys()))
        selected_type = model_options[selected_name]
        
        # Tampilkan metrik
        metrics = st.session_state.results[selected_type]['metrics']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['test_accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
        
        st.markdown("---")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            umur = st.number_input("Umur", min_value=17, max_value=50, value=20)
            status_menikah = st.selectbox("Status Menikah", ["Belum Menikah", "Menikah"])
            kehadiran = st.slider("Kehadiran (%)", 0, 100, 80)
        
        with col2:
            partisipasi = st.number_input("Partisipasi Diskusi", 0, 100, 75)
            nilai_tugas = st.number_input("Nilai Tugas", 0.0, 100.0, 80.0)
            aktivitas = st.number_input("Aktivitas E-Learning", 0, 100, 70)
        
        if st.button("🎯 Prediksi", type="primary"):
            model_data = st.session_state.models[selected_type]
            
            # Encode
            gender_enc = model_data['le_gender'].transform([jenis_kelamin])[0]
            married_enc = model_data['le_married'].transform([status_menikah])[0]
            
            # Feature engineering
            rata_akademik = (nilai_tugas + partisipasi + aktivitas) / 3
            engagement = (kehadiran * 0.4 + partisipasi * 0.3 + aktivitas * 0.3)
            kehadiran_tugas = (kehadiran / 100 * nilai_tugas)
            
            # Input
            input_data = pd.DataFrame({
                'jenis_kelamin_encoded': [gender_enc],
                'umur': [umur],
                'status_menikah_encoded': [married_enc],
                'kehadiran': [kehadiran],
                'partisipasi_diskusi': [partisipasi],
                'nilai_tugas': [nilai_tugas],
                'aktivitas_elearning': [aktivitas],
                'rata_rata_akademik': [rata_akademik],
                'engagement_score': [engagement],
                'kehadiran_x_tugas': [kehadiran_tugas]
            })
            
            # Scale jika LR
            if selected_type == 'logistic_regression':
                input_scaled = model_data['scaler'].transform(input_data)
            else:
                input_scaled = input_data
            
            # Prediksi
            pred_encoded = model_data['model'].predict(input_scaled)[0]
            pred_label = model_data['le_target'].inverse_transform([pred_encoded])[0]
            pred_proba = model_data['model'].predict_proba(input_scaled)[0]
            
            # Hasil
            st.subheader("📊 Hasil Prediksi")
            
            # Warna dan emoji berdasarkan prediksi
            color_map = {
                'Lulus': 'green',
                'Tidak Lulus': 'red',
                'Cumlaude': 'green',
                'Sangat Memuaskan': 'blue',
                'Memuaskan': 'orange'
            }
            
            emoji_map = {
                'Lulus': '✅',
                'Tidak Lulus': '❌',
                'Cumlaude': '🏆',
                'Sangat Memuaskan': '⭐',
                'Memuaskan': '👍'
            }
            
            color = color_map.get(pred_label, 'gray')
            emoji = emoji_map.get(pred_label, '📊')
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {emoji} Prediksi: :{color}[{pred_label}]")
                confidence = pred_proba[pred_encoded]
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col2:
                st.write("**Probabilitas per Kelas:**")
                for idx, cls_name in enumerate(model_data['class_names']):
                    cls_label = model_data['le_target'].inverse_transform([idx])[0]
                    st.progress(pred_proba[idx], text=f"{cls_label}: {pred_proba[idx]:.1%}")
            
            # Prediksi semua model
            if len(st.session_state.models) > 1:
                if st.checkbox("📊 Lihat prediksi semua model"):
                    all_preds = []
                    for mtype, mdata in st.session_state.models.items():
                        if mtype == 'logistic_regression':
                            inp = mdata['scaler'].transform(input_data)
                        else:
                            inp = input_data
                        
                        pred_enc = mdata['model'].predict(inp)[0]
                        pred_lbl = mdata['le_target'].inverse_transform([pred_enc])[0]
                        pred_prob = mdata['model'].predict_proba(inp)[0][pred_enc]
                        
                        all_preds.append({
                            'Model': st.session_state.results[mtype]['name'],
                            'Prediksi': pred_lbl,
                            'Confidence': f"{pred_prob:.1%}"
                        })
                    
                    st.dataframe(pd.DataFrame(all_preds), use_container_width=True)

# MENU 3: Visualisasi
elif menu == "Visualisasi":
    st.header("📊 Visualisasi Model")
    
    if not st.session_state.trained:
        st.warning("⚠️ Silakan training model terlebih dahulu")
    else:
        # Pilih model
        model_options = {v['name']: k for k, v in st.session_state.results.items()}
        selected_name = st.selectbox("Pilih Model:", list(model_options.keys()))
        selected_type = model_options[selected_name]
        
        model_data = st.session_state.models[selected_type]
        
        tab1, tab2, tab3 = st.tabs(["🎯 Confusion Matrix", "📈 Feature Importance", "📊 Classification Report"])
        
        with tab1:
            st.subheader("Confusion Matrix")
            
            cm = model_data['confusion_matrix']
            class_labels = [model_data['le_target'].inverse_transform([i])[0] 
                           for i in range(len(model_data['class_names']))]
            
            fig = px.imshow(cm, 
                           x=class_labels, 
                           y=class_labels,
                           labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                           text_auto=True,
                           aspect="auto",
                           color_continuous_scale='Blues')
            fig.update_layout(title=f'Confusion Matrix - {selected_name}')
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretasi
            st.info("""
            **Cara Membaca Confusion Matrix:**
            - Diagonal (kiri atas ke kanan bawah) = Prediksi benar
            - Off-diagonal = Prediksi salah
            - Semakin gelap warna di diagonal = semakin bagus
            """)
        
        with tab2:
            if selected_type in ['random_forest', 'xgboost']:
                st.subheader("Feature Importance")
                
                feature_names = [
                    'Jenis Kelamin', 'Umur', 'Status Menikah', 
                    'Kehadiran', 'Partisipasi', 'Nilai Tugas', 
                    'Aktivitas E-Learning', 'Rata² Akademik',
                    'Engagement Score', 'Kehadiran×Tugas'
                ]
                
                importance = model_data['model'].feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df, x='Importance', y='Feature',
                           orientation='h',
                           title=f'Feature Importance - {selected_name}',
                           color='Importance',
                           color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"🎯 Fitur paling penting: **{importance_df.iloc[0]['Feature']}**")
            else:
                st.info("Feature importance hanya tersedia untuk Random Forest dan XGBoost")
        
        with tab3:
            st.subheader("Classification Report")
            
            class_report = model_data['classification_report']
            
            # Buat dataframe dari classification report
            report_data = []
            for class_name, metrics in class_report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    report_data.append({
                        'Kelas': class_name,
                        'Precision': f"{metrics['precision']:.2%}",
                        'Recall': f"{metrics['recall']:.2%}",
                        'F1-Score': f"{metrics['f1-score']:.2%}",
                        'Support': int(metrics['support'])
                    })
            
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True)
            
            # Overall metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{class_report['accuracy']:.2%}")
            with col2:
                st.metric("Macro Avg F1", f"{class_report['macro avg']['f1-score']:.2%}")
            with col3:
                st.metric("Weighted Avg F1", f"{class_report['weighted avg']['f1-score']:.2%}")
            
            with st.expander("💡 Penjelasan Metrik"):
                st.markdown("""
                **Precision (Presisi):**
                - Dari semua prediksi positif, berapa yang benar?
                - Tinggi = sedikit False Positive
                
                **Recall (Sensitivitas):**
                - Dari semua data positif aktual, berapa yang terdeteksi?
                - Tinggi = sedikit False Negative
                
                **F1-Score:**
                - Harmonic mean dari Precision dan Recall
                - Metrik seimbang antara keduanya
                
                **Support:**
                - Jumlah data aktual untuk kelas tersebut
                
                **Macro Average:**
                - Rata-rata sederhana dari semua kelas
                - Semua kelas diberi bobot sama
                
                **Weighted Average:**
                - Rata-rata tertimbang berdasarkan jumlah data per kelas
                - Lebih representative untuk imbalanced data
                """)

# MENU 4: Perbandingan Model
elif menu == "Perbandingan Model":
    st.header("⚖️ Perbandingan Model")
    
    if not st.session_state.trained:
        st.warning("⚠️ Silakan training model terlebih dahulu")
    else:
        mode_text = "Binary" if st.session_state.classification_mode == 'binary' else "Multi-class"
        st.info(f"📊 Mode: **{mode_text} Classification**")
        
        st.subheader("📊 Tabel Perbandingan Lengkap")
        
        # Buat tabel perbandingan
        comparison_data = []
        for model_type, result in st.session_state.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': result['name'],
                'Train Accuracy': metrics['train_accuracy'],
                'Test Accuracy': metrics['test_accuracy'],
                'CV Accuracy': metrics['cv_accuracy_mean'],
                'CV Std': metrics['cv_accuracy_std'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Overfitting Gap': metrics['train_accuracy'] - metrics['test_accuracy']
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Format dan highlight
        st.dataframe(
            comp_df.style.format({
                'Train Accuracy': '{:.2%}',
                'Test Accuracy': '{:.2%}',
                'CV Accuracy': '{:.2%}',
                'CV Std': '{:.4f}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'Overfitting Gap': '{:.4f}'
            }).background_gradient(subset=['Test Accuracy'], cmap='RdYlGn')
              .background_gradient(subset=['F1-Score'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Model terbaik
        best_idx = comp_df['Test Accuracy'].idxmax()
        best_model = comp_df.iloc[best_idx]
        
        st.success(f"""
        🏆 **Model Terbaik: {best_model['Model']}**
        - Test Accuracy: {best_model['Test Accuracy']:.2%}
        - F1-Score: {best_model['F1-Score']:.2%}
        - Cross-Validation: {best_model['CV Accuracy']:.2%} ± {best_model['CV Std']:.4f}
        """)
        
        # Warning
        for _, row in comp_df.iterrows():
            if row['Overfitting Gap'] > 0.15:
                st.warning(f"⚠️ **{row['Model']}**: Overfitting terdeteksi (gap={row['Overfitting Gap']:.2%})")
        
        # Visualisasi perbandingan
        st.subheader("📈 Visualisasi Perbandingan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Perbandingan accuracy
            fig = px.bar(comp_df, x='Model', 
                        y=['Train Accuracy', 'Test Accuracy', 'CV Accuracy'],
                        barmode='group',
                        title='Perbandingan Accuracy',
                        labels={'value': 'Accuracy', 'variable': 'Metric'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Perbandingan metrik klasifikasi
            fig = px.bar(comp_df, x='Model',
                        y=['Precision', 'Recall', 'F1-Score'],
                        barmode='group',
                        title='Perbandingan Precision, Recall, F1',
                        labels={'value': 'Score', 'variable': 'Metric'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart untuk perbandingan multi-dimensi
        st.subheader("🎯 Radar Chart Perbandingan")
        
        fig = go.Figure()
        
        categories = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for _, row in comp_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Test Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
                theta=categories,
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Perbandingan Multi-Dimensi'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix Comparison
        st.subheader("🎯 Perbandingan Confusion Matrix")
        
        cols = st.columns(len(st.session_state.models))
        
        for idx, (model_type, model_data) in enumerate(st.session_state.models.items()):
            with cols[idx]:
                cm = model_data['confusion_matrix']
                class_labels = [model_data['le_target'].inverse_transform([i])[0] 
                               for i in range(len(model_data['class_names']))]
                
                fig = px.imshow(cm,
                               x=class_labels,
                               y=class_labels,
                               text_auto=True,
                               aspect="auto",
                               color_continuous_scale='Blues')
                
                model_name = st.session_state.results[model_type]['name']
                fig.update_layout(
                    title=model_name,
                    width=350,
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tips interpretasi
        with st.expander("💡 Tips Memilih Model Terbaik"):
            st.markdown("""
            **Pertimbangan Memilih Model:**
            
            1. **Accuracy Tinggi** ✅
               - Model memprediksi dengan benar
               - Perhatikan Test Accuracy, bukan Train!
            
            2. **F1-Score Seimbang** ⚖️
               - Balance antara Precision dan Recall
               - Penting untuk imbalanced data
            
            3. **Low Overfitting Gap** 📉
               - Gap < 10% = bagus
               - Gap > 15% = overfitting
            
            4. **Stable CV Score** 📊
               - CV Std kecil = model stabil
               - Tidak tergantung data split
            
            5. **Confusion Matrix Bersih** 🎯
               - Diagonal kuat (prediksi benar tinggi)
               - Off-diagonal lemah (error rendah)
            
            **Rekomendasi untuk Dataset Anda:**
            - Jika akurasi prioritas → pilih model dengan Test Accuracy tertinggi
            - Jika stabilitas penting → pilih model dengan CV Std terendah
            - Jika balanced performance → pilih model dengan F1-Score tertinggi
            """)
        
        # Perbandingan dengan Regresi
        with st.expander("🔄 Klasifikasi vs Regresi - Mana Lebih Baik?"):
            st.markdown("""
            **Keuntungan Klasifikasi untuk Kasus Anda:**
            
            ✅ **Lebih Robust terhadap Noise**
            - Data Anda memiliki anomali (kehadiran tinggi tapi IPK rendah)
            - Klasifikasi lebih toleran karena hanya prediksi kategori
            
            ✅ **Output Lebih Actionable**
            - "Lulus" atau "Tidak Lulus" → langsung bisa ambil keputusan
            - IPK 3.45 vs 3.47 → sulit interpretasi praktis
            
            ✅ **Metrik Lebih Jelas**
            - Accuracy, Precision, Recall mudah dipahami
            - R² negatif di regresi membingungkan
            
            ✅ **Handling Imbalanced Data**
            - Class weights bisa disesuaikan
            - Fokus pada kelas minoritas (misalnya "Tidak Lulus")
            
            **Kapan Pakai Regresi?**
            - Butuh nilai IPK exact untuk ranking
            - Data bersih dan konsisten
            - Hubungan linear jelas antara fitur dan IPK
            
            **Kapan Pakai Klasifikasi?**
            - Data noisy/tidak konsisten ✅ (kasus Anda!)
            - Output kategorikal sudah cukup
            - Imbalanced data
            - Fokus pada decision making
            """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🎓 Aplikasi Klasifikasi v1.0")
st.sidebar.caption("""
**Model yang Digunakan:**
- Random Forest Classifier
- XGBoost Classifier  
- Logistic Regression
""")

# Tips di sidebar
with st.sidebar.expander("💡 Tips Penggunaan"):
    st.markdown("""
    **Mode Binary vs Multi-class:**
    
    🔵 **Binary (Recommended)**
    - Lebih simple
    - Akurasi lebih tinggi
    - Cocok untuk keputusan Lulus/Tidak
    
    🟣 **Multi-class**
    - Lebih detail (4 kategori)
    - Butuh data lebih banyak
    - Cocok untuk klasifikasi predikat
    
    **Metrik Penting:**
    - **Accuracy**: Seberapa sering benar
    - **Precision**: Sedikit false alarm
    - **Recall**: Tidak miss yang penting
    - **F1-Score**: Balance keduanya
    
    **Model Recommendation:**
    - 🌲 Random Forest: Robust & stabil
    - 🚀 XGBoost: Performa terbaik
    - 📈 Logistic Regression: Simple & cepat
    """)
