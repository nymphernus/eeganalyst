import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime

from __init__ import setup_project_paths
setup_project_paths()

from core.data.loader import load_edf, load_csv
from core.data.manager import DataManager
from core.data.sample import EEGSample
st.set_page_config(
    page_title="EEG Insights Agent",
    page_icon="🧬",
    layout="wide")

st.title("🔗 Множественная загрузка файлов")

manager = DataManager()

st.subheader("📂 Выберите файлы")
uploaded_files = st.file_uploader(
    "Загрузите несколько EEG файлов (.edf или .csv)", 
    type=["edf", "csv"], 
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("📥 Выберите несколько файлов одного типа")
else:
    file_types = [f.name.split('.')[-1].lower() for f in uploaded_files]
    if len(set(file_types)) > 1:
        st.error("❌ Все файлы должны быть одного типа (.edf или .csv)")
    else:
        file_type = file_types[0]
        st.success(f"✅ Выбрано {len(uploaded_files)} файлов типа .{file_type}")

        has_labels = False
        if file_type == "csv":
            st.subheader("⚙️ Параметры CSV")
            col1, col2 = st.columns(2)
            with col1:
                sfreq = st.number_input("Частота дискретизации (Hz)", min_value=1.0, value=256.0)
            with col2:
                has_labels = st.checkbox("Файлы содержат метки классов")
        
        if st.button("🔍 Проверить совместимость", type="primary"):
            try:
                with st.spinner("Проверка файлов..."):
                    samples = []
                    temp_files = []
                    
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                            temp_files.append(tmp_file_path)
                        
                        if file_type == "edf":
                            sample = load_edf(tmp_file_path)
                        else:
                            sample = load_csv(tmp_file_path, sfreq=sfreq, has_labels=has_labels)
                        
                        samples.append(sample)
                    
                    for tmp_file in temp_files:
                        try:
                            os.unlink(tmp_file)
                        except:
                            pass
                    
                    st.subheader("📊 Результаты проверки")
                    compatible = True
                    first_sample = samples[0]
                    report_data = []
                    
                    for i, sample in enumerate(samples):
                        file_name = uploaded_files[i].name
                        issues = []
                        
                        if sample.data.shape[0] != first_sample.data.shape[0]:
                            compatible = False
                            issues.append(f"каналы: {sample.data.shape[0]}≠{first_sample.data.shape[0]}")
                        
                        if set(sample.ch_names) != set(first_sample.ch_names):
                            compatible = False
                            issues.append("названия каналов")
                        
                        if abs(sample.sfreq - first_sample.sfreq) > 0.1:
                            compatible = False
                            issues.append(f"частота: {sample.sfreq}≠{first_sample.sfreq}")
                        
                        status = "✅ OK" if not issues else "❌ " + ", ".join(issues)
                        report_data.append({
                            'Файл': file_name,
                            'Каналов': sample.data.shape[0],
                            'Отсчётов': f"{sample.data.shape[1]:,}",
                            'Статус': status
                        })
                    
                    report_df = pd.DataFrame(report_data)
                    st.dataframe(report_df, use_container_width=True, hide_index=True)
                    
                    if compatible:
                        st.success("✅ Все файлы совместимы!")
                        st.session_state.merge_samples = samples
                        st.session_state.merge_files = uploaded_files
                        st.session_state.merge_ready = True
                        st.session_state.merge_file_type = file_type
                    else:
                        st.error("❌ Найдены несовместимые файлы")
                        if 'merge_ready' in st.session_state:
                            del st.session_state.merge_ready
                        
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")

if st.session_state.get('merge_ready', False):
    
    merged_name = st.text_input("Имя результата:", value=f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if st.button("💾 Объединить файлы", type="secondary", use_container_width=True):
        try:
            with st.spinner("Объединение файлов..."):
                samples = st.session_state.merge_samples
                uploaded_files = st.session_state.merge_files
                file_type = st.session_state.merge_file_type
                
                file_names = [f.name for f in uploaded_files]
                sorted_indices = sorted(range(len(file_names)), key=lambda i: file_names[i])
                samples = [samples[i] for i in sorted_indices]
                uploaded_files = [uploaded_files[i] for i in sorted_indices]
                
                combined_data = np.concatenate([sample.data for sample in samples], axis=1)
                
                final_filename = f"{merged_name}.{file_type}"
                merged_sample = EEGSample(
                    data=combined_data,
                    sfreq=samples[0].sfreq,
                    ch_names=samples[0].ch_names,
                    raw_path=final_filename,
                    metadata={
                        "source_files": [f.name for f in uploaded_files],
                        "n_files": len(samples),
                        "total_samples": combined_data.shape[1],
                        "merged_at": datetime.now().isoformat()
                    }
                )
                
                merged_id = manager.add_sample(merged_sample, final_filename)
                
                for key in ['merge_samples', 'merge_files', 'merge_ready', 'merge_file_type']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success(f"✅ Файлы успешно объединены! ID: {merged_id}")
                
        except Exception as e:
            st.error(f"❌ Ошибка при объединении: {str(e)}")
