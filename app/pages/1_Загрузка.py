import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import os
import pandas as pd
import hashlib

from __init__ import setup_project_paths
setup_project_paths()

from core.data.loader import load_edf, load_csv
from core.data.manager import DataManager
from core.utils.hashing import compute_file_hash

st.set_page_config(
        layout="wide",
        page_title="EEG Insights Agent",
        page_icon="🧬"
    )

if 'page_initialized' not in st.session_state:
    st.session_state.page_initialized = True
    
st.title("📂 Загрузка файлов ЭЭГ")

manager = DataManager()

def file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

st.sidebar.header("⚙️ Настройки графика")
amplitude_scale = st.sidebar.slider("Масштаб амплитуды", min_value=0.5, max_value=10000.0, value=10.0, step=0.000001, format="%.1f")
vertical_spacing = st.sidebar.slider("Вертикальное расстояние между каналами", min_value=0.5, max_value=1000.0, value=10.0, step=0.00001, format="%.1f")
channels_to_show = st.sidebar.slider("Количество каналов", min_value=1, max_value=64, value=32, step=1)
time_samples = st.sidebar.slider("Количество отсчётов", min_value=100, max_value=50000, value=5000, step=100)

uploaded_file = st.file_uploader("Загрузите EEG файл (.edf или .csv)", type=["edf", "csv"])
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    fhash = compute_file_hash(file_bytes)
    file_type = uploaded_file.name.split(".")[-1]

    existing_id = None
    catalog = manager.list_samples()
    for item in catalog:
        if item.get('file_hash') == fhash:
            existing_id = item['id']
            break

    if existing_id:
        st.warning(f"⚠️ Файл `{uploaded_file.name}` уже загружен (ID={existing_id})")
        if st.button("👁 Показать"):
            st.session_state.selected_id = existing_id
            st.rerun()
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        try:
            if file_type == "edf":
                sample = load_edf(tmp_file_path)
            elif file_type == "csv":
                df_temp = pd.read_csv(tmp_file_path, nrows=1)
                has_labels_flag = 'label' in [col.lower() for col in df_temp.columns]
                sample = load_csv(tmp_file_path, sfreq=256, has_labels=has_labels_flag)
            else:
                st.error("Неподдерживаемый формат")
                st.stop()
            
            sample_id = manager.add_sample(sample, uploaded_file.name)
            
            st.success(f"✅ Файл загружен (ID={sample_id})")
            if st.button("👁 Показать"):
                st.session_state.selected_id = sample_id
                st.rerun()
                
        finally:
            os.unlink(tmp_file_path)

if "selected_id" in st.session_state:
    sample = manager.get_sample(st.session_state.selected_id)
    
    if sample is None:
        st.error("❌ Ошибка загрузки данных")
        if 'selected_id' in st.session_state:
            del st.session_state.selected_id
    else:
        st.divider()
        st.subheader(f"🔎 Запись {st.session_state.selected_id}")
        
        st.write(f"Файл: {sample.raw_path}")
        st.write(f"Частота дискретизации: {sample.sfreq} Hz")
        st.write(f"Количество отсчётов: {sample.data.shape[1]}")
        
        st.write(f"**Каналы** ({len(sample.ch_names)} всего):")
        with st.expander("Показать все каналы"):
            channels_text = "  ".join([f"`{ch}`" for ch in sample.ch_names])
            st.markdown(channels_text)

        fig, ax = plt.subplots(figsize=(14, 7))
        
        time_limit = min(time_samples, sample.data.shape[1])
        channel_limit = min(channels_to_show, sample.data.shape[0])
        
        for i in range(channel_limit):
            data_segment = sample.data[i][:time_limit] * amplitude_scale
            ax.plot(data_segment + i * vertical_spacing, linewidth=0.8)
        
        ax.set_xlabel('Время (отсчёты)')
        ax.set_ylabel('Амплитуда')
        ax.set_title(f'EEG сигналы (масштаб: {amplitude_scale}x, каналы: {channel_limit}, отсчёты: {time_limit})')
        ax.grid(True, alpha=0.3)

        if st.button("❌ Закрыть просмотр"):
            del st.session_state.selected_id
            st.rerun()
        
        st.pyplot(fig)
        
# Список загруженных данных
st.header("📋 Сохраненные записи")

catalog = manager.list_samples()
if not catalog:
    st.info("📥 Список пуст. Загрузите файлы.")
else:
    header_cols = st.columns([3, 1, 1, 1, 2, 1, 1])
    header_cols[0].write("**Файл**")
    header_cols[1].write("**Гц**")
    header_cols[2].write("**Каналы**")
    header_cols[3].write("**Отсчёты**")
    header_cols[4].write("**Добавлен**")
    header_cols[5].write("**ID**")
    header_cols[6].write("**Действия**")
    
    for item in catalog:
        cols = st.columns([3, 1, 1, 1, 2, 1, 1])
        
        cols[0].write(item['filename'])
        cols[1].write(f"{item['sfreq']:.0f}")
        cols[2].write(f"{item['n_channels']}")
        cols[3].write(f"{item['n_samples']:,}")
        
        date_str = item.get('created_at_formatted', item.get('created_at', ''))[:16]
        cols[4].write(date_str)
        
        short_id = item['id'][:8]
        cols[5].write(f"`{short_id}`")
        
        with cols[6]:
            button_cols = st.columns([1, 1], gap="small")
            with button_cols[0]:
                if st.button("👁", key=f"show_{item['id']}", help="Показать"):
                    st.session_state.selected_id = item['id']
                    st.rerun()
            with button_cols[1]:
                if st.button("🗑", key=f"del_{item['id']}", help="Удалить"):
                    manager.delete_sample(item['id'])
                    if 'selected_id' in st.session_state and st.session_state.selected_id == item['id']:
                        del st.session_state.selected_id
                    st.rerun()