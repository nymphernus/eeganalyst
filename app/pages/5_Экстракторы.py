from __init__ import setup_project_paths
setup_project_paths()

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.data.manager import DataManager
from core.features.spectral import PSDExtractor, BandPowerExtractor
from core.features.time_domain import TimeDomainExtractor
from core.features.rocket import MiniRocketExtractor

st.set_page_config(
        layout="wide",
        page_title="EEG Insights Agent",
        page_icon="🧬"
    )

if "page_initialized" not in st.session_state:
    st.session_state.page_initialized = True

st.title("🔬 Извлечение признаков")

manager = DataManager()

proc_list = manager.list_processed()
if not proc_list:
    st.info("📥 Нет обработанных данных.")
    st.stop()

from collections import defaultdict
grouped_data = defaultdict(list)
for item in proc_list:
    grouped_data[item['parent_id']].append(item)

source_files = {}
for parent_id in grouped_data.keys():
    source_info = manager.get_sample_info(parent_id)
    if source_info:
        source_files[parent_id] = source_info

selected_parent_id = st.selectbox(
    "📁 Выберите исходный файл",
    options=list(source_files.keys()),
    format_func=lambda pid: source_files[pid]['filename']
)

selected_processed_list = grouped_data[selected_parent_id]
selected_proc_id = st.selectbox(
    "⚙️ Выберите обработку",
    options=[p['id'] for p in selected_processed_list],
    format_func=lambda pid: f"{pid[:8]} • {next((p['created_at_formatted'] for p in selected_processed_list if p['id'] == pid), '')}"
)


proc_sample = manager.get_processed_sample(selected_proc_id)
if proc_sample is None:
    st.error("❌ Не удалось загрузить обработанный датасет")
    st.stop()

st.sidebar.header("⚙️ Экстракторы")

use_psd = st.sidebar.checkbox("PSD", value=False)
if use_psd:
    psd_fmin = st.sidebar.number_input(
        "PSD fmin (Hz)", min_value=0.0, max_value=float(proc_sample.sfreq/2), value=1.0, step=0.5
    )
    psd_fmax = st.sidebar.number_input(
        "PSD fmax (Hz)", min_value=0.0, max_value=float(proc_sample.sfreq/2), value=40.0, step=0.5
    )
    psd_nperseg = st.sidebar.number_input("PSD nperseg", min_value=16, max_value=65536, value=256, step=1)

use_band = st.sidebar.checkbox("BandPower", value=False)
use_time = st.sidebar.checkbox("TimeDomain", value=False)
use_rocket = st.sidebar.checkbox("MiniRocket", value=False)

if st.sidebar.button("🧪 Извлечь"):
    extractors = []
    errors = []

    try:
        if use_psd:
            extractors.append(PSDExtractor(
                sfreq=float(proc_sample.sfreq),
                fmin=float(psd_fmin),
                fmax=float(psd_fmax),
                nperseg=int(psd_nperseg)
            ))
        if use_band:
            extractors.append(BandPowerExtractor(sfreq=float(proc_sample.sfreq)))
        if use_time:
            extractors.append(TimeDomainExtractor())
        if use_rocket:
            try:
                extractors.append(MiniRocketExtractor())
            except Exception as e:
                errors.append(("MiniRocket", str(e)))
    except Exception as e:
        errors.append(("Инициализация", str(e)))

    if errors:
        for name, msg in errors:
            st.error(f"❌ Ошибка в {name}: {msg}")
    elif not extractors:
        st.warning("⚠️ Выберите хотя бы один экстрактор")
    else:
        X_parts = []
        for ext in extractors:
            try:
                Xp = ext.fit_transform(proc_sample.data)
                if Xp.ndim == 1:
                    Xp = np.expand_dims(Xp, 0)
                X_parts.append(Xp)
            except Exception as e:
                st.error(f"❌ Ошибка в {ext.name}: {e}")

        if X_parts:
            try:
                X_all = np.concatenate(X_parts, axis=1)
                extractor_config = {"composed": [e.to_dict() for e in extractors]}
                feat_id = manager.save_features_from_array(
                    parent_id=selected_proc_id,
                    X=X_all,
                    y=None,
                    extractor_config=extractor_config,
                    metadata={"from_preview": False}
                )
                st.success(f"✅ Фичи сохранены (ID: `{feat_id[:8]}`), форма: {X_all.shape}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Ошибка сохранения: {e}")

st.header("💾 Сохранённые признаки")
features_list = manager.list_features(parent_id=selected_proc_id)
if not features_list:
    st.info("📥 Нет сохранённых фичей.")
else:
    for f in features_list:
        with st.expander(f"🧬 ID: `{f['id'][:8]}` • X_shape: {f.get('X_shape')} • {f.get('created_at_formatted', '')}", expanded=False):
            feat_info = manager.get_features_info(f['id'])
            if feat_info and feat_info.get("extractor_config"):
                extractor_config = feat_info["extractor_config"]
                if isinstance(extractor_config, dict):
                    extractors_list = []
                    for ext in extractor_config.get("composed", []):
                        name = ext.get("name", "Unknown")
                        params = ext.get("params", {})
                        if params:
                            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                            extractors_list.append(f"{name} ({param_str})")
                        else:
                            extractors_list.append(name)
                    if extractors_list:
                        st.markdown("**🛠 Использованные экстракторы:**")
                        for e in extractors_list:
                            st.markdown(f"- {e}")

            col1, col2, col3 = st.columns(3)
            loaded_key = "loaded_features"
            if col1.button("👁 Предпросмотр", key=f"preview_{f['id']}"):
                loaded = manager.get_features_data(f['id'])
                if loaded:
                    X_l, _ = loaded
                    df = pd.DataFrame(np.array(X_l)[:10, :min(20, X_l.shape[1])])
                    st.dataframe(df)
            if col2.button("📊 График", key=f"plot_{f['id']}"):
                loaded = manager.get_features_data(f['id'])
                if loaded:
                    X_l, _ = loaded
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(X_l[:10], aspect='auto', cmap='viridis')
                    ax.set_title("Первые 10 строк признаков")
                    st.pyplot(fig)
            if col3.button("🗑 Удалить", key=f"del_{f['id']}"):
                manager.delete_features(f['id'])
                st.success("✅ Удалено")
                st.rerun()