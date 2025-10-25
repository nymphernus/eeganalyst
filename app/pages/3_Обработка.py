from __init__ import setup_project_paths
setup_project_paths()

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from core.data.manager import DataManager
from core.preprocess.pipeline import PreprocessPipeline
from core.preprocess.steps.filters import BandpassFilter, NotchFilter
from core.preprocess.steps.resample_normalize import Resample, Normalize
from core.preprocess.steps.artifacts import ReReference, ICAFilter, Epoching

st.set_page_config(
        layout="wide",
        page_title="EEG Insights Agent",
        page_icon="🧬"
    )

if "page_initialized" not in st.session_state:
    st.session_state.page_initialized = True

st.title("⚙️ Pipeline предобработки сигналов")

manager = DataManager()


st.sidebar.header("🛠️ Параметры")
use_notch = st.sidebar.checkbox("🔌 Notch", value=False)
notch_freq = st.sidebar.number_input("Частота (Hz) для Notch", min_value=40.0, max_value=70.0, value=50.0, step=1.0) if use_notch else None
notch_Q = st.sidebar.number_input("Q для Notch", min_value=5.0, max_value=200.0, value=30.0, step=1.0) if use_notch else None

use_band = st.sidebar.checkbox("🔊 Bandpass", value=False)
band_low = st.sidebar.number_input("Low (Hz)", min_value=0.1, max_value=100.0, value=1.0, step=0.1) if use_band else None
band_high = st.sidebar.number_input("High (Hz)", min_value=1.0, max_value=200.0, value=40.0, step=0.5) if use_band else None
band_order = st.sidebar.slider("Order", min_value=2, max_value=8, value=5) if use_band else None

use_resample = st.sidebar.checkbox("⏱️ Resample", value=False)
resample_target = st.sidebar.number_input("Target rate (Hz)", min_value=32, max_value=1024, value=128, step=1) if use_resample else None

use_normalize = st.sidebar.checkbox("🧼 Normalize", value=False)
normalize_method = st.sidebar.selectbox("Method", ["zscore", "minmax"]) if use_normalize else None

use_reref = st.sidebar.checkbox("🔁 Re-reference", value=False)
reref_method = st.sidebar.selectbox("Rereference method", ["average", "mastoid"]) if use_reref else None

use_ica = st.sidebar.checkbox("⚠️ ICA (compute)", value=False)
ica_ncomp = st.sidebar.slider("ICA n_components", min_value=5, max_value=50, value=15, step=1) if use_ica else None

use_epoch = st.sidebar.checkbox("◻️ Epoching", value=False)
epoch_len = st.sidebar.number_input("Epoch length (s)", min_value=0.5, max_value=10.0, value=2.0, step=0.5) if use_epoch else None


st.header("📂 Выбор сырого сигнала")
datasets = manager.list_samples()
if not datasets:
    st.warning("Нет загруженных сырых данных.")
    st.stop()

label_to_id = {f"{d['filename']} · {d['sfreq']:.0f}Hz · {d['id'][:8]}": d["id"] for d in datasets}
selected_label = st.selectbox("Выберите файл для предобработки", list(label_to_id.keys()))
raw_id = label_to_id[selected_label]

if st.session_state.get("selected_raw_id") != raw_id:
    st.session_state["selected_raw_id"] = raw_id
    st.session_state.pop("last_processed_id", None)

raw_sample = manager.get_sample(raw_id)
sfreq = float(raw_sample.sfreq)

steps = []
if use_notch:
    steps.append(NotchFilter(freq=float(notch_freq), sfreq=sfreq, Q=float(notch_Q)))
if use_band:
    steps.append(BandpassFilter(low=float(band_low), high=float(band_high), sfreq=sfreq, order=int(band_order)))
if use_resample:
    steps.append(Resample(target_rate=float(resample_target), orig_rate=sfreq))
if use_normalize:
    steps.append(Normalize(method=normalize_method))
if use_reref:
    steps.append(ReReference(method=reref_method))
if use_ica:
    steps.append(ICAFilter(n_components=int(ica_ncomp)))
if use_epoch:
    steps.append(Epoching(sfreq=sfreq, epoch_length=float(epoch_len)))

pipeline = PreprocessPipeline(steps=steps)

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🚀 Применить и сохранить"):
        try:
            proc_id = manager.apply_pipeline(raw_id, pipeline, save=True)
            st.session_state["last_processed_id"] = proc_id
            st.success(f"Готово! (ID={proc_id[:8]})")
        except Exception as e:
            st.error(f"Ошибка при применении пайплайна: {e}")

def plot_time(data: np.ndarray, sf: float, title: str, n_channels: int = 8, n_samples: int = 3000, offset: float = 100.0):
    if data.ndim == 3:
        display = data[:, 0, :]
    else:
        display = data
    n_samples = min(n_samples, display.shape[-1])
    t = np.arange(n_samples) / sf
    fig, ax = plt.subplots(figsize=(12, 4))
    for i in range(min(n_channels, display.shape[0])):
        ax.plot(t, display[i, :n_samples] + i*offset, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Амплитуда + offset")
    ax.grid(alpha=0.3)
    return fig

def plot_psd(data: np.ndarray, sf: float, title: str, n_fft: int = 1024):
    if data.ndim == 3:
        display = data[:, 0, :]
    else:
        display = data
    fig, ax = plt.subplots(figsize=(12, 4))
    freqs, psds = [], []
    for i in range(display.shape[0]):
        f, p = welch(display[i], fs=sf, nperseg=min(n_fft, display.shape[-1]))
        if i == 0:
            freqs = f
        psds.append(p)
    mean_psd = np.mean(np.stack(psds, axis=0), axis=0)
    ax.semilogy(freqs, mean_psd)
    ax.set_title(title)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("PSD")
    ax.grid(alpha=0.3)
    return fig

c_raw, c_proc = st.columns(2, gap="large")
with c_raw:
    st.markdown("**Сырые данные**")
    st.write(f"Файл: `{raw_sample.raw_path}`  ·  {raw_sample.data.shape[0]} ch · {raw_sample.data.shape[-1]} сэмплов · {raw_sample.sfreq} Hz")
    st.pyplot(plot_time(raw_sample.data, raw_sample.sfreq, "До предобработки"))
    st.pyplot(plot_psd(raw_sample.data, raw_sample.sfreq, "PSD до"))

with c_proc:
    st.markdown("**Обработанные версии**")
    processed_list = manager.list_processed(parent_id=raw_id)
    if processed_list:
        options = {f"{p['id'][:8]} · {p['created_at_formatted'] if p.get('created_at_formatted') else p.get('created_at', '')}": p["id"] for p in processed_list}
        default = None
        if st.session_state.get("last_processed_id") and any(p["id"] == st.session_state["last_processed_id"] for p in processed_list):
            default = st.session_state["last_processed_id"]
        else:
            default = processed_list[0]["id"]
            if st.session_state.get("selected_raw_id") != raw_id:
                default = None

        selected_proc_label = st.selectbox("Выберите обработанную версию", list(options.keys()), index=0 if default else 0)
        selected_proc_id = options[selected_proc_label]
        st.session_state["last_processed_id"] = selected_proc_id

        proc_sample = manager.get_processed_sample(selected_proc_id)
        if proc_sample is not None:
            st.write(f"Processed from: `{proc_sample.raw_path}`  ·  shape: {proc_sample.data.shape}")
            st.pyplot(plot_time(proc_sample.data, proc_sample.sfreq, "После предобработки"))
            st.pyplot(plot_psd(proc_sample.data, proc_sample.sfreq, "PSD после"))
        else:
            st.error("Не удалось загрузить")
    else:
        st.info("Для этого файла нет обработанных версий. Примените пайплайн и сохраните результат.")
