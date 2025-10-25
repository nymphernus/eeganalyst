from __init__ import setup_project_paths
setup_project_paths()

import streamlit as st
import matplotlib.pyplot as plt

from core.data.manager import DataManager

st.set_page_config(
        layout="wide",
        page_title="EEG Insights Agent",
        page_icon="🧬"
    )

if 'page_initialized' not in st.session_state:
    st.session_state.page_initialized = True


st.title("📊 Обработанные данные")


manager = DataManager()

st.sidebar.header("⚙️ Настройки")
amplitude_scale = st.sidebar.slider("Масштаб амплитуды", min_value=0.5, max_value=10000.0, value=10.0, step=0.000001, format="%.1f")
vertical_spacing = st.sidebar.slider("Вертикальное расстояние между каналами", min_value=0.5, max_value=1000.0, value=10.0, step=0.00001, format="%.1f")
channels_to_show = st.sidebar.slider("Количество каналов", min_value=1, max_value=64, value=32, step=1)
time_samples = st.sidebar.slider("Количество отсчётов", min_value=100, max_value=50000, value=5000, step=100)

processed_data = manager.list_all_processed()

if not processed_data:
    st.info("📥 Нет обработанных данных.")
else:
    from collections import defaultdict
    grouped_data = defaultdict(list)
    for item in processed_data:
        grouped_data[item['parent_id']].append(item)
    
    source_files = {}
    for parent_id in grouped_data.keys():
        source_info = manager.get_sample_info(parent_id)
        if source_info:
            source_files[parent_id] = source_info
    
    for parent_id, processed_list in grouped_data.items():
        source_info = source_files.get(parent_id, {})
        source_name = source_info.get('filename', f'Unknown ({parent_id[:8]})')
        
        with st.expander(f"📁 {source_name} ({len(processed_list)} обработок)", expanded=False):
            if source_info:
                st.markdown(f"""
                **Исходный файл:** `{source_name}`  [ **Частота:** {source_info.get('sfreq', 'N/A')} Hz • **Каналов:** {source_info.get('n_channels', 'N/A')} • **Отсчётов:** {source_info.get('n_samples', 'N/A'):,} ]""")
            st.markdown("")
            
            for item in processed_list:
                proc_id = item['id']
                short_id = proc_id[:8]
                
                created_at = item.get('created_at_formatted', item.get('created_at', ''))
                
                col1, col2 = st.columns([8, 1])
                
                with col1:
                    st.markdown(f"**ID обработки:** `{short_id}` • {created_at} [ **Частота:** {item.get('sfreq', 'N/A')} Hz • **Каналов:** {item.get('n_channels', 'N/A')} • **Отсчётов:** {item.get('n_samples', 'N/A'):,} ]")
                
                with col2:
                    btn_cols = st.columns(2, gap="small")
                    with btn_cols[0]:
                        is_selected = 'selected_processed_id' in st.session_state and st.session_state.selected_processed_id == proc_id
                        btn_label = "❌" if is_selected else "👁"
                        if st.button(btn_label, key=f"toggle_{proc_id}"):
                            if is_selected:
                                del st.session_state.selected_processed_id
                            else:
                                st.session_state.selected_processed_id = proc_id
                            st.rerun()
                    with btn_cols[1]:
                        if st.button("🗑", key=f"delete_{proc_id}"):
                            if manager.delete_processed_sample(proc_id):
                                if 'selected_processed_id' in st.session_state and st.session_state.selected_processed_id == proc_id:
                                    del st.session_state.selected_processed_id
                                st.success(f"✅ Обработка {short_id} удалена")
                                st.rerun()
                            else:
                                st.error(f"❌ Ошибка удаления {short_id}")
                
                if 'selected_processed_id' in st.session_state and st.session_state.selected_processed_id == proc_id:
                    try:
                        with st.expander("📊 График", expanded=False):
                            proc_sample = manager.get_processed_sample(proc_id)
                            if proc_sample is not None and proc_sample.data is not None:
                                data_shape = proc_sample.data.shape
                                n_channels = data_shape[0]
                                n_samples = data_shape[1] if len(data_shape) == 2 else data_shape[2]
                                
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                display_channels = min(channels_to_show, n_channels)
                                display_samples = min(time_samples, n_samples)
                                
                                if len(data_shape) == 3:
                                    data_to_plot = proc_sample.data[:, 0, :]
                                else:
                                    data_to_plot = proc_sample.data
                                
                                for i in range(display_channels):
                                    channel_data = data_to_plot[i][:display_samples] * amplitude_scale
                                    ax.plot(channel_data + i * vertical_spacing, 
                                        linewidth=0.8, alpha=0.8)
                                
                                ax.set_xlabel('Время (отсчёты)')
                                ax.set_ylabel('Амплитуда')
                                ax.set_title(f'Обработанные данные (ID: {short_id})')
                                ax.grid(True, alpha=0.3)
                                
                                st.pyplot(fig)
                            else:
                                st.error("❌ Ошибка загрузки данных для визуализации")
                        with st.expander("📋 Пайплайн обработки", expanded=False):
                            proc_info = manager.get_processed_info(proc_id)
                            if proc_info and proc_info.get('pipeline_config'):
                                pipeline_cfg = proc_info['pipeline_config']
                                steps = pipeline_cfg.get('steps', [])
                                if steps:
                                    for i, step in enumerate(steps, 1):
                                        step_name = step.get('name', 'Unknown')
                                        step_params = step.get('params', {})
                                        st.markdown(f"{i}. **{step_name}**: {step_params}")
                            else:
                                st.error("❌ Ошибка загрузки данных пайплайна")
                    except Exception as e:
                        st.error(f"❌ Ошибка загрузки данных обработки: {str(e)}")
                