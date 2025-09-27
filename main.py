from dataset_loader import load_dataset, datasets_list as dtl

import dash
from dash import dcc, html, Input, Output, State, callback, no_update
from dash.dependencies import ALL

import plotly.graph_objs as go
import pandas as pd
import numpy as np

from base64 import b64decode
from io import BytesIO
from datetime import datetime
from os import path as ospath, makedirs
from mne import pick_types

# === Настройки ===
SAVE_FOLDER = "saved_annotations"
makedirs(SAVE_FOLDER, exist_ok=True)

# Инициализация приложения
app = dash.Dash(__name__)
app.title = "Разметка ЭЭГ"


# === Макет ===
app.layout = html.Div([

    # Выбор датасета и загрузка аннотаций
    html.Div([
        html.H3("📁 Данные", className="section-title"),
        html.Div([
            # Левая колонка: выбор датасета
            html.Div([
                html.Label("Выберите датасет:", className="slider-label"),
                dcc.Dropdown(
                    id='dataset-dropdown',
                    options=[{'label': f"{k} — {v}", 'value': k} for k, v in dtl.items()],
                    value='sample',
                    clearable=False
                ),
                html.Div(id='loading-status')
            ], className="data-col-left"),

            # Правая колонка: загрузка аннотаций
            html.Div([
                html.Label("Загрузить аннотации из CSV:", className="slider-label"),
                dcc.Upload(
                    id='upload-annotations',
                    children=html.Div([
                        'Перетащите файл сюда или ',
                        html.A('выберите файл')
                    ]),
                    className="upload-area",
                    multiple=False
                ),
                html.Div(id='upload-status')
            ], className="data-col-right")
        ], className="data-row")
    ], className="card"),

    # Настройки отображения
    html.Div([
        html.H3("⚙️ Настройки отображения", className="section-title"),
        html.Div([
            html.Div([
                html.Label("Канал:", className="slider-label"),
                dcc.Dropdown(id='channel-dropdown', clearable=False)
            ], className="settings-col settings-col-left"),
            html.Div([
                html.Label("Длительность окна (сек):", className="slider-label"),
                dcc.Slider(
                    id='window-slider',
                    min=5, max=60, step=1, value=10,
                    marks={i: str(i) for i in range(5, 61, 5)},
                    tooltip={"placement": "bottom"}
                ),
                html.Label("Начало окна (сек):", className="slider-label"),
                dcc.Slider(
                    id='start-slider',
                    min=0, max=100, step=0.1, value=0,
                    tooltip={"placement": "bottom"},
                    className="custom-slider"
                )
            ], className="settings-col settings-col-right")
        ], className="settings-row")
    ], className="card"),

    # График
    html.Div([
        html.H3("📊 Сигнал", className="section-title"),
        dcc.Graph(
            id='eeg-graph',
            config={
                'modeBarButtonsToRemove': [
                    'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d', 'toImage',
                ],
                'displayModeBar': True,
                'displaylogo': False,
                'toImageButtonOptions': {'format': 'png', 'filename': 'eeg_plot'}
            }
        ),
        html.Div(id='annotation-input-area', className="annotation-input-area hidden", children=[
            html.Div(id='selection-info', className="feedback"),
            html.Div([
                dcc.Input(
                    id='label-input',
                    type='text',
                    placeholder='Введите метку (например: eye_blink)',
                    className="label-input"
                ),
                html.Button(
                    '➕ Добавить аннотацию',
                    id='add-annotation-btn',
                    n_clicks=0,
                    className="add-annotation-btn"
                )
            ]),
            html.Div(id='annotation-feedback', className="feedback")
        ])
    ], className="card graph-container"),

    # Аннотации и сохранение
    html.Div([
        html.H3("📝 Аннотации", className="section-title"),
        html.Div(id='annotations-table'),
        html.Div([
            html.Button('🗑️ Очистить все', id='clear-annotations-btn', n_clicks=0, className="btn btn-clear"),
            html.Button('💾 Сохранить', id='save-local-btn', n_clicks=0, className="btn btn-save")
        ], className="btn-div"),
        html.Div(id='save-feedback', className="feedback")
    ], className="card"),

    # Хранилище
    dcc.Store(id='annotations-store', data=[]),
    dcc.Store(id='current-selection', data=None),
    dcc.Store(id='raw-info', data={}),
    dcc.Store(id='current-dataset', data='sample')
])


# === Callbacks ===

@app.callback(
    [Output('channel-dropdown', 'options'),
     Output('channel-dropdown', 'value'),
     Output('start-slider', 'max'),
     Output('loading-status', 'children'),
     Output('raw-info', 'data'),
     Output('current-dataset', 'data')],
    Input('dataset-dropdown', 'value')
)
def update_dataset(dataset_name):
    try:
        raw = load_dataset(dataset_name)
        picks = pick_types(raw.info, eeg=True, meg=False, exclude='bads')
        if len(picks) == 0:
            picks = pick_types(raw.info, eeg=False, meg=True, exclude='bads')
        if len(picks) == 0:
            picks = pick_types(raw.info, eeg=True, meg=True, exclude='bads')
        if len(picks) == 0:
            picks = list(range(len(raw.ch_names)))
        channels = [raw.ch_names[i] for i in picks]
        duration = float(raw.times[-1])
        return (
            [{'label': ch, 'value': ch} for ch in channels],
            channels[0],
            max(10, duration - 10),
            f"✅ Загружен: {dataset_name}",
            {'sfreq': raw.info['sfreq'], 'duration': duration},
            dataset_name
        )
    except Exception as e:
        return no_update, no_update, no_update, f"❌ Ошибка: {str(e)}", no_update, no_update


def parse_csv(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = b64decode(content_string)
    try:
        df = pd.read_csv(BytesIO(decoded)) if 'csv' in filename else None
    except:
        return None, "Ошибка чтения CSV"
    if df is None:
        return None, "Поддерживается только CSV"
    
    cols = df.columns.str.lower()
    df.columns = cols
    onset = offset = label = None
    for c in cols:
        if 'onset' in c or 'start' in c: onset = c
        if 'offset' in c or 'end' in c: offset = c
        if 'label' in c or 'desc' in c: label = c
    if not all([onset, offset, label]):
        if len(cols) >= 3:
            onset, offset, label = cols[0], cols[1], cols[2]
        else:
            return None, "Нужны 3 колонки: начало, конец, метка"
    anns = []
    for _, r in df.iterrows():
        try:
            t0, t1 = float(r[onset]), float(r[offset])
            if t0 < t1:
                anns.append({'t0': t0, 't1': t1, 'label': str(r[label]).strip()})
        except:
            continue
    return anns, f"✅ Загружено {len(anns)} аннотаций"


@app.callback(
    [Output('annotations-store', 'data', allow_duplicate=True),
     Output('upload-status', 'children')],
    Input('upload-annotations', 'contents'),
    State('upload-annotations', 'filename'),
    prevent_initial_call=True
)
def upload_annotations(contents, filename):
    if not contents:
        return dash.no_update, ""
    anns, msg = parse_csv(contents, filename)
    color = 'green' if anns is not None else 'red'
    return (anns, html.Div(msg, style={'color': color})) if anns is not None else (dash.no_update, html.Div(msg, style={'color': color}))


@app.callback(
    Output('eeg-graph', 'figure'),
    [Input('channel-dropdown', 'value'),
     Input('window-slider', 'value'),
     Input('start-slider', 'value'),
     Input('annotations-store', 'data'),
     Input('raw-info', 'data')],
    State('dataset-dropdown', 'value'),
    prevent_initial_call=False
)
def update_graph(channel, window_dur, start_time, annotations, raw_info, dataset_name):
    if not all([channel, raw_info, dataset_name]):
        return go.Figure()
    
    raw = load_dataset(dataset_name)
    sfreq = raw_info['sfreq']
    end_time = min(start_time + window_dur, raw_info['duration'])
    start_samp, end_samp = int(start_time * sfreq), int(end_time * sfreq)
    data, times = raw[channel, start_samp:end_samp]
    data = data.flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=data, mode='lines', line=dict(color='#2D3748', width=1)))
    
    for ann in annotations:
        t0, t1 = ann['t0'], ann['t1']
        if t1 < start_time or t0 > end_time:
            continue
        fig.add_vrect(x0=t0, x1=t1, fillcolor='#FF6B6B', opacity=0.2, layer="below")
        fig.add_annotation(x=t0, y=np.max(data), text=ann['label'], showarrow=False, font=dict(size=10, color="#FF6B6B"))

    fig.update_layout(
        title=f"Канал: {channel}",
        xaxis_title="Время (сек)",
        yaxis_title="Амплитуда (мкВ)",
        dragmode='select',
        height=400,
        margin=dict(l=50, r=30, t=40, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'color': '#2D3748'}
    )
    return fig


@app.callback(
    [Output('current-selection', 'data'),
     Output('selection-info', 'children'),
     Output('annotation-input-area', 'className')],
    Input('eeg-graph', 'selectedData'),
    State('start-slider', 'value'),
    State('window-slider', 'value'),
    prevent_initial_call=True
)
def handle_selection(sel, start, window):
    if not sel or 'range' not in sel:
        return None, "", "annotation-input-area hidden"
    x0, x1 = sel['range']['x']
    if x0 > x1: x0, x1 = x1, x0
    x0 = max(x0, start)
    x1 = min(x1, start + window)
    if x1 <= x0:
        return None, "", "annotation-input-area hidden"
    return {'x0': x0, 'x1': x1}, f"Выделен участок: {x0:.3f} – {x1:.3f} сек", "annotation-input-area"


@app.callback(
    [Output('annotations-store', 'data', allow_duplicate=True),
     Output('label-input', 'value'),
     Output('annotation-feedback', 'children')],
    Input('add-annotation-btn', 'n_clicks'),
    [State('current-selection', 'data'),
     State('label-input', 'value'),
     State('annotations-store', 'data')],
    prevent_initial_call=True
)
def add_annotation(_, sel, label, anns):
    if not sel or not label or not label.strip():
        return anns, "", "⚠️ Метка не может быть пустой!"
    new = {'t0': float(sel['x0']), 't1': float(sel['x1']), 'label': label.strip()}
    return anns + [new], "", "✅ Аннотация добавлена!"


@app.callback(
    Output('annotations-store', 'data', allow_duplicate=True),
    Input('clear-annotations-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_annotations(_):
    return []


@app.callback(
    Output('annotations-table', 'children'),
    Input('annotations-store', 'data')
)
def update_table(anns):
    if not anns:
        return html.P("Нет аннотаций", className="feedback")
    df = pd.DataFrame(anns)[['t0', 't1', 'label']]
    return html.Table([
        html.Thead(html.Tr([
            html.Th("Начало (сек)"),
            html.Th("Конец (сек)"),
            html.Th("Метка"),
            html.Th("Действие")
        ])),
        html.Tbody([
            html.Tr([
                html.Td(f"{r['t0']:.3f}"),
                html.Td(f"{r['t1']:.3f}"),
                html.Td(r['label']),
                html.Td(html.Button("❌", id={'type': 'delete-btn', 'index': i}, n_clicks=0,
                                    className="btn btn-delete"))
            ]) for i, r in df.iterrows()
        ])
    ], className="annotations-table")

@app.callback(
    Output('annotations-store', 'data', allow_duplicate=True),
    Input({'type': 'delete-btn', 'index': ALL}, 'n_clicks'),
    State('annotations-store', 'data'),
    prevent_initial_call=True
)
def delete_annotation(n_clicks_list, anns):
    if not anns:
        return anns
    # ищем индекс кнопки, по которой кликнули
    for i, n in enumerate(n_clicks_list):
        if n and n > 0:
            anns.pop(i)
            break
    return anns

@app.callback(
    Output('save-feedback', 'children'),
    Input('save-local-btn', 'n_clicks'),
    State('annotations-store', 'data'),
    State('current-dataset', 'data'),
    prevent_initial_call=True
)
def save_to_disk(_, anns, dataset):
    if not anns:
        return "⚠️ Нет аннотаций для сохранения"
    try:
        df = pd.DataFrame(anns)[['t0', 't1', 'label']].rename(columns={
            't0': 'onset', 't1': 'offset', 'label': 'description'
        })
        fname = f"{dataset}_annotations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(ospath.join(SAVE_FOLDER, fname), index=False, header=True, encoding='utf-8')
        return f"✅ Сохранено: {fname}"
    except Exception as e:
        return f"❌ Ошибка: {str(e)[:50]}"


if __name__ == '__main__':
    app.run(debug=True)