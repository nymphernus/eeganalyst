from dataset_loader import load_dataset, datasets_list as dtl

import dash
from dash import dcc, html, Input, Output, State, no_update
from dash.dependencies import ALL

import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json

from base64 import b64decode
from io import BytesIO
from datetime import datetime
from os import path as ospath, makedirs
from mne import pick_types, Annotations

# === Настройки приложения ===
SAVE_FOLDER = "saved_annotations"
AUTOSAVE_FOLDER = "autosave"
makedirs(SAVE_FOLDER, exist_ok=True)
makedirs(AUTOSAVE_FOLDER, exist_ok=True)

STEP_SECONDS = 5.0
DEFAULT_DECIM = 1

app = dash.Dash(__name__)
app.title = "Разметка ЭЭГ"

# --- Вспомогательные функции ---

def label_to_color(label: str) -> str:
    """Создаёт цвет для метки."""
    h = abs(hash(label)) % 360
    s = 65 + (abs(hash(label + 's')) % 20)
    l = 45 + (abs(hash(label + 'l')) % 10)
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h/360.0, l/100.0, s/100.0)
    return '#{0:02x}{1:02x}{2:02x}'.format(int(r*255), int(g*255), int(b*255))


def _normalize_annotation_entry(entry: dict):
    """Преобразует запись в формат:
    {'onset': float, 'duration': float, 'description': str}
    """
    if not isinstance(entry, dict):
        return None
    try:
        onset = float(entry['onset'])
        duration = float(entry['duration'])
        description = str(entry['description'])
        if duration <= 0:
            return None
        return {'onset': onset, 'duration': duration, 'description': description}
    except (KeyError, ValueError, TypeError):
        return None


def anns_list_to_mne(anns_list):
    """Превращает список аннотаций в формат MNE."""
    if not anns_list:
        return Annotations([], [], [])
    onsets = [float(a['onset']) for a in anns_list]
    durations = [float(a['duration']) for a in anns_list]
    descriptions = [str(a['description']) for a in anns_list]
    return Annotations(onsets=onsets, durations=durations, description=descriptions)


def mne_annotations_to_list(anns: Annotations):
    """Превращает аннотации MNE обратно в наш список."""
    if anns is None:
        return []
    return [{'onset': float(onset), 'duration': float(dur), 'description': desc}
            for onset, dur, desc in zip(anns.onset, anns.duration, anns.description)]


def parse_csv_or_json(contents, filename):
    """
    Загружает аннотации из CSV или JSON в формате:
    [{'onset':..., 'duration':..., 'description':...}]
    """
    if not contents:
        return None, "Нет содержимого"
    try:
        content_type, content_string = contents.split(',')
    except Exception:
        return None, "Неправильный формат contents"
    decoded = b64decode(content_string)
    txt = decoded.decode('utf-8', errors='ignore')

    # ---- JSON ----
    if filename.lower().endswith('.json'):
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict) and 'annotations' in obj:
                obj = obj['annotations']
            if not isinstance(obj, list):
                return None, "JSON должен содержать список аннотаций"
            anns = []
            for it in obj:
                new = _normalize_annotation_entry(it)
                if new is not None:
                    anns.append(new)
            return anns, f"✅ Загружено {len(anns)} аннотаций (JSON)"
        except Exception as e:
            return None, f"Ошибка парсинга JSON: {str(e)[:100]}"

    # ---- CSV ----
    try:
        df = pd.read_csv(BytesIO(decoded))
    except Exception as e:
        return None, f"Ошибка чтения CSV: {str(e)[:120]}"
    if df is None or df.shape[1] < 3:
        return None, "CSV должен содержать как минимум 3 колонки"

    cols = [c.lower() for c in df.columns]
    df.columns = cols

    required = ['onset', 'duration', 'description']
    if not all(c in cols for c in required):
        return None, "CSV должен содержать колонки: onset, duration, description"

    anns = []
    for _, r in df.iterrows():
        try:
            t0 = float(r['onset'])
            dur = float(r['duration'])
            lab = str(r['description']).strip()
            if dur > 0:
                anns.append({'onset': t0, 'duration': dur, 'description': lab})
        except Exception:
            continue
    return anns, f"✅ Загружено {len(anns)} аннотаций (CSV)"


def autosave_write(dataset_name, anns):
    """Автосохранение в формате onset/duration/description."""
    fname = ospath.join(AUTOSAVE_FOLDER, f"autosave_{dataset_name}.json")
    try:
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump({'annotations': anns, 'saved_at': datetime.now().isoformat()},
                      f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def autosave_read(dataset_name):
    """Загрузка автосохранения (только новый формат)."""
    fname = ospath.join(AUTOSAVE_FOLDER, f"autosave_{dataset_name}.json")
    try:
        if not ospath.exists(fname):
            return []
        with open(fname, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        anns = obj.get('annotations', [])
        out = []
        for a in anns:
            new = _normalize_annotation_entry(a)
            if new is not None:
                out.append(new)
        return out
    except Exception:
        return []


# --- Макет интерфейса ---

app.layout = html.Div([
    html.Div([
        html.H3("📁 Данные", className="section-title"),
        html.Div([
            html.Div([
                html.Label("Выберите датасет:", className="slider-label"),
                dcc.Dropdown(
                    id='dataset-dropdown',
                    options=[{'label': f"{k} — {v}", 'value': k} for k, v in dtl.items()],
                    value='sample',
                    clearable=False
                ),
                html.Div(id='loading-status')
            ], className="data-col"),
            html.Div([
                html.Label("Загрузить аннотации (CSV или JSON):", className="slider-label"),
                dcc.Upload(
                    id='upload-annotations',
                    children=html.Div(['Перетащите файл сюда или ', html.A('выберите файл')]),
                    className="upload-area",
                    multiple=False
                ),
                html.Div(id='upload-status')
            ], className="data-col")
        ], className="data-row")
    ], className="card"),

    html.Div([
        html.H3("⚙️ Настройки", className="section-title"),
        html.Div([
            html.Div([
                html.Label("Каналы:"),
                dcc.Dropdown(id='channel-dropdown', clearable=False, multi=True)
            ], className="settings-col settings-col-left"),
            html.Div([
                html.Label("Длительность окна (сек):"),
                dcc.Slider(id='window-slider', min=2, max=120, step=1, value=10,
                           marks={i: str(i) for i in range(2,121,10)}),
                html.Br(),
                html.Label("Начало окна (сек):"),
                dcc.Slider(id='start-slider', min=0, max=100, step=0.1, value=0,
                           tooltip={"placement":"bottom"}),
                html.Br(),
                html.Label("Децимация:"),
                dcc.Slider(id='decim-slider', min=1, max=20, step=1, value=DEFAULT_DECIM,
                           marks={1:'1',2:'2',5:'5',10:'10',20:'20'})
            ], className="settings-col settings-col-right"),
        ], className="settings-row")
    ], className="card"),

    html.Div([
        html.H3("📊 Сигнал", className="section-title"),
        html.Div([
            html.Button('⏮ Назад', id='seek-back-btn', n_clicks=0, className="seek-btn"),
            html.Button('⏭ Вперёд', id='seek-forward-btn', n_clicks=0, className="seek-btn"),
            html.Span(id='export-feedback')
        ], className="seek-controls"),
        dcc.Graph(id='eeg-graph', config={
            'modeBarButtonsToRemove': ['zoom2d','pan2d','zoomIn2d','zoomOut2d',
                                       'autoScale2d','resetScale2d','toImage'],
            'displayModeBar': True, 'displaylogo': False
        }),
        html.Div(id='annotation-input-area', className="annotation-input-area hidden", children=[
            html.Div(id='selection-info', className="feedback"),
            html.Div([
                dcc.Input(id='label-input', type='text', placeholder='Метка (например: eye_blink)', className="label-input"),
                html.Button('+ Добавить метку', id='add-annotation-btn', n_clicks=0, className="btn-add")
            ], className="annotation-form"),
            html.Div(id='annotation-feedback', className="feedback")
        ])
    ], className="card graph-container"),

    html.Div([
        html.H3("📝 Аннотации", className="section-title"),
        html.Div(id='annotations-table'),
        html.Div([
            html.Button('🗑️ Очистить все', id='clear-annotations-btn', n_clicks=0, className="btn"),
            html.Button('💾 Сохранить CSV', id='save-local-csv-btn', n_clicks=0, className="btn"),
            html.Button('💾 Сохранить JSON', id='save-local-json-btn', n_clicks=0, className="btn"),
        ], className="save-buttons"),
        html.Div(id='save-feedback', className="feedback")
    ], className="card"),

    dcc.Store(id='annotations-store', data=[]),
    dcc.Store(id='current-selection', data=None),
    dcc.Store(id='raw-info', data={}),
    dcc.Store(id='current-dataset', data='sample'),
    dcc.Store(id='last-delete-idx', data=None)
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
        autos = autosave_read(dataset_name)
        msg = f"✅ Загружен: {dataset_name} | Каналов: {len(channels)}"
        if autos:
            msg += f" | Найдено автосохранение ({len(autos)} аннотаций)."
        return (
            [{'label': ch, 'value': ch} for ch in channels],
            [channels[0]] if channels else None,
            max(10, duration - 10),
            msg,
            {'sfreq': raw.info['sfreq'], 'duration': duration},
            dataset_name
        )
    except Exception as e:
        return no_update, no_update, no_update, f"❌ Ошибка: {str(e)}", no_update, no_update


@app.callback(
    [Output('annotations-store', 'data', allow_duplicate=True),
     Output('upload-status', 'children')],
    Input('upload-annotations', 'contents'),
    State('upload-annotations', 'filename'),
    State('current-dataset', 'data'),
    prevent_initial_call=True
)
def upload_annotations(contents, filename, dataset):
    if not contents:
        return dash.no_update, ""
    anns, msg = parse_csv_or_json(contents, filename)
    color = 'green' if anns is not None else 'red'
    if anns is None:
        return dash.no_update, html.Div(msg, style={'color':color})
    autosave_write(dataset, anns)
    return anns, html.Div(msg, style={'color':color})


@app.callback(
    Output('eeg-graph', 'figure'),
    [Input('channel-dropdown', 'value'),
     Input('window-slider', 'value'),
     Input('start-slider', 'value'),
     Input('annotations-store', 'data'),
     Input('raw-info', 'data'),
     Input('decim-slider', 'value')],
    State('current-dataset', 'data'),
    prevent_initial_call=False
)
def update_graph(channels, window_dur, start_time, annotations, raw_info, decim, dataset_name):
    if not all([channels, raw_info, dataset_name]):
        return go.Figure()
    if isinstance(channels, str):
        channels = [channels]

    raw = load_dataset(dataset_name)
    sfreq = raw_info['sfreq']
    duration = raw_info['duration']
    end_time = min(start_time + window_dur, duration)
    start_samp = int(start_time * sfreq)
    end_samp = int(end_time * sfreq)

    picks = [raw.ch_names.index(ch) for ch in channels]
    full_data = raw.get_data(picks=picks)
    global_max_ampl = np.max(np.abs(full_data)) if full_data.size > 0 else 1.0
    separation = global_max_ampl * 3.0

    window_data = raw.get_data(picks=picks, start=start_samp, stop=end_samp)
    times = np.arange(start_samp, end_samp) / sfreq

    if decim > 1:
        window_data = window_data[:, ::decim]
        times = times[::decim]

    fig = go.Figure()
    offsets = [i * separation for i in range(len(channels))]

    for i, ch in enumerate(channels):
        y = window_data[i, :] + offsets[i]
        fig.add_trace(go.Scatter(x=times, y=y, mode='lines', name=ch, line=dict(width=1)))

    for ann in (annotations or []):
        try:
            t0 = float(ann['onset'])
            dur = float(ann['duration'])
        except Exception:
            continue
        t1 = t0 + dur
        if t1 < start_time or t0 > end_time:
            continue
        color = label_to_color(ann['description'])
        fig.add_vrect(x0=t0, x1=t1, fillcolor=color, opacity=0.25, layer="below", line_width=0)
        fig.add_annotation(
            x=max(t0, start_time),
            y=offsets[-1] + global_max_ampl if offsets else global_max_ampl,
            text=ann['description'],
            showarrow=False,
            font=dict(size=10, color=color)
        )

    total_offset = offsets[-1] if offsets else 0
    y_min = -global_max_ampl
    y_max = total_offset + global_max_ampl

    fig.update_layout(
        title=f"Каналы: {', '.join(channels)}",
        xaxis_title="Время (сек)",
        yaxis=dict(showticklabels=False, range=[y_min, y_max]),
        dragmode='select',
        height=600,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white'
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
    return {'x0': float(x0), 'x1': float(x1)}, f"Выделен участок: {x0:.3f} – {x1:.3f} сек", "annotation-input-area"


@app.callback(
    [Output('annotations-store', 'data', allow_duplicate=True),
     Output('annotation-feedback', 'children')],
    Input('add-annotation-btn', 'n_clicks'),
    [State('current-selection', 'data'),
     State('label-input', 'value'),
     State('annotations-store', 'data'),
     State('current-dataset', 'data')],
    prevent_initial_call=True
)
def add_annotation(nc, sel, label, anns, dataset):
    if not sel:
        return anns, "⚠️ Нет выделения для добавления"
    if not label or not label.strip():
        return anns, "⚠️ Метка пустая"
    onset = float(sel['x0'])
    duration = float(sel['x1']) - float(sel['x0'])
    if duration <= 0:
        return anns, "⚠️ Неправильная длительность"
    new = {'onset': onset, 'duration': duration, 'description': label.strip()}
    out = (anns or []) + [new]
    autosave_write(dataset, out)
    return out, "✅ Добавлено"


@app.callback(
    Output('annotations-store', 'data', allow_duplicate=True),
    Input('clear-annotations-btn', 'n_clicks'),
    State('current-dataset', 'data'),
    prevent_initial_call=True
)
def clear_annotations(_n, dataset):
    autosave_write(dataset, [])
    return []


@app.callback(
    Output('annotations-table', 'children'),
    Input('annotations-store', 'data')
)
def update_table(anns):
    if not anns:
        return html.P("Нет аннотаций", className="feedback")
    rows = []
    rows.append(html.Thead(html.Tr([
        html.Th("№"),
        html.Th("Начало (сек)"),
        html.Th("Длительность (сек)"),
        html.Th("Метка"),
        html.Th("Действие")
    ])))
    body_trs = []
    for i, a in enumerate(anns):
        onset_in = dcc.Input(
            id={'type': 'edit-onset', 'index': i},
            value=f"{a['onset']:.3f}",
            type='number',
            step=0.001,
            className="edit-input"
        )
        dur_in = dcc.Input(
            id={'type': 'edit-duration', 'index': i},
            value=f"{a['duration']:.3f}",
            type='number',
            step=0.001,
            min=0.001,
            className="edit-input"
        )
        label_in = dcc.Input(
            id={'type': 'edit-description', 'index': i},
            value=a['description'],
            type='text',
            className="edit-label-input"
        )
        delete_btn = html.Button("❌", id={'type': 'delete-btn', 'index': i}, n_clicks=0, className="delete-btn")
        body_trs.append(html.Tr([
            html.Td(str(i)),
            html.Td(onset_in),
            html.Td(dur_in),
            html.Td(label_in),
            html.Td(delete_btn)
        ]))
    rows.append(html.Tbody(body_trs))
    return html.Table(rows, className="annotations-table")


@app.callback(
    Output('annotations-store', 'data', allow_duplicate=True),
    Input({'type':'delete-btn','index': ALL}, 'n_clicks'),
    State('annotations-store', 'data'),
    State('current-dataset', 'data'),
    prevent_initial_call=True
)
def delete_annotation(n_clicks_list, anns, dataset):
    if not anns:
        return anns
    for i, n in enumerate(n_clicks_list):
        if n and n > 0:
            new = list(anns)
            if 0 <= i < len(new):
                new.pop(i)
                autosave_write(dataset, new)
                return new
    return anns


@app.callback(
    Output('annotations-store', 'data', allow_duplicate=True),
    Input({'type':'edit-onset','index': ALL}, 'value'),
    State('annotations-store', 'data'),
    State('current-dataset', 'data'),
    prevent_initial_call=True
)
def edit_onset(values, anns, dataset):
    if anns is None:
        return anns
    new = list(anns)
    updated = False
    for i, v in enumerate(values):
        try:
            vf = float(v)
        except Exception:
            continue
        if 0 <= i < len(new):
            new[i]['onset'] = float(vf)
            updated = True
    if updated:
        autosave_write(dataset, new)
    return new


@app.callback(
    Output('annotations-store', 'data', allow_duplicate=True),
    Input({'type':'edit-duration','index': ALL}, 'value'),
    State('annotations-store', 'data'),
    State('current-dataset', 'data'),
    prevent_initial_call=True
)
def edit_duration(values, anns, dataset):
    if anns is None:
        return anns
    new = list(anns)
    updated = False
    for i, v in enumerate(values):
        try:
            vf = float(v)
        except Exception:
            continue
        if 0 <= i < len(new) and vf > 0:
            new[i]['duration'] = float(vf)
            updated = True
    if updated:
        autosave_write(dataset, new)
    return new


@app.callback(
    Output('annotations-store', 'data', allow_duplicate=True),
    Input({'type':'edit-description','index': ALL}, 'value'),
    State('annotations-store', 'data'),
    State('current-dataset', 'data'),
    prevent_initial_call=True
)
def edit_description(values, anns, dataset):
    if anns is None:
        return anns
    new = list(anns)
    updated = False
    for i, v in enumerate(values):
        if 0 <= i < len(new) and v != new[i]['description']:
            new[i]['description'] = str(v)
            updated = True
    if updated:
        autosave_write(dataset, new)
    return new


@app.callback(
    Output('start-slider', 'value'),
    [Input('seek-back-btn', 'n_clicks'),
     Input('seek-forward-btn', 'n_clicks')],
    State('start-slider', 'value'),
    prevent_initial_call=True
)
def seek(back, forward, start_val):
    ctx = dash.callback_context
    if not ctx.triggered:
        return start_val
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'seek-back-btn':
        new = max(0.0, start_val - STEP_SECONDS)
    elif button_id == 'seek-forward-btn':
        new = start_val + STEP_SECONDS
    else:
        new = start_val
    return float(new)


@app.callback(
    Output('save-feedback', 'children'),
    [Input('save-local-csv-btn', 'n_clicks'),
     Input('save-local-json-btn', 'n_clicks')],
    [State('annotations-store', 'data'),
     State('current-dataset', 'data')],
    prevent_initial_call=True
)
def save_to_disk(save_csv, save_json, anns, dataset):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    which = ctx.triggered[0]['prop_id'].split('.')[0]
    if not anns:
        return "⚠️ Нет аннотаций для сохранения"
    df = pd.DataFrame(anns)[['onset','duration','description']]
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        if which == 'save-local-csv-btn':
            fname = f"{dataset}_annotations_{now}.csv"
            df.to_csv(ospath.join(SAVE_FOLDER, fname), index=False, encoding='utf-8')
            return f"✅ CSV сохранён: {fname}"
        elif which == 'save-local-json-btn':
            fname = f"{dataset}_annotations_{now}.json"
            j = df.to_dict(orient='records')
            with open(ospath.join(SAVE_FOLDER, fname), 'w', encoding='utf-8') as f:
                json.dump(j, f, ensure_ascii=False, indent=2)
            return f"✅ JSON сохранён: {fname}"
    except Exception as e:
        return f"❌ Ошибка: {str(e)[:100]}"
    return ""


@app.callback(
    Output('export-feedback', 'children'),
    Input('annotations-store', 'data'),
    State('current-dataset', 'data'),
    prevent_initial_call=False
)
def autosave_callback(anns, dataset):
    ok = autosave_write(dataset, anns or [])
    return "" if ok else "Autosave failed"


@app.callback(
    Output('annotations-store', 'data', allow_duplicate=True),
    Input('dataset-dropdown', 'value'),
    State('annotations-store', 'data'),
    prevent_initial_call=True
)
def load_autosave(dataset, current):
    autos = autosave_read(dataset)
    if autos and (not current or len(current)==0):
        return autos
    return current


if __name__ == '__main__':
    app.run(debug=True)