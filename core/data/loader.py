import mne
import pandas as pd
from .sample import EEGSample


def load_edf(path: str) -> EEGSample:
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    data, times = raw.get_data(return_times=True)
    return EEGSample(
        data=data,
        sfreq=raw.info['sfreq'],
        ch_names=raw.ch_names,
        raw_path=path,
        metadata={"n_times": len(times)}
    )


def load_csv(path: str, sfreq: float, has_labels: bool = False) -> EEGSample:
    df = pd.read_csv(path)
    
    label_column = None
    if has_labels:
        possible_labels = ['label', 'Label', 'labels', 'Labels', 'target', 'Target']
        for col in possible_labels:
            if col in df.columns:
                label_column = col
                break
    
    if has_labels and label_column:
        ch_names = df.columns.drop(label_column).tolist()
        labels = df[label_column].to_numpy()
    else:
        ch_names = df.columns.tolist()
        labels = None
    
    data = df[ch_names].to_numpy().T
    
    return EEGSample(
        data=data,
        sfreq=sfreq,
        ch_names=ch_names,
        labels=labels,
        raw_path=path,
        metadata={"n_samples": len(df)}
    )