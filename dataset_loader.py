import mne
from functools import lru_cache

datasets_list = {
    "sample": "Стандартный EEG (auditory & visual)",
    "eegbci": "Motor Imagery BCI (EEG, 64 канала, движения рук/ног)",
}

@lru_cache(maxsize=3)
def load_dataset(name: str):
    """
    Загружает датасет из MNE с кэшированием.
    """
    if name == "sample":
        data_path = mne.datasets.sample.data_path()
        raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
        raw = mne.io.read_raw_fif(raw_fname, preload=True)

    elif name == "eegbci":
        from mne.datasets import eegbci
        files = eegbci.load_data(1, [2])
        raw = mne.io.read_raw_edf(files[0], preload=True)

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return raw