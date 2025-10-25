import numpy as np
import mne
from core.preprocess.base import Transform

class ReReference(Transform):
    def __init__(self, method: str = "average"):
        super().__init__("rereference", {"method": method})

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        info = mne.create_info(ch_names=[f"ch{i}" for i in range(X.shape[0])], sfreq=256, ch_types="eeg")
        raw = mne.io.RawArray(X, info, verbose=False)
        if self.params["method"] == "average":
            raw.set_eeg_reference("average", verbose=False)
        elif self.params["method"] == "mastoid":
            raw.set_eeg_reference(["ch0", "ch1"], verbose=False)
        return raw.get_data()

class ICAFilter(Transform):
    def __init__(self, n_components: int = 15, random_state: int = 42):
        super().__init__("ica", {"n_components": n_components, "random_state": random_state})

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        info = mne.create_info([f"ch{i}" for i in range(X.shape[0])], sfreq=256, ch_types="eeg")
        raw = mne.io.RawArray(X, info, verbose=False)
        ica = mne.preprocessing.ICA(n_components=self.params["n_components"], random_state=self.params["random_state"], verbose=False)
        ica.fit(raw)
        raw_clean = ica.apply(raw.copy(), verbose=False)
        return raw_clean.get_data()

class Epoching(Transform):
    def __init__(self, sfreq: float, epoch_length: float = 2.0):
        super().__init__("epoching", {"sfreq": sfreq, "epoch_length": epoch_length})

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        n_samples = X.shape[-1]
        win = int(self.params["sfreq"] * self.params["epoch_length"])
        n_epochs = n_samples // win
        return X[:, :n_epochs * win].reshape(X.shape[0], n_epochs, win)