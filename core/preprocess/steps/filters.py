import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from core.preprocess.base import Transform

class BandpassFilter(Transform):
    def __init__(self, low: float, high: float, sfreq: float, order: int = 5):
        super().__init__("bandpass", {"low": low, "high": high, "sfreq": sfreq, "order": order})

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        nyq = 0.5 * self.params["sfreq"]
        low = self.params["low"] / nyq
        high = self.params["high"] / nyq
        b, a = butter(self.params["order"], [low, high], btype="band")
        return filtfilt(b, a, X, axis=-1)

class NotchFilter(Transform):
    def __init__(self, freq: float, sfreq: float, Q: float = 30.0):
        super().__init__("notch", {"freq": freq, "sfreq": sfreq, "Q": Q})

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        b, a = iirnotch(self.params["freq"], self.params["Q"], self.params["sfreq"])
        return filtfilt(b, a, X, axis=-1)
