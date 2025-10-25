import numpy as np
from .base import FeatureExtractor
from scipy.signal import welch

class PSDExtractor(FeatureExtractor):
    def __init__(self, sfreq: float, fmin: float = 1.0, fmax: float = 40.0, nperseg: int = 256):
        super().__init__("PSD", {"sfreq": sfreq, "fmin": fmin, "fmax": fmax, "nperseg": nperseg})
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.nperseg = nperseg

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X_in = np.expand_dims(X, axis=0)
        elif X.ndim == 3:
            X_in = np.transpose(X, (1, 0, 2))
        else:
            raise ValueError("Unsupported X shape for PSDExtractor")

        feats = []
        for inst in X_in:
            ch_psds = []
            for ch in inst:
                f, p = welch(ch, fs=self.sfreq, nperseg=min(self.nperseg, ch.shape[0]))
                idx = np.where((f >= self.fmin) & (f <= self.fmax))[0]
                ch_psds.append(p[idx])
            feats.append(np.concatenate(ch_psds, axis=0))
        return np.asarray(feats)


class BandPowerExtractor(FeatureExtractor):
    DEFAULT_BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
    }

    def __init__(self, sfreq: float, bands: dict = None, nperseg: int = 256):
        bands = bands or self.DEFAULT_BANDS
        super().__init__("BandPower", {"sfreq": sfreq, "bands": bands, "nperseg": nperseg})
        self.sfreq = sfreq
        self.bands = bands
        self.nperseg = nperseg

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X_in = np.expand_dims(X, axis=0)
        elif X.ndim == 3:
            X_in = np.transpose(X, (1, 0, 2))
        else:
            raise ValueError("Unsupported X shape for BandPowerExtractor")

        feats = []
        for inst in X_in:
            inst_feats = []
            for ch in inst:
                f, p = welch(ch, fs=self.sfreq, nperseg=min(self.nperseg, ch.shape[0]))
                for (low, high) in self.bands.values():
                    idx = np.logical_and(f >= low, f <= high)
                    band_power = np.trapz(p[idx], f[idx]) if np.any(idx) else 0.0
                    inst_feats.append(band_power)
            feats.append(np.array(inst_feats))
        return np.asarray(feats)
