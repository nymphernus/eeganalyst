import numpy as np
from scipy.signal import resample
from core.preprocess.base import Transform

class Resample(Transform):
    def __init__(self, target_rate: float, orig_rate: float):
        super().__init__("resample", {"target_rate": target_rate, "orig_rate": orig_rate})

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        factor = self.params["target_rate"] / self.params["orig_rate"]
        new_len = int(X.shape[-1] * factor)
        return resample(X, new_len, axis=-1)

class Normalize(Transform):
    def __init__(self, method: str = "zscore"):
        super().__init__("normalize", {"method": method})

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        if self.params["method"] == "zscore":
            return (X - np.mean(X, axis=-1, keepdims=True)) / (np.std(X, axis=-1, keepdims=True) + 1e-8)
        elif self.params["method"] == "minmax":
            Xmin = X.min(axis=-1, keepdims=True)
            Xmax = X.max(axis=-1, keepdims=True)
            return (X - Xmin) / (Xmax - Xmin + 1e-8)
        else:
            return X
