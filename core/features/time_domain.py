import numpy as np
from .base import FeatureExtractor
from scipy.stats import skew, kurtosis

class TimeDomainExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__("TimeDomain", {})

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X_in = np.expand_dims(X, axis=0)
        elif X.ndim == 3:
            X_in = np.transpose(X, (1, 0, 2))
        else:
            raise ValueError("Unsupported X shape for TimeDomainExtractor")

        feats = []
        for inst in X_in:
            inst_feats = []
            for ch in inst:
                arr = ch
                mean = np.mean(arr)
                std = np.std(arr)
                sk = float(skew(arr))
                kurt = float(kurtosis(arr))
                ptp = float(np.ptp(arr))
                median = float(np.median(arr))
                q75, q25 = np.percentile(arr, [75, 25])
                iqr = float(q75 - q25)
                inst_feats.extend([mean, std, sk, kurt, ptp, median, iqr])
            feats.append(np.array(inst_feats))
        return np.asarray(feats)
