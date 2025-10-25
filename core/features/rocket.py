import numpy as np
from .base import FeatureExtractor

class MiniRocketExtractor(FeatureExtractor):
    def __init__(self, random_state: int = 42, n_jobs: int = 1):
        super().__init__("MiniRocket", {"random_state": random_state, "n_jobs": n_jobs})
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        self._check_sktime()

    def _check_sktime(self):
        try:
            from sktime.transformations.panel.rocket import MiniRocketMultivariate
            self.model_cls = MiniRocketMultivariate
        except Exception:
            try:
                from sktime.transformations.panel.rocket import MiniRocket
                self.model_cls = MiniRocket
            except Exception as e:
                raise ImportError("sktime MiniRocket is required for MiniRocketExtractor. Install sktime.") from e

    def fit(self, X: np.ndarray, y=None):
        X_in = self._prepare_X(X)
        self.model = self.model_cls(random_state=self.random_state, n_jobs=self.n_jobs)
        try:
            self.model.fit(X_in)
        except Exception:
            self.model.fit(X_in)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_in = self._prepare_X(X)
        feats = self.model.transform(X_in)
        return np.asarray(feats)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            X_in = np.expand_dims(X, axis=0)
        elif X.ndim == 3:
            X_in = np.transpose(X, (1, 0, 2))
        else:
            raise ValueError("Unsupported X shape for MiniRocketExtractor")
        return X_in
