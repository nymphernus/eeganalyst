import numpy as np
from typing import Dict, Any

class FeatureExtractor:
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "params": self.params}