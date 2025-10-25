from typing import Dict, Any
import numpy as np

class Transform:
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X, y)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "params": self.params}