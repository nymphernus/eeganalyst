import yaml, json
import numpy as np
from typing import List
from core.preprocess.base import Transform
from core.preprocess.steps.filters import BandpassFilter, NotchFilter
from core.preprocess.steps.resample_normalize import Resample, Normalize
from core.preprocess.steps.artifacts import ReReference, ICAFilter, Epoching

STEP_REGISTRY = {
    "bandpass": BandpassFilter,
    "notch": NotchFilter,
    "resample": Resample,
    "normalize": Normalize,
    "rereference": ReReference,
    "ica": ICAFilter,
    "epoching": Epoching,
}

class PreprocessPipeline:
    def __init__(self, steps: List[Transform] = None, random_state: int = 42):
        self.steps = steps or []
        self.random_state = random_state

    def add_step(self, step: Transform):
        self.steps.append(step)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        for step in self.steps:
            X = step.fit_transform(X, y)
        return X

    def to_dict(self):
        return {
            "random_state": self.random_state,
            "steps": [s.to_dict() for s in self.steps]
        }