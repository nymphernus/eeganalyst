import numpy as np
from typing import Optional, Tuple
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except Exception:
    torch = None
    TorchDataset = None
try:
    import tensorflow as tf
except Exception:
    tf = None

from storage.eeg_database import EEGDatabase

db = EEGDatabase()

def load_features(feat_id: str) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    data = db.get_features_data(feat_id)
    if data is None:
        return None
    X, y = data
    return (np.array(X), None if y is None else np.array(y))

def to_torch_dataset(X: np.ndarray, y: Optional[np.ndarray] = None):
    if torch is None:
        raise ImportError("PyTorch not available")
    class _TDS(TorchDataset):
        def __init__(self, X, y=None):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.int64) if y is not None else None
        def __len__(self):
            return self.X.shape[0]
        def __getitem__(self, idx):
            if self.y is None:
                return self.X[idx]
            return self.X[idx], self.y[idx]
    return _TDS(X, y)

def to_tf_dataset(X: np.ndarray, y: Optional[np.ndarray] = None, batch_size: int = 32, shuffle: bool = True):
    if tf is None:
        raise ImportError("TensorFlow not available")
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    else:
        ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int64)))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1024, X.shape[0]))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
