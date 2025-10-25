import hashlib
import numpy as np

def compute_file_hash(file_bytes: bytes, algo: str = "md5") -> str:
    if algo == "sha256":
        return hashlib.sha256(file_bytes).hexdigest()
    return hashlib.md5(file_bytes).hexdigest()

def compute_array_hash(data: np.ndarray, algo: str = "md5") -> str:
    if algo == "sha256":
        return hashlib.sha256(data.tobytes()).hexdigest()
    return hashlib.md5(data.tobytes()).hexdigest()
