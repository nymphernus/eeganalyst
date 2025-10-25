from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Dict

@dataclass
class EEGSample:
    data: np.ndarray  # (channels, time)
    sfreq: float      # частота дискретизации
    ch_names: List[str]
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    task: Optional[str] = None
    labels: Optional[np.ndarray] = None  # классы (если есть)
    raw_path: Optional[str] = None       # путь к исходному файлу
    metadata: Optional[Dict] = None      # любые доп. атрибуты
