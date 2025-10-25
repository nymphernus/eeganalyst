import sqlite3
import json
import os
import uuid
from typing import Optional, Dict, List
import numpy as np
from datetime import datetime, timezone
import time

class EEGDatabase:
    def __init__(self, db_path: str = "storage/raw.db"):
        self.db_path = db_path
        self.data_dir = "storage/raw_data"
        self._init_storage()
        self._init_database()
        self.timezone_offset = self._get_local_timezone_offset()
    
    def _get_local_timezone_offset(self) -> int:
        local_time = time.localtime()
        utc_time = time.gmtime()
        
        offset_seconds = time.mktime(local_time) - time.mktime(utc_time)
        offset_hours = int(offset_seconds / 3600)
        return offset_hours
    
    def _init_storage(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _init_database(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE,
                    sfreq REAL,
                    n_channels INTEGER,
                    n_samples INTEGER,
                    ch_names TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS processed_datasets (
                    id TEXT PRIMARY KEY,
                    parent_id TEXT NOT NULL,
                    pipeline_config TEXT NOT NULL,
                    sfreq REAL,
                    n_channels INTEGER,
                    n_samples INTEGER,
                    ch_names TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(parent_id) REFERENCES datasets(id)
                );
                CREATE TABLE IF NOT EXISTS features_datasets (
                        id TEXT PRIMARY KEY,
                        parent_id TEXT NOT NULL,
                        extractor_config TEXT NOT NULL,
                        X_shape TEXT,
                        y_shape TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(parent_id) REFERENCES processed_datasets(id)
                    );
                
                CREATE INDEX IF NOT EXISTS idx_file_hash ON datasets(file_hash);
                CREATE INDEX IF NOT EXISTS idx_created_at ON datasets(created_at);
                CREATE INDEX IF NOT EXISTS idx_proc_parent ON processed_datasets(parent_id);
                CREATE INDEX IF NOT EXISTS idx_proc_created ON processed_datasets(created_at);
                CREATE INDEX IF NOT EXISTS idx_features_parent ON features_datasets(parent_id);
            """)
    
    def _format_datetime(self, dt_string: str) -> str:
        if not dt_string:
            return ""
        try:
            if 'T' in dt_string:
                if 'Z' in dt_string:
                    dt = datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
                else:
                    dt = datetime.fromisoformat(dt_string)
            else:
                dt = datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S')
            
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            local_dt = dt.astimezone()
            return local_dt.strftime('%d.%m.%Y %H:%M')
        except Exception as e:
            return dt_string
    
    def _save_data_file(self, data_id: str, data: np.ndarray) -> str:
        file_path = os.path.join(self.data_dir, f"{data_id}.npy")
        np.save(file_path, data, allow_pickle=False)
        return file_path
    
    def _load_data_file(self, data_id: str) -> Optional[np.ndarray]:
        file_path = os.path.join(self.data_dir, f"{data_id}.npy")
        return np.load(file_path, mmap_mode="r") if os.path.exists(file_path) else None
    
    def _remove_data_file(self, data_id: str) -> bool:
        file_path = os.path.join(self.data_dir, f"{data_id}.npy")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    
    def add_dataset(self, filename: str, file_hash: str, 
                   sfreq: float, n_channels: int, n_samples: int, 
                   ch_names: list, data: np.ndarray, metadata: dict = None) -> str:
        dataset_id = str(uuid.uuid4())
        
        self._save_data_file(dataset_id, data)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO datasets 
                    (id, filename, file_hash, sfreq, n_channels, n_samples, ch_names, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dataset_id, filename, file_hash,
                    sfreq, n_channels, n_samples,
                    json.dumps(ch_names),
                    json.dumps(metadata) if metadata else None
                ))
            return dataset_id
        except sqlite3.IntegrityError:
            self._remove_data_file(dataset_id)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT id FROM datasets WHERE file_hash = ?", (file_hash,))
                row = cursor.fetchone()
                return row[0] if row else None
   
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
                
            return {
                'id': row['id'],
                'filename': row['filename'],
                'file_hash': row['file_hash'],
                'sfreq': row['sfreq'],
                'n_channels': row['n_channels'],
                'n_samples': row['n_samples'],
                'ch_names': json.loads(row['ch_names']),
                'metadata': json.loads(row['metadata']) if row['metadata'] else None,
                'created_at': row['created_at']
            }
    
    def get_dataset_data(self, dataset_id: str) -> Optional[np.ndarray]:
        return self._load_data_file(dataset_id)
    
    def list_datasets(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, filename, sfreq, n_channels, n_samples, file_hash, created_at
                FROM datasets 
                ORDER BY created_at DESC
            """)
            
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                row_dict['created_at_formatted'] = self._format_datetime(row_dict['created_at'])
                results.append(row_dict)
            return results
    
    def delete_dataset(self, dataset_id: str) -> bool:
        self._remove_data_file(dataset_id)
        
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            return result.rowcount > 0
    
    def dataset_exists(self, file_hash: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id FROM datasets WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def add_processed_dataset(self, parent_id: str, sample, pipeline_cfg: dict) -> str:
        proc_id = str(uuid.uuid4())
        
        self._save_data_file(proc_id, sample.data)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO processed_datasets
                (id, parent_id, pipeline_config, sfreq, n_channels, n_samples, ch_names, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                proc_id,
                parent_id,
                json.dumps(pipeline_cfg),
                float(sample.sfreq),
                int(sample.data.shape[0]),
                int(sample.data.shape[-1]),
                json.dumps(sample.ch_names),
                json.dumps(sample.metadata or {})
            ))
        return proc_id
    
    def list_processed(self, parent_id: Optional[str] = None) -> List[Dict]:
        query = """
            SELECT id, parent_id, sfreq, n_channels, n_samples, created_at
            FROM processed_datasets
            {where}
            ORDER BY created_at DESC
        """
        
        where_clause = "WHERE parent_id = ?" if parent_id else ""
        params = (parent_id,) if parent_id else ()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query.format(where=where_clause), params)
            
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                row_dict['created_at_formatted'] = self._format_datetime(row_dict['created_at'])
                results.append(row_dict)
            return results

    def get_processed_info(self, proc_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM processed_datasets WHERE id = ?", (proc_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
                
            return {
                "id": row["id"],
                "parent_id": row["parent_id"],
                "pipeline_config": json.loads(row["pipeline_config"]),
                "sfreq": row["sfreq"],
                "n_channels": row["n_channels"],
                "n_samples": row["n_samples"],
                "ch_names": json.loads(row["ch_names"]),
                "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                "created_at": row["created_at"],
            }

    def get_processed_data(self, proc_id: str) -> Optional[np.ndarray]:
        return self._load_data_file(proc_id)

    def delete_processed(self, proc_id: str) -> bool:
        self._remove_data_file(proc_id)
        
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("DELETE FROM processed_datasets WHERE id = ?", (proc_id,))
            return result.rowcount > 0
        
    def add_features_dataset(self, parent_id: str, extractor_config: dict, X: np.ndarray, y: Optional[np.ndarray], metadata: dict = None) -> str:
        feat_id = str(uuid.uuid4())
        self._save_data_file(f"{feat_id}_X", X)
        if y is not None:
            self._save_data_file(f"{feat_id}_y", y)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO features_datasets 
                (id, parent_id, extractor_config, X_shape, y_shape, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                feat_id,
                parent_id,
                json.dumps(extractor_config),
                json.dumps(X.shape),
                json.dumps(y.shape) if y is not None else None,
                json.dumps(metadata) if metadata else None
            ))
        return feat_id

    def list_features(self, parent_id: Optional[str] = None) -> List[Dict]:
        query = """
            SELECT id, parent_id, X_shape, y_shape, created_at
            FROM features_datasets
            {where}
            ORDER BY created_at DESC
        """
        where_clause = "WHERE parent_id = ?" if parent_id else ""
        params = (parent_id,) if parent_id else ()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query.format(where=where_clause), params)
            return [dict(row) for row in cursor.fetchall()]

    def get_features_info(self, feat_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM features_datasets WHERE id = ?", (feat_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "parent_id": row["parent_id"],
                "extractor_config": json.loads(row["extractor_config"]),
                "X_shape": json.loads(row["X_shape"]),
                "y_shape": json.loads(row["y_shape"]) if row["y_shape"] else None,
                "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                "created_at": row["created_at"]
            }

    def get_features_data(self, feat_id: str) -> Optional[tuple]:
        X = self._load_data_file(f"{feat_id}_X")
        y = self._load_data_file(f"{feat_id}_y")
        return (X, y) if X is not None else None

    def delete_features(self, feat_id: str) -> bool:
        self._remove_data_file(f"{feat_id}_X")
        self._remove_data_file(f"{feat_id}_y")
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("DELETE FROM features_datasets WHERE id = ?", (feat_id,))
            return result.rowcount > 0