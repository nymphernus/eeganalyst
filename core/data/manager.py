from typing import Optional, List, Dict, Union
from storage.eeg_database import EEGDatabase
from .sample import EEGSample
from core.utils.hashing import compute_array_hash
from core.preprocess.pipeline import PreprocessPipeline
from core.features.base import FeatureExtractor


class DataManager:
    def __init__(self):
        self.db = EEGDatabase()

    def add_sample(self, sample: EEGSample, filename: str) -> str:
        file_hash = compute_array_hash(sample.data)
        existing_id = self.db.dataset_exists(file_hash)
        if existing_id:
            return existing_id
        return self.db.add_dataset(
            filename=filename,
            file_hash=file_hash,
            sfreq=float(sample.sfreq),
            n_channels=int(sample.data.shape[0]),
            n_samples=int(sample.data.shape[-1]),
            ch_names=sample.ch_names,
            data=sample.data,
            metadata=sample.metadata,
        )

    def get_sample(self, dataset_id: str) -> Optional[EEGSample]:
        info = self.db.get_dataset_info(dataset_id)
        if not info:
            return None
        data = self.db.get_dataset_data(dataset_id)
        if data is None:
            return None
        return EEGSample(
            data=data,
            sfreq=info["sfreq"],
            ch_names=info["ch_names"],
            metadata=info.get("metadata"),
            raw_path=info.get("filename"),
        )

    def list_samples(self) -> list:
        return self.db.list_datasets()

    def delete_sample(self, dataset_id: str) -> bool:
        return self.db.delete_dataset(dataset_id)

    def apply_pipeline(self, dataset_id: str, pipeline: PreprocessPipeline, save: bool = True) -> Union[str, EEGSample]:
        sample = self.get_sample(dataset_id)
        if sample is None:
            raise ValueError(f"Dataset {dataset_id} not found")

        X = pipeline.fit_transform(sample.data)

        final_sfreq = float(sample.sfreq)
        for step in getattr(pipeline, "steps", []):
            if step.name.lower() in ("resample",):
                params = step.params
                if "target_rate" in params:
                    final_sfreq = float(params["target_rate"])
                elif "new_sfreq" in params:
                    final_sfreq = float(params["new_sfreq"])
                elif "target_sfreq" in params:
                    final_sfreq = float(params["target_sfreq"])

        processed_sample = EEGSample(
            data=X,
            sfreq=final_sfreq,
            ch_names=sample.ch_names,
            subject_id=sample.subject_id,
            session_id=sample.session_id,
            task=sample.task,
            labels=sample.labels,
            metadata={
                "input_shape": list(sample.data.shape),
                "output_shape": list(X.shape),
            },
        )

        if not save:
            return processed_sample

        proc_id = self.db.add_processed_dataset(
            parent_id=dataset_id,
            sample=processed_sample,
            pipeline_cfg=pipeline.to_dict()
        )
        return proc_id

    def get_processed_sample(self, proc_id: str) -> Optional[EEGSample]:
        info = self.db.get_processed_info(proc_id)
        if not info:
            return None
        data = self.db.get_processed_data(proc_id)
        if data is None:
            return None
        return EEGSample(
            data=data,
            sfreq=info["sfreq"],
            ch_names=info["ch_names"],
            metadata=info.get("metadata"),
            raw_path=None,
        )

    def list_processed(self, parent_id: Optional[str] = None) -> List[Dict]:
        return self.db.list_processed(parent_id=parent_id)

    def delete_processed(self, proc_id: str) -> bool:
        return self.db.delete_processed(proc_id)

    def list_all_processed(self) -> List[Dict]:
        return self.db.list_processed()

    def get_processed_info(self, proc_id: str) -> Optional[Dict]:
        return self.db.get_processed_info(proc_id)

    def delete_processed_sample(self, proc_id: str) -> bool:
        return self.db.delete_processed(proc_id)

    def get_sample_info(self, sample_id: str) -> Optional[Dict]:
        return self.db.get_dataset_info(sample_id)

    def get_processed_parent_info(self, proc_id: str) -> Optional[Dict]:
        proc_info = self.get_processed_info(proc_id)
        if proc_info and proc_info.get('parent_id'):
            return self.get_sample_info(proc_info['parent_id'])
        return None

    def extract_features(self, proc_id: str, extractor: FeatureExtractor, y=None, save: bool = True) -> Union[str, tuple]:
        info = self.db.get_processed_info(proc_id)
        if not info:
            raise ValueError(f"Processed dataset {proc_id} not found")

        data = self.db.get_processed_data(proc_id)
        if data is None:
            raise ValueError("Failed to load processed data")

        X = extractor.fit_transform(data, y)

        if not save:
            return X, y

        feat_id = self.db.add_features_dataset(
            parent_id=proc_id,
            extractor_config=extractor.to_dict(),
            X=X,
            y=y,
            metadata={"input_shape": data.shape}
        )
        return feat_id

    def save_features_from_array(self, parent_id: str, X, y=None, extractor_config: dict = None, metadata: dict = None) -> str:
        if extractor_config is None:
            extractor_config = {}
        feat_id = self.db.add_features_dataset(
            parent_id=parent_id,
            extractor_config=extractor_config,
            X=X,
            y=y,
            metadata=metadata
        )
        return feat_id

    def list_features(self, parent_id: Optional[str] = None) -> List[Dict]:
        return self.db.list_features(parent_id=parent_id)

    def get_features_data(self, feat_id: str):
        return self.db.get_features_data(feat_id)

    def get_features_info(self, feat_id: str):
        return self.db.get_features_info(feat_id)

    def delete_features(self, feat_id: str) -> bool:
        return self.db.delete_features(feat_id)
