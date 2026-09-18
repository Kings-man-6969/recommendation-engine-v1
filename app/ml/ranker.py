import os
from typing import Any
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

DEFAULT_MODEL_PATH = os.path.join("models", "ranker.joblib")

class MLRanker:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model: HistGradientBoostingClassifier | None = None
        self.metadata: dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def score(self, feature_matrix: np.ndarray | list[list[float]]) -> np.ndarray | None:
        if self.model is None:
            return None
        try:
            X = np.asarray(feature_matrix, dtype=np.float32)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[0] == 0:
                return np.array([], dtype=np.float32)
            # Probability of engagement (class 1)
            probas = self.model.predict_proba(X)
            # If binary classifier with 2 classes
            if probas.shape[1] >= 2:
                return probas[:, 1]
            return probas[:, 0]
        except Exception:
            return None

    def save(self, path: str | None = None) -> str:
        target_path = path or self.model_path
        dirname = os.path.dirname(os.path.abspath(target_path))
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        joblib.dump({"model": self.model, "metadata": self.metadata}, target_path)
        return target_path

    def load(self, path: str | None = None) -> bool:
        target_path = path or self.model_path
        if not os.path.exists(target_path):
            return False
        try:
            payload = joblib.load(target_path)
            if isinstance(payload, dict) and "model" in payload:
                self.model = payload["model"]
                self.metadata = payload.get("metadata", {})
                return True
        except Exception:
            self.model = None
            self.metadata = {}
            return False
        return False
