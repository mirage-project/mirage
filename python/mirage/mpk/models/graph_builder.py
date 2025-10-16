import abc
from typing import Optional, Dict, Any


class GraphBuilder(abc.ABC):
    def __init__(self, mpk, weights: Optional[Dict[str, Any]] = None):
        self.mpk = mpk
        self.weights = weights or {}

    @abc.abstractmethod
    def build_from_model(self, model_path: str | None = None):
        raise NotImplementedError("build_from_model is not implemented")
