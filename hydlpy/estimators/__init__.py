from .base import BaseEstimator, DynamicEstimator, StaticEstimator
from .implements.lstm import LstmEstimator
from .implements.gru import GruEstimator
from .implements.mlp import MlpEstimator
from .implements.direct import DirectEstimator

DYNAMIC_ESTIMATORS = {"lstm": LstmEstimator}

STATIC_ESTIMATORS = {"direct": DirectEstimator, "mlp": MlpEstimator}

__all__ = [
    "BaseEstimator",
    "DynamicEstimator",
    "StaticEstimator",
    "DirectEstimator",
    "MlpEstimator",
    "GruEstimator",
    "LstmEstimator",
    "DYNAMIC_ESTIMATORS",
    "STATIC_ESTIMATORS",
]
