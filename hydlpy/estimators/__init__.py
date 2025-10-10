from .base import BaseEstimator, DynamicEstimator, StaticEstimator
from .implements.lstm import LstmEstimator
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
    "LstmEstimator",
    "DYNAMIC_ESTIMATORS",
    "STATIC_ESTIMATORS",
]
