from .base import BaseEstimator, DynamicEstimator, StaticEstimator
from .dynamic import LSTMEstimator
from .static import DirectEstimator, MLPEstimator

DYNAMIC_ESTIMATORS = {"lstm": LSTMEstimator}

STATIC_ESTIMATORS = {"direct": DirectEstimator, "mlp": MLPEstimator}

__all__ = [
    "BaseEstimator",
    "DynamicEstimator",
    "StaticEstimator",
    "DirectEstimator",
    "MLPEstimator",
    "LSTMEstimator",
    "DYNAMIC_ESTIMATORS",
    "STATIC_ESTIMATORS",
]
