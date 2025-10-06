from typing import Any, Dict, Optional, Union

import torch.nn as nn

from .estimators.dynamic import LSTMEstimator
from .estimators.static import DirectEstimator, MLPEstimator
# Import from the new structured directories
from .hydrology_cores import GR4H, ExpHydro, Hbv
from .routing_modules.dmc import DmcRouting
from .routing_modules.routing import LSTMRouting, MeanRouting, MLPRouting


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_names = []

    def forward(self, x):
        return {}


def _create_module(config: Optional[Dict[str, Any]], module_map: Dict[str, nn.Module]) -> Union[nn.Module, None]:
    """Generic factory function to create a module.
    
    If the config is None or empty, it returns an Identity module.
    """
    if not config:
        return None
    model_type = config.get('type', '').lower()
    params = config.get('params', {})
    
    model_class = module_map.get(model_type)
    
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model_class(**params)


def create_static_parameter_estimator(config: Optional[Dict[str, Any]]) -> Union[nn.Module, None]:
    """Factory for the static parameter estimator."""
    module_map = {
        'direct': DirectEstimator,
        'mlp': MLPEstimator
    }
    return _create_module(config, module_map)


def create_dynamic_parameter_estimator(config: Optional[Dict[str, Any]]) -> Union[nn.Module, None]:
    """Factory for the dynamic parameter estimator."""
    module_map = {
        'lstm': LSTMEstimator
    }
    return _create_module(config, module_map)


def create_hydrology_core(config: Optional[Dict[str, Any]]) -> nn.Module:
    """Factory for the core hydrological model."""
    module_map = {
        'gr4h': GR4H,
        'hbv': Hbv,
        'exphydro': ExpHydro
    }
    # Core model is not optional, so we don't use the generic creator
    if not config:
        raise ValueError("hydrology_core configuration is required.")
    return _create_module(config, module_map)

def create_routing_module(config: Optional[Dict[str, Any]]) -> Union[nn.Module, None]:
    """Factory for the routing model."""
    module_map = {
        'mean': MeanRouting,
        'mlp': MLPRouting,
        'lstm': LSTMRouting,
        'dmc': DmcRouting
    }
    return _create_module(config, module_map)