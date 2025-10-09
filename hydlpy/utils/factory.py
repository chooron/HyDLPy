from typing import Any, Dict, Optional, Union

import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_names = []

    def forward(self, x):
        return {}


def _create_module(
    config: Optional[Dict[str, Any]], module_map: Dict[str, nn.Module]
) -> Union[nn.Module, None]:
    """Generic factory function to create a module.

    If the config is None or empty, it returns an Identity module.
    """
    if not config:
        return None
    model_type = config.get("type", "").lower()
    params = config.get("params", {})

    model_class = module_map.get(model_type)

    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_class(**params)


def create_static_parameter_estimator(
    config: Optional[Dict[str, Any]],
) -> Union[nn.Module, None]:
    """Factory for the static parameter estimator."""
    module_map = {"direct": DirectEstimator, "mlp": MLPEstimator}

    return _create_module(config, module_map)


def create_dynamic_parameter_estimator(
    config: Optional[Dict[str, Any]],
) -> Union[nn.Module, None]:
    """Factory for the dynamic parameter estimator."""
    module_map = {"lstm": LSTMEstimator}
    return _create_module(config, module_map)


def create_hydrology_core(config: Dict[str, Any]) -> Union[nn.Module, None]:
    """Factory for the core hydrological model."""
    hydrology_module = HYDROLOGY_MODELS[config["model_name"]]
    return hydrology_module(**config)


def create_routing_module(config: Optional[Dict[str, Any]]) -> Union[nn.Module, None]:
    """Factory for the routing model."""
    module_map = {
        "mean": MeanRouting,
        "mlp": MLPRouting,
        "lstm": LSTMRouting,
        "dmc": DmcRouting,
    }
    return _create_module(config, module_map)
