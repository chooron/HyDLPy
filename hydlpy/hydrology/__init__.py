from .hydrological_model import HydrologicalModel
from .hydrosymolic_model import HydroSymolicModel
from .implements import ExpHydro, HBV, XAJ
from .symbol_toolkit import HydroParameter, HydroVariable, variables, parameters

AVAILABLE_MODELS = ["ExpHydro", "HBV", "XAJ"]

HYDROLOGY_MODELS = {
    "exphydro": ExpHydro,
    "hbv": HBV,
    "xaj": XAJ,
}

__all__ = [
    "HydrologicalModel",
    "HydroSymolicModel",
    "HydroParameter",
    "HydroVariable",
    "variables",
    "parameters",
    "AVAILABLE_MODELS",
]
