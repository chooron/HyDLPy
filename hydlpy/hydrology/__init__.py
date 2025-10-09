from .hydrological_model import HydrologicalModel
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
    "HydroParameter",
    "HydroVariable",
    "variables",
    "parameters",
    "AVAILABLE_MODELS",
]
