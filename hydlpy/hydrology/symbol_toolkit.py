"""
A custom SymPy extension for creating distinguished symbols for modeling:
Parameters and Variables, with support for metadata and bulk creation.
"""

from typing import Callable
import sympy
import torch
import torch.nn as nn

class MetaSymbol(sympy.Symbol):
    """
    A base Symbol class that can accept and store arbitrary metadata.
    """
    def __new__(cls, name, **kwargs):
        metadata_keys = getattr(cls, '_metadata_keys', [])
        metadata = {}
        for key in metadata_keys:
            if key in kwargs:
                metadata[key] = kwargs.pop(key)
        
        obj = super().__new__(cls, name, **kwargs)
        obj._metadata = metadata
        return obj

    def get_description(self):
        """Gets the description of the symbol."""
        return self._metadata.get('description', None)

    def get_bounds(self):
        """Gets the bounds of the parameter."""
        return self._metadata.get('bounds', None)

    def get_unit(self):
        """Gets the unit of the symbol."""
        return self._metadata.get('unit', None)

    def get_default(self):
        """Gets the default value of the parameter."""
        return self._metadata.get('default', None)
        
    def get_metadata(self):
        """Gets the entire metadata dictionary."""
        return self._metadata

class HydroParameter(MetaSymbol):
    """
    Parameter class, carrying metadata like description, bounds, unit, and default value.
    """
    _is_parameter = True
    _metadata_keys = ['description', 'bounds', 'unit', 'default']

class HydroVariable(MetaSymbol):
    """
    Variable class, can carry metadata like description and unit.
    """
    _is_parameter = False
    _metadata_keys = ['description', 'unit']

def is_parameter(s):
    """Checks if a given sympy object is a Parameter."""
    return getattr(s, '_is_parameter', False)

def parameters(names, **kwargs):
    """
    Creates multiple Parameter objects from a string, similar to sympy.symbols.
    Example: a, b, c = parameters('a b c', positive=True)
    """
    return sympy.symbols(names, cls=HydroParameter, **kwargs)

def variables(names, **kwargs):
    """
    Creates multiple Variable objects from a string, similar to sympy.symbols.
    Example: x, y, z = variables('x y z')
    """
    return sympy.symbols(names, cls=HydroVariable, **kwargs)


class SympyFunction(nn.Module):
    """ """

    def __init__(self, func, name=None):
        super().__init__()
        self.func = func
        self.name = name or func.__name__

    def forward(self, *args):
        return self.func(*args)

    def __repr__(self):
        return f"{self.__class__.__name__}(func={self.name})"

ZERO = sympy.Symbol("ZERO")
ONE = sympy.Symbol("ONE")


def create_nn_module_wrapper(model: nn.Module) -> Callable:
    """
    Creates a wrapper function for an nn.Module to make it compatible
    with the multi-argument output of SymPy's lambdify.

    Args:
        model (nn.Module): The PyTorch module to wrap.

    Returns:
        Callable: A new function that accepts multiple tensor arguments,
                    stacks them, passes them to the model, and returns the result.
    """

    def wrapper(*args: torch.Tensor) -> torch.Tensor:
        """
        This inner function will be called by lambdify.
        It takes individual tensors, preprocesses them, and calls the nn.Module.
        """
        if not args:
            raise ValueError("Custom nn.Module function was called with no arguments.")

        device, dtype = args[0].device, args[0].dtype
        inputs = torch.stack([torch.atleast_1d(arg) for arg in args], dim=-1).to(
            device=device, dtype=dtype
        )

        return model(inputs).squeeze(-1)

    return wrapper

# Helper function to ensure inputs to min/max are Tensors
def _to_tensor(val, ref=None):
    if torch.is_tensor(val):
        return val
    device = ref.device if (ref is not None and torch.is_tensor(ref)) else None
    return torch.tensor(val, dtype=torch.float32, device=device)


# Custom module for lambdify to handle mixed-type min/max
TORCH_EXTEND_MODULE = {
    "Max": lambda a, b: torch.maximum(_to_tensor(a, b), _to_tensor(b, a)),
    "max": lambda a, b: torch.maximum(_to_tensor(a, b), _to_tensor(b, a)),
    "Min": lambda a, b: torch.minimum(_to_tensor(a, b), _to_tensor(b, a)),
    "min": lambda a, b: torch.minimum(_to_tensor(a, b), _to_tensor(b, a)),
}