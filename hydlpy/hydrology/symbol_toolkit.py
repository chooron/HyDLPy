"""
A custom SymPy extension for creating distinguished symbols for modeling:
Parameters and Variables, with support for metadata and bulk creation.
"""

import sympy

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