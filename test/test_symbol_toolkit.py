# file: test_symbol_toolkit.py (or any name starting with test_)

import os
import sys
import sympy

# Add the parent directory to the path to find the hydlpy module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the components to be tested
from hydlpy.hydrology import HydroParameter, HydroVariable, parameters, variables
from hydlpy.hydrology.symbol_toolkit import is_parameter


def test_individual_parameter_creation():
    """Tests the creation of a single Parameter and its metadata."""
    amp = HydroParameter(
        "amp",
        description="Amplitude",
        bounds=(0, 10),
        unit="V",
        default=5.0,
        real=True,  # Test passing standard sympy assumptions
    )
    assert isinstance(amp, HydroParameter)
    assert amp.name == "amp"
    assert amp.get_description() == "Amplitude"
    assert amp.get_bounds() == (0, 10)
    assert amp.get_unit() == "V"
    assert amp.get_default() == 5.0
    assert amp.is_real is True
    # Test that metadata is None when not provided
    assert HydroParameter("p").get_description() is None


def test_individual_variable_creation():
    """Tests the creation of a single Variable."""
    t = HydroVariable("t", unit="s")
    assert isinstance(t, HydroVariable)
    assert t.name == "t"
    assert t.get_unit() == "s"
    assert t.get_description() is None


def test_is_parameter_utility():
    """Tests the is_parameter helper function."""
    p = HydroParameter("p")
    v = HydroVariable("v")
    s = sympy.Symbol("s")

    assert is_parameter(p) is True
    assert is_parameter(v) is False
    assert is_parameter(s) is False
    assert is_parameter(5) is False  # Test with a non-symbol object


def test_bulk_creation_parameters():
    """Tests the 'parameters' bulk creation function."""
    k, m, g = parameters("k m g", positive=True)
    assert isinstance(k, HydroParameter)
    assert isinstance(m, HydroParameter)
    assert isinstance(g, HydroParameter)
    assert k.is_positive is True

    # Test range creation
    p_vec = parameters("p:3")
    assert len(p_vec) == 3
    assert isinstance(p_vec[0], HydroParameter)
    assert p_vec[1].name == "p1"


def test_bulk_creation_variables():
    """Tests the 'variables' bulk creation function."""
    # Test comma-separated creation
    x, y = variables("x,y")
    assert isinstance(x, HydroVariable)
    assert isinstance(y, HydroVariable)
    assert not is_parameter(x)


def test_expression_integration():
    """Tests the integration of custom symbols within a SymPy expression."""
    amp = HydroParameter("amp", default=1)
    k = HydroParameter("k")
    t = HydroVariable("t")

    # Create an expression
    expr = amp * sympy.sin(k * t)

    # Check that free_symbols finds all custom symbols
    symbols_in_expr = expr.free_symbols
    assert len(symbols_in_expr) == 3

    # Separate parameters and variables from the expression
    found_params = {s for s in symbols_in_expr if is_parameter(s)}
    found_vars = {s for s in symbols_in_expr if not is_parameter(s)}

    # Assert that the sets contain the correct symbols
    assert found_params == {amp, k}
    assert found_vars == {t}
