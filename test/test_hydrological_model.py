# file: test_hydro_model.py

import pytest
import torch
import sympy
from sympy import S, Min, Max, exp, tanh
import sys

# Assume the main module is in a discoverable path
sys.path.append("E:\\PyCode\\HyDLPy")
from hydlpy.hydrology import (
    HydrologicalModel,
    HydroParameter,
    HydroVariable,
    variables,
)


def step_func(x):
    """Helper function used in the model equations."""
    return (tanh(5.0 * x) + 1.0) * 0.5


@pytest.fixture(scope="module")
def model_equations():
    """
    A pytest fixture that sets up the symbolic model definition once for all tests.
    This replaces the unittest setUp method.
    """
    # 1. Define all symbols using the custom classes
    Tmin = HydroParameter("Tmin", default=-1.0, bounds=(-5.0, 5.0))
    Tmax = HydroParameter("Tmax", default=1.0, bounds=(-5.0, 5.0))
    Df = HydroParameter("Df", default=2.5, bounds=(0.0, 10.0))
    Smax = HydroParameter("Smax", default=250.0, bounds=(100.0, 400.0))
    Qmax = HydroParameter("Qmax", default=10.0, bounds=(0.0, 50.0))
    f = HydroParameter("f", default=0.05, bounds=(0.0, 0.2))

    temp = HydroVariable("temp")
    prcp = HydroVariable("prcp")
    lday = HydroVariable("lday")
    snowpack = HydroVariable("snowpack")
    soilwater = HydroVariable("soilwater")

    # Define symbols for intermediate fluxes
    rainfall, snowfall, melt, pet, evap, baseflow, surfaceflow, flow = variables(
        "rainfall, snowfall, melt, pet, evap, baseflow, surfaceflow, flow"
    )

    # 2. Define the list of equations
    fluxes = [
        sympy.Eq(
            pet,
            S(29.8)
            * lday
            * 24
            * 0.611
            * exp((S(17.3) * temp) / (temp + 237.3))
            / (temp + 273.2),
        ),
        sympy.Eq(rainfall, step_func(Tmin - temp) * prcp),
        sympy.Eq(snowfall, step_func(temp - Tmax) * prcp),
        sympy.Eq(melt, step_func(temp - Tmax) * Min(snowpack, Df * (temp - Tmax))),
        sympy.Eq(evap, step_func(soilwater) * pet * Min(soilwater, Smax) / Smax),
        sympy.Eq(
            baseflow,
            step_func(soilwater) * Qmax * exp(-f * (Smax - Min(Smax, soilwater))),
        ),
        sympy.Eq(surfaceflow, Max(soilwater, Smax) - Smax),
        sympy.Eq(flow, baseflow + surfaceflow),
    ]

    dfluxes = [
        sympy.Eq(snowpack, snowfall - melt),
        sympy.Eq(soilwater, (rainfall + melt) - (evap + flow)),
    ]

    # The fixture provides the equations to the tests that request it.
    return fluxes, dfluxes


@pytest.mark.parametrize("hidden_size", [1, 16])
def test_initialization(model_equations, hidden_size):
    """
    Tests model initialization for both single and multiple HRUs.
    This single function replaces test_initialization_single_hru and test_initialization_multi_hru.
    """
    print(f"\n--- Running Test: Initialization (hidden_size={hidden_size}) ---")
    fluxes, dfluxes = model_equations
    model = HydrologicalModel(fluxes=fluxes, dfluxes=dfluxes, hidden_size=hidden_size)

    assert model.hidden_size == hidden_size
    assert model.Smax.shape == (hidden_size,)
    assert "snowpack" in model.state_names
    assert "temp" in model.forcing_names


@pytest.mark.parametrize("hidden_size", [1, 16])
def test_core(model_equations, hidden_size):
    """
    Tests the forward pass using an externally provided parameter tensor.
    This verifies that the model can correctly use parameter values passed at runtime,
    bypassing its internal nn.Parameter members.
    """
    print(
        f"\n--- Running Test: Forward Pass with External Parameters (hidden_size={hidden_size}) ---"
    )
    fluxes, dfluxes = model_equations
    model = HydrologicalModel(fluxes=fluxes, dfluxes=dfluxes, hidden_size=hidden_size)

    # 1. Prepare input tensors for states and forcings
    states = torch.rand(hidden_size, len(model.state_names))
    forcings = torch.rand(hidden_size, len(model.forcing_names))

    # 2. Create a correctly shaped external parameter tensor to override the model's internal ones
    # The required shape is (hidden_size, num_parameters)
    external_params = torch.rand(hidden_size, len(model.parameter_names))

    # 3. Call the model's forward pass, providing the external `parameters` tensor
    output_fluxes, new_states = model._core(
        forcings, states, parameters=external_params
    )

    # 4. Assert that the output tensors have the correct shapes
    expected_fluxes_shape = (hidden_size, len(model.flux_names))
    expected_states_shape = (hidden_size, len(model.state_names))

    assert output_fluxes.shape == expected_fluxes_shape
    assert new_states.shape == expected_states_shape

    # 5. Check for any NaN values in the output
    assert not torch.isnan(output_fluxes).any()
    assert not torch.isnan(new_states).any()


@pytest.mark.parametrize(
    ("hidden_size", "basin_num"),
    [
        (1, 1),
        (16, 12),
    ],
)
def test_forward(model_equations, hidden_size, basin_num):
    """
    Tests the forward pass using an externally provided parameter tensor.
    This verifies that the model can correctly use parameter values passed at runtime,
    bypassing its internal nn.Parameter members.
    """
    print(
        f"\n--- Running Test: Forward Pass with External Parameters (hidden_size={hidden_size}) ---"
    )
    fluxes, dfluxes = model_equations
    model = HydrologicalModel(fluxes=fluxes, dfluxes=dfluxes, hidden_size=hidden_size)
    time_len= 365
    # 1. Prepare input tensors for states and forcings
    states = torch.rand(basin_num, hidden_size, len(model.state_names))
    forcings = torch.rand(time_len, basin_num, hidden_size, len(model.forcing_names))

    # 2. Create a correctly shaped external parameter tensor to override the model's internal ones
    # The required shape is (hidden_size, num_parameters)
    external_params = torch.rand(
        time_len, basin_num, hidden_size, len(model.parameter_names)
    )

    # 3. Call the model's forward pass, providing the external `parameters` tensor
    output_fluxes, new_states = model(forcings, states, external_params)

    # 4. Assert that the output tensors have the correct shapes
    expected_fluxes_shape = (time_len, basin_num, hidden_size, len(model.flux_names))
    expected_states_shape = (time_len, basin_num, hidden_size, len(model.state_names))

    assert output_fluxes.shape == expected_fluxes_shape
    assert new_states.shape == expected_states_shape

    # 5. Check for any NaN values in the output
    assert not torch.isnan(output_fluxes).any()
    assert not torch.isnan(new_states).any()
