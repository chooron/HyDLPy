# Import the necessary base class and symbolic tools
import torch
import torch.nn as nn

from ..hydrological_model import HydrologicalModel


# Helper function can be defined at the module level
def step_func(x):
    """A smooth approximation of the heaviside step function."""
    return (torch.tanh(5.0 * x) + 1.0) * 0.5


def _flux_caculator(
    states: torch.Tensor, forcings: torch.Tensor, params: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """
    Calculates hydrological fluxes using PyTorch.

    Args:
        states (tuple or list): A sequence of state tensors.
                                Order: (snowpack, soilwater).
        forcings (tuple or list): A sequence of forcing tensors.
                                  Order: (prcp, temp, pet).
        params (tuple or list): A sequence of parameter tensors.
                                Order: (Tmin, Tmax, Df, Smax, Qmax, f).

    Returns:
        A dictionary containing the calculated flux tensors.
    """
    # Unpack variables by index
    snowpack, soilwater = states[0], states[1]
    prcp, temp, pet = forcings[0], forcings[1], forcings[2]
    Tmin, Tmax, Df, Smax, Qmax, f = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
    )

    # Calculate fluxes
    rainfall = (temp < Tmin).float() * prcp
    snowfall = (temp > Tmax).float() * prcp

    potential_melt = Df * (temp - Tmax)
    melt = (temp > Tmax).float() * torch.min(snowpack, torch.relu(potential_melt))

    relative_soilwater = torch.clamp(soilwater / Smax, min=0.0, max=1.0)
    evap = pet * relative_soilwater

    soilwater_clipped = torch.clamp(soilwater, max=Smax)
    deficit = Smax - soilwater_clipped
    baseflow = (soilwater > 0).float() * Qmax * torch.exp(-f * deficit)

    surfaceflow = torch.relu(soilwater - Smax)

    flow = baseflow + surfaceflow

    return (rainfall, snowfall, melt, evap, baseflow, surfaceflow, flow)


def _state_updator(
    states: torch.Tensor,
    fluxes: torch.Tensor,
    forcings: torch.Tensor,
    params: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """
    Calculates hydrological fluxes using PyTorch.

    Args:
        states (tuple or list): A sequence of state tensors.
                                Order: (snowpack, soilwater).
        forcings (tuple or list): A sequence of forcing tensors.
                                  Order: (prcp, temp, pet).
        params (tuple or list): A sequence of parameter tensors.
                                Order: (Tmin, Tmax, Df, Smax, Qmax, f).

    Returns:
        A dictionary containing the calculated flux tensors.
    """
    # Unpack variables by index
    snowpack, soilwater = states[0], states[1]
    prcp, temp, pet = forcings[0], forcings[1], forcings[2]
    Tmin, Tmax, Df, Smax, Qmax, f = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
    )
    rainfall, snowfall, melt, evap, baseflow, surfaceflow, flow = (
        fluxes[0],
        fluxes[1],
        fluxes[2],
        fluxes[3],
        fluxes[4],
        fluxes[5],
        fluxes[6],
    )
    new_snowpack = snowpack + snowfall - melt
    new_soilwater = rainfall + melt - evap - flow

    return (new_snowpack, new_soilwater)

class ExpHydroManual(HydrologicalModel):
    """
    A pre-packaged, ready-to-use hydrological model with a defined set of equations.

    This class inherits from the generic HydrologicalModel engine and encapsulates
    a specific model structure, simplifying its instantiation and use.
    """

    def __init__(self, hru_num: int = 1, **kwargs):
        super().__init__(
            hru_num=hru_num,
            state_names=["snowpack", "soilwater"],
            forcing_names=["prcp", "temp", "pet"],
            parameter_names=["Tmin", "Tmax", "Df", "Smax", "Qmax", "f"],
            flux_names=[
                "rainfall",
                " snowfall",
                " melt",
                " evap",
                " baseflow",
                " surfaceflow",
                " flow",
            ],
            parameter_bounds={
                "Tmin": (-5.0, 5.0),
                "Tmax": (-5.0, 5.0),
                "Df": (0.0, 10.0),
                "Smax": (100.0, 400.0),
                "Qmax": (0.0, 50.0),
                "f": (0.0, 0.2),
            },
            flux_calculator=_flux_caculator,
            state_updater=_state_updator,
            **kwargs,
        )
