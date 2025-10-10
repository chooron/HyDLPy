# Import the necessary base class and symbolic tools
from ..hydrosymolic_model import HydroSymolicModel
from ..symbol_toolkit import HydroVariable, HydroParameter, variables
from sympy import Min, Max, exp, tanh, Eq


# Helper function can be defined at the module level
def step_func(x):
    """A smooth approximation of the heaviside step function."""
    return (tanh(5.0 * x) + 1.0) * 0.5


class ExpHydro(HydroSymolicModel):
    """
    A pre-packaged, ready-to-use hydrological model with a defined set of equations.

    This class inherits from the generic HydrologicalModel engine and encapsulates
    a specific model structure, simplifying its instantiation and use.
    """

    def __init__(self, hru_num: int = 1, **kwargs):
        # Step 1: Define all symbols and equations for this specific model
        # -----------------------------------------------------------------

        # --- Define Parameters ---
        Tmin = HydroParameter("Tmin", default=-1.0, bounds=(-5.0, 5.0))
        Tmax = HydroParameter("Tmax", default=1.0, bounds=(-5.0, 5.0))
        Df = HydroParameter("Df", default=2.5, bounds=(0.0, 10.0))
        Smax = HydroParameter("Smax", default=250.0, bounds=(100.0, 400.0))
        Qmax = HydroParameter("Qmax", default=10.0, bounds=(0.0, 50.0))
        f = HydroParameter("f", default=0.05, bounds=(0.0, 0.2))

        # --- Define State and Forcing Variables ---
        temp = HydroVariable("temp")
        prcp = HydroVariable("prcp")
        pet = HydroVariable("pet")
        snowpack = HydroVariable("snowpack")
        soilwater = HydroVariable("soilwater")

        # --- Define intermediate flux symbols ---
        rainfall, snowfall, melt, evap, baseflow, surfaceflow, flow = variables(
            "rainfall, snowfall, melt, evap, baseflow, surfaceflow, flow"
        )

        # --- Define the list of equations ---
        fluxes = [
            Eq(rainfall, step_func(Tmin - temp) * prcp),
            Eq(snowfall, step_func(temp - Tmax) * prcp),
            Eq(melt, step_func(temp - Tmax) * Min(snowpack, Df * (temp - Tmax))),
            Eq(evap, step_func(soilwater) * pet * Min(soilwater, Smax) / Smax),
            Eq(
                baseflow,
                step_func(soilwater) * Qmax * exp(-f * (Smax - Min(Smax, soilwater))),
            ),
            Eq(surfaceflow, Max(soilwater, Smax) - Smax),
            Eq(flow, baseflow + surfaceflow),
        ]

        dfluxes = [
            Eq(snowpack, snowfall - melt),
            Eq(soilwater, (rainfall + melt) - (evap + flow)),
        ]

        # Step 2: Call the parent class's __init__ method
        # ----------------------------------------------------
        # Pass the now-defined equations and the user-provided hru_num
        # to the parent engine for analysis and compilation.
        super().__init__(fluxes=fluxes, dfluxes=dfluxes, hru_num=hru_num)
