# Import the necessary base class and symbolic tools
from ..hydrological_model import HydrologicalModel
from ..symbol_toolkit import HydroParameter, HydroVariable, variables
from sympy import S, Min, Max, Eq, tanh


# Helper function can be defined at the module level
def step_func(x):
    """A smooth approximation of the heaviside step function."""
    return (tanh(5.0 * x) + 1.0) * 0.5


class HBV(HydrologicalModel):
    """
    A pre-packaged implementation of the HBV hydrological model.

    This class inherits from the generic HydrologicalModel engine and encapsulates
    the complete set of symbols and equations for the HBV conceptual model.
    """

    def __init__(self, hidden_size: int = 1):
        # Step 1: Define all symbols and equations for the HBV model
        # -----------------------------------------------------------------

        # --- Define Parameters with defaults chosen from their bounds ---
        TT = HydroParameter("TT", default=0.0, bounds=(-1.5, 1.2), unit="C")
        CFMAX = HydroParameter("CFMAX", default=3.0, bounds=(1.0, 8.0), unit="mm")
        CWH = HydroParameter("CWH", default=0.1, bounds=(0.0, 0.2), unit="mm")
        CFR = HydroParameter("CFR", default=0.05, bounds=(0.0, 0.1), unit="mm")
        FC = HydroParameter("FC", default=250.0, bounds=(50.0, 500.0), unit="mm")
        LP = HydroParameter("LP", default=0.7, bounds=(0.3, 1.0), unit="mm")
        BETA = HydroParameter("BETA", default=2.0, bounds=(1.0, 6.0), unit="-")
        PPERC = HydroParameter("PPERC", default=1.0, bounds=(0.0, 3.0), unit="mm")
        UZL = HydroParameter("UZL", default=40.0, bounds=(0.0, 70.0), unit="d")
        k0 = HydroParameter("k0", default=0.2, bounds=(0.05, 0.5), unit="1/d")
        k1 = HydroParameter("k1", default=0.1, bounds=(0.01, 0.3), unit="1/d")
        k2 = HydroParameter("k2", default=0.05, bounds=(0.001, 0.15), unit="1/d")

        # --- Define State and Forcing Variables ---
        # State Variables
        soilwater = HydroVariable("soilwater")
        snowpack = HydroVariable("snowpack")
        meltwater = HydroVariable("meltwater")
        suz = HydroVariable("suz")
        slz = HydroVariable("slz")
        # Forcing Variables
        P = HydroVariable("P")
        Ep = HydroVariable("Ep")
        T = HydroVariable("T")

        # --- Define intermediate flux symbols ---
        (
            rainfall,
            snowfall,
            melt,
            refreeze,
            infil,
            excess,
            recharge,
            evap,
            q0,
            q1,
            q2,
            Qt,
            perc,
        ) = variables(
            "rainfall, snowfall, melt, refreeze, infil, "
            "excess, recharge, evap, q0, q1, q2, Qt, perc"
        )

        # --- Translate Julia equations into SymPy equations ---
        fluxes = [
            # Precipitation splitting
            Eq(snowfall, step_func(TT - T) * P),
            Eq(rainfall, step_func(T - TT) * P),
            # Snow bucket
            Eq(melt, Min(snowpack, Max(S(0), T - TT) * CFMAX)),
            Eq(refreeze, Min(max((TT - T), S(0)) * CFR * CFMAX, meltwater)),
            Eq(infil, Max(S(0), meltwater - snowpack * CWH)),
            # Soil bucket
            Eq(
                recharge,
                (rainfall + infil) * Min(S(1), Max(S(0), (soilwater / FC)) ** BETA),
            ),
            Eq(excess, Max(soilwater - FC, S(0))),
            Eq(evap, Min(S(1), Max(S(0), soilwater / (LP * FC))) * Ep),
            # Response (Zone) bucket
            Eq(perc, suz * PPERC),
            Eq(q0, Max(S(0), suz - UZL) * k0),
            Eq(q1, suz * k1),
            Eq(q2, slz * k2),
            Eq(Qt, q0 + q1 + q2),
        ]

        dfluxes = [
            # State updates
            Eq(snowpack, snowfall + refreeze - melt),
            Eq(meltwater, melt - refreeze - infil),
            Eq(soilwater, rainfall + infil - (recharge + excess + evap)),
            Eq(suz, recharge + excess - (perc + q0 + q1)),
            Eq(slz, perc - q2),
        ]

        # Step 2: Call the parent class's constructor
        # ----------------------------------------------------
        super().__init__(fluxes=fluxes, dfluxes=dfluxes, hidden_size=hidden_size)
