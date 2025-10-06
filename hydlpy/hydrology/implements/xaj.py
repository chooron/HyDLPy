# --- Imports ---
from ..hydrological_model import HydrologicalModel
from ..symbol_toolkit import HydroParameter, HydroVariable, variables
from sympy import S, Min, Max, Eq, tanh, log


# --- Helper Functions ---
# This can be defined at the module level or imported
def step_func(x):
    """A smooth approximation of the heaviside step function."""
    return (tanh(5.0 * x) + 1.0) * 0.5


class XAJ(HydrologicalModel):
    """
    A pre-packaged implementation of the Xinanjiang (XAJ) hydrological model.

    This class inherits from the generic HydrologicalModel engine and encapsulates
    the complete set of symbols and equations for the XAJ conceptual model.
    """

    def __init__(self, hidden_size: int = 1):
        # Step 1: Define all symbols and equations for the XAJ model
        # -----------------------------------------------------------------

        # --- Define Parameters with reasonable default values ---
        Ke = HydroParameter("Ke", default=1.0, bounds=(0.6, 1.5))
        c = HydroParameter("c", default=0.1, bounds=(0.01, 0.2))
        Wum = HydroParameter("Wum", default=15.0, bounds=(5.0, 30.0))
        Wlm = HydroParameter("Wlm", default=75.0, bounds=(60.0, 90.0))
        Wdm = HydroParameter("Wdm", default=30.0, bounds=(15.0, 60.0))
        Aimp = HydroParameter("Aimp", default=0.05, bounds=(0.01, 0.2))
        b = HydroParameter("b", default=0.2, bounds=(0.1, 0.4))
        Smax = HydroParameter("Smax", default=20.0, bounds=(10.0, 50.0))
        ex = HydroParameter("ex", default=1.2, bounds=(1.0, 1.5))
        Ki = HydroParameter("Ki", default=0.3, bounds=(0.1, 0.55))
        Kg = HydroParameter("Kg", default=0.3, bounds=(0.1, 0.55))
        ci = HydroParameter("ci", default=0.7, bounds=(0.5, 0.9))
        cg = HydroParameter("cg", default=0.99, bounds=(0.98, 0.998))
        Kf = HydroParameter("Kf", default=2.5, bounds=(1.0, 5.0))

        # --- Define State and Forcing Variables ---
        # State Variables
        wu = HydroVariable("wu")
        wl = HydroVariable("wl")
        wd = HydroVariable("wd")
        s0 = HydroVariable("s0")
        oi = HydroVariable("oi")
        og = HydroVariable("og")
        F1 = HydroVariable("F1")
        F2 = HydroVariable("F2")
        F3 = HydroVariable("F3")
        # Forcing Variables
        P = HydroVariable("P")
        Ep = HydroVariable("Ep")

        # --- Define intermediate flux symbols ---
        (
            pn,
            iu,
            il,
            fw,
            en,
            eu,
            el,
            ed,
            et,
            r,
            rs,
            ri,
            rg,
            rt,
            qi,
            qg,
            qt,
            q1,
            q2,
            q3,
            Qt,
        ) = variables(
            "pn, iu, il, fw, en, eu, el, ed, et, r, rs, ri, rg, rt, "
            "qi, qg, qt, q1, q2, q3, Qt"
        )

        # --- Translate Julia equations into SymPy equations ---
        fluxes = [
            # Soil water component
            Eq(pn, Max(S(0), P - Ke * Ep)),
            Eq(en, Max(S(0), Ke * Ep - P)),
            Eq(eu, step_func(wu) * en),
            Eq(el, step_func(wl) * Max(c, wl / Wlm) * (en - eu)),
            Eq(ed, step_func(wd) * Max(c * (en - eu) - el, S(0))),
            Eq(et, eu + el + ed),
            Eq(
                fw,
                (1 - Aimp)
                * (
                    1
                    - Min(S(1), Max(S(0), 1 - (wu + wl + wd) / (Wum + Wlm + Wdm)))
                    ** (b / (1 + b))
                ),
            ),
            Eq(r, pn * (fw + Aimp)),
            Eq(iu, step_func(wu - Wum) * (pn - r - eu)),
            Eq(il, step_func(wl - Wlm) * (iu - el)),
            # Free water component
            Eq(
                rs,
                pn * fw * (1 - Min(S(1), Max(S(0), 1 - s0 / Smax)) ** (ex / (1 + ex))),
            ),
            Eq(ri, fw * s0 * (-Ki * log(Max(S("1e-6"), 1 - Ki - Kg)) / (Ki + Kg))),
            Eq(rg, fw * s0 * (-Kg * log(Max(S("1e-6"), 1 - Ki - Kg)) / (Ki + Kg))),
            Eq(rt, pn * Aimp + rs),
            # Land routing component
            Eq(qi, -oi * log(ci)),
            Eq(qg, -og * log(cg)),
            Eq(qt, rt + qi + qg),
            # River routing component (Nash Cascade)
            Eq(q1, Max(S(0), F1) / Kf),
            Eq(q2, Max(S(0), F2) / Kf),
            Eq(q3, Max(S(0), F3) / Kf),
            Eq(Qt, q2),  # Translated literally from the source code
        ]

        dfluxes = [
            # State updates
            Eq(wu, pn - (r + eu + iu)),
            Eq(wl, iu - (el + il)),
            Eq(wd, il - ed),
            Eq(s0, pn - (rs + ri + rg) / fw),
            Eq(oi, ri - qi),
            Eq(og, rg - qg),
            Eq(F1, qt - q1),
            Eq(F2, q1 - q2),
            Eq(F3, q2 - q3),
        ]

        # Step 2: Call the parent class's constructor
        # ----------------------------------------------------
        super().__init__(fluxes=fluxes, dfluxes=dfluxes, hidden_size=hidden_size)
