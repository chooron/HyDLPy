from typing import Dict, Optional

import torch

from .base import BaseHydrologyModel


class Hbv(BaseHydrologyModel):
    """HBV 1.0 ~.

    This class implements the HBV model-specific logic, inheriting shared
    functionality from the BaseHydrologyModel.
    """

    _model_name = "hbv"

    _parameter_bounds = {
        "parBETA": [1.0, 6.0],
        "parFC": [50, 1000],
        "parK0": [0.05, 0.9],
        "parK1": [0.01, 0.5],
        "parK2": [0.001, 0.2],
        "parLP": [0.2, 1],
        "parPERC": [0, 10],
        "parUZL": [0, 100],
        "parTT": [-2.5, 2.5],
        "parCFMAX": [0.5, 10],
        "parCFR": [0, 0.1],
        "parCWH": [0, 0.2],
        "parBETAET": [0.3, 5],
    }

    def __init__(
        self,
        device: Optional[torch.device] = None,
        **config,
    ):
        """Initializes the HBV model, setting its name and parameter bounds."""
        super().__init__(device=device, **config)

    def _initialize_states(self) -> Dict[str, torch.Tensor]:
        """Implements the creation of initial states for the HBV model."""
        SNOWPACK = torch.zeros(self.nmul, dtype=torch.float32, device=self.device) + 0.001
        MELTWATER = (
            torch.zeros(self.nmul, dtype=torch.float32, device=self.device) + 0.001
        )
        SM = torch.zeros(self.nmul, dtype=torch.float32, device=self.device) + 0.001
        SUZ = torch.zeros( self.nmul, dtype=torch.float32, device=self.device) + 0.001
        SLZ = torch.zeros(self.nmul, dtype=torch.float32, device=self.device) + 0.001
        return {"SNOWPACK": SNOWPACK, "MELTWATER": MELTWATER, "SM": SM, "SUZ": SUZ, "SLZ": SLZ}

    def _process_core(
        self,
        x_dict: Dict[str, torch.Tensor],
        states: Dict[str, torch.Tensor] = {},
        static_params: Dict[str, torch.Tensor] = {},
        dynamic_params: Dict[str, torch.Tensor] = {},
        warm_up: int = 0,
        return_states: bool = False,
    ) -> Dict[str, torch.Tensor]:

        # Forcings n_steps, nmul, n_vars
        Pm = x_dict["prcp"]  # (n_timesteps, n_basins)
        PETm = x_dict["pet"]  # (n_timesteps, n_basins)
        Tm = torch.ones(size=(Pm.shape[0], Pm.shape[1]), device=self.device) * 10.0  # (n_timesteps, n_basins)

        n_steps, nmul = Pm.size()

        """Implements the core process-based calculations for the HBV model."""
        default_states = (
            torch.zeros([1, self.nmul], dtype=torch.float32, device=self.device) + 0.001
        )
        SNOWPACK = states.get("SNOWPACK", default_states)
        MELTWATER = states.get("MELTWATER", default_states)
        SM = states.get("SM", default_states)
        SUZ = states.get("SUZ", default_states)
        SLZ = states.get("SLZ", default_states)

        Qsimmu = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.001

        param_dict = static_params if static_params else {}

        for t in range(n_steps):
            if dynamic_params:
                for key in dynamic_params.keys():
                    param_dict[key] = dynamic_params[key][t, :]

            # Separate precipitation into liquid and solid components.
            PRECIP = Pm[t, :]
            RAIN = torch.mul(PRECIP, (Tm[t, :] >= param_dict["parTT"]).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :] < param_dict["parTT"]).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = param_dict["parCFMAX"] * (Tm[t, :] - param_dict["parTT"])
            melt = torch.clamp(melt, min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = (
                param_dict["parCFR"] * param_dict["parCFMAX"] * (param_dict["parTT"] - Tm[t, :])
            )
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (param_dict["parCWH"] * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / param_dict["parFC"]) ** param_dict["parBETA"]
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness
            SM = SM + RAIN + tosoil - recharge
            excess = SM - param_dict["parFC"]
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            evapfactor = SM / (param_dict["parLP"] * param_dict["parFC"])
            if "parBETAET" in param_dict:
                evapfactor = evapfactor ** param_dict["parBETAET"]
            evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = PETm[t, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=self.nearzero)

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, param_dict["parPERC"])
            SUZ = SUZ - PERC
            Q0 = param_dict["parK0"] * torch.clamp(SUZ - param_dict["parUZL"], min=0.0)
            SUZ = SUZ - Q0
            Q1 = param_dict["parK1"] * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = param_dict["parK2"] * SLZ
            SLZ = SLZ - Q2

            Qsimmu[t, :] = Q0 + Q1 + Q2

        if warm_up:
            return {"SNOWPACK": SNOWPACK, "MELTWATER": MELTWATER, "SM": SM, "SUZ": SUZ, "SLZ": SLZ}

        return {"q_sim": Qsimmu}
