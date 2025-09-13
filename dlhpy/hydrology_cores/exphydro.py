from typing import Dict

import torch

from .base import BaseHydrologyModel


class ExpHydro(BaseHydrologyModel):
    """
    Exp-Hydro model implementation based on the provided formulas.
    This model simulates hydrological processes including evaporation, baseflow, and surface flow.
    """

    _name = "ExpHydro"
    _parameter_bounds = {
        "Smax": (0.1, 500.0),
        "Qmax": (5.0, 50.0),
        "f": (-3.0, -1.0),
    }

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **config,
    ) -> None:
        """
        Initializes the ExpHydro model.

        Parameters
        ----------
        device
            The device on which to run the model.
        config
            A dictionary of model configurations.
        """
        super().__init__(device=device, **config)
        # self.nmul is initialized in the base class and represents the number of parallel units.

    def _initialize_states(self) -> Dict[str, torch.Tensor]:
        """Initializes the soilwater state tensor."""
        return {"soilwater": torch.zeros(self.nmul, device=self.device)}

    def _step_func(self, x: torch.Tensor) -> torch.Tensor:
        """The step function: (tanh(5.0 * x) + 1.0) * 0.5"""
        return (torch.tanh(5.0 * x) + 1.0) * 0.5

    def _process_core(
        self,
        x_dict: Dict[str, torch.Tensor],
        states: Dict[str, torch.Tensor],
        static_params: Dict[str, torch.Tensor],
        dynamic_params: Dict[str, torch.Tensor],
        return_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs the core of the Exp-Hydro model for one or more time steps.
        """
        prcp = x_dict["prcp"]
        pet = x_dict["pet"]

        n_timesteps = prcp.shape[0]

        # Initialize output tensor for flow
        flow_out = torch.zeros_like(prcp, device=self.device)
        soilwater_arr = torch.zeros_like(prcp, device=self.device)

        # Get initial state
        soilwater = states["soilwater"]

        # Get parameters
        Smax = static_params["Smax"]
        Qmax = static_params["Qmax"]
        f = static_params["f"]

        for t in range(n_timesteps):
            # Evaporation
            evap = pet[t] * torch.min(
                torch.ones_like(soilwater, device=self.device), soilwater / (Smax + self.nearzero)
            )
            # Baseflow
            step_val = self._step_func(soilwater)
            exp_term = torch.exp(
                -torch.pow(10, f) * torch.max(torch.zeros_like(soilwater, device=self.device), Smax - soilwater)
            )
            baseflow_potential = step_val * Qmax * exp_term
            baseflow = torch.min(soilwater, baseflow_potential)

            # Surface flow
            surfaceflow = torch.max(torch.zeros_like(soilwater, device=self.device), soilwater - Smax)

            # Total flow
            flow = baseflow + surfaceflow
            flow_out[t] = flow

            # Update soilwater state
            soilwater = soilwater + prcp[t] - (evap + flow)
            soilwater = torch.max(torch.zeros_like(soilwater, device=self.device), soilwater)
            soilwater_arr[t] = soilwater

        if return_states:
            return {"soilwater": soilwater}
        else:
            return {"q_sim": flow_out}
