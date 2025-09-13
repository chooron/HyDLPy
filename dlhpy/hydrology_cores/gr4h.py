from typing import Dict, Union

import torch
import torch.nn.functional as F

from .base import BaseHydrologyModel


class GR4H(BaseHydrologyModel):
    """GR4H model implementation.

    This model is a variant of the GR4J model, adapted for hourly time steps.
    It consists of a production store, a routing store, and two unit hydrographs.

    Parameters
    ----------
    x1 : float
        Production store capacity (mm).
    x2 : float
        Groundwater exchange coefficient (mm/h).
    x3 : float
        Routing store capacity (mm).
    x4 : float
        Time base of the unit hydrograph (h).
    beta : float
        Coefficient for the percolation from the production store.
    """

    _name = "GR4H"
    _parameter_bounds = {
        "x1": (1.0, 500.0),  # Production store capacity
        "x2": (-5.0, 5.0),  # Water exchange coefficient
        "x3": (1.0, 500.0),  # Routing store capacity
        "x4": (0.5, 4.0),  # Time constant of unit hydrograph
        "beta": (0.1, 2.0),  # Percolation coefficient
    }

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **config,
    ) -> None:
        super().__init__(device=device, **config)
        self.nh = config.get("nh", 480)  # number of steps for unit hydrograph
        self.d = config.get("d", 1.25)  # exponent for unit hydrograph

    def _initialize_states(self) -> Dict[str, torch.Tensor]:
        """Return initial states for the model."""
        return {
            "slw": torch.zeros(self.nmul, device=self.device) + self.nearzero,
            "rts": torch.zeros(self.nmul, device=self.device) + self.nearzero,
        }

    def _ss1(self, i: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        """S-curve of the first unit hydrograph."""
        i = i.unsqueeze(-1)  # (nh, 1)
        x4 = x4.unsqueeze(0)  # (1, n_basins)

        zeros = torch.zeros_like(i, device=self.device)
        ones = torch.ones_like(i, device=self.device)

        cond1 = i <= 0
        cond2 = i < x4

        res = torch.where(cond1, zeros, torch.where(cond2, (i / x4) ** self.d, ones))
        return res

    def _ss2(self, i: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        """S-curve of the second unit hydrograph."""
        i = i.unsqueeze(-1)  # (nh, 1)
        x4 = x4.unsqueeze(0)  # (1, n_basins)

        zeros = torch.zeros_like(i, device=self.device)
        ones = torch.ones_like(i, device=self.device)

        cond1 = i <= 0
        cond2 = i <= x4
        cond3 = i < 2 * x4

        term1 = 0.5 * (i / x4) ** self.d
        term2 = 1.0 - 0.5 * (2.0 - i / x4) ** self.d

        res = torch.where(cond1, zeros, torch.where(cond2, term1, torch.where(cond3, term2, ones)))
        return res

    def _uh(self, ss_fun, x4: torch.Tensor, nh: int) -> torch.Tensor:
        """Calculate the unit hydrograph from the S-curve."""
        i = torch.arange(1, nh + 1, device=self.device, dtype=torch.float32)
        ss_i = ss_fun(i, x4)

        i_minus_1 = torch.arange(0, nh, device=self.device, dtype=torch.float32)
        ss_i_minus_1 = ss_fun(i_minus_1, x4)

        uh = ss_i - ss_i_minus_1
        return uh  # shape (nh, n_basins)

    def _convolution(self, q: torch.Tensor, uh: torch.Tensor) -> torch.Tensor:
        """Performs convolution using 1D convolution for each basin.

        q: (n_timesteps, n_basins)
        uh: (nh, n_basins)

        returns: (n_timesteps, n_basins)
        """
        n_timesteps, n_basins = q.shape
        nh = uh.shape[0]

        q_routed_list = []
        for i in range(n_basins):
            q_i = q[:, i].reshape(1, 1, -1)  # (1, 1, n_timesteps)
            uh_i = uh[:, i].reshape(1, 1, -1)  # (1, 1, nh)
            uh_i_flipped = torch.flip(uh_i, [2])

            q_i_padded = F.pad(q_i, (nh - 1, 0))
            q_routed_i = F.conv1d(q_i_padded, uh_i_flipped, padding=0)
            q_routed_list.append(q_routed_i.squeeze())

        return torch.stack(q_routed_list, dim=1)

    def _process_core(
        self,
        x_dict: Dict[str, torch.Tensor],
        states: Dict[str, torch.Tensor],
        static_params: Dict[str, torch.Tensor],
        dynamic_params: Dict[str, torch.Tensor],
        return_states: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        prcp = x_dict["prcp"]  # (n_timesteps, n_basins)
        pet = x_dict["pet"]  # (n_timesteps, n_basins)

        n_timesteps, n_basins = prcp.shape

        x1 = static_params["x1"]
        x2 = static_params["x2"]
        x3 = static_params["x3"]
        x4 = static_params["x4"]
        beta = static_params["beta"]

        slw = states["slw"]
        rts = states["rts"]

        # --- Soil moisture accounting part ---
        pr_list = []
        slw_list = []
        for t in range(n_timesteps):
            prcp_t = prcp[t, :]
            pet_t = pet[t, :]

            en = pet_t - torch.min(prcp_t, pet_t)
            pn = prcp_t - torch.min(prcp_t, pet_t)

            slw = torch.max(slw, torch.tensor(1e-6, device=self.device))

            slw_x1_ratio = slw / x1
            tanh_en_x1 = torch.tanh(en / x1)
            tanh_pn_x1 = torch.tanh(pn / x1)

            evap = slw * (2.0 - slw_x1_ratio) * tanh_en_x1 / (1.0 + (1.0 - slw_x1_ratio) * tanh_en_x1)

            ps_denominator = 1.0 + slw_x1_ratio * tanh_pn_x1
            ps = x1 * (1.0 - torch.pow(slw_x1_ratio, 2)) * tanh_pn_x1 / ps_denominator

            perc_term = torch.pow(slw / (beta * x1), 4)
            perc = slw * (1.0 - torch.pow(1.0 + perc_term, -0.25))

            slw = slw + ps - evap - perc
            pr = pn - ps + perc

            pr_list.append(pr)
            slw_list.append(slw)

        pr_arr = torch.stack(pr_list)

        # --- Routing part ---
        q9_arr = pr_arr * 0.9
        q1_arr = pr_arr * 0.1

        uh_q9 = self._uh(self._ss1, x4, self.nh)
        uh_q1 = self._uh(self._ss2, x4, 2 * self.nh)

        q9_routed = self._convolution(q9_arr, uh_q9)
        q1_routed = self._convolution(q1_arr, uh_q1)

        # --- Routing store part ---
        q_list = []
        for t in range(n_timesteps):
            q9_t = q9_routed[t, :]
            q1_t = q1_routed[t, :]

            exch = x2 * (rts / x3) ** 3.5
            rts = torch.max(rts, torch.tensor(1e-6, device=self.device))

            rts = rts + q9_t + exch
            rts = torch.max(torch.tensor(0.0, device=self.device), rts)

            qr = rts * (1.0 - (1.0 + (rts / x3) ** 4) ** (-0.25))

            rts = rts - qr
            qd = torch.max(torch.tensor(0.0, device=self.device), q1_t + exch)
            q = qr + qd

            q_list.append(q)

        q_sim = torch.stack(q_list)

        final_states = {"slw": slw, "rts": rts}
        outputs = {"q_sim": q_sim}

        if return_states:
            return final_states
        else:
            return outputs