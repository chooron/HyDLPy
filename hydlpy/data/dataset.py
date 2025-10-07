# hydro_dataset.py

import torch
from torch.utils.data import Dataset
from typing import Dict

class HydroDataset(Dataset):
    """
    PyTorch Dataset for hydrological time-sequence data.

    Each item in this dataset represents a single training sequence of length
    (rho + warm_up) from a specific basin.
    """
    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        rho: int,
        warm_up: int,
    ) -> None:
        """
        Initializes the Dataset.

        Parameters
        ----------
        data
            A dictionary of pre-processed and normalized data tensors.
            Dynamic tensors should have shape [time, basin, features].
            Static tensors should have shape [basin, features].
        rho
            The length of the prediction sequence.
        warm_up
            The length of the warm-up period.
        """
        super().__init__()
        self.data = data
        self.rho = rho
        self.warm_up = warm_up
        self.sequence_length = self.rho + self.warm_up
        
        # Get dimensions from the data
        # Assuming 'x_phy' is always present and has shape [time, basins, features]
        self.num_timesteps, self.num_basins, _ = self.data['x_phy'].shape
        
        # The total number of samples is every possible start time in every basin
        self.num_possible_starts = self.num_timesteps - self.sequence_length + 1
        if self.num_possible_starts <= 0:
            raise ValueError(
                "The total number of timesteps is smaller than the required "
                f"sequence length (rho + warm_up = {self.sequence_length})."
            )

    def __len__(self) -> int:
        """Returns the total number of possible sequences."""
        return self.num_basins * self.num_possible_starts

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Fetches a single training sequence.

        The index `idx` is mapped to a specific basin and a start time.
        
        Parameters
        ----------
        idx
            A flat index representing a basin-time combination.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing all data tensors for a single training sample.
        """
        # Deconstruct the flat index to a basin index and time index
        basin_idx = idx // self.num_possible_starts
        start_time_idx = idx % self.num_possible_starts
        end_time_idx = start_time_idx + self.sequence_length

        # --- Slice the data to create the sample ---
        # Dynamic variables (time-series)
        x_phy = self.data['x_phy'][start_time_idx:end_time_idx, basin_idx, :]
        x_nn_norm = self.data['x_nn_norm'][start_time_idx:end_time_idx, basin_idx, :]
        xc_nn_norm = self.data['xc_nn_norm'][start_time_idx:end_time_idx, basin_idx, :]
        
        # Target variable (slice warm-up period)
        target = self.data['target'][start_time_idx:end_time_idx, basin_idx, :]
        target_final = target[self.warm_up:, :]
        
        # Static variables (attributes)
        c_phy = self.data['c_phy'][basin_idx, :]
        c_nn = self.data['c_nn'][basin_idx, :]
        c_nn_norm = self.data['c_nn_norm'][basin_idx, :]

        return {
            'x_phy': x_phy,
            'c_phy': c_phy,
            'c_nn': c_nn,
            'c_nn_norm': c_nn_norm,
            'x_nn_norm': x_nn_norm,
            'xc_nn_norm': xc_nn_norm,
            'target': target_final,
        }