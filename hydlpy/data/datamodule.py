import json
import os
import pickle
from typing import Any, Optional, Dict

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from .dataset import HydroDataset 
# Import other dependencies from your project

class HydroDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for hydrological data from the CAMELS dataset.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        """
        Initializes the DataModule.

        Parameters
        ----------
        config
            Configuration dictionary, same as the one used in HydroLoader.
        batch_size
            The size of each data batch.
        num_workers
            Number of subprocesses to use for data loading.
        """
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Extract parameters from config
        self.phy_forcings = config['delta_model']['phy_model'].get('forcings', [])
        self.phy_attributes = config['delta_model']['phy_model'].get('attributes', [])
        self.nn_forcings = config['delta_model']['nn_model'].get('forcings', [])
        self.nn_attributes = config['delta_model']['nn_model'].get('attributes', [])
        self.forcing_names = config['observations']['all_forcings']
        self.attribute_names = config['observations']['all_attributes']
        self.target = config['train']['target']
        self.log_norm_vars = config['delta_model']['phy_model']['use_log_norm']
        self.device = config['device']
        
        self.rho = config['delta_model']['rho']
        self.warm_up = config['delta_model']['phy_model']['warm_up']
        
        self.norm_stats = {}
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Loads, preprocesses, and splits the data. This is called automatically by PL.
        
        - Calculates normalization stats ONLY on the training data.
        - Applies the same stats to validation and test data.
        """
        if stage == "fit" or stage is None:
            # Load and process training data
            train_data_raw = self._read_data(scope='train')
            
            # Calculate and save normalization stats
            self.norm_stats = self._init_norm_stats(
                x_nn=train_data_raw['x_nn'], 
                c_nn=train_data_raw['c_nn'], 
                target=train_data_raw['target']
            )

            # Normalize and create the final training dataset
            train_data_processed = self._process_data(train_data_raw)
            self.train_dataset = HydroDataset(train_data_processed, self.rho, self.warm_up)

            # Load, process, and create the validation dataset
            val_data_raw = self._read_data(scope='test')
            val_data_processed = self._process_data(val_data_raw)
            self.val_dataset = HydroDataset(val_data_processed, self.rho, self.warm_up)

        if stage == "test" or stage is None:
            # Load, process, and create the test dataset
            # Ensure norm_stats have been loaded (e.g., from a 'fit' stage or file)
            if not self.norm_stats:
                norm_path = os.path.join(self.config['model_path'], 'normalization_statistics.json')
                with open(norm_path, 'r') as f:
                    self.norm_stats = json.load(f)
            
            test_data_raw = self._read_data(scope='test')
            test_data_processed = self._process_data(test_data_raw)
            self.test_dataset = HydroDataset(test_data_processed, self.rho, self.warm_up)

    def _read_data(self, scope: str) -> Dict[str, NDArray[np.float32]]:
        """
        Reads raw data from pickle files based on the scope (train/test).
        This is adapted from the original `read_data` method.
        """
        # This logic is mostly copied from your original `read_data` method
        data_path = self.config['observations'].get('data_path')
        if scope == 'train':
            path = data_path or self.config['observations']['train_path']
            time = self.config['train_time']
        elif scope == 'test':
            path = data_path or self.config['observations']['test_path']
            time = self.config['test_time']
        else:
            raise ValueError("Scope must be 'train' or 'test'.")

        all_time = pd.date_range(
            self.config['all_time'][0], self.config['all_time'][-1], freq='d'
        )
        idx_start = all_time.get_loc(time[0])
        idx_end = all_time.get_loc(time[-1]) + 1

        with open(path, 'rb') as f:
            forcings, target, attributes = pickle.load(f)

        forcings = np.transpose(forcings[:, idx_start:idx_end], (1, 0, 2))
        target = np.transpose(target[:, idx_start:idx_end], (1, 0, 2))
        
        # Index selection logic (copied and simplified)
        phy_forc_idx = [self.forcing_names.index(f) for f in self.phy_forcings]
        phy_attr_idx = [self.attribute_names.index(a) for a in self.phy_attributes]
        nn_forc_idx = [self.forcing_names.index(f) for f in self.nn_forcings]
        nn_attr_idx = [self.attribute_names.index(a) for a in self.nn_attributes]

        x_phy = forcings[:, :, phy_forc_idx]
        c_phy = attributes[:, phy_attr_idx]
        x_nn = forcings[:, :, nn_forc_idx]
        c_nn = attributes[:, nn_attr_idx]

        # Convert flow to mm/day
        target = self._flow_conversion(c_nn, target)

        return {
            'x_phy': x_phy, 'c_phy': c_phy,
            'x_nn': x_nn, 'c_nn': c_nn,
            'target': target
        }

    def _process_data(self, raw_data: Dict) -> Dict[str, torch.Tensor]:
        """
        Normalizes raw data and converts it to a dictionary of torch tensors.
        """
        x_nn_norm, xc_nn_norm, c_nn_norm = self._normalize(
            raw_data['x_nn'], raw_data['c_nn']
        )
        
        # Convert all numpy arrays to torch tensors
        processed_data = {
            'x_phy': torch.from_numpy(raw_data['x_phy']).float(),
            'c_phy': torch.from_numpy(raw_data['c_phy']).float(),
            'x_nn': torch.from_numpy(raw_data['x_nn']).float(),
            'c_nn': torch.from_numpy(raw_data['c_nn']).float(),
            'target': torch.from_numpy(raw_data['target']).float(),
            'x_nn_norm': torch.from_numpy(x_nn_norm).float(),
            'xc_nn_norm': torch.from_numpy(xc_nn_norm).float(),
            'c_nn_norm': torch.from_numpy(c_nn_norm).float(),
        }
        return processed_data
    
    # ====================================================================
    # Helper methods (mostly copied from the original HydroLoader)
    # These could be further refactored into a separate utility class.
    # ====================================================================

    def _normalize(self, x_nn, c_nn):
        # This method is copied from the original `normalize`
        x_nn_norm = self._to_norm(np.swapaxes(x_nn, 1, 0).copy(), self.nn_forcings)
        c_nn_norm = self._to_norm(c_nn, self.nn_attributes)
        x_nn_norm[np.isnan(x_nn_norm)] = 0
        c_nn_norm[np.isnan(c_nn_norm)] = 0
        c_nn_norm_expand = np.repeat(np.expand_dims(c_nn_norm, 0), x_nn_norm.shape[0], axis=0)
        xc_nn_norm = np.concatenate((x_nn_norm, c_nn_norm_expand), axis=2)
        return x_nn_norm, xc_nn_norm, c_nn_norm
    
    def _init_norm_stats(self, x_nn, c_nn, target):
        # This method is copied from the original `_init_norm_stats`
        stat_dict = {}
        for k, var in enumerate(self.nn_forcings):
            if var in self.log_norm_vars:
                stat_dict[var] = self._calc_gamma_stats(x_nn[:, :, k])
            else:
                stat_dict[var] = self._calc_norm_stats(x_nn[:, :, k])
        for k, var in enumerate(self.nn_attributes):
            stat_dict[var] = self._calc_norm_stats(c_nn[:, k])
        for i, name in enumerate(self.target):
            stat_dict[name] = self._calc_norm_stats(np.swapaxes(target[:, :, i:i + 1], 1, 0))
        
        # Save stats to file
        out_path = os.path.join(self.config['model_path'], 'normalization_statistics.json')
        with open(out_path, 'w') as f:
            json.dump(stat_dict, f, indent=4)
        return stat_dict

    def _flow_conversion(self, c_nn, target):
        # This method is copied from the original `_flow_conversion`
        for name in ['flow_sim', 'streamflow', 'sf']:
            if name in self.target:
                target_idx = self.target.index(name)
                area_name = self.config['observations']['area_name']
                basin_area = c_nn[:, self.nn_attributes.index(area_name)]
                area = np.expand_dims(basin_area, axis=0).repeat(target.shape[0], 0)
                # Conversion factor: cfs to m3/s -> m3/day -> mm/day
                target[:, :, target_idx] = (target[:, :, target_idx] * 0.0283168 * 86400 * 1000) / (area * 1e6)
        return target

    # ... include _calc_norm_stats, _calc_gamma_stats, and _to_norm here ...
    # (These methods can be copied directly from your HydroLoader class)
    def _calc_norm_stats(self, x, basin_area=None):
        x[x == -999] = np.nan
        a = x.flatten()
        b = a[~np.isnan(a)]
        if b.size == 0:
            b = np.array([0])
        p10, p90 = np.percentile(b, [10, 90]).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)
        return [p10, p90, mean, max(std, 0.001)]

    def _calc_gamma_stats(self, x):
        a = np.swapaxes(x, 1, 0).flatten()
        b = a[~np.isnan(a)]
        b = np.log10(np.sqrt(b) + 0.1)
        p10, p90 = np.percentile(b, [10, 90]).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)
        return [p10, p90, mean, max(std, 0.001)]
    
    def _to_norm(self, data, vars):
        data_norm = np.zeros_like(data, dtype=np.float32)
        for k, var in enumerate(vars):
            stat = self.norm_stats[var]
            if data.ndim == 3:
                if var in self.log_norm_vars:
                    data[:, :, k] = np.log10(np.sqrt(data[:, :, k]) + 0.1)
                data_norm[:, :, k] = (data[:, :, k] - stat[2]) / stat[3]
            elif data.ndim == 2:
                if var in self.log_norm_vars:
                    data[:, k] = np.log10(np.sqrt(data[:, k]) + 0.1)
                data_norm[:, k] = (data[:, k] - stat[2]) / stat[3]
        return np.swapaxes(data_norm, 1, 0) if data_norm.ndim == 3 else data_norm
    
    # ====================================================================
    # DataLoader creation methods
    # ====================================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )