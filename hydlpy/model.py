import inspect
from typing import Any, Dict

import pytorch_lightning as pl
import torch

# Use the updated factory functions
from .utils.factory import (
    create_dynamic_parameter_estimator,
    create_hydrology_core,
    create_routing_module,
    create_static_parameter_estimator,
)
from .hydrology import HydrologicalModel
from .utils.metrics import LogNSELoss


class DplHydroModel(pl.LightningModule):
    """
    A highly modular PyTorch Lightning wrapper for a differentiable hydrological model.

    This model is composed of optional and required modules:
    - Optional: Initial State Estimator (e.g., GRU)
    - Optional: Static Parameter Estimator (e.g., MLP from basin attributes)
    - Optional: Dynamic Parameter Estimator (e.g., LSTM from meteorological data)
    - Required: Hydrology Core (differentiable physics-based model)
    - Optional: Routing Module (e.g., MLP, Mean)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)

        # Initialize all components, using .get() to handle optional configs
        # self.initial_state_estimator = create_initial_state_estimator(
        #     self.hparams.get("initial_state_estimator")
        # )
        self.static_param_estimator = create_static_parameter_estimator(
            self.hparams.get("static_parameter_estimator")
        )
        self.dynamic_param_estimator = create_dynamic_parameter_estimator(
            self.hparams.get("dynamic_parameter_estimator")
        )
        self.hydrology_core: HydrologicalModel = create_hydrology_core(
            self.hparams.get("hydrology")
        )
        self.routing_module = create_routing_module(self.hparams.get("routing"))
        self.loss_fn = LogNSELoss()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Defines the modular forward pass of the complete model.
        """
        # 1. Estimate initial states (optional)
        # initial_states = None
        # if self.initial_state_estimator:
        #     initial_states = self.initial_state_estimator(
        #         x_dynamic_lookback=batch.get("x_dynamic_lookback")
        #     )

        # 2. Estimate and denormalize static and dynamic parameters (optional)
        static_params = {}
        if self.static_param_estimator:
            raw_static_params = self.static_param_estimator(batch.get("x_static", {}))
            static_params = self.hydrology_core._denormalize_parameters(
                raw_static_params, self.static_param_estimator.param_names
            )

        dynamic_params = {}
        if self.dynamic_param_estimator:
            raw_dynamic_params = self.dynamic_param_estimator(
                batch.get("x_dynamic", {})
            )
            dynamic_params = self.hydrology_core._denormalize_parameters(
                raw_dynamic_params, self.dynamic_param_estimator.param_names
            )

        # 3. Run the core hydrological model
        runoff = self.hydrology_core(
            x_dict=batch.get("x_forcing", {}),  # Pass the nested forcing dictionary
            static_params=static_params,
            dynamic_params=dynamic_params,
            initial_states={},
        )

        # 4. Run the routing module to get final streamflow
        routing_args = inspect.signature(self.routing_module.forward).parameters
        if "batch" in routing_args and "static_params" in routing_args:
            # This routing module needs more context (e.g., hydrofabric, parameters)
            predicted_streamflow = self.routing_module(
                runoff=runoff,
                batch=batch,
                static_params=static_params,
                dynamic_params=dynamic_params,
            )
        else:
            # Standard routing module just needs the runoff
            predicted_streamflow = self.routing_module(runoff)

        return predicted_streamflow

    def _calculate_loss(self, batch: Dict[str, torch.Tensor]):
        y_true = batch["y"]
        y_pred = self.forward(batch)["y"]
        mask = ~torch.isnan(y_true)
        loss = self.loss_fn(y_pred[mask], y_true[mask])
        return loss

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._calculate_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss = self._calculate_loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss = self._calculate_loss(batch)
        self.log("test_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.optimizer.get("lr", 1e-3)
        )
        return optimizer
