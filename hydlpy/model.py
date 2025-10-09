import inspect
from typing import Any, Dict

import pytorch_lightning as pl
import torch

from torchmetrics import MeanSquaredError
from .hydrology import HydrologicalModel
from .hydrology import HYDROLOGY_MODELS
from .routing import ROUTING_MODELS
from .estimators import DYNAMIC_ESTIMATORS, STATIC_ESTIMATORS


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
        self.static_estimator_name = self.hparams.get("static_estimator").get(
            "name", None
        )
        self.dynamic_estimator_name = self.hparams.get("dynamic_estimator").get(
            "name", None
        )
        self.hydrology_model_name = self.hparams.get("hydrology_model").get(
            "name", None
        )
        self.routing_model_name = self.hparams.get("routing_model").get("name", None)

        if self.static_estimator_name is not None:
            self.static_estimator = STATIC_ESTIMATORS[self.dynamic_estimator_name](
                **self.hparams.get("static_estimator")
            )
        else:
            self.static_estimator = None

        if self.dynamic_estimator is not None:
            self.dynamic_estimator = STATIC_ESTIMATORS[self.dynamic_estimator_name](
                **self.hparams.get("dynamic_estimator")
            )
        else:
            self.dynamic_estimator = None

        if self.hydrology_model_name is not None:
            self.hydrology_model = HYDROLOGY_MODELS[self.hydrology_model_name](
                **self.hparams.get("hydrology_model")
            )
        else:
            self.hydrology_model = None

        if self.routing_model_name is not None:
            self.routing_model = ROUTING_MODELS[self.routing_model_name](
                **self.hparams.get("routing_model")
            )
        else:
            self.routing_model = None

        self.warm_up = self.hparams["warmup"]
        self.loss_function = MeanSquaredError()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Defines the modular forward pass of the complete model.
        """
        x_forcing = batch["x_forcing"]
        x_forcing = x_forcing.unsqueeze(2).repeat(
            1, 1, self.hydrology_core.hidden_size, 1
        )

        parameters_dict = {}
        if self.static_param_estimator is not None:
            est_static_params = self.static_param_estimator(batch["x_static"])
            for key in est_static_params.keys():
                parameters_dict[key] = (
                    est_static_params[key]
                    .unsqueeze(0)
                    .repeat(x_forcing.shape[0], 1, 1, 1)
                )
        if self.dynamic_param_estimator is not None:
            est_dynamic_params = self.dynamic_param_estimator(batch["x_dynamic"])
            for key in est_dynamic_params.keys():
                parameters_dict[key] = est_static_params[key]
        parameters = torch.stack(
            parameters_dict[self.hydrology_core.parameter_names], dim=-1
        )

        # model warm up
        if self.warm_up > 0:
            _, states_ = self.hydrology_core(
                x_forcing[: self.warm_up, :, :, :],
                parameters[: self.warm_up, :, :, :],
            )
            warmup_states = states_[-1, :, :, :]

        # model forward
        fluxes, states = self.hydrology_core(
            x_forcing[: self.warm_up, :, :, :],
            warmup_states,
            parameters[: self.warm_up, :, :, :],
        )
        fluxes_dict = {
            k: v for (k, v) in zip(self.hydrology_core.flux_names, fluxes.unbind(-1))
        }
        states_dict = {
            k: v for (k, v) in zip(self.hydrology_core.state_names, states.unbind(-1))
        }
        fluxes_dict.update(states_dict)

        # routing module
        if self.routing_module is not None:
            routing_output = self.routing_module(fluxes_dict)
            fluxes_dict.update({"routing_output": routing_output})

        return fluxes_dict

    def _calculate_loss(self, batch: Dict[str, torch.Tensor]):
        y_true = batch["y"]
        y_pred = self.forward(batch)["y"]
        mask = ~torch.isnan(y_true)
        loss = self.loss_function(y_pred[mask], y_true[mask])
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
