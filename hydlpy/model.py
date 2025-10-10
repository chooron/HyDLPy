from typing import Any, Dict

import pytorch_lightning as pl
import torch

from torchmetrics import MeanSquaredError
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
        self.warm_up = self.hparams["warm_up"]
        self.hru_num = self.hparams["hru_num"]
        self.loss_function = MeanSquaredError()

        self.static_estimator = self._build_component(
            self.hparams.get("static_estimator", None), STATIC_ESTIMATORS
        )
        self.dynamic_estimator = self._build_component(
            self.hparams.get("dynamic_estimator", None), DYNAMIC_ESTIMATORS
        )
        self.hydrology_model = self._build_component(
            self.hparams.get("hydrology_model", None), HYDROLOGY_MODELS
        )
        self.routing_model = self._build_component(
            self.hparams.get("routing_model", None), ROUTING_MODELS
        )
        if self.hparams.get("alias_mapping", None) is not None:
            alias_mapping_config = self.hparams.get("alias_mapping")
            self.alias_mapping = {
                "x_phy": alias_mapping_config.get("x_phy", "x_phy"),
                "c_phy": alias_mapping_config.get("c_phy", "c_phy"),
                "x_nn_norm": alias_mapping_config.get("x_nn_norm", "x_nn_norm"),
                "c_nn_norm": alias_mapping_config.get("c_nn_norm", "c_nn_norm"),
                "xc_nn_norm": alias_mapping_config.get("xc_nn_norm", "xc_nn_norm"),
                "target": alias_mapping_config.get("target", "target"),
            }
        else:
            self.alias_mapping = {
                "x_phy": "x_phy",
                "c_phy": "c_phy",
                "x_nn_norm": "x_nn_norm",
                "c_nn_norm": "c_nn_norm",
                "xc_nn_norm": "xc_nn_norm",
                "target": "target",
            }

        # Ensure estimator parameters cover all hydrology parameters
        hydro_params = set(self.hydrology_model.parameter_names)
        static_params = (
            set(self.static_estimator.estimate_parameters)
            if self.static_estimator is not None
            else set()
        )
        dynamic_params = (
            set(self.dynamic_estimator.estimate_parameters)
            if self.dynamic_estimator is not None
            else set()
        )
        provided_params = static_params.union(dynamic_params)
        missing_params = hydro_params.difference(provided_params)

        assert not missing_params, (
            f"The following required hydrology model parameters are missing from the estimators: "
            f"{sorted(list(missing_params))}"
        )

    def _build_component(self, config, component_mapping):
        if config is not None:
            component_name = config.get("name", None)
            if component_name is not None:
                component = component_mapping[component_name](hru_num=self.hru_num, **config)
            else:
                component = None
        else:
            component = None
        return component

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Defines the modular forward pass of the complete model.
        """
        x_phy = batch[self.alias_mapping["x_phy"]]
        x_phy = x_phy.unsqueeze(2).repeat(1, 1, self.hru_num, 1)

        parameters_dict = {}
        if self.static_estimator is not None:
            est_static_params = self.static_estimator(
                batch[self.alias_mapping["c_nn_norm"]]
            )
            for key in est_static_params.keys():
                parameters_dict[key] = (
                    est_static_params[key].unsqueeze(0).repeat(x_phy.shape[0], 1, 1)
                )
            
        if self.dynamic_estimator is not None:
            est_dynamic_params = self.dynamic_estimator(
                batch[self.alias_mapping["xc_nn_norm"]]
            )
            for key in est_dynamic_params.keys():
                parameters_dict[key] = est_dynamic_params[key]

        parameters = torch.stack(
            [parameters_dict[k] for k in self.hydrology_model.parameter_names], dim=-1
        )
        # model warm up
        if self.warm_up > 0:
            _, states_ = self.hydrology_model(
                x_phy[: self.warm_up, :, :, :],
                parameters=parameters[: self.warm_up, :, :, :],
            )
            warmup_states = states_[-1, :, :, :]

        # model forward
        fluxes, states = self.hydrology_model(
            x_phy[: self.warm_up, :, :, :],
            states=warmup_states,
            parameters=parameters[: self.warm_up, :, :, :],
        )
        fluxes_dict = {
            k: v for (k, v) in zip(self.hydrology_model.flux_names, fluxes.unbind(-1))
        }
        states_dict = {
            k: v for (k, v) in zip(self.hydrology_model.state_names, states.unbind(-1))
        }
        fluxes_dict.update(states_dict)

        # routing module
        if self.routing_model is not None:
            routing_output = self.routing_model(fluxes_dict)
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
