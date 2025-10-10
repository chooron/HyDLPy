# Training and Evaluation

This section shows how to train `DplHydroModel` with PyTorch Lightning (optional).

## Optimizer and Loss

- Built-in optimizer: AdamW (set learning rate via `optimizer.lr`)
- Built-in loss: MSE (`torchmetrics.MeanSquaredError`); replaceable in a custom loop

```python
from hydlpy.model import DplHydroModel

config = {
    # ... 同前文配置 ...
    "optimizer": {"lr": 1e-3},
}
model = DplHydroModel(config)
opt = model.configure_optimizers()
```

## Lightning Integration

```python
import pytorch_lightning as pl
# from hydlpy.data import HydroDataModule  # 可选

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
)

# data_module = HydroDataModule(...)
# trainer.fit(model, data_module)
```

## Evaluation and Inference

- Use `with torch.no_grad(): outputs = model(batch)` for inference
- Output is a dict: keys are flux/state names such as `flow`, `soilwater`, etc.

## Tips
- Ensure batch keys and shapes satisfy configuration requirements
- Validate forward stability on a small batch before full training

