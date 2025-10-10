# 训练与评估

本节介绍如何用 PyTorch Lightning 训练 DplHydroModel（可选）。

## 优化器与损失

- 内置优化器：AdamW（可通过 `optimizer.lr` 配置学习率）
- 内置损失：MSE（`torchmetrics.MeanSquaredError`），可在自定义训练循环中替换

```python
from hydlpy.model import DplHydroModel

config = {
    # ... 同前文配置 ...
    "optimizer": {"lr": 1e-3},
}
model = DplHydroModel(config)
opt = model.configure_optimizers()
```

## 与 Lightning 集成

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

## 评估与推理

- 使用 `with torch.no_grad(): outputs = model(batch)` 获取推理结果
- 输出为字典：键为通量名/状态名，例如 `flow`, `soilwater` 等

## 常见问题
- 确保数据 batch 的键与形状满足配置要求
- 训练前先通过小批次验证前向是否稳定

