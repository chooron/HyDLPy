# 内置物理模型

本节列出内置模型与关键参数名，便于编写正确的估计器配置。

## ExpHydro
- 输入名：`[prcp, pet, temp]`
- 典型参数名：`[Tmin, Tmax, Df, Smax, Qmax, f]`
  - 可拆分为：
    - 静态：`[Tmin, Tmax, Df, Smax]`
    - 动态：`[Qmax, f]`

## HBV
- 输入名：`[P, Ep, T]`
- 参考参数名（根据实现）：
  - `TT`, `CFMAX`, `CWH`, `CFR`, `FC`, `LP`, `BETA`, `PPERC`, `UZL`, `k0`, `k1`, `k2`
- 示例拆分：
  - 静态：`TT, CFMAX, CWH, CFR, FC, LP, BETA, k1, k2, UZL`
  - 动态：`BETA, PPERC, k0`
  - 注意：静态与动态合并后须覆盖全部物理参数集合，允许交叉（例如 BETA）

> 提示：实际参数名以 `hydlpy/hydrology/implements/*.py` 为准，可直接查看源码。
