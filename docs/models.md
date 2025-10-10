# Built-in Hydrology Models

This section lists built-in models and key parameter names to help you configure estimators correctly.

## ExpHydro
- Inputs: `[prcp, pet, temp]`
- Typical parameters: `[Tmin, Tmax, Df, Smax, Qmax, f]`
  - Split example:
    - Static: `[Tmin, Tmax, Df, Smax]`
    - Dynamic: `[Qmax, f]`

## HBV
- Inputs: `[P, Ep, T]`
- Reference parameters (based on implementation):
  - `TT`, `CFMAX`, `CWH`, `CFR`, `FC`, `LP`, `BETA`, `PPERC`, `UZL`, `k0`, `k1`, `k2`
- Split example:
  - Static: `TT, CFMAX, CWH, CFR, FC, LP, BETA, k1, k2, UZL`
  - Dynamic: `BETA, PPERC, k0`
  - Note: The union of static and dynamic must cover the full hydrology parameter set; overlap is allowed (e.g., BETA)

> Tip: the actual parameter names are defined in `hydlpy/hydrology/implements/*.py`. Check the source for ground truth.
