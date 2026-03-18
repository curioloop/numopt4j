# MISSING_PARAM

A required parameter was not set before calling `.solve()`.

## Cause

`OptimizationFailure` with code `MISSING_PARAM` is thrown when:

- `objective` (or `residuals`) was not provided
- `initialPoint` was not provided
- For `TRFProblem`: number of residuals was not set via `.residuals(fn, m)`

## Examples

```java
// Missing objective
new LBFGSBProblem()
    .initialPoint(1.0, 1.0)
    .solve(); // throws MISSING_PARAM: objective is required

// Missing initialPoint
new LBFGSBProblem()
    .objective(x -> x[0] * x[0])
    .solve(); // throws MISSING_PARAM: initialPoint is required

// TRF: missing residuals
new TRFProblem()
    .initialPoint(0.0, 0.0)
    .solve(); // throws MISSING_PARAM: residuals/objective is required
```

## Fix

Ensure all required parameters are set before calling `.solve()`:

```java
// L-BFGS-B
OptimizationResult r = Minimizer.lbfgsb()
    .objective(x -> x[0] * x[0] + x[1] * x[1])  // required
    .initialPoint(1.0, 1.0)                        // required
    .solve();

// TRF
OptimizationResult r = Minimizer.trf()
    .residuals((x, res) -> { res[0] = x[0] - 1; res[1] = x[1] - 2; }, 2)  // required
    .initialPoint(0.0, 0.0)                                                   // required
    .solve();
```
