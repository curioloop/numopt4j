# INVALID_VALUE

A configuration parameter value violates its documented constraint.

## Cause

Fluent setter methods validate their arguments immediately. `INVALID_VALUE` is thrown when a value is out of the allowed range, for example:

| Parameter | Constraint |
|-----------|-----------|
| `maxIterations` | must be > 0 |
| `maxEvaluations` | must be > 0 |
| `gradientTolerance` | must be > 0 and finite |
| `functionTolerance` | must be > 0 and finite |
| `parameterTolerance` | must be > 0 |
| `corrections` (L-BFGS-B) | must be > 0 |
| `lossScale` (TRF) | must be > 0 |

## Examples

```java
new LBFGSBProblem()
    .maxIterations(-1);       // throws: maxIterations must be positive, got -1

new LBFGSBProblem()
    .gradientTolerance(0.0);  // throws: gradientTolerance must be positive, got 0.0

new TRFProblem()
    .lossScale(-1.0);         // throws: lossScale must be positive, got -1.0
```

## Fix

Use positive, finite values for all tolerance and limit parameters:

```java
OptimizationResult r = Minimizer.lbfgsb()
    .objective(x -> x[0] * x[0] + x[1] * x[1])
    .initialPoint(1.0, 1.0)
    .maxIterations(500)          // > 0 ✓
    .gradientTolerance(1e-8)     // > 0 ✓
    .functionTolerance(1e-10)    // > 0 ✓
    .solve();
```
