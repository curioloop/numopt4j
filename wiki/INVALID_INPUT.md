# INVALID_INPUT

The initial point contains a non-finite value (`NaN` or `Infinity`).

## Cause

All optimizers require a finite starting point. `INVALID_INPUT` is thrown when any element of `initialPoint` is `NaN` or `±Infinity`.

Common sources:
- Computing the initial point from data that contains missing values or division by zero
- Passing a pre-allocated array that was never initialized (Java initializes `double[]` to `0.0`, so this is rare)
- Propagating `NaN` from a previous failed computation

## Example

```java
double[] x0 = computeInitialGuess(); // returns [1.0, NaN] due to a bug

new LBFGSBProblem()
    .objective(x -> x[0] * x[0] + x[1] * x[1])
    .initialPoint(x0)
    .solve(); // throws INVALID_INPUT: initialPoint[1] is NaN
```

## Fix

Validate or sanitize the initial point before passing it:

```java
double[] x0 = computeInitialGuess();
for (int i = 0; i < x0.length; i++) {
    if (!Double.isFinite(x0[i])) {
        x0[i] = 0.0; // fallback to a safe default
    }
}

OptimizationResult r = Minimizer.lbfgsb()
    .objective(x -> x[0] * x[0] + x[1] * x[1])
    .initialPoint(x0)
    .solve();
```
