# DIMENSION_MISMATCH

An array length does not match the problem dimension.

## Cause

The problem dimension `n` is inferred from the first call to `.initialPoint(x0)` as `x0.length`.
`DIMENSION_MISMATCH` is thrown when a subsequent array has a different length, for example:

- `bounds` array length ≠ `n`
- Calling `.initialPoint()` again with a different length after workspace allocation

## Example

```java
new LBFGSBProblem()
    .objective(x -> x[0] * x[0] + x[1] * x[1])
    .initialPoint(1.0, 1.0)          // n = 2
    .bounds(Bound.atLeast(0))        // length = 1, but n = 2 → DIMENSION_MISMATCH
    .solve();
```

## Fix

Make sure every array matches the dimension set by `initialPoint`:

```java
OptimizationResult r = Minimizer.lbfgsb()
    .objective(x -> x[0] * x[0] + x[1] * x[1])
    .initialPoint(1.0, 1.0)                          // n = 2
    .bounds(Bound.atLeast(0), Bound.atLeast(0))      // length = 2 ✓
    .solve();
```

For workspace reuse, always call `.initialPoint()` before changing dimension:

```java
LBFGSBProblem problem = new LBFGSBProblem()
    .objective(fn)
    .initialPoint(new double[n]);  // fixes dimension to n

LBFGSBWorkspace ws = problem.alloc();
for (double[] x0 : startPoints) {
    // x0.length must equal n
    OptimizationResult r = problem.initialPoint(x0).solve(ws);
}
```
