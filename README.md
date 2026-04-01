# numopt4j

High-performance numerical optimization library for Java.

## Features

- **Subplex**: Derivative-free optimization (Nelder-Mead + subspace decomposition, no gradient required)
- **L-BFGS-B**: Limited-memory BFGS with bound constraints
- **SLSQP**: Sequential Least Squares Programming with equality/inequality constraints
- **TRF**: Trust Region Reflective for nonlinear least squares
- **Root finding**: Brentq (1-D), HYBR and Broyden (N-D) via `RootFinder`
- **Numerical integration**: adaptive GK15, fixed Gauss-Legendre, oscillatory, improper, endpoint-singular, Cauchy principal value, and sampled-data quadrature via `Integrator`
- **Linear regression**: OLS and WLS with SVD/QR solvers, full statistical output via `Regressor`
- **Matrix decompositions**: LU, QR, LQ, SVD, Cholesky/LDLᵀ, Schur, Eigen, GEVD, GGEVD, GSVD via `Decomposer`
- Workspace reuse for high-frequency scenarios
- Multiple numerical gradient/Jacobian methods with different accuracy/speed tradeoffs

## Requirements

- Java 8+
- Native library for your platform (included for darwin-aarch64)

## Installation

```xml
<dependency>
    <groupId>com.curioloop</groupId>
    <artifactId>numopt4j</artifactId>
    <version>${version}</version>
</dependency>
```

## AI Assistant Integration

If you are using an AI coding assistant (e.g. GitHub Copilot, Cursor, Claude), you can provide the full API documentation by referencing `llms.txt` or `llms-full.txt` in the project root.

## Quick Start

### Derivative-Free Optimization (Subplex)

```java
// No gradient required — works for any dimension
Optimization result = Minimizer.subplex()
    .objective(x -> x[0]*x[0] + x[1]*x[1])
    .initialPoint(1.0, 1.0)
    .solve();

// High-dimensional with bounds
Optimization result = Minimizer.subplex()
    .objective(x -> { double s = 0; for (double v : x) s += v*v; return s; })
    .initialPoint(new double[20])
    .bounds(...)
    .functionTolerance(1e-8)
    .maxEvaluations(50000)
    .solve();
```

### Unconstrained Optimization (L-BFGS-B)

```java
Optimization result = Minimizer.lbfgsb()
    .objective(x -> x[0]*x[0] + x[1]*x[1])
    .initialPoint(1.0, 1.0)
    .solve();

if (result.isSuccessful()) {
    System.out.println("Solution: " + Arrays.toString(result.getSolution()));
}
```

### With Analytical Gradient

```java
Optimization result = Minimizer.lbfgsb()
    .objective((x, g) -> {
        double f = x[0]*x[0] + x[1]*x[1];
        if (g != null) { g[0] = 2*x[0]; g[1] = 2*x[1]; }
        return f;
    })
    .initialPoint(1.0, 1.0)
    .solve();
```

### Bound Constraints

```java
Optimization result = Minimizer.lbfgsb()
    .objective(x -> x[0]*x[0] + x[1]*x[1])
    .bounds(Bound.between(0, 10), Bound.between(0, 10))
    .initialPoint(1.0, 1.0)
    .solve();
```

### Constrained Optimization (SLSQP)

```java
// Equality constraint: x[0] + x[1] = 1
// Inequality constraint: x[0] >= 0.5
Optimization result = Minimizer.slsqp()
    .objective(x -> x[0]*x[0] + x[1]*x[1])
    .equalityConstraints(x -> x[0] + x[1] - 1)
    .inequalityConstraints(x -> x[0] - 0.5)
    .initialPoint(0.5, 0.5)
    .solve();
```

### Nonlinear Least Squares (TRF)

```java
// Fit y = a * exp(-b * t)
double[] tData = {0.0, 1.0, 2.0, 3.0};
double[] yData = {2.0, 1.2, 0.7, 0.4};

Optimization result = Minimizer.trf()
    .residuals((x, r) -> {
        for (int i = 0; i < tData.length; i++) {
            r[i] = yData[i] - x[0] * Math.exp(-x[1] * tData[i]);
        }
    }, tData.length)
    .bounds(Bound.atLeast(0), Bound.atLeast(0))
    .initialPoint(1.0, 0.5)
    .solve();
```

### Linear Regression (OLS / WLS)

```java
// OLS with SVD solver (X is overwritten in-place)
OLS r = Regressor.ols(y, X, n, k, Regressor.Opts.PINV);

// OLS with QR solver (faster when X is full rank)
OLS r = Regressor.ols(y, X, n, k, Regressor.Opts.QR);

// WLS with per-observation weights (X is overwritten in-place)
WLS r = Regressor.wls(y, X, weights, n, k, Regressor.Opts.PINV);

// Workspace reuse across multiple fits
OLS.Pool ws = new OLS.Pool();
for (double[] Xi : series) {
    OLS r = Regressor.ols(y, Xi.clone(), n, k, ws, Regressor.Opts.PINV);
    double[] beta = r.params();
    double   r2   = r.r2(false);
}

// Statistical output
double[] beta = r.params();   // β̂
double[] bse  = r.bse();          // standard errors
double   r2   = r.r2(false);      // R²
double   r2a  = r.r2(true);       // adjusted R²
double   llf  = r.logLike();      // log-likelihood
double   aic  = r.aic();
double   bic  = r.bic();

// Prediction intervals
Prediction pred = r.predict(newX, m, null);
double[][] ci   = pred.confInt(0.05);  // 95% prediction interval
// ci[0] = lower bounds, ci[1] = upper bounds, ci[2] = std errors
```

### Root Finding (1-D Brentq)

```java
// Find root of sin(x) in [3, 4] → π
Optimization result = RootFinder.brentq(Math::sin)
    .bracket(Bound.between(3.0, 4.0))
    .solve();

double root = result.getRoot(); // ≈ π
```

### Root Finding (N-D HYBR / Broyden)

```java
// Powell hybrid method (HYBR)
Optimization result = RootFinder.hybr((x, f) -> {
        f[0] = x[0]*x[0] - 2;
        f[1] = x[1] - x[0];
    }, 2)
    .initialPoint(1.0, 1.0)
    .solve();

double[] solution = result.getSolution(); // [√2, √2]

// Broyden (Jacobian-free)
result = RootFinder.broyden((x, f) -> {
        f[0] = x[0]*x[0] - 2;
        f[1] = x[1] - x[0];
    }, 2)
    .initialPoint(1.0, 1.0)
    .solve();

// Use central differences for Jacobian (HYBR only)
result = RootFinder.hybr(fn, 2)
    .jacobian(NumericalJacobian.CENTRAL)
    .initialPoint(1.0, 1.0)
    .solve();
```

### Workspace Reuse

For high-frequency optimization, reuse workspace to reduce allocation overhead:

```java
LBFGSBProblem problem = Minimizer.lbfgsb()
    .objective(x -> x[0]*x[0] + x[1]*x[1])
    .initialPoint(new double[n]);

LBFGSBWorkspace workspace = problem.alloc();  // allocate once
for (double[] point : points) {
    Optimization result = problem.initialPoint(point).solve(workspace);
    // process result
}

// Root finding workspace reuse
HYBRProblem finder = RootFinder.hybr(fn, 2).initialPoint(0.0, 0.0);
HYBRWorkspace ws = finder.alloc();
for (double[] x0 : initialPoints) {
    Optimization r = finder.initialPoint(x0).solve(ws);
}
```

### Quadrature (Numerical Integration)

```java
import com.curioloop.numopt4j.quad.*;
import com.curioloop.numopt4j.quad.gauss.*;
import com.curioloop.numopt4j.quad.adapt.*;
import com.curioloop.numopt4j.quad.special.*;
import com.curioloop.numopt4j.quad.sampled.*;

// Fixed Gauss-Legendre on [a, b]
double v = Integrator.fixed()
    .function(x -> Math.exp(-x * x)).bounds(0.0, 1.0).points(8).integrate();

// Adaptive GK15 on [a, b] with error estimate
Quadrature r = Integrator.adaptive()
    .function(Math::sin).bounds(0.0, Math.PI).tolerances(1e-10, 1e-10).integrate();
System.out.printf("value=%.10f  error=%.2e%n", r.getValue(), r.getEstimatedError());

// Oscillatory: ∫₀^∞ e^{-x}·cos(2x) dx
Quadrature r2 = Integrator.oscillatory(OscillatoryOpts.COS_UPPER)
    .function(x -> Math.exp(-x)).lowerBound(0.0).omega(2.0)
    .tolerances(1e-10, 1e-10).integrate();

// Improper: ∫₀^∞ e^{-x} dx  (adaptive with error control)
Quadrature r3 = Integrator.improper(ImproperOpts.UPPER)
    .function(x -> Math.exp(-x)).lowerBound(0.0).tolerances(1e-10, 1e-10).integrate();

// Endpoint-singular: ∫₋₁^1 (1-x)^{-0.5}(1+x)^{-0.5} f(x) dx
Quadrature r4 = Integrator.endpointSingular(EndpointOpts.ALGEBRAIC)
    .function(x -> 1.0).bounds(-1.0, 1.0).exponents(-0.5, -0.5)
    .tolerances(1e-10, 1e-10).integrate();

// Cauchy principal value: P.V. ∫₀^1 f(x)/(x-0.5) dx
Quadrature r5 = Integrator.principalValue()
    .function(x -> 1.0).bounds(0.0, 1.0).pole(0.5).tolerances(1e-12, 1e-12).integrate();

// Sampled data
double total = Integrator.sampled(SampledRule.SIMPSON).samples(y, dx).integrate();
double[] cumulative = Integrator.cumulative(SampledRule.TRAPEZOIDAL).samples(y, dx).integrate();

// Workspace reuse
AdaptiveIntegral problem = Integrator.adaptive()
    .function(Math::sin).bounds(0.0, Math.PI).tolerances(1e-10, 1e-10);
AdaptivePool ws = problem.alloc();
for (double[] bounds : intervals) {
    Quadrature result = problem.bounds(bounds[0], bounds[1]).integrate(ws);
}
```

## API Reference

### Minimizer (facade — static factory entry point)

```java
Minimizer.subplex()  // → SubplexProblem (Nelder-Mead)
Minimizer.lbfgsb()   // → LBFGSBProblem
Minimizer.slsqp()    // → SLSQPProblem
Minimizer.trf()      // → TRFProblem
```

### RootFinder (facade — static factory entry point)

```java
RootFinder.brentq(DoubleUnaryOperator f)                    // → BrentqProblem
RootFinder.hybr(BiConsumer<double[],double[]> fn, int n)    // → HYBRProblem
RootFinder.broyden(BiConsumer<double[],double[]> fn, int n) // → BroydenProblem
```

### Regressor (facade — linear regression)

```java
Regressor.ols(y, X, n, k, Opts...)           // OLS, must specify Opts.QR or Opts.PINV
Regressor.ols(y, X, n, k, Pool, Opts...)     // OLS with workspace reuse
Regressor.wls(y, X, w, n, k, Opts...)        // WLS
Regressor.wls(y, X, w, n, k, Pool, Opts...)  // WLS with workspace reuse
```

`Opts.QR` — QR factorization (faster, full-rank X); `Opts.PINV` — SVD pseudoinverse (robust, rank-deficient X); `Opts.HAS_CONST` — declare X has a constant column (kConst=1, skip detection); `Opts.NO_CONST` — declare X has no constant column (kConst=0, skip detection).

**Both OLS and WLS overwrite X in-place.** y is never modified. WLS additionally writes whitened y~ into `WLS.Pool.yWhiten`.

Key result methods on `Regression` (base of `OLS`/`WLS`):

| Method | Description |
|---|---|
| `params()` | β̂ (length k) |
| `bse()` | standard errors √diag(Cov(β̂)) |
| `paramCov()` | Cov(β̂) = σ̂²·(XᵀX)⁻¹, k×k |
| `ssr()` | sum of squared residuals |
| `scale()` | σ̂² = SSR / (n − rank) |
| `r2(boolean)` | R² (pass `true` for adjusted) |
| `mse()` | double[3] = {MSE_model, MSE_residual, MSE_total} |
| `logLike()` | Gaussian log-likelihood |
| `aic()` / `bic()` | information criteria |
| `fitted(boolean)` | ŷ = Xβ̂ (pass `true` for whitened) |
| `residual(boolean)` | e = y − ŷ (pass `true` for whitened) |
| `predict(newX, m, w)` | `Prediction` with mean(), paramVar(), residualVar() |
| `nObs()` / `nParams()` / `kConst()` | dimensions |
| `rank()` / `condNum()` | numerical rank and condition number |

### Decomposer (facade — matrix decompositions)

```java
// Standard decompositions
LU       lu  = Decomposer.lu(A, n);                          // LU with partial pivoting
QR       qr  = Decomposer.qr(A, m, n);                      // QR for tall/square matrices (m >= n)
LQ       lq  = Decomposer.lq(A, m, n);                      // LQ for wide/square matrices (m <= n)
SVD      svd = Decomposer.svd(A, m, n);                      // SVD, thin U and Vᵀ by default
Cholesky ch  = Decomposer.cholesky(A, n);                    // Cholesky (or LDLᵀ with PIVOTING)
Schur    sc  = Decomposer.schur(A, n);                       // Real Schur: A = Z·T·Zᵀ

// Eigenvalue decompositions
Eigen    eg  = Decomposer.eigen(A, n);                       // General eigen (right vectors)
Eigen    egs = Decomposer.eigen(A, n, Eigen.Opts.SYMMETRIC_LOWER); // Symmetric eigen
GEVD     gv  = Decomposer.gevd(A, B, n);                    // Generalized symmetric-definite
GGEVD    gg  = Decomposer.ggevd(A, B, n);                   // Generalized non-symmetric

// Generalized SVD
GSVD     gs  = Decomposer.gsvd(A, m, n, B, p);              // GSVD of A (m×n) and B (p×n)

// With options
QR  qrp  = Decomposer.qr(A, m, n, QR.Opts.PIVOTING);        // rank-revealing QR for m >= n
SVD svdU = Decomposer.svd(A, m, n, SVD.Opts.FULL_U, SVD.Opts.FULL_V);
GEVD gv2 = Decomposer.gevd(A, B, n, GEVD.Opts.UPPER, GEVD.Opts.TYPE2);

// Workspace reuse
LU.Pool ws = Decomposer.lu(A, n).pool();
for (double[] mat : matrices) {
    LU result = Decomposer.lu(mat, n, ws);
}

#### Decomposition result methods

| Class | Key result methods |
|---|---|
| `LU` | `toL()`, `toU()`, `toP()`, `solve(b,x)`, `inverse(Ainv)`, `determinant()`, `cond()` |
| `QR` | `toQ()` full `m×m`, `toR()` full `m×n`, `toP()` (pivoted only), `solve(b,x)` for square systems, `leastSquares(b,x)` for tall/square systems, `rank()`, `cond()` |
| `LQ` | `toL()` full `m×n`, `toQ()` full `n×n`, `solve(b,x)` minimum-norm solve for wide/square systems, `leastSquares(b,x)` alias of `solve(b,x)`, `cond()` |
| `SVD` | `toU()`, `toVT()`, `singularValues()`, `solve(b,x)`, `rank()`, `cond()` |
| `Cholesky` | `toL()`, `toD()` (LDLᵀ only), `solve(b,x)`, `inverse(Ainv)`, `determinant()`, `cond()` |
| `Schur` | `toT()`, `toZ()`, `toS()`, `eigenvalues()`, `lyapunov(Q)`, `lyapunov(Q,sign)`, `discreteLyapunov(A,Q)` |
| `Eigen` | `toV()`, `toS()`, `eigenvalues()`, `eigenvector(j)`, `cond()` |
| `GEVD` | `toV()`, `toS()`, `eigenvalues()`, `cond()` |
| `GGEVD` | `toVR()`, `toVL()`, `toS()`, `alphar()`, `alphai()`, `beta()` |
| `GSVD` | `toU()`, `toV()`, `toQ()`, `toS()`, `sigma()`, `rank()`, `cond()` |

### Integrator (facade — numerical integration)

```java
Integrator.fixed()                              // → FixedIntegral (Gauss-Legendre on [a,b])
Integrator.weighted()                           // → WeightedIntegral (rule's natural domain)
Integrator.adaptive()                           // → AdaptiveIntegral (adaptive GK15)
Integrator.oscillatory(OscillatoryOpts)         // → OscillatoryIntegral
Integrator.principalValue()                     // → PrincipalValueIntegral (Cauchy P.V.)
Integrator.endpointSingular(EndpointOpts)       // → EndpointSingularIntegral
Integrator.improper(ImproperOpts)               // → ImproperIntegral.Adaptive (with error)
Integrator.improperFixed(ImproperOpts)          // → ImproperIntegral.Fixed (fast, no error)
Integrator.sampled(SampledRule)                 // → SampledIntegral (scalar total)
Integrator.cumulative(SampledRule)              // → CumulativeIntegral (running total array)
```

**OscillatoryOpts**: `COS`, `SIN` (finite interval); `COS_UPPER`, `SIN_UPPER` (semi-infinite)

**EndpointOpts**: `ALGEBRAIC` (Gauss-Jacobi); `LOG_LEFT`, `LOG_RIGHT`, `LOG_BOTH` (tanh-sinh)

**ImproperOpts**: `UPPER` (∫ₐ^∞), `LOWER` (∫₋∞^b), `WHOLE_LINE` (∫₋∞^∞)

**SampledRule**: `TRAPEZOIDAL`, `SIMPSON`, `ROMBERG` (ROMBERG not supported for cumulative)

**Quadrature result**:
```java
r.getValue()           // integral estimate
r.getEstimatedError()  // absolute error bound
r.getStatus()          // Quadrature.Status enum
r.isSuccessful()       // true if CONVERGED
r.getIterations()      // adaptive subdivisions or refinement levels
r.getEvaluations()     // total function evaluations
```

### Bound

```java
Bound.unbounded()           // No constraint
Bound.between(lower, upper) // lower <= x <= upper
Bound.atLeast(lower)        // x >= lower
Bound.atMost(upper)         // x <= upper
Bound.exactly(value)        // x == value
Bound.nonNegative()         // x >= 0
Bound.nonPositive()         // x <= 0
```

### NumericalGradient

Four methods available with different accuracy/performance tradeoffs:

| Method | Formula | Accuracy | Evals/dim |
|--------|---------|----------|-----------|
| `FORWARD` | `(f(x+h) - f(x)) / h` | O(h) | 1 |
| `BACKWARD` | `(f(x) - f(x-h)) / h` | O(h) | 1 |
| `CENTRAL` | `(f(x+h) - f(x-h)) / 2h` | O(h²) | 2 |
| `FIVE_POINT` | `(-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / 12h` | O(h⁴) | 4 |

## Building Native Library

```bash
mkdir build && cd build
cmake ..
make
```

The native library will be placed in `src/main/resources/native/<platform>/`.

## License

MIT License
