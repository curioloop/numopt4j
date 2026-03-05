# numopt4j

High-performance numerical optimization library for Java.

## Features

- **L-BFGS-B**: Limited-memory BFGS with bound constraints
- **SLSQP**: Sequential Least Squares Programming with equality/inequality constraints
- **TRF**: Trust Region Reflective for nonlinear least squares
- **Root finding**: Brentq (1-D), HYBR and Broyden (N-D) via `FindRoot`
- Workspace reuse for high-frequency optimization scenarios
- Multiple numerical gradient/Jacobian methods with different accuracy/speed tradeoffs

## Requirements

- Java 8+
- Native library for your platform (included for darwin-aarch64)

## Installation

```xml
<dependency>
    <groupId>com.curioloop</groupId>
    <artifactId>numopt4j</artifactId>
    <version>0.0.3</version>
</dependency>
```

## AI 助手使用

如果你在使用 AI 编程助手（如 GitHub Copilot、Cursor、Claude 等），可以通过以下方式获取 numopt4j 的完整 API 文档：

- 在 AI 对话中输入 `use context7` 并提及 numopt4j，AI 助手将自动加载最新文档
- 或直接引用项目根目录的 `llms.txt` 或 `llms-full.txt` 文件

### 推荐的 AI 友好用法

```java
// 最简单的用法：只需提供目标函数，无需梯度
OptimizationResult r = Minimize.objective(x -> x[0]*x[0] + x[1]*x[1])
    .startingFrom(1.0, 1.0)
    .run();

if (r.isSuccessful()) {
    System.out.println(Arrays.toString(r.getSolution()));
} else {
    System.out.println(r.getErrorMessage());
}
```

## Quick Start

### Unconstrained Optimization (L-BFGS-B)

```java
// Simplest usage: no gradient required
OptimizationResult result = LBFGSBProblem.create()
    .objective(x -> x[0]*x[0] + x[1]*x[1])
    .initialPoint(1.0, 1.0)
    .solve();

if (result.isSuccessful()) {
    System.out.println("Solution: " + Arrays.toString(result.getSolution()));
}
```

### With Analytical Gradient

```java
// Provide analytical gradient for best performance
OptimizationResult result = LBFGSBProblem.create()
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
OptimizationResult result = LBFGSBProblem.create()
    .objective(x -> x[0]*x[0] + x[1]*x[1])
    .bounds(Bound.between(0, 10), Bound.between(0, 10))  // 0 <= x <= 10
    .initialPoint(1.0, 1.0)
    .solve();
```

### Constrained Optimization (SLSQP)

```java
// Equality constraint: x[0] + x[1] = 1
// Inequality constraint: x[0] >= 0.5
OptimizationResult result = SLSQPProblem.create()
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

OptimizationResult result = TRFProblem.create()
    .residuals((x, r) -> {
        for (int i = 0; i < tData.length; i++) {
            r[i] = yData[i] - x[0] * Math.exp(-x[1] * tData[i]);
        }
    }, tData.length)
    .bounds(Bound.atLeast(0), Bound.atLeast(0))
    .initialPoint(1.0, 0.5)
    .solve();
```

### Root Finding (1-D Brentq)

```java
// Find root of sin(x) in [3, 4] → π
OptimizationResult result = FindRoot.scalar(Math::sin)
    .bracket(3.0, 4.0)
    .solve();

double root = result.getRoot(); // ≈ π
```

### Root Finding (N-D HYBR / Broyden)

```java
// Solve F(x) = 0 for a system of equations
OptimizationResult result = FindRoot.equations((x, f) -> {
        f[0] = x[0]*x[0] - 2;
        f[1] = x[1] - x[0];
    }, 2)
    .initialPoint(1.0, 1.0)
    .solve();

double[] solution = result.getSolution(); // [√2, √2]

// Use Broyden instead of HYBR (default)
result = FindRoot.equations(fn, 2)
    .initialPoint(1.0, 1.0)
    .method(RootMethod.BROYDEN)
    .solve();

// Use central differences for Jacobian
result = FindRoot.equations(fn, 2)
    .jacobian(NumericalJacobian.CENTRAL)
    .initialPoint(1.0, 1.0)
    .solve();
```

### Workspace Reuse

For high-frequency optimization, reuse workspace to reduce allocation overhead:

```java
LBFGSBProblem problem = LBFGSBProblem.create()
    .objective(x -> x[0]*x[0] + x[1]*x[1])
    .initialPoint(new double[n]);

LBFGSBWorkspace workspace = problem.alloc();  // allocate once
for (double[] point : points) {
    OptimizationResult result = problem.initialPoint(point).solve(workspace);
    // process result
}
```

## API Reference

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

### Termination

```java
Termination.defaults()  // Default criteria

Termination.builder()
    .maxIterations(100)
    .maxEvaluations(1000)
    .maxComputations(5000000)  // Time limit in microseconds
    .accuracy(1e-6)
    .gradientTolerance(1e-5)
    .build();
```

### NumericalGradient

Four methods available with different accuracy/performance tradeoffs:

| Method | Formula | Accuracy | Evals/dim |
|--------|---------|----------|-----------|
| `FORWARD` | `(f(x+h) - f(x)) / h` | O(h) | 1 |
| `BACKWARD` | `(f(x) - f(x-h)) / h` | O(h) | 1 |
| `CENTRAL` | `(f(x+h) - f(x-h)) / 2h` | O(h²) | 2 |
| `FIVE_POINT` | `(-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / 12h` | O(h⁴) | 4 |

```java
// Fastest (1 eval per dimension)
Univariate fast = NumericalGradient.FORWARD.wrap(func);

// Good balance of accuracy and speed
Univariate balanced = NumericalGradient.CENTRAL.wrap(func);

// Highest accuracy (4 evals per dimension)
Univariate accurate = NumericalGradient.FIVE_POINT.wrap(func);
```

Typical error comparison:
```
FORWARD/BACKWARD: ~1e-7
CENTRAL:          ~1e-11
FIVE_POINT:       ~1e-12
```

## Building Native Library

```bash
mkdir build && cd build
cmake ..
make
```

The native library will be placed in `src/main/resources/native/<platform>/`.

## License

MIT License
