# numopt4j

High-performance numerical optimization library for Java.

## Features

- **L-BFGS-B**: Limited-memory BFGS with bound constraints
- **SLSQP**: Sequential Least Squares Programming with equality/inequality constraints
- Native C implementation via JNI for performance
- Workspace reuse for high-frequency optimization scenarios
- Numerical gradient support (forward/central difference)

## Requirements

- Java 8+
- Native library for your platform (included for darwin-aarch64)

## Installation

```xml
<dependency>
    <groupId>com.curioloop</groupId>
    <artifactId>numopt4j</artifactId>
    <version>1.0.0-SNAPSHOT</version>
</dependency>
```

## Quick Start

### Unconstrained Optimization (L-BFGS-B)

```java
// With analytical gradient
ObjectiveFunction objective = (x, g) -> {
    double f = x[0]*x[0] + x[1]*x[1];
    if (g != null) {
        g[0] = 2*x[0];
        g[1] = 2*x[1];
    }
    return f;
};

OptimizationResult result = LbfgsbOptimizer.minimize(objective, new double[]{1, 1});
System.out.println("Solution: " + Arrays.toString(result.getSolution()));
```

### With Numerical Gradient

```java
OptimizationResult result = LbfgsbOptimizer.builder()
    .dimension(2)
    .objective(x -> x[0]*x[0] + x[1]*x[1], NumericalGradient.CENTRAL)
    .build()
    .optimize(new double[]{1, 1});
```

### Bound Constraints

```java
LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
    .dimension(2)
    .objective(objective)
    .bounds(Bound.between(0, 10))  // 0 <= x <= 10 for all variables
    .build();
```

### Constrained Optimization (SLSQP)

```java
ObjectiveFunction objective = (x, g) -> {
    double f = x[0]*x[0] + x[1]*x[1];
    if (g != null) { g[0] = 2*x[0]; g[1] = 2*x[1]; }
    return f;
};

// Equality constraint: x[0] + x[1] = 1
ConstraintFunction eq = (x, g) -> {
    if (g != null) { g[0] = 1; g[1] = 1; }
    return x[0] + x[1] - 1;
};

// Inequality constraint: x[0] >= 0.5
ConstraintFunction ineq = (x, g) -> {
    if (g != null) { g[0] = 1; g[1] = 0; }
    return x[0] - 0.5;
};

SlsqpOptimizer optimizer = SlsqpOptimizer.builder()
    .dimension(2)
    .objective(objective)
    .equalityConstraints(eq)
    .inequalityConstraints(ineq)
    .build();

OptimizationResult result = optimizer.optimize(new double[]{0, 0});
```

### Workspace Reuse

For high-frequency optimization, reuse workspace to reduce allocation overhead:

```java
LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
    .dimension(n)
    .objective(objective)
    .build();

try (LbfgsbWorkspace workspace = LbfgsbWorkspace.allocate(n, 5)) {
    for (double[] point : points) {
        OptimizationResult result = optimizer.optimize(point, workspace);
        // process result
    }
}
```

## API Reference

### Bound

```java
Bound.unbounded()           // No constraint
Bound.between(lower, upper) // lower <= x <= upper
Bound.atLeast(lower)        // x >= lower
Bound.atMost(upper)         // x <= upper
Bound.fixed(value)          // x == value
Bound.nonNegative()         // x >= 0
Bound.nonPositive()         // x <= 0
```

### Termination

```java
Termination.defaults()  // Default criteria

Termination.builder()
    .maxIterations(100)
    .maxEvaluations(1000)
    .accuracy(1e-6)
    .gradientTolerance(1e-5)
    .build();
```

### NumericalGradient

```java
NumericalGradient.CENTRAL  // Central difference (more accurate)
NumericalGradient.FORWARD  // Forward difference (faster)
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
