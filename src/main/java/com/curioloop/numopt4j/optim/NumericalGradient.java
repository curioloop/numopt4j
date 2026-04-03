/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;
import java.util.Objects;

/**
 * Numerical gradient computation methods for approximating ∇f(x) when analytical gradients
 * are unavailable.
 *
 * <p>Provides four finite-difference methods with different accuracy/cost trade-offs.
 * Use {@link #wrap(Univariate.Objective)} to create a {@link Univariate} that computes both
 * the function value and its numerical gradient.</p>
 *
 * <h2>Method Comparison</h2>
 * <table border="1" summary="Comparison of numerical gradient methods">
 *   <tr><th>Method</th><th>Formula</th><th>Accuracy Order</th><th>Evaluations/Dimension</th><th>Recommended Scenario</th></tr>
 *   <tr><td>{@link #FORWARD}</td><td>(f(x+h) - f(x)) / h</td><td>O(h)</td><td>1 (+ 1 base)</td><td>Fast evaluation, low accuracy needed</td></tr>
 *   <tr><td>{@link #BACKWARD}</td><td>(f(x) - f(x-h)) / h</td><td>O(h)</td><td>1 (+ 1 base)</td><td>Function better-behaved for smaller x</td></tr>
 *   <tr><td>{@link #CENTRAL}</td><td>(f(x+h) - f(x-h)) / (2h)</td><td>O(h²)</td><td>2</td><td>Default choice — good accuracy/cost balance</td></tr>
 *   <tr><td>{@link #FIVE_POINT}</td><td>(-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)) / (12h)</td><td>O(h⁴)</td><td>4</td><td>High accuracy needed, expensive function OK</td></tr>
 * </table>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Default choice: central difference (O(h²) accuracy, 2 evaluations per dimension)
 * Univariate balanced = NumericalGradient.CENTRAL.wrap((x, n) -> x[0]*x[0] + x[1]*x[1]);
 *
 * // Fastest: forward difference (O(h) accuracy, 1 evaluation per dimension)
 * Univariate fast = NumericalGradient.FORWARD.wrap((x, n) -> x[0]*x[0] + x[1]*x[1]);
 *
 * // Most accurate: five-point stencil (O(h⁴) accuracy, 4 evaluations per dimension)
 * Univariate accurate = NumericalGradient.FIVE_POINT.wrap((x, n) -> x[0]*x[0] + x[1]*x[1]);
 * }</pre>
 *
 * @see Univariate
 */
public enum NumericalGradient {

    /**
     * Forward difference method.
     *
     * <p>Formula: g[i] ≈ (f(x + h·eᵢ) - f(x)) / h</p>
     * <p>Step size: h = √ε · max(1, |xᵢ|) ≈ 1.49e-8 · scale
     * (where ε ≈ 2.22e-16 is machine epsilon)</p>
     * <p>Typical error: O(h) ≈ 1e-8 — suitable when function evaluations are expensive
     * and moderate accuracy is acceptable. Requires 1 extra evaluation per dimension
     * (plus 1 base evaluation shared across all dimensions).</p>
     */
    FORWARD {
        @Override
        public Univariate wrap(Univariate.Objective func) {
            Objects.requireNonNull(func, "func must not be null");
            return (x, n, g) -> {
                double f = func.evaluate(x, n);
                if (g != null) {
                    forwardDifference(func, x, n, f, g);
                }
                return f;
            };
        }
    },

    /**
     * Backward difference method.
     *
     * <p>Formula: g[i] ≈ (f(x) - f(x - h·eᵢ)) / h</p>
     * <p>Step size: h = √ε · max(1, |xᵢ|) ≈ 1.49e-8 · scale</p>
     * <p>Typical error: O(h) ≈ 1e-8 — same accuracy as forward difference.
     * Useful when the function behaves better for smaller x values (e.g., near a boundary).
     * Requires 1 extra evaluation per dimension (plus 1 base evaluation).</p>
     */
    BACKWARD {
        @Override
        public Univariate wrap(Univariate.Objective func) {
            Objects.requireNonNull(func, "func must not be null");
            return (x, n, g) -> {
                double f = func.evaluate(x, n);
                if (g != null) {
                    backwardDifference(func, x, n, f, g);
                }
                return f;
            };
        }
    },

    /**
     * Central difference method (recommended default).
     *
     * <p>Formula: g[i] ≈ (f(x + h·eᵢ) - f(x - h·eᵢ)) / (2h)</p>
     * <p>Step size: h = ∛ε · max(1, |xᵢ|) ≈ 6.06e-6 · scale
     * (cube root of machine epsilon, optimal for O(h²) methods)</p>
     * <p>Typical error: O(h²) ≈ 1e-11 — significantly more accurate than forward/backward
     * difference at the cost of 2 evaluations per dimension (no base evaluation needed).
     * This is the default method used by {@code Problem.objective(Univariate.Objective)}.</p>
     */
    CENTRAL {
        @Override
        public Univariate wrap(Univariate.Objective func) {
            Objects.requireNonNull(func, "func must not be null");
            return (x, n, g) -> {
                double f = func.evaluate(x, n);
                if (g != null) {
                    centralDifference(func, x, n, g);
                }
                return f;
            };
        }
    },

    /**
     * Five-point stencil method (highest accuracy).
     *
     * <p>Formula: g[i] ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)</p>
     * <p>Step size: h = ε^(1/4) · max(1, |xᵢ|) ≈ 1.22e-4 · scale
     * (fourth root of machine epsilon, optimal for O(h⁴) methods)</p>
     * <p>Typical error: O(h⁴) ≈ 1e-15 — near machine precision accuracy.
     * Requires 4 evaluations per dimension. Use when maximum gradient accuracy is needed
     * and function evaluations are not the bottleneck.</p>
     */
    FIVE_POINT {
        @Override
        public Univariate wrap(Univariate.Objective func) {
            Objects.requireNonNull(func, "func must not be null");
            return (x, n, g) -> {
                double f = func.evaluate(x, n);
                if (g != null) {
                    fivePointDifference(func, x, n, g);
                }
                return f;
            };
        }
    };

    /** Machine epsilon (ε ≈ 2.22e-16) */
    public static final double EPSILON = Math.ulp(1.0);

    /** Default step size for forward difference (√ε ≈ 1.49e-8)  */
    public static final double SQRT_EPSILON = Math.sqrt(EPSILON);

    /** Default step size for central difference (∛ε ≈ 6.8e-6) */
    public static final double CBRT_EPSILON = Math.cbrt(EPSILON);

    /** Default step size for five-point stencil (∜ε ≈ 1.2e-4) */
    private static final double FOURTH_ROOT_EPSILON = Math.pow(EPSILON, 0.25);

    /**
     * Wraps a function-only objective to include numerical gradient.
     * @param func Function that computes only the objective value
     * @return Univariate that computes both value and gradient
     */
    public abstract Univariate wrap(Univariate.Objective func);

    /**
     * Computes gradient using forward difference.
     */
    private static void forwardDifference(Univariate.Objective func, double[] x, int n, double f0, double[] g) {
        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double h = SQRT_EPSILON * Math.max(1.0, Math.abs(xi));
            if (xi < 0) h = -h;

            // Ensure h is representable
            double temp = xi + h;
            h = temp - xi;

            x[i] = xi + h;
            double f1 = func.evaluate(x, n);
            x[i] = xi;

            g[i] = (f1 - f0) / h;
        }
    }

    /**
     * Computes gradient using backward difference.
     */
    private static void backwardDifference(Univariate.Objective func, double[] x, int n, double f0, double[] g) {
        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double h = SQRT_EPSILON * Math.max(1.0, Math.abs(xi));
            if (xi < 0) h = -h;

            // Ensure h is representable
            double temp = xi - h;
            h = xi - temp;

            x[i] = xi - h;
            double f1 = func.evaluate(x, n);
            x[i] = xi;

            g[i] = (f0 - f1) / h;
        }
    }

    /**
     * Computes gradient using central difference.
     */
    private static void centralDifference(Univariate.Objective func, double[] x, int n, double[] g) {
        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double h = CBRT_EPSILON * Math.max(1.0, Math.abs(xi));

            x[i] = xi + h;
            double f1 = func.evaluate(x, n);

            x[i] = xi - h;
            double f2 = func.evaluate(x, n);

            x[i] = xi;

            g[i] = (f1 - f2) / (2.0 * h);
        }
    }

    /**
     * Computes gradient using five-point stencil.
     */
    private static void fivePointDifference(Univariate.Objective func, double[] x, int n, double[] g) {
        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double h = FOURTH_ROOT_EPSILON * Math.max(1.0, Math.abs(xi));

            x[i] = xi + 2*h;
            double f1 = func.evaluate(x, n);

            x[i] = xi + h;
            double f2 = func.evaluate(x, n);

            x[i] = xi - h;
            double f3 = func.evaluate(x, n);

            x[i] = xi - 2*h;
            double f4 = func.evaluate(x, n);

            x[i] = xi;

            g[i] = (-f1 + 8*f2 - 8*f3 + f4) / (12.0 * h);
        }
    }

}
