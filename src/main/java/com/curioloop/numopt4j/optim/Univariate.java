/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

/**
 * Functional interface for a scalar-valued function ℝⁿ → ℝ with optional gradient.
 *
 * <p>Represents a mapping from an n-dimensional input vector to a single scalar output,
 * with optional gradient computation. Used for objective functions and constraint functions
 * in optimization (both share the same signature: compute a scalar value and optionally
 * its gradient).</p>
 *
 * <h2>gradient=null behavior</h2>
 * <p>When the {@code gradient} parameter is {@code null}, only the function value needs
 * to be computed — gradient computation can be skipped entirely. The optimizer passes
 * {@code null} when it only needs the function value (e.g., during line search).
 * Always check {@code if (gradient != null)} before writing to the gradient array.</p>
 *
 * <h2>Using NumericalGradient when analytical gradient is unavailable</h2>
 * <pre>{@code
 * // Wrap a pure function (no gradient) using central difference approximation
 * Univariate objective = NumericalGradient.CENTRAL.wrap(x -> x[0]*x[0] + x[1]*x[1]);
 *
 * // Or use forward difference (faster, less accurate)
 * Univariate fast = NumericalGradient.FORWARD.wrap(x -> Math.sin(x[0]) + x[1]*x[1]);
 * }</pre>
 *
 * <h2>Implementing with analytical gradient</h2>
 * <pre>{@code
 * // Analytical gradient is preferred for performance
 * Univariate rosenbrock = (x, g) -> {
 *     double a = 1 - x[0], b = x[1] - x[0]*x[0];
 *     if (g != null) {
 *         g[0] = -2*a - 4*x[0]*b;
 *         g[1] = 2*b;
 *     }
 *     return a*a + 100*b*b;
 * };
 * }</pre>
 *
 * @see Multivariate
 * @see NumericalGradient
 */
@FunctionalInterface
public interface Univariate {

    /**
     * Functional interface for a pure scalar-valued function ℝⁿ → ℝ without gradient.
     *
     * <p>Used as the input to {@link NumericalGradient#wrap(Objective)} to produce a
     * full {@link Univariate} with numerically approximated gradient, and accepted
     * directly by derivative-free optimizers (CMA-ES, Subplex, SLSQP, L-BFGS-B).</p>
     *
     * <p>Only the first {@code n} elements of {@code x} are considered effective;
     * {@code x.length} may be greater than {@code n}.</p>
     *
     * @see NumericalGradient
     */
    @FunctionalInterface
    interface Objective {
        /**
         * Evaluates the scalar function at point {@code x}.
         *
         * @param x Current point (read-only); only {@code x[0..n-1]} are read
         * @param n Number of effective dimensions
         * @return Function value at x
         */
        double evaluate(double[] x, int n);
    }

    /**
     * Evaluates the function and optionally its gradient.
     * <p>
     * When gradient is not null, the implementation should compute and store
     * the partial derivatives in gradient[0..n-1]. When gradient is null,
     * only the function value needs to be computed.
     * </p>
     *
     * @param x Current point (read-only); x.length may be >= n
     * @param n Number of effective dimensions; only x[0..n-1] are read
     * @param gradient Output array for gradient (may be null); only gradient[0..n-1] are written
     * @return Function value at x
     */
    double evaluate(double[] x, int n, double[] gradient);


}
