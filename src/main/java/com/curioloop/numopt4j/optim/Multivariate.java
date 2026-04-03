/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

/**
 * Functional interface for a vector-valued function ℝⁿ → ℝᵐ with optional Jacobian.
 *
 * <p>Represents a mapping from an n-dimensional input vector to an m-dimensional output vector,
 * with optional Jacobian matrix computation. Used for residual functions in nonlinear
 * least-squares problems (TRF algorithm).</p>
 *
 * <p>This is the multi-output version of {@link Univariate}. While {@link Univariate}
 * computes a single scalar output with an optional gradient vector, this interface
 * computes multiple outputs (stored in array {@code y}) with an optional Jacobian matrix.</p>
 *
 * <h2>Jacobian storage format (row-major)</h2>
 * <p>The Jacobian matrix J has elements J[i,j] = ∂y[i]/∂x[j], stored in row-major order
 * as a flat 1D array of length m×n:</p>
 * <pre>
 *   jacobian[i*n + j] = ∂y[i]/∂x[j]
 *
 *   where i = output index (0 to m-1)
 *         j = input index  (0 to n-1)
 * </pre>
 *
 * <h2>jacobian=null behavior</h2>
 * <p>When the {@code jacobian} parameter is {@code null}, only the residual values in
 * {@code y} need to be computed — Jacobian computation can be skipped entirely.
 * Always check {@code if (jacobian != null)} before writing to the Jacobian array.</p>
 *
 * <h2>Using NumericalJacobian when analytical Jacobian is unavailable</h2>
 * <pre>{@code
 * // Define a residual function (multi-output, no Jacobian)
 * BiConsumer<double[], double[]> residuals = (coeffs, r) -> {
 *     for (int i = 0; i < data.length; i++) {
 *         r[i] = observed[i] - model(coeffs, data[i]);
 *     }
 * };
 *
 * // Wrap with numerical Jacobian using central difference (more accurate)
 * Multivariate eval = NumericalJacobian.CENTRAL.wrap(residuals, m, n);
 *
 * // Or use forward difference (faster, 1 extra evaluation per parameter)
 * Multivariate fast = NumericalJacobian.FORWARD.wrap(residuals, m, n);
 * }</pre>
 *
 * <h2>Implementing with analytical Jacobian</h2>
 * <pre>{@code
 * // Analytical Jacobian is preferred for performance
 * Multivariate linear = (x, y, jac) -> {
 *     // y = A*x - b  (linear residuals)
 *     for (int i = 0; i < m; i++) {
 *         y[i] = -b[i];
 *         for (int j = 0; j < n; j++) y[i] += A[i][j] * x[j];
 *     }
 *     if (jac != null) {
 *         for (int i = 0; i < m; i++)
 *             for (int j = 0; j < n; j++)
 *                 jac[i*n + j] = A[i][j];  // row-major: jac[i*n + j]
 *     }
 * };
 * }</pre>
 *
 * @see Univariate
 * @see NumericalJacobian
 */
@FunctionalInterface
public interface Multivariate {

    /**
     * Functional interface for a pure vector-valued function ℝⁿ → ℝᵐ without Jacobian.
     *
     * <p>Used as the input to {@link NumericalJacobian#wrap(Objective, int, int)} to produce a
     * full {@link Multivariate} with numerically approximated Jacobian, and accepted
     * directly by residual-based solvers (TRF, HYBR, Broyden).</p>
     *
     * <p>Only the first {@code n} elements of {@code x} and first {@code m} elements of
     * {@code y} are considered effective; array lengths may exceed {@code n} and {@code m}.</p>
     *
     * @see NumericalJacobian
     */
    @FunctionalInterface
    interface Objective {
        /**
         * Evaluates the vector function at point {@code x}, writing results into {@code y}.
         *
         * @param x Current point (read-only); only {@code x[0..n-1]} are read
         * @param n Number of effective input dimensions
         * @param y Output array; only {@code y[0..m-1]} are written
         * @param m Number of effective output dimensions
         */
        void evaluate(double[] x, int n, double[] y, int m);
    }

    /**
     * Evaluates the function outputs and optionally the Jacobian matrix.
     * <p>
     * When {@code jacobian} is not null, the implementation should compute and store
     * the partial derivatives in row-major order: jacobian[i * n + j] = ∂y[i]/∂x[j].
     * When {@code jacobian} is null, only the function outputs need to be computed.
     * </p>
     *
     * @param x Current point (length &ge; n, only x[0..n-1] are read)
     * @param n Number of effective input dimensions
     * @param y Output array for function values (length &ge; m, only y[0..m-1] are written)
     * @param m Number of effective output dimensions
     * @param jacobian Output array for Jacobian matrix (length m&times;n, row-major; may be null)
     */
    void evaluate(double[] x, int n, double[] y, int m, double[] jacobian);


}
