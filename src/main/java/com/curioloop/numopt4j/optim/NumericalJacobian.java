/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;
import java.util.Objects;

import java.util.Arrays;

import static com.curioloop.numopt4j.optim.NumericalGradient.CBRT_EPSILON;
import static com.curioloop.numopt4j.optim.NumericalGradient.SQRT_EPSILON;

/**
 * Numerical Jacobian computation methods for vector-valued functions ℝⁿ → ℝᵐ.
 *
 * <p>Analogous to {@link NumericalGradient} for scalar functions, this enum provides
 * finite-difference methods for approximating the Jacobian matrix J where
 * J[i,j] = ∂y[i]/∂x[j], when an analytical Jacobian is not available.</p>
 *
 * <h2>Relationship with NumericalGradient</h2>
 * <p>{@link NumericalGradient} handles scalar functions (ℝⁿ → ℝ) and produces a gradient
 * vector. {@code NumericalJacobian} handles vector-valued functions (ℝⁿ → ℝᵐ) and produces
 * a full Jacobian matrix. Both use the same step-size formulas and difference schemes.</p>
 *
 * <h2>Jacobian storage format</h2>
 * <p>Default ({@code transpose=false}): row-major</p>
 * <pre>
 *   jacobian[i*n + j] = ∂y[i]/∂x[j]
 * </pre>
 * <p>Transposed ({@code transpose=true}): col-major — useful for solvers that store
 * the Jacobian column-by-column (e.g. MINPACK-style col-major QR factorization):</p>
 * <pre>
 *   jacobian[i + m*j] = ∂y[i]/∂x[j]
 *
 *   where i = output/residual index (0 to m-1)
 *         j = input/parameter index (0 to n-1)
 *         total length = m * n
 * </pre>
 *
 * <h2>Workspace requirements</h2>
 * <ul>
 *   <li>{@link #FORWARD}: 1 temporary array of length m (for perturbed residuals)</li>
 *   <li>{@link #CENTRAL}: 1 temporary array of length m (reused for +h and -h evaluations)</li>
 * </ul>
 * <p>Workspace is allocated internally by default. For high-frequency optimization,
 * use {@link TRFProblem} with workspace reuse to avoid repeated allocation.</p>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Row-major (default) — for TRF and most solvers
 * Multivariate eval = NumericalJacobian.CENTRAL.wrap(residuals, m, n);
 *
 * // Col-major (transpose=true) — for MINPACK-style col-major solvers
 * Multivariate colMajor = NumericalJacobian.FORWARD.wrap(residuals, n, n, true);
 * }</pre>
 *
 * @see Multivariate
 * @see NumericalGradient
 */
public enum NumericalJacobian {

    /**
     * Forward difference method.
     *
     * <p>Formula: J[i,j] = (r(x + h·eⱼ)[i] - r(x)[i]) / h</p>
     * <p>Step size: h = √ε · (1 + |xⱼ|) ≈ 1.49e-8 · scale</p>
     * <p>Requires n+1 function evaluations total (1 base + 1 per parameter).
     * Truncation error is O(h) ≈ 1e-8. Use when function evaluations are expensive
     * and moderate Jacobian accuracy is acceptable.</p>
     */
    FORWARD {
        @Override
        public Multivariate wrap(Multivariate.Objective func, int m, int n, boolean transpose) {
            Objects.requireNonNull(func, "func must not be null");
            return (x, xn, y, ym, jacobian) -> {
                if (jacobian != null) {
                    forwardDifference(func, x, jacobian, y, m, n, transpose);
                }
                func.evaluate(x, n, y, m);
            };
        }
    },

    /**
     * Central difference method (recommended default).
     *
     * <p>Formula: J[i,j] = (r(x + h·eⱼ)[i] - r(x - h·eⱼ)[i]) / (2h)</p>
     * <p>Step size: h = ∛ε · (1 + |xⱼ|) ≈ 6.06e-6 · scale</p>
     * <p>Requires 2n function evaluations total (2 per parameter, no base evaluation).
     * Truncation error is O(h²) ≈ 1e-11. Significantly more accurate than forward
     * difference at the cost of twice as many evaluations. This is the default method
     * used by {@link com.curioloop.numopt4j.optim.trf.TRFProblem#residuals(Multivariate.Objective, int)}.</p>
     */
    CENTRAL {
        @Override
        public Multivariate wrap(Multivariate.Objective func, int m, int n, boolean transpose) {
            Objects.requireNonNull(func, "func must not be null");
            double[] work = new double[m];
            return (x, xn, y, ym, jacobian) -> {
                if (jacobian != null) {
                    centralDifference(func, x, jacobian, work, m, n, transpose);
                }
                func.evaluate(x, n, y, m);
            };
        }
    };

    /**
     * Wraps a residual function to create a {@link Multivariate} that computes both
     * residuals and numerical Jacobian, with row-major output.
     *
     * @param func Residual function: {@code (x, n, r, m) -> void}
     * @param m    Number of residuals (outputs)
     * @param n    Number of parameters (inputs)
     * @return {@link Multivariate} with row-major Jacobian: {@code jacobian[i*n + j]}
     */
    public Multivariate wrap(Multivariate.Objective func, int m, int n) {
        return wrap(func, m, n, false);
    }

    /**
     * Wraps a residual function to create a {@link Multivariate} that computes both
     * residuals and numerical Jacobian.
     *
     * <p>When {@code transpose=false} (default), the Jacobian is stored row-major:
     * {@code jacobian[i*n + j] = ∂r[i]/∂x[j]}.</p>
     * <p>When {@code transpose=true}, the Jacobian is stored col-major:
     * {@code jacobian[i + m*j] = ∂r[i]/∂x[j]}. The difference computation writes
     * directly in col-major order — no post-hoc transposition is performed.</p>
     *
     * @param func      Residual function: {@code (x, n, r, m) -> void}
     * @param m         Number of residuals (outputs)
     * @param n         Number of parameters (inputs)
     * @param transpose {@code true} for col-major output, {@code false} for row-major
     * @return {@link Multivariate} with the requested Jacobian layout
     */
    public abstract Multivariate wrap(Multivariate.Objective func, int m, int n, boolean transpose);

    // ── Private difference kernels ────────────────────────────────────────────

    private static void forwardDifference(Multivariate.Objective func, double[] x,
                                          double[] jacobian, double[] work,
                                          int m, int n, boolean transpose) {
        // jacobian temporarily stores base f(x), work holds f(x+h)
        func.evaluate(x, n, work, m);
        if (transpose) {
            for (int j = 0; j < n; j++)
                System.arraycopy(work, 0, jacobian, m * j, m);
        } else {
            for (int i = 0; i < m; i++) 
                Arrays.fill(jacobian, i * n, (i + 1) * n, work[i]);
        }
        for (int j = 0; j < n; j++) {
            double aj = x[j];
            double h = SQRT_EPSILON * (1.0 + Math.abs(aj));
            x[j] = aj + h;
            func.evaluate(x, n, work, m);
            x[j] = aj;
            double invH = 1.0 / h;
            if (transpose) {
                for (int i = 0, base = m * j; i < m; i++)
                    jacobian[base + i] = (work[i] - jacobian[base + i]) * invH;
            } else {
                for (int i = 0; i < m; i++)
                    jacobian[i * n + j] = (work[i] - jacobian[i * n + j]) * invH;
            }
        }
    }

    private static void centralDifference(Multivariate.Objective func, double[] x,
                                          double[] jacobian, double[] work,
                                          int m, int n, boolean transpose) {
        if (transpose) {
            for (int j = 0; j < n; j++) {
                double aj = x[j];
                double h = CBRT_EPSILON * (1.0 + Math.abs(aj));
                int base = m * j;
                x[j] = aj + h;
                func.evaluate(x, n, work, m);
                for (int i = 0; i < m; i++) jacobian[base + i] = work[i];
                x[j] = aj - h;
                func.evaluate(x, n, work, m);
                x[j] = aj;
                double invTwoH = 1.0 / (2.0 * h);
                for (int i = 0; i < m; i++)
                    jacobian[base + i] = (jacobian[base + i] - work[i]) * invTwoH;
            }
        } else {
            for (int j = 0; j < n; j++) {
                double aj = x[j];
                double h = CBRT_EPSILON * (1.0 + Math.abs(aj));
                x[j] = aj + h;
                func.evaluate(x, n, work, m);
                for (int i = 0; i < m; i++) jacobian[i * n + j] = work[i];
                x[j] = aj - h;
                func.evaluate(x, n, work, m);
                x[j] = aj;
                double invTwoH = 1.0 / (2.0 * h);
                for (int i = 0; i < m; i++)
                    jacobian[i * n + j] = (jacobian[i * n + j] - work[i]) * invTwoH;
            }
        }
    }
}
