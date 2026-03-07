/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBProblem;
import com.curioloop.numopt4j.optim.slsqp.SLSQPProblem;
import com.curioloop.numopt4j.optim.trf.TRFProblem;

/**
 * Abstract base for minimization problem builders.
 *
 * <p>Holds the three fields shared by every solver:</p>
 * <ul>
 *   <li>{@code initialPoint} — starting point x₀</li>
 *   <li>{@code bounds}       — variable bounds lb ≤ x ≤ ub (optional)</li>
 *   <li>{@code objective}    — objective / residual function</li>
 * </ul>
 *
 * <p>Use the static factory methods as the primary entry point:</p>
 * <pre>{@code
 * // Bound-constrained (L-BFGS-B)
 * OptimizationResult r = Minimizer.lbfgsb()
 *     .objective(x -> x[0]*x[0] + x[1]*x[1])
 *     .initialPoint(1.0, 1.0)
 *     .solve();
 *
 * // Constrained (SLSQP)
 * OptimizationResult r = Minimizer.slsqp()
 *     .objective(x -> x[0] + x[1])
 *     .equalityConstraints(x -> x[0]*x[0] + x[1]*x[1] - 1)
 *     .initialPoint(0.5, 0.5)
 *     .solve();
 *
 * // Nonlinear least squares (TRF)
 * OptimizationResult r = Minimizer.trf()
 *     .residuals((x, r) -> { r[0] = x[0] - 1; r[1] = x[1] - 2; }, 2)
 *     .initialPoint(0.0, 0.0)
 *     .solve();
 * }</pre>
 *
 * @param <O> objective function type ({@link Univariate} or {@link Multivariate})
 * @param <W> workspace type
 * @param <S> self type for fluent builder chaining
 * @see LBFGSBProblem
 * @see SLSQPProblem
 * @see TRFProblem
 */
public abstract class Minimizer<O, W, S extends Minimizer<O, W, S>> implements OptimizationProblem<OptimizationResult, W> {

    /** Problem dimension (number of variables, inferred from initialPoint) */
    protected int dimension;

    /** Initial point (x₀) */
    protected double[] initialPoint;

    /** Variable bounds (l ≤ x ≤ u) */
    protected Bound[] bounds;

    /** Objective / residual function */
    protected O objective;

    /** Cached workspace for reuse across multiple solve calls */
    protected transient W workspace;

    protected Minimizer() {}

    // ── Common fluent setters ─────────────────────────────────────────────────

    /**
     * Sets the initial point. Also infers {@code dimension} from the array length.
     *
     * @param x0 initial point values
     * @return this builder
     */
    @SuppressWarnings("unchecked")
    public S initialPoint(double... x0) {
        if (x0 == null || x0.length == 0)
            throw new IllegalArgumentException("initialPoint must not be null or empty");
        this.initialPoint = x0;
        this.dimension = x0.length;
        return (S) this;
    }

    /**
     * Sets variable bounds (lb ≤ x ≤ ub).
     * Length must match {@code dimension} when both are known.
     *
     * @param bounds bounds for each variable
     * @return this builder
     */
    @SuppressWarnings("unchecked")
    public S bounds(Bound... bounds) {
        if (bounds != null && initialPoint != null && bounds.length != initialPoint.length)
            throw new IllegalArgumentException(
                "bounds.length=" + bounds.length + " but dimension=" + initialPoint.length);
        this.bounds = bounds;
        return (S) this;
    }

    // ── Common getters ────────────────────────────────────────────────────────

    /** Returns the problem dimension (number of variables). */
    public int dimension() { return dimension; }

    /** Returns the initial point. */
    public double[] initialPoint() { return initialPoint; }

    /** Returns the variable bounds, or {@code null} if unconstrained. */
    public Bound[] bounds() { return bounds; }

    // ── Static factory methods (facade) ──────────────────────────────────────

    /**
     * Creates an {@link LBFGSBProblem} for bound-constrained optimization.
     *
     * <p>L-BFGS-B solves:
     * <pre>  minimize f(x)  subject to  l ≤ x ≤ u</pre>
     * Gradient is computed numerically ({@link NumericalGradient#CENTRAL}) when not provided.</p>
     *
     * @return new {@link LBFGSBProblem} builder
     */
    public static LBFGSBProblem lbfgsb() {
        return new LBFGSBProblem();
    }

    /**
     * Creates a {@link SLSQPProblem} for general constrained optimization.
     *
     * <p>SLSQP solves:
     * <pre>  minimize f(x)
     *   subject to  c_eq(x) = 0,  c_ineq(x) ≥ 0,  l ≤ x ≤ u</pre>
     * </p>
     *
     * @return new {@link SLSQPProblem} builder
     */
    public static SLSQPProblem slsqp() {
        return new SLSQPProblem();
    }

    /**
     * Creates a {@link TRFProblem} for bounded nonlinear least-squares.
     *
     * <p>TRF solves:
     * <pre>  min  ½‖f(x)‖²   subject to  lb ≤ x ≤ ub</pre>
     * Jacobian is computed numerically ({@link NumericalJacobian#CENTRAL}) when not provided.</p>
     *
     * @return new {@link TRFProblem} builder
     */
    public static TRFProblem trf() {
        return new TRFProblem();
    }
}
