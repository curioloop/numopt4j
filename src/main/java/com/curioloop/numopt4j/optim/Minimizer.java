/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

import com.curioloop.numopt4j.optim.cmaes.CMAESProblem;
import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBProblem;
import com.curioloop.numopt4j.optim.subplex.SubplexProblem;
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
 * // Global derivative-free (CMA-ES)
 * Optimization r = Minimizer.cmaes()
 *     .objective((x, n) -> x[0]*x[0] + x[1]*x[1])
 *     .initialPoint(1.0, 1.0)
 *     .solve();
 *
 * // Derivative-free (Nelder-Mead)
 * Optimization r = Minimizer.subplex()
 *     .objective((x, n) -> x[0]*x[0] + x[1]*x[1])
 *     .initialPoint(1.0, 1.0)
 *     .solve();
 *
 * // Bound-constrained (L-BFGS-B)
 * Optimization r = Minimizer.lbfgsb()
 *     .objective((x, n) -> x[0]*x[0] + x[1]*x[1])
 *     .initialPoint(1.0, 1.0)
 *     .solve();
 *
 * // Constrained (SLSQP)
 * Optimization r = Minimizer.slsqp()
 *     .objective((x, n) -> x[0] + x[1])
 *     .equalityConstraints((x, n) -> x[0]*x[0] + x[1]*x[1] - 1)
 *     .initialPoint(0.5, 0.5)
 *     .solve();
 *
 * // Nonlinear least squares (TRF)
 * Optimization r = Minimizer.trf()
 *     .residuals((x, n, r, m) -> { r[0] = x[0] - 1; r[1] = x[1] - 2; }, 2)
 *     .initialPoint(0.0, 0.0)
 *     .solve();
 * }</pre>
 *
 * @param <O> objective function type ({@link Univariate}, {@link Multivariate},
 *            or {@code ToDoubleFunction<double[]>} for derivative-free solvers)
 * @param <W> workspace type
 * @param <S> self type for fluent builder chaining
 * @see CMAESProblem
 * @see SubplexProblem
 * @see LBFGSBProblem
 * @see SLSQPProblem
 * @see TRFProblem
 */
public abstract class Minimizer<O, W, S extends Minimizer<O, W, S>> implements Problem<W> {

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

    // ── Common validation helpers ─────────────────────────────────────────────

    /** Throws {@link IllegalStateException} if {@code objective} is null. */
    protected void requireObjective() {
        if (objective == null)
            throw new IllegalStateException("objective is required. Call .objective(fn) before .solve().");
    }

    /** Throws if {@code initialPoint} is null/empty or contains non-finite values. */
    protected void requireInitialPoint() {
        if (initialPoint == null || initialPoint.length == 0)
            throw new IllegalStateException("initialPoint is required. Call .initialPoint(x0) before .solve().");
        for (int i = 0; i < initialPoint.length; i++) {
            double v = initialPoint[i];
            if (!Double.isFinite(v))
                throw new IllegalArgumentException(
                    "initialPoint[" + i + "] is " + v + ". All initial values must be finite.");
        }
    }

    /**
     * Returns {@code external} if non-null; otherwise returns (and caches) the internal workspace,
     * creating it via {@code ctor} on first use.
     */
    protected W resolveWorkspace(W external, java.util.function.Supplier<W> ctor) {
        if (external != null) return external;
        if (workspace == null) workspace = ctor.get();
        return workspace;
    }

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

    /**
     * Creates a {@link SubplexProblem} for derivative-free optimization.
     *
     * <p>No gradient is required. Uses the Subplex algorithm (Rowan 1990) which
     * decomposes the problem into low-dimensional Nelder-Mead subproblems.</p>
     *
     * @return new {@link SubplexProblem} builder
     */
    public static SubplexProblem subplex() {
        return new SubplexProblem();
    }

    /**
     * Creates a {@link CMAESProblem} for derivative-free global optimization.
     *
     * <p>CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a stochastic
     * optimizer suitable for non-convex, non-smooth, and noisy objective functions.
     * Supports sep-CMA-ES (diagonal mode) and IPOP/BIPOP restart strategies.</p>
     *
     * @return new {@link CMAESProblem} builder
     */
    public static CMAESProblem cmaes() {
        return new CMAESProblem();
    }
}
