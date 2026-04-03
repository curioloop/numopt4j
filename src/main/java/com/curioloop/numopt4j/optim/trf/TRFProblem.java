/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.Multivariate;
import com.curioloop.numopt4j.optim.NumericalJacobian;
import com.curioloop.numopt4j.optim.Optimization;

import static com.curioloop.numopt4j.optim.trf.TRFConstants.*;

/**
 * Fluent API for defining and solving TRF bounded nonlinear least-squares problems.
 *
 * <p>TRF solves bounded nonlinear least-squares problems:</p>
 * <pre>
 *   min  F(x) = (1/2) ||f(x)||²
 *    x
 *   s.t. lb ≤ x ≤ ub
 * </pre>
 *
 * <p>{@code TRFProblem} is the public fluent API that validates inputs, manages
 * workspaces, and stores the solution in
 * {@link Optimization#getSolution()}.</p>
 *
 * <h2>Basic Usage</h2>
 * <pre>{@code
 * // Recommended entry point: BiConsumer lambda (no Jacobian required)
 * Optimization result = new TRFProblem()
 *     .residuals((x, r) -> { r[0] = x[0] - 1; r[1] = x[1] - 2; }, 2)
 *     .initialPoint(0.0, 0.0)
 *     .solve();
 *
 * if (result.isSuccessful()) {
 *     double[] solution = result.getSolution();  // [1.0, 2.0]
 * }
 * }</pre>
 *
 * <h2>Advanced Usage</h2>
 * <pre>{@code
 * // Curve fitting: y = a * exp(-b * t), with bounds on parameters
 * double[] tData = {0.0, 1.0, 2.0, 3.0};
 * double[] yData = {2.0, 1.2, 0.7, 0.4};
 *
 * Optimization result = new TRFProblem()
 *     .residuals((x, r) -> {
 *         for (int i = 0; i < tData.length; i++) {
 *             r[i] = yData[i] - x[0] * Math.exp(-x[1] * tData[i]);
 *         }
 *     }, tData.length)
 *     .bounds(Bound.atLeast(0), Bound.atLeast(0))
 *     .initialPoint(1.0, 0.5)
 *     .functionTolerance(1e-10)
 *     .maxEvaluations(500)
 *     .solve();
 *
 * // With workspace reuse for high-frequency optimization
 * TRFProblem problem = new TRFProblem()
 *     .residuals(fn, m)
 *     .initialPoint(x0);
 * TRFWorkspace ws = problem.alloc();
 * for (Data d : dataList) {
 *     problem.residuals(d::residuals, m).solve(ws);
 * }
 * }</pre>
 *
 * @see com.curioloop.numopt4j.optim.Minimizer
 * @see TRFWorkspace
 */
public final class TRFProblem extends Minimizer<Multivariate, TRFWorkspace, TRFProblem> {

    /** Number of residuals */
    private int numResiduals;

    /** Convergence tolerances */
    private double functionTolerance    = DEFAULT_FTOL;
    private double parameterTolerance       = DEFAULT_XTOL;
    private double gradientTolerance    = DEFAULT_GTOL;

    /** Maximum function evaluations */
    private int maxEvaluations = 0; // 0 = auto: 100 * n

    /** Trust-region initial scale factor */
    private double factor = DEFAULT_FACTOR;

    /** Robust loss function (default: LINEAR = standard least-squares) */
    private RobustLoss loss = RobustLoss.LINEAR;

    /** Soft margin C for loss scaling — rho(z) evaluated at z=(f/C)² (default: 1.0) */
    private double lossScale = 1.0;

    /** User-supplied diagonal scaling (optional) */
    private double[] diag;

    /** Cached workspace for reuse — inherited from {@link Minimizer} */

    public TRFProblem() {}

    private Multivariate resolveObjective() {
        if (objective != null) return objective;
        if (pendingFn != null) return pendingJac.wrap(pendingFn, numResiduals, dimension);
        return null;
    }

    private void validate() {
        if (objective == null && pendingFn == null) {
            throw new IllegalStateException(
                "residuals/objective is required. Call .residuals(fn, m) before .solve().");
        }
        if (initialPoint == null || initialPoint.length == 0) {
            throw new IllegalStateException(
                "initialPoint is required. Call .initialPoint(x0) before .solve().");
        }
        for (int i = 0; i < initialPoint.length; i++) {
            double v = initialPoint[i];
            if (Double.isNaN(v) || Double.isInfinite(v)) {
                throw new IllegalArgumentException(
                    "initialPoint[" + i + "] is " + v + ". All initial values must be finite.");
            }
        }
        if (numResiduals <= 0) {
            throw new IllegalStateException(
                "number of residuals m must be set. Call .residuals(fn, m) before .solve().");
        }
        if (numResiduals < dimension) {
            throw new IllegalStateException(
                "Cannot solve: need m >= n (residuals >= parameters), got m=" + numResiduals + ", n=" + dimension);
        }
    }

    @Override
    public TRFWorkspace alloc() {
        validate();
        if (workspace == null) {
            workspace = new TRFWorkspace();
        }
        workspace.ensure(numResiduals, dimension);
        return workspace;
    }

    @Override
    public Optimization solve() {
        return solve(null);
    }

    /**
     * Solves the optimization problem.
     *
     * <p>The initial point is cloned internally; {@code initialPoint} is not modified.
     * The solution is stored in {@link Optimization#getSolution()} and returned
     * as a direct reference (no defensive copy). The caller owns the returned array.</p>
     *
     * @param workspace optional pre-allocated workspace for reuse
     * @return optimization result
     */
    @Override
    public Optimization solve(TRFWorkspace workspace) {
        validate();
        TRFWorkspace ws = workspace != null ? workspace : (this.workspace != null ? this.workspace : new TRFWorkspace());
        ws.ensure(numResiduals, dimension);

        double[] x = initialPoint.clone();
        int maxfev = (maxEvaluations > 0) ? maxEvaluations : 100 * dimension;
        Optimization r = TRFCore.optimize(numResiduals, dimension, resolveObjective(),
                functionTolerance, parameterTolerance, gradientTolerance,
                maxfev, factor, diag, bounds, loss, lossScale, x, ws);
        return new Optimization(Double.NaN, x, r.getCost(), r.getStatus(), r.getIterations(), r.getEvaluations());
    }

    // ── Getters ──────────────────────────────────────────────────────────────

    public int residualCount() { return numResiduals; }
    public int maxEvaluations() { return maxEvaluations; }
    public double functionTolerance() { return functionTolerance; }
    public double parameterTolerance() { return parameterTolerance; }
    public double gradientTolerance() { return gradientTolerance; }
    public RobustLoss loss() { return loss; }
    public double lossScale() { return lossScale; }

    // ── Fluent setters ────────────────────────────────────────────────────────

    /**
     * Sets the residual function using a numerical Jacobian (central differences).
     *
     * @param fn residual function: (x, n, r, m) -> void
     * @param m  number of residuals
     * @return this problem instance
     */
    public TRFProblem residuals(Multivariate.Objective fn, int m) {
        if (fn == null) throw new IllegalArgumentException("fn must not be null");
        if (m <= 0) {
            throw new IllegalArgumentException("number of residuals m must be positive, got " + m);
        }
        return residuals(NumericalJacobian.CENTRAL, fn, m);
    }

    /**
     * Sets the residual function with a specified numerical Jacobian method.
     *
     * @param jac numerical Jacobian method
     * @param fn  residual function: (x, n, r, m) -> void
     * @param m   number of residuals
     * @return this problem instance
     */
    public TRFProblem residuals(NumericalJacobian jac, Multivariate.Objective fn, int m) {
        if (fn == null) throw new IllegalArgumentException("fn must not be null");
        if (m <= 0) {
            throw new IllegalArgumentException("number of residuals m must be positive, got " + m);
        }
        this.numResiduals = m;
        this.pendingJac = jac;
        this.pendingFn  = fn;
        this.objective   = null; // cleared; will be built lazily in solve()
        return this;
    }

    /**
     * Sets the objective function with analytical Jacobian.
     *
     * @param f multivariate function computing residuals and Jacobian
     * @return this problem instance
     */
    public TRFProblem objective(Multivariate f) {
        this.objective = f;
        this.pendingFn = null;
        return this;
    }

    /**
     * Sets the function (chi-squared) tolerance.
     *
     * <p>Valid range: &gt; 0. Default: 1e-8.
     * Convergence criterion: relative reduction in the cost function (chi-squared) is below this value.
     * Tighter values yield more accurate solutions but require more evaluations.</p>
     *
     * @param value tolerance (must be positive)
     * @return this problem instance
     */
    public TRFProblem functionTolerance(double value) {
        if (value <= 0) throw new IllegalArgumentException("functionTolerance must be positive, got " + value);
        this.functionTolerance = value;
        return this;
    }

    /**
     * Sets the step tolerance.
     *
     * <p>Valid range: &gt; 0. Default: 1e-8.
     * Convergence criterion: relative change in the solution vector is below this value.
     * Controls how precisely the solution parameters are determined.</p>
     *
     * @param value tolerance (must be positive)
     * @return this problem instance
     */
    public TRFProblem parameterTolerance(double value) {
        if (value <= 0) throw new IllegalArgumentException("parameterTolerance must be positive, got " + value);
        this.parameterTolerance = value;
        return this;
    }

    /**
     * Sets the gradient tolerance.
     *
     * <p>Valid range: &gt; 0. Default: 1e-8.
     * Convergence criterion: maximum absolute value of the scaled gradient is below this value.
     * Indicates the solution is at a stationary point of the cost function.</p>
     *
     * @param value tolerance (must be positive)
     * @return this problem instance
     */
    public TRFProblem gradientTolerance(double value) {
        if (value <= 0) throw new IllegalArgumentException("gradientTolerance must be positive, got " + value);
        this.gradientTolerance = value;
        return this;
    }

    /**
     * Sets the maximum number of function evaluations.
     *
     * <p>Valid range: &gt; 0. Default: 0 (auto = 100 * n, where n is the number of parameters).
     * Limits total residual function calls. Useful for expensive functions where
     * computation budget is constrained. If the optimizer hits this limit,
     * {@link Optimization.Status#MAX_EVALUATIONS_REACHED} is returned.</p>
     *
     * @param value maximum evaluations (must be positive)
     * @return this problem instance
     */
    public TRFProblem maxEvaluations(int value) {
        if (value <= 0) throw new IllegalArgumentException("maxEvaluations must be positive, got " + value);
        this.maxEvaluations = value;
        return this;
    }

    /**
     * Sets the trust-region initial scale factor.
     *
     * @param value factor (must be positive)
     * @return this problem instance
     */
    public TRFProblem factor(double value) {
        if (value <= 0) throw new IllegalArgumentException("factor must be positive");
        this.factor = value;
        return this;
    }

    /**
     * Sets the robust loss function.
     *
     * <p>Default: {@link RobustLoss#LINEAR} (standard least-squares).
     * Use non-linear losses to reduce the influence of outliers.</p>
     *
     * @param loss loss function (must not be null)
     * @return this problem instance
     * @see RobustLoss
     */
    public TRFProblem loss(RobustLoss loss) {
        if (loss == null) throw new IllegalArgumentException("loss must not be null");
        this.loss = loss;
        return this;
    }

    /**
     * Sets the soft margin (f_scale) for the robust loss function.
     *
     * <p>The loss is evaluated as ρ(f²/C²) where C is this value.
     * Residuals with |f| ≪ C are treated as inliers; |f| ≫ C as outliers.
     * Has no effect when {@link RobustLoss#LINEAR} is used.
     * Default: 1.0.</p>
     *
     * @param value soft margin C (must be positive)
     * @return this problem instance
     */
    public TRFProblem lossScale(double value) {
        if (value <= 0) throw new IllegalArgumentException("lossScale must be positive, got " + value);
        this.lossScale = value;
        return this;
    }

    /**
     * Sets user-supplied diagonal scaling vector.
     *
     * @param diag scaling vector (length must equal n)
     * @return this problem instance
     */
    public TRFProblem diag(double[] diag) {
        this.diag = diag;
        return this;
    }

    // ── Internal state for lazy Jacobian wrapping ─────────────────────────────

    private transient NumericalJacobian pendingJac;
    private transient Multivariate.Objective pendingFn;
}