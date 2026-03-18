/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.NumericalGradient;
import com.curioloop.numopt4j.optim.OptimizationFailure;
import com.curioloop.numopt4j.optim.OptimizationResult;
import com.curioloop.numopt4j.optim.OptimizationStatus;
import com.curioloop.numopt4j.optim.Univariate;

import java.time.Duration;
import java.util.function.ToDoubleFunction;

/**
 * Fluent API for defining and solving L-BFGS-B bound-constrained optimization problems.
 *
 * <p>L-BFGS-B solves bound-constrained optimization problems:</p>
 * <pre>
 *   minimize f(x)
 *   subject to l ≤ x ≤ u
 * </pre>
 *
 * <p>{@code LBFGSBProblem} is the public fluent API that validates inputs, manages
 * workspaces, and stores the solution in
 * {@link com.curioloop.numopt4j.optim.OptimizationResult#getSolution()}.</p>
 *
 * <h2>Basic Usage</h2>
 * <pre>{@code
 * // Recommended entry point: ToDoubleFunction lambda (no gradient required)
 * OptimizationResult result = new LBFGSBProblem()
 *     .objective(x -> x[0]*x[0] + x[1]*x[1])
 *     .initialPoint(1.0, 1.0)
 *     .solve();
 *
 * if (result.isSuccessful()) {
 *     double[] solution = result.getSolution();  // [0.0, 0.0]
 * }
 * }</pre>
 *
 * <h2>Advanced Usage</h2>
 * <pre>{@code
 * // Bound-constrained with analytical gradient for best performance
 * OptimizationResult result = new LBFGSBProblem()
 *     .objective((x, g) -> {
 *         double f = x[0]*x[0] + x[1]*x[1];
 *         if (g != null) { g[0] = 2*x[0]; g[1] = 2*x[1]; }
 *         return f;
 *     })
 *     .bounds(Bound.atLeast(0), Bound.atLeast(0))
 *     .initialPoint(1.0, 1.0)
 *     .maxIterations(200)
 *     .gradientTolerance(1e-10)
 *     .solve();
 *
 * // Workspace reuse for high-frequency optimization (e.g., in a loop)
 * LBFGSBProblem problem = new LBFGSBProblem()
 *     .objective(x -> x[0]*x[0])
 *     .initialPoint(1.0);
 * LBFGSBWorkspace ws = problem.alloc();  // allocate workspace once
 * for (Data data : dataList) {
 *     problem.objective(data::objective).solve(ws);  // reuse workspace
 * }
 * }</pre>
 *
 * @see com.curioloop.numopt4j.optim.Minimize
 * @see com.curioloop.numopt4j.optim.Bound
 */
public final class LBFGSBProblem extends Minimizer<Univariate, LBFGSBWorkspace, LBFGSBProblem> {

    private static final double EPSILON = Math.ulp(1.0);

    /** Maximum number of iterations (default: 15000) */
    private int maxIterations = 15000;

    /** Maximum number of function evaluations (default: 15000) */
    private int maxEvaluations = 15000;

    /** Maximum number of floating-point operations (default: null = unlimited) */
    private Duration maxComputations = null;

    /**
     * Projected gradient tolerance (pgtol) for convergence.
     * <p>
     * Convergence criterion: ‖proj(g)‖∞ ≤ pgtol where proj(g) is the projected gradient
     * and ‖·‖∞ denotes the infinity norm (maximum absolute component).
     * </p>
     * <p>
     * The projected gradient accounts for bound constraints by setting components
     * to zero when the corresponding variable is at a bound and the gradient
     * points outward from the feasible region.
     * </p>
     * <p>Default: 1e-5 (matches scipy's default pgtol=1e-5)</p>
     */
    private double gradientTolerance = 1e-5;

    /** Problem dimension (number of variables) — inherited as {@code dimension} from {@link Minimizer} */

    /**
     * Number of L-BFGS corrections (limited memory parameter m).
     * <p>
     * Controls the amount of curvature information stored. Typical range is 3-20.
     * Higher values use more memory but may converge faster.
     * </p>
     * <p>Default: 10</p>
     */
    private int corrections = 10;

    /**
     * Maximum number of line search steps per iteration.
     * <p>
     * Controls how many function evaluations are allowed during the backtracking
     * line search within each iteration. Increasing this may help convergence on
     * difficult problems at the cost of more evaluations per iteration.
     * </p>
     * <p>Default: 20 (matches scipy's default maxls=20)</p>
     */
    private int maxLineSearchSteps = 20;

    /**
     * Function tolerance (ftol) for convergence.
     * <p>
     * Convergence criterion for function value change:
     * <pre>
     *   |fₖ - fₖ₊₁| / max(|fₖ|, |fₖ₊₁|, 1) ≤ ftol
     * </pre>
     * where fₖ is the function value at iteration k.
     * </p>
     * <p>
     * Matches scipy's {@code ftol} parameter. Internally converted to
     * {@code factr = ftol / ε} where ε ≈ 2.2e-16 is machine precision.
     * </p>
     * <p>Default: 1e7 * ε ≈ 2.22e-9 (matches scipy's default factr=1e7)</p>
     */
    private double functionTolerance = 1e7 * EPSILON;

    /** Cached workspace for reuse — inherited from {@link Minimizer} */

    public LBFGSBProblem() {}

    private void validate() {
        if (objective == null) {
            throw new OptimizationFailure("MISSING_PARAM",
                "objective is required. Call .objective(fn) before .solve().");
        }
        if (initialPoint == null || initialPoint.length == 0) {
            throw new OptimizationFailure("MISSING_PARAM",
                "initialPoint is required. Call .initialPoint(x0) before .solve().");
        }
        for (int i = 0; i < initialPoint.length; i++) {
            double v = initialPoint[i];
            if (Double.isNaN(v) || Double.isInfinite(v)) {
                throw new OptimizationFailure("INVALID_INPUT",
                    "initialPoint[" + i + "] is " + v + ". All initial values must be finite.");
            }
        }
    }

    @Override
    public LBFGSBWorkspace alloc() {
        validate();
        if (workspace == null || !workspace.ensureCapacity(dimension, corrections)) {
            workspace = new LBFGSBWorkspace(dimension, corrections);
        }
        return workspace;
    }

    /**
     * Solves the optimization problem.
     *
     * <p>The initial point is cloned internally; {@code initialPoint} is not modified.
     * The solution is stored in {@link OptimizationResult#getSolution()} and returned
     * as a direct reference (no defensive copy). The caller owns the returned array.</p>
     *
     * @param workspace optional pre-allocated workspace for reuse
     * @return optimization result
     */
    @Override
    public OptimizationResult solve() {
        return solve((LBFGSBWorkspace) null);
    }

    @Override
    public OptimizationResult solve(LBFGSBWorkspace workspace) {
        validate();
        LBFGSBWorkspace ws = workspace;
        if (ws != null && !ws.ensureCapacity(dimension, corrections)) {
            throw new IllegalArgumentException(
                    "Workspace dimensions (" + ws.getDimension() + ", " + ws.getCorrections() +
                            ") do not match problem dimensions (" + dimension + ", " + corrections + ")");
        }
        if (ws == null) {
            ws = this.workspace;
            if (ws == null || !ws.ensureCapacity(dimension, corrections)) {
                ws = new LBFGSBWorkspace(dimension, corrections);
            }
        }
        ws.reset();
        double[] x = initialPoint.clone();

        LBFGSBCore.OptimizeResult coreResult = LBFGSBCore.optimize(
                dimension, corrections,
                x,
                objective,
                bounds,
                functionTolerance / EPSILON,  // convert ftol to factr = ftol / eps
                gradientTolerance,
                maxIterations,
                maxEvaluations,
                maxComputations != null ? maxComputations.toNanos() : 0,
                maxLineSearchSteps,
                ws
        );

        OptimizationStatus status = convertStatus(coreResult.status);
        return new OptimizationResult(Double.NaN, x, coreResult.f, status, coreResult.iterations, coreResult.evaluations);
    }

    private static OptimizationStatus convertStatus(int statusCode) {
        if (statusCode == LBFGSBConstants.CONV_GRAD_PROG_NORM)  return OptimizationStatus.GRADIENT_TOLERANCE_REACHED;
        if (statusCode == LBFGSBConstants.CONV_ENOUGH_ACCURACY) return OptimizationStatus.FUNCTION_TOLERANCE_REACHED;
        if (statusCode == LBFGSBConstants.OVER_ITER_LIMIT)      return OptimizationStatus.MAX_ITERATIONS_REACHED;
        if (statusCode == LBFGSBConstants.OVER_EVAL_LIMIT)      return OptimizationStatus.MAX_EVALUATIONS_REACHED;
        if (statusCode == LBFGSBConstants.OVER_TIME_LIMIT)      return OptimizationStatus.MAX_COMPUTATIONS_REACHED;
        if (statusCode == LBFGSBConstants.OVER_GRAD_THRESH)     return OptimizationStatus.GRADIENT_TOLERANCE_REACHED;
        if (statusCode == LBFGSBConstants.STOP_ABNORMAL_SEARCH) return OptimizationStatus.LINE_SEARCH_FAILED;
        if (statusCode == LBFGSBConstants.HALT_EVAL_PANIC)      return OptimizationStatus.CALLBACK_ERROR;
        return OptimizationStatus.ABNORMAL_TERMINATION;
    }

    public int corrections() { return corrections; }
    public int maxIterations() { return maxIterations; }
    public int maxEvaluations() { return maxEvaluations; }
    public int maxLineSearchSteps() { return maxLineSearchSteps; }
    public double functionTolerance() { return functionTolerance; }
    public double gradientTolerance() { return gradientTolerance; }

    /**
     * Sets the maximum number of iterations.
     *
     * <p>Valid range: &gt; 0. Default: 15000.
     * Increasing this allows more iterations for convergence on difficult problems,
     * at the cost of longer runtime. If the optimizer hits this limit,
     * {@link com.curioloop.numopt4j.optim.OptimizationStatus#MAX_ITERATIONS_REACHED} is returned.</p>
     *
     * @param value maximum number of iterations (must be positive)
     * @return this problem instance
     * @throws IllegalArgumentException if value &lt;= 0
     */
    public LBFGSBProblem maxIterations(int value) {
        if (value <= 0) {
            throw new IllegalArgumentException("maxIterations must be positive, got " + value);
        }
        this.maxIterations = value;
        return this;
    }

    /**
     * Sets the maximum number of function evaluations.
     *
     * <p>Valid range: &gt; 0. Default: 15000.
     * Limits total objective function calls. Useful for expensive functions where
     * computation budget is constrained. If the optimizer hits this limit,
     * {@link com.curioloop.numopt4j.optim.OptimizationStatus#MAX_EVALUATIONS_REACHED} is returned.</p>
     *
     * @param value maximum number of function evaluations (must be positive)
     * @return this problem instance
     * @throws IllegalArgumentException if value &lt;= 0
     */
    public LBFGSBProblem maxEvaluations(int value) {
        if (value <= 0) {
            throw new IllegalArgumentException("maxEvaluations must be positive, got " + value);
        }
        this.maxEvaluations = value;
        return this;
    }

    public LBFGSBProblem maxComputations(Duration value) {
        if (value != null && value.isNegative()) {
            throw new IllegalArgumentException("Max computations must be non-negative");
        }
        this.maxComputations = value;
        return this;
    }

    /**
     * Sets the projected gradient tolerance (pgtol) for convergence.
     * <p>
     * Convergence criterion: ‖proj(g)‖∞ ≤ pgtol where proj(g) is the projected gradient.
     * Matches scipy's {@code pgtol} parameter. Default: 1e-5.
     * </p>
     *
     * @param value projected gradient tolerance (must be positive)
     * @return this problem instance
     * @throws IllegalArgumentException if value is not positive or is NaN
     */
    public LBFGSBProblem gradientTolerance(double value) {
        if (value <= 0 || Double.isNaN(value)) {
            throw new IllegalArgumentException("gradientTolerance must be positive, got " + value);
        }
        this.gradientTolerance = value;
        return this;
    }

    /**
     * Sets the objective function with automatic numerical gradient (central difference).
     *
     * <p><b>Recommended entry point for AI use - no gradient required.</b>
     * Automatically wraps the function using {@link NumericalGradient#CENTRAL} (O(h²) accuracy,
     * 2 evaluations per dimension). Suitable for most problems where analytical gradients
     * are unavailable or inconvenient to compute.</p>
     *
     * @param f function that computes only the objective value (ℝⁿ → ℝ)
     * @return this problem instance
     * @throws IllegalArgumentException if f is null
     */
    public LBFGSBProblem objective(ToDoubleFunction<double[]> f) {
        if (f == null) {
            throw new IllegalArgumentException("objective function must not be null");
        }
        return objective(NumericalGradient.CENTRAL, f);
    }

    /**
     * Sets the objective function with specified numerical gradient method.
     *
     * @param grad numerical gradient method
     * @param f function that computes only the objective value
     * @return this problem instance
     */
    public LBFGSBProblem objective(NumericalGradient grad, ToDoubleFunction<double[]> f) {
        this.objective = grad.wrap(f);
        return this;
    }

    /**
     * Sets the objective function with analytical gradient.
     * <p>
     * Use this method for best performance when analytical gradients are available.
     * </p>
     *
     * @param f function that computes both value and gradient
     * @return this problem instance
     */
    public LBFGSBProblem objective(Univariate f) {
        if (f == null) {
            throw new IllegalArgumentException("objective function must not be null");
        }
        this.objective = f;
        return this;
    }

    /**
     * Sets the number of L-BFGS corrections (limited memory parameter).
     * <p>
     * Default value is 10. Typical range is 3-20.
     * </p>
     *
     * @param m number of corrections
     * @return this problem instance
     */
    public LBFGSBProblem corrections(int m) {
        if (m <= 0) {
            throw new IllegalArgumentException("Corrections must be positive");
        }
        this.corrections = m;
        return this;
    }

    /**
     * Sets the maximum number of line search steps per iteration.
     * <p>
     * Controls how many function evaluations are allowed during the backtracking
     * line search within each iteration. Increasing this may help convergence on
     * difficult problems at the cost of more evaluations per iteration.
     * </p>
     *
     * @param value maximum line search steps per iteration (must be positive)
     * @return this problem instance
     * @throws IllegalArgumentException if value &lt;= 0
     */
    public LBFGSBProblem maxLineSearchSteps(int value) {
        if (value <= 0) {
            throw new IllegalArgumentException("maxLineSearchSteps must be positive, got " + value);
        }
        this.maxLineSearchSteps = value;
        return this;
    }

    /**
     * Sets the function tolerance (ftol) for convergence.
     * <p>
     * Convergence criterion for function value change:
     * <pre>
     *   |fₖ - fₖ₊₁| / max(|fₖ|, |fₖ₊₁|, 1) ≤ ftol
     * </pre>
     * Matches scipy's {@code ftol} parameter. Internally converted to
     * {@code factr = ftol / ε} where ε ≈ 2.2e-16 is machine precision.
     * </p>
     *
     * @param value function tolerance (must be positive)
     * @return this problem instance
     * @throws IllegalArgumentException if value &lt;= 0 or is NaN
     */
    public LBFGSBProblem functionTolerance(double value) {
        if (value <= 0 || Double.isNaN(value)) {
            throw new IllegalArgumentException("functionTolerance must be positive, got " + value);
        }
        this.functionTolerance = value;
        return this;
    }
}
