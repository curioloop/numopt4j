/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.slsqp;

import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.NumericalGradient;
import com.curioloop.numopt4j.optim.Optimization;
import com.curioloop.numopt4j.optim.Univariate;

import java.time.Duration;
import java.util.function.ToDoubleFunction;

/**
 * Fluent API for defining and solving SLSQP constrained optimization problems.
 *
 * <p>SLSQP solves general constrained optimization problems:</p>
 * <pre>
 *   minimize f(x)
 *   subject to c_eq(x) = 0
 *              c_ineq(x) &gt;= 0
 *              l ≤ x ≤ u
 * </pre>
 *
 * <p>{@code SLSQPProblem} is the public fluent API that validates inputs, manages
 * workspaces, and stores the solution in
 * {@link Optimization#getSolution()}.</p>
 *
 * <h2>Basic Usage</h2>
 * <pre>{@code
 * // Recommended entry point: ToDoubleFunction lambda (no gradient required)
 * // Equality constraint: x[0] + x[1] = 1
 * Optimization result = new SLSQPProblem()
 *     .objective(x -> x[0]*x[0] + x[1]*x[1])
 *     .equalityConstraints(x -> x[0] + x[1] - 1)
 *     .initialPoint(0.5, 0.5)
 *     .solve();
 *
 * if (result.isSuccessful()) {
 *     double[] solution = result.getSolution();  // [0.5, 0.5]
 * }
 * }</pre>
 *
 * <h2>Advanced Usage</h2>
 * <pre>{@code
 * // Mixed constraints with analytical gradients for best performance
 * Optimization result = new SLSQPProblem()
 *     .objective((x, g) -> {
 *         double f = x[0]*x[0] + x[1]*x[1];
 *         if (g != null) { g[0] = 2*x[0]; g[1] = 2*x[1]; }
 *         return f;
 *     })
 *     .equalityConstraints((x, g) -> {
 *         if (g != null) { g[0] = 1; g[1] = 1; }
 *         return x[0] + x[1] - 1;
 *     })
 *     .inequalityConstraints(x -> x[0] - 0.1)  // x[0] >= 0.1
 *     .bounds(Bound.atLeast(0), Bound.atLeast(0))
 *     .initialPoint(0.5, 0.5)
 * .functionTolerance(1e-8)
 *     .maxIterations(200)
 *     .solve();
 * }</pre>
 *
 * @see com.curioloop.numopt4j.optim.Minimizer
 * @see com.curioloop.numopt4j.optim.Bound
 */
public final class SLSQPProblem extends Minimizer<Univariate, SLSQPWorkspace, SLSQPProblem> {

    /** Maximum number of iterations (default: 100) */
    private int maxIterations = 100;

    /** Maximum number of function evaluations (default: 1000) */
    private int maxEvaluations = 1000;

    /** Maximum number of floating-point operations (default: null = unlimited) */
    private Duration maxComputations = null;

    /** Equality constraint functions (c(x) = 0) */
    private Univariate[] equalityConstraints;

    /** Inequality constraint functions (c(x) ≥ 0) */
    private Univariate[] inequalityConstraints;

    /**
     * Convergence tolerance (functionTolerance) — primary convergence criterion.
     * <p>
     * Primary convergence check:
     * <pre>
     *   h1 < functionTolerance && h2 < functionTolerance
     * </pre>
     * where:
     * <ul>
     *   <li>h1 = |gᵀd| - optimality measure (gradient projected on search direction)</li>
     *   <li>h2 = Σⱼ max(-cⱼ, 0) - constraint violation (L1 norm for inequalities, 0 for equalities)</li>
     * </ul>
     * </p>
     * <p>Matches scipy's {@code functionTolerance} parameter for {@code method='SLSQP'}.</p>
     * <p>Default: 1e-6</p>
     */
    private double functionTolerance = 1e-6;

    /**
     * Maximum iterations for NNLS (Non-Negative Least Squares) subproblem solver.
     * <p>
     * When set to 0, uses the default value of 3×n where n is the problem dimension.
     * </p>
     * <p>Default: 0 (auto)</p>
     */
    private int nnlsIterations = 0;

    /** Whether to use exact line search (golden section + quadratic interpolation) */
    private boolean exactLineSearch = false;

    /**
     * Function evaluation tolerance for secondary convergence check.
     * <p>
     * Convergence criterion: |f| < ε₁ where ε₁ is this tolerance.
     * Disabled when set to -1.
     * </p>
     * <p>Default: -1 (disabled)</p>
     */
    private double functionEvaluationTolerance = -1;

    /**
     * Function difference tolerance for secondary convergence check.
     * <p>
     * Convergence criterion: |f - f₀| < ε₂ where f₀ is the previous function value.
     * Disabled when set to -1.
     * </p>
     * <p>Default: -1 (disabled)</p>
     */
    private double functionDifferenceTolerance = -1;

    /**
     * Variable difference tolerance for secondary convergence check.
     * <p>
     * Convergence criterion: ‖x - x₀‖₂ < ε₃ where x₀ is the previous point.
     * Disabled when set to -1.
     * </p>
     * <p>Default: -1 (disabled)</p>
     */
    private double variableDifferenceTolerance = -1;

    /** Cached workspace for reuse — inherited from {@link Minimizer} */

    public SLSQPProblem() {}

    private void validate() {
        if (objective == null) {
            throw new IllegalStateException(
                "objective is required. Call .objective(fn) before .solve().");
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
    }

    @Override
    public SLSQPWorkspace alloc() {
        validate();
        if (workspace == null || !workspace.ensureCapacity(dimension, numEqualityConstraints(), numInequalityConstraints())) {
            workspace = new SLSQPWorkspace(dimension, numEqualityConstraints(), numInequalityConstraints());
        }
        return workspace;
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
    public Optimization solve(SLSQPWorkspace workspace) {
        validate();
        SLSQPWorkspace ws = workspace;
        if (ws != null) {
            if (!ws.ensureCapacity(dimension, numEqualityConstraints(), numInequalityConstraints())) {
                throw new IllegalArgumentException("Workspace is incompatible with problem dimensions");
            }
        } else {
            ws = this.workspace;
            if (ws == null || !ws.ensureCapacity(dimension, numEqualityConstraints(), numInequalityConstraints())) {
                ws = new SLSQPWorkspace(dimension, numEqualityConstraints(), numInequalityConstraints());
            }
        }

        ws.reset();
        double[] x = initialPoint.clone();

        Optimization r = SLSQPCore.optimize(
                x,
                objective,
                equalityConstraints,
                inequalityConstraints,
                bounds,
                functionTolerance,
                maxIterations,
                maxEvaluations,
                nnlsIterations > 0 ? nnlsIterations : 0,
                maxComputations != null ? maxComputations.toNanos() : 0,
                exactLineSearch,
                functionEvaluationTolerance,
                functionDifferenceTolerance,
                variableDifferenceTolerance,
                ws
        );
        return new Optimization(Double.NaN, x, r.getCost(), r.getStatus(), r.getIterations(), r.getEvaluations());
    }

    private int numEqualityConstraints() { return equalityConstraints != null ? equalityConstraints.length : 0; }
    private int numInequalityConstraints() { return inequalityConstraints != null ? inequalityConstraints.length : 0; }
    public int maxIterations() { return maxIterations; }
    public int maxEvaluations() { return maxEvaluations; }
    public int nnlsIterations() { return nnlsIterations; }
    public double functionTolerance() { return functionTolerance; }

    /**
     * Sets the maximum number of iterations allowed for the optimization algorithm.
     *
     * <p>Valid range: &gt; 0. Default: 100.
     * Termination criterion: algorithm stops when iteration count reaches maxIterations.
     * This prevents infinite loops in case of non-convergent problems.
     * If the optimizer hits this limit,
     * {@link Optimization.Status#MAX_ITERATIONS_REACHED} is returned.</p>
     *
     * @param value maximum number of iterations (must be positive)
     * @return this problem instance
     * @throws IllegalArgumentException if value is not positive
     */
    public SLSQPProblem maxIterations(int value) {
        if (value <= 0) {
            throw new IllegalArgumentException("maxIterations must be positive, got " + value);
        }
        this.maxIterations = value;
        return this;
    }

    /**
     * Sets the maximum number of function evaluations allowed for the optimization algorithm.
     *
     * <p>Valid range: &gt; 0. Default: 1000.
     * Termination criterion: algorithm stops when function evaluation count reaches maxEvaluations.
     * This limits computational cost and prevents excessive resource consumption.
     * If the optimizer hits this limit,
     * {@link Optimization.Status#MAX_EVALUATIONS_REACHED} is returned.</p>
     *
     * @param value maximum number of function evaluations (must be positive)
     * @return this problem instance
     * @throws IllegalArgumentException if value is not positive
     */
    public SLSQPProblem maxEvaluations(int value) {
        if (value <= 0) {
            throw new IllegalArgumentException("maxEvaluations must be positive, got " + value);
        }
        this.maxEvaluations = value;
        return this;
    }

    /**
     * Sets the maximum computation time allowed for the optimization algorithm.
     * <p>
     * Termination criterion: algorithm stops when elapsed time reaches the specified duration.
     * When set to null, no time limit is enforced.
     * </p>
     *
     * @param value maximum computation time (must be non-negative)
     * @return this problem instance
     * @throws IllegalArgumentException if value is negative
     */
    public SLSQPProblem maxComputations(Duration value) {
        if (value != null && value.isNegative()) {
            throw new IllegalArgumentException("Max computations must be non-negative");
        }
        this.maxComputations = value;
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
    public SLSQPProblem objective(ToDoubleFunction<double[]> f) {
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
    public SLSQPProblem objective(NumericalGradient grad, ToDoubleFunction<double[]> f) {
        this.objective = grad.wrap(f);
        return this;
    }

    /**
     * Sets the objective function with analytical gradient.
     *
     * @param f function that computes both value and gradient
     * @return this problem instance
     */
    public SLSQPProblem objective(Univariate f) {
        if (f == null) {
            throw new IllegalArgumentException("objective function must not be null");
        }
        this.objective = f;
        return this;
    }

    /**
     * Sets equality constraints with automatic numerical gradient.
     *
     * @param constraints constraint functions (c(x) = 0)
     * @return this problem instance
     */
    @SafeVarargs
    public final SLSQPProblem equalityConstraints(ToDoubleFunction<double[]>... constraints) {
        return equalityConstraints(NumericalGradient.CENTRAL, constraints);
    }

    /**
     * Sets equality constraints with specified numerical gradient method.
     *
     * @param grad numerical gradient method
     * @param constraints constraint functions (c(x) = 0)
     * @return this problem instance
     */
    @SafeVarargs
    public final SLSQPProblem equalityConstraints(NumericalGradient grad, ToDoubleFunction<double[]>... constraints) {
        this.equalityConstraints = wrapConstraints(grad, constraints);
        return this;
    }

    /**
     * Sets equality constraints with analytical gradients.
     *
     * @param constraints constraint functions (c(x) = 0)
     * @return this problem instance
     */
    public SLSQPProblem equalityConstraints(Univariate... constraints) {
        this.equalityConstraints = constraints;
        return this;
    }

    /**
     * Sets inequality constraints with automatic numerical gradient.
     *
     * @param constraints constraint functions (c(x) >= 0)
     * @return this problem instance
     */
    @SafeVarargs
    public final SLSQPProblem inequalityConstraints(ToDoubleFunction<double[]>... constraints) {
        return inequalityConstraints(NumericalGradient.CENTRAL, constraints);
    }

    /**
     * Sets inequality constraints with specified numerical gradient method.
     *
     * @param grad numerical gradient method
     * @param constraints constraint functions (c(x) >= 0)
     * @return this problem instance
     */
    @SafeVarargs
    public final SLSQPProblem inequalityConstraints(NumericalGradient grad, ToDoubleFunction<double[]>... constraints) {
        this.inequalityConstraints = wrapConstraints(grad, constraints);
        return this;
    }

    /**
     * Sets inequality constraints with analytical gradients.
     *
     * @param constraints constraint functions (c(x) >= 0)
     * @return this problem instance
     */
    public SLSQPProblem inequalityConstraints(Univariate... constraints) {
        this.inequalityConstraints = constraints;
        return this;
    }

    /**
     * Sets the convergence tolerance (functionTolerance) — primary convergence criterion.
     *
     * <p>Valid range: &gt; 0. Default: 1e-6.
     * Primary convergence check:
     * <pre>
     *   h1 &lt; functionTolerance &amp;&amp; h2 &lt; functionTolerance
     * </pre>
     * where:
     * <ul>
     *   <li>h1 = |gᵀd| - optimality measure (gradient projected on search direction)</li>
     *   <li>h2 = Σⱼ max(-cⱼ, 0) - constraint violation (L1 norm)</li>
     * </ul>
     * Tighter values (e.g., 1e-9) yield more accurate solutions but require more iterations.
     * Looser values (e.g., 1e-4) converge faster but may be less precise.</p>
     *
     * @param value convergence tolerance (must be positive)
     * @return this problem instance
     * @throws IllegalArgumentException if value is not positive
     */
    public SLSQPProblem functionTolerance(double value) {
        if (value <= 0) {
            throw new IllegalArgumentException("functionTolerance must be positive, got " + value);
        }
        this.functionTolerance = value;
        return this;
    }

    /**
     * Sets the maximum number of iterations for the NNLS (Non-Negative Least Squares) solver.
     * <p>
     * When set to 0, uses the default value of 3*n where n is the problem dimension.
     * </p>
     *
     * @param iterations maximum number of NNLS iterations (non-negative)
     * @return this problem instance
     * @throws IllegalArgumentException if iterations is negative
     */
    public SLSQPProblem nnlsIterations(int iterations) {
        if (iterations < 0) {
            throw new IllegalArgumentException("NNLS iterations must be non-negative");
        }
        this.nnlsIterations = iterations;
        return this;
    }

    /**
     * Enables exact line search (golden section + quadratic interpolation).
     *
     * @return this problem instance
     */
    public SLSQPProblem exactLineSearch() {
        this.exactLineSearch = true;
        return this;
    }

    /**
     * Sets function evaluation tolerance for convergence.
     * <p>
     * Convergence criterion: |f| < ε₁ where ε₁ is the function evaluation tolerance.
     * This convergence condition is disabled when tolerance is set to -1.
     * </p>
     *
     * @param tol tolerance (converge if |f| < tol, -1 to disable)
     * @return this problem instance
     */
    public SLSQPProblem functionEvaluationTolerance(double tol) {
        this.functionEvaluationTolerance = tol;
        return this;
    }

    /**
     * Sets function difference tolerance for convergence.
     * <p>
     * Convergence criterion: |f - f₀| < ε₂ where ε₂ is the function difference tolerance,
     * and f₀ is the previous function value. This convergence condition is disabled
     * when tolerance is set to -1.
     * </p>
     *
     * @param tol tolerance (converge if |f - f0| < tol, -1 to disable)
     * @return this problem instance
     */
    public SLSQPProblem functionDifferenceTolerance(double tol) {
        this.functionDifferenceTolerance = tol;
        return this;
    }

    /**
     * Sets variable difference tolerance for convergence.
     * <p>
     * Convergence criterion: ||x - x₀|| < ε₃ where ε₃ is the variable difference tolerance,
     * x₀ is the previous variable vector, and ||·|| denotes the Euclidean norm.
     * This convergence condition is disabled when tolerance is set to -1.
     * </p>
     *
     * @param tol tolerance (converge if ||x - x0|| < tol, -1 to disable)
     * @return this problem instance
     */
    public SLSQPProblem variableDifferenceTolerance(double tol) {
        this.variableDifferenceTolerance = tol;
        return this;
    }

    @SafeVarargs
    private static Univariate[] wrapConstraints(NumericalGradient grad, ToDoubleFunction<double[]>... constraints) {
        if (constraints == null || constraints.length == 0) {
            return null;
        }
        Univariate[] result = new Univariate[constraints.length];
        for (int i = 0; i < constraints.length; i++) {
            result[i] = grad.wrap(constraints[i]);
        }
        return result;
    }
}
