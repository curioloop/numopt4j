package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.optim.Multivariate;
import com.curioloop.numopt4j.optim.NumericalJacobian;
import com.curioloop.numopt4j.optim.Optimization;
import com.curioloop.numopt4j.optim.RootFinder;

/**
 * Multi-dimensional root finder using the Powell hybrid method (MINPACK {@code hybrd}).
 *
 * <p>Workspace type: {@link HYBRWorkspace}. Create once with {@link #workspace()} and reuse across calls.</p>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * HYBRProblem finder = RootFinder.hybr((x, f) -> { f[0] = x[0] - 1; }, 1)
 *     .initialPoint(0.0);
 * HYBRWorkspace ws = HYBRProblem.workspace();
 * Optimization r = finder.solve(ws);
 * }</pre>
 */
public final class HYBRProblem extends RootFinder<Multivariate.Objective, HYBRWorkspace, HYBRProblem> {

    private NumericalJacobian jacobian = NumericalJacobian.FORWARD;
    private double ftol = HYBRSolver.DEFAULT_FTOL;
    private int maxEvaluations = 0;

    public HYBRProblem() {}

    /** Sets the system of equations {@code F(x) = 0}. */
    public HYBRProblem equations(Multivariate.Objective fn, int n) {
        if (fn == null) throw new IllegalArgumentException("fn must not be null");
        if (n < 1) throw new IllegalArgumentException("n must be >= 1, got " + n);
        this.function = fn;
        this.dimension = n;
        return this;
    }

    /**
     * Sets the numerical Jacobian method.
     * Default: {@link NumericalJacobian#FORWARD}.
     */
    public HYBRProblem jacobian(NumericalJacobian jac) {
        this.jacobian = jac != null ? jac : NumericalJacobian.FORWARD;
        return this;
    }

    /** Sets the function-value tolerance. Convergence when {@code ||F(x)||₂ <= ftol}. */
    public HYBRProblem functionTolerance(double ftol) {
        if (ftol <= 0) throw new IllegalArgumentException("ftol must be > 0");
        this.ftol = ftol;
        return this;
    }

    public double functionTolerance() { return ftol; }

    /** Sets the maximum number of function evaluations. */
    public HYBRProblem maxEvaluations(int k) {
        if (k <= 0) throw new IllegalArgumentException("maxEvaluations must be > 0");
        this.maxEvaluations = k;
        return this;
    }

    public int maxEvaluations() { return maxEvaluations; }

    /**
     * Creates a new {@link HYBRWorkspace} for use with {@link #solve(HYBRWorkspace)}.
     * Memory is allocated lazily on the first {@code solve()} call.
     */
    public static HYBRWorkspace workspace() {
        return new HYBRWorkspace();
    }

    @Override
    public Optimization solve(HYBRWorkspace ws) {
        if (function == null)
            throw new IllegalStateException(
                "Missing required parameter: equations. Call .equations(fn, n) before .solve().");
        if (initialPoint == null)
            throw new IllegalStateException(
                "Missing required parameter: initialPoint. Call .initialPoint(x0) before .solve().");
        HYBRWorkspace ws0 = ws != null ? ws : workspace();
        ws0.ensure(dimension);
        int maxfev = maxEvaluations > 0 ? maxEvaluations : HYBRSolver.DEFAULT_MAXFEV_FACTOR * (dimension + 1);
        return HYBRSolver.solve(jacobian.wrap(function, dimension, dimension, true), initialPoint, ftol, maxfev, ws0);
    }
}
