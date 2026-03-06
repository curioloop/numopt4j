package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.optim.OptimizationResult;
import com.curioloop.numopt4j.optim.RootFinder;

import java.util.function.BiConsumer;

/**
 * Multi-dimensional root finder using Broyden's method ({@code broyden1}).
 *
 * <p>Jacobian-free; maintains a rank-1 inverse-Jacobian approximation.
 * Workspace type: {@link BroydenWorkspace}.</p>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * OptimizationResult r = RootFinder.broyden((x, f) -> { f[0] = x[0] - 1; f[1] = x[1] - 2; }, 2)
 *     .initialPoint(0.0, 0.0)
 *     .solve();
 * }</pre>
 */
public final class BroydenProblem extends RootFinder<BiConsumer<double[], double[]>, BroydenWorkspace, BroydenProblem> {

    private double ftol = HYBRSolver.DEFAULT_FTOL;
    private int maxIterations = 0;

    public BroydenProblem() {}

    /** Sets the system of equations {@code F(x) = 0}. */
    public BroydenProblem equations(BiConsumer<double[], double[]> fn, int n) {
        if (n < 1) throw new IllegalArgumentException("n must be >= 1, got " + n);
        this.function = fn;
        this.dimension = n;
        return this;
    }

    /** Sets the function-value tolerance. Convergence when {@code ||F(x)||₂ <= ftol}. */
    public BroydenProblem functionTolerance(double ftol) {
        if (ftol <= 0) throw new IllegalArgumentException("ftol must be > 0");
        this.ftol = ftol;
        return this;
    }

    public double functionTolerance() { return ftol; }

    /** Sets the maximum number of iterations. */
    public BroydenProblem maxIterations(int k) {
        if (k <= 0) throw new IllegalArgumentException("maxIterations must be > 0");
        this.maxIterations = k;
        return this;
    }

    public int maxIterations() { return maxIterations; }

    /** Allocates a {@link BroydenWorkspace} for dimension {@code n} and caches it. */
    @Override
    public BroydenWorkspace alloc() {
        if (dimension < 1) throw new IllegalStateException(
            "Problem dimension n is unknown; call equations(fn, n) or initialPoint(x0) first");
        if (workspace == null) workspace = new BroydenWorkspace(dimension);
        return workspace;
    }

    @Override
    public OptimizationResult solve(BroydenWorkspace... ws) {
        if (function == null)
            throw new IllegalStateException(
                "Missing required parameter: equations. Call .equations(fn, n) before .solve().");
        if (initialPoint == null)
            throw new IllegalStateException(
                "Missing required parameter: initialPoint. Call .initialPoint(x0) before .solve().");
        BroydenWorkspace ws0 = (ws != null && ws.length > 0 && ws[0] != null) ? ws[0] : alloc();
        ws0.reset();
        int maxIter = maxIterations > 0 ? maxIterations : 100 * (dimension + 1);
        return BroydenSolver.solve(function, initialPoint, ftol, maxIter, ws0);
    }
}
