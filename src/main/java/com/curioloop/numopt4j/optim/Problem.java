/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

/**
 * Interface for optimization problems.
 * <p>
 * Defines the core contract for all optimization algorithms.
 * Each implementation provides its own configuration options and workspace management.
 * </p>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Create and solve (workspace allocated internally)
 * Optimization result = Minimizer.lbfgsb()
 *     .objective((x, n) -> x[0]*x[0] + x[1]*x[1])
 *     .initialPoint(1.0, 1.0)
 *     .solve();
 *
 * // With workspace reuse (allocate once, reuse across calls)
 * LBFGSBProblem problem = Minimizer.lbfgsb()
 *     .objective(fn)
 *     .initialPoint(1.0, 1.0);
 * LBFGSBWorkspace ws = LBFGSBProblem.workspace();
 * problem.solve(ws);  // reuse workspace
 * }</pre>
 *
 * @param <W> the workspace type
 */
public interface Problem<W> {

    /**
     * Solves the optimization problem using a temporary workspace.
     *
     * @return optimization result
     */
    default Optimization solve() {
        return solve(null);
    }

    /**
     * Solves the optimization problem with a pre-allocated workspace for reuse.
     *
     * @param workspace pre-allocated workspace (pass {@code null} to allocate internally)
     * @return optimization result
     * @throws IllegalArgumentException if workspace is incompatible
     */
    Optimization solve(W workspace);
}
