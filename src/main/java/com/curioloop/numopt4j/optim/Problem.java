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
 * // Create and solve
 * LMResult result = LMProblem.create()
 *     .residuals(residuals, 10)
 *     .initialCoefficients(1.0, 1.0)
 *     .solve();
 *
 * // With workspace reuse
 * LMProblem problem = LMProblem.create()
 *     .residuals(residuals, 10)
 *     .initialCoefficients(1.0, 1.0);
 * LMWorkspace ws = problem.alloc();
 * problem.solve(ws);  // reuse workspace
 * }</pre>
 *
 * @param <W> the workspace type
 */
public interface Problem<W> {

    /**
     * Allocates a workspace for this optimization problem.
     * <p>
     * The workspace can be reused across multiple optimization runs to reduce
     * memory allocation overhead.
     * </p>
     *
     * @return allocated workspace instance
     * @throws IllegalStateException if required fields are missing
     */
    W alloc();

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
     * @param workspace pre-allocated workspace
     * @return optimization result
     * @throws IllegalArgumentException if workspace is incompatible
     */
    Optimization solve(W workspace);
}
