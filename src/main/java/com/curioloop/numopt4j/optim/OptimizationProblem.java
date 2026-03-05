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
 * @param <R> the result type returned by solve()
 * @param <W> the workspace type
 */
public interface OptimizationProblem<R extends OptimizationResult, W> {

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
     * Solves the optimization problem.
     * <p>
     * If no workspace is provided, a temporary workspace is created for this solve only.
     * </p>
     *
     * @param workspace optional pre-allocated workspace (omit to create temporary)
     * @return optimization result
     * @throws IllegalArgumentException if workspace is incompatible
     */
    @SuppressWarnings("unchecked")
    R solve(W... workspace);
}
