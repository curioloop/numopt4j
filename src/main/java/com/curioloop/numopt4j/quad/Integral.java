/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad;

/**
 * Contract for configured quadrature problems that can allocate reusable workspace and execute.
 *
 * <p>Typical usage pattern:</p>
 * <pre>{@code
 * AdaptiveIntegral problem = Integrator.adaptive()
 *     .function(Math::sin).bounds(0, Math.PI).tolerances(1e-10, 1e-10);
 *
 * // Option A: one-shot
 * Quadrature result = problem.integrate();
 *
 * // Option B: reuse workspace across multiple calls
 * AdaptivePool ws = problem.alloc();
 * Quadrature r1 = problem.integrate(ws);
 * Quadrature r2 = problem.integrate(ws);
 * }</pre>
 *
 * @param <R> result type returned by {@link #integrate()}
 * @param <W> workspace type; use {@link Void} when no workspace is needed
 */
public interface Integral<R, W> {

    /**
     * Allocates and caches a reusable workspace sized for this problem configuration.
     * Subsequent calls return the same instance if the configuration has not changed.
     */
    W alloc();

    /** Computes the integral using an internally managed workspace. */
    default R integrate() {
        return integrate(null);
    }

    /** Computes the integral using the caller-provided workspace, avoiding internal allocation. */
    R integrate(W workspace);
}