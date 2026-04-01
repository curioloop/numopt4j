/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad;

/**
 * Contract for a configured quadrature builder that can allocate a reusable
 * workspace and execute the integration.
 *
 * <p>All builders returned by {@link Integrator} implement this interface.
 * The two-phase lifecycle separates configuration from execution and allows
 * workspace reuse across repeated calls:</p>
 *
 * <pre>{@code
 * // One-shot (workspace allocated and discarded internally)
 * Quadrature r = Integrator.adaptive()
 *     .function(Math::sin).bounds(0, Math.PI).tolerances(1e-10, 1e-10)
 *     .integrate();
 *
 * // Workspace reuse (allocate once, integrate many times)
 * AdaptiveIntegral problem = Integrator.adaptive()
 *     .function(Math::sin).bounds(0, Math.PI).tolerances(1e-10, 1e-10);
 * AdaptivePool ws = problem.alloc();
 * for (double[] interval : intervals) {
 *     Quadrature r = problem.bounds(interval[0], interval[1]).integrate(ws);
 * }
 * }</pre>
 *
 * @param <R> result type — {@link Quadrature} for function-based integrators,
 *            {@link Double} for fixed/weighted Gauss rules,
 *            {@code double[]} for cumulative sampled integration
 * @param <W> workspace type — {@link com.curioloop.numopt4j.quad.adapt.AdaptivePool},
 *            {@link com.curioloop.numopt4j.quad.gauss.GaussPool}, or {@link Void}
 *            when no workspace is needed (sampled data)
 */
public interface Integral<R, W> {

    /**
     * Allocates and caches a reusable workspace sized for the current configuration.
     * Subsequent calls return the same cached instance unless the configuration changes.
     *
     * @return workspace instance ready for use with {@link #integrate(Object)}
     * @throws IllegalStateException if a required parameter has not been set
     */
    W alloc();

    /**
     * Computes the integral using an internally managed workspace.
     *
     * <p>Equivalent to {@code integrate(null)}: the implementation allocates
     * (or reuses a cached) workspace automatically.</p>
     *
     * @return integration result
     */
    default R integrate() {
        return integrate(null);
    }

    /**
     * Computes the integral using the caller-provided workspace.
     *
     * <p>Pass the workspace returned by {@link #alloc()} to avoid repeated
     * allocation when calling the same builder in a loop.</p>
     *
     * @param workspace pre-allocated workspace, or {@code null} to use an internal one
     * @return integration result
     */
    R integrate(W workspace);
}
