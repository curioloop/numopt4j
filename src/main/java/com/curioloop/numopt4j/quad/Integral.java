/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad;

/**
 * Contract for a configured quadrature builder that can execute the integration
 * with an optional reusable workspace.
 *
 * <p>All builders returned by {@link Integrator} implement this interface.</p>
 *
 * <pre>{@code
 * // One-shot (workspace allocated internally)
 * Quadrature r = Integrator.adaptive()
 *     .function(Math::sin).bounds(0, Math.PI).tolerances(1e-10, 1e-10)
 *     .integrate();
 *
 * // Workspace reuse (create once, integrate many times)
 * AdaptiveIntegral problem = Integrator.adaptive()
 *     .function(Math::sin).bounds(0, Math.PI).tolerances(1e-10, 1e-10);
 * AdaptivePool ws = new AdaptivePool();
 * for (double[] interval : intervals) {
 *     Quadrature r = problem.bounds(interval[0], interval[1]).integrate(ws);
 * }
 * }</pre>
 *
 * @param <R> result type — {@link Quadrature} for function-based integrators,
 *            {@link Double} for fixed/weighted Gauss rules,
 *            {@code double[]} for cumulative sampled integration
 * @param <W> workspace type — {@link com.curioloop.numopt4j.quad.adapt.AdaptivePool},
 *            {@link com.curioloop.numopt4j.quad.special.OscillatoryPool},
 *            {@link com.curioloop.numopt4j.quad.gauss.GaussPool}, or {@link Void}
 *            when no workspace is needed (sampled data)
 */
public interface Integral<R, W> {

    /**
     * Computes the integral using an internally managed workspace.
     *
     * <p>Equivalent to {@code integrate(null)}.</p>
     *
     * @return integration result
     */
    default R integrate() {
        return integrate(null);
    }

    /**
     * Computes the integral using the caller-provided workspace.
     *
     * <p>Pass a pre-allocated workspace to avoid repeated allocation when calling
     * the same builder in a loop. Pass {@code null} to allocate internally.</p>
     *
     * @param workspace pre-allocated workspace, or {@code null} to use an internal one
     * @return integration result
     */
    R integrate(W workspace);
}
