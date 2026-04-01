/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.adapt;

/**
 * Underlying quadrature rule used by {@link AdaptiveIntegral}.
 *
 * <ul>
 *   <li>{@link #GK15} — 15-point Gauss-Kronrod rule (default).
 *       Exact for polynomials of degree ≤ 29. Embeds a 7-point Gauss rule
 *       for error estimation. 15 function evaluations per interval.</li>
 *   <li>{@link #GAUSS_LOBATTO} — 4-point Gauss-Lobatto rule with endpoint reuse.
 *       Exact for polynomials of degree ≤ 5. Endpoints are fixed at the interval
 *       boundaries and shared between adjacent sub-intervals, so each bisection
 *       requires only 2 new function evaluations instead of 4.
 *       Preferred when function evaluations are expensive (e.g. Heston characteristic
 *       function integrals).</li>
 * </ul>
 */
public enum AdaptiveRule { GK15, GAUSS_LOBATTO }
