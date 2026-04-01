/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;

/**
 * Specifies the algorithm and logarithmic endpoint factors for an endpoint-singular integral
 * of the form ∫ (x−a)^α (b−x)^β · log-factor · f(x) dx.
 *
 * <ul>
 *   <li>{@link #ALGEBRAIC}  — no logarithmic factor; uses Gauss-Jacobi quadrature</li>
 *   <li>{@link #LOG_LEFT}   — multiply by ln(x−a); uses tanh-sinh quadrature</li>
 *   <li>{@link #LOG_RIGHT}  — multiply by ln(b−x); uses tanh-sinh quadrature</li>
 *   <li>{@link #LOG_BOTH}   — multiply by ln(x−a)·ln(b−x); uses tanh-sinh quadrature</li>
 * </ul>
 */
public enum EndpointOpts {
    ALGEBRAIC(false, false),
    LOG_LEFT(true, false),
    LOG_RIGHT(false, true),
    LOG_BOTH(true, true);

    final boolean logLeft;
    final boolean logRight;

    EndpointOpts(boolean logLeft, boolean logRight) {
        this.logLeft = logLeft;
        this.logRight = logRight;
    }
}
