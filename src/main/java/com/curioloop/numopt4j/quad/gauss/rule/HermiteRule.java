/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss.rule;

import com.curioloop.numopt4j.quad.gauss.GaussRule;

/**
 * Standard Gauss-Hermite quadrature rule for
 *   ∫_{−∞}^{+∞} e^{−x²} · f(x) dx.
 *
 * <p>Three-term recurrence:
 *   diagonal αₖ = 0  (symmetric rule)
 *   off-diagonal βₖ = √(k/2)
 * Zero-th moment: μ₀ = √π</p>
 *
 */
public final class HermiteRule implements GaussRule {

    public static final HermiteRule INSTANCE = new HermiteRule();

    private static final double SQRT_PI = 1.7724538509055160272981674833411451;

    @Override
    public double zeroMoment() { return SQRT_PI; }

    @Override
    public void fillJacobi(int points, double[] arena, int diag, int offDiag) {
        for (int i = 0; i < points; i++) arena[diag + i] = 0.0;
        for (int i = 1; i < points; i++) arena[offDiag + i - 1] = Math.sqrt(0.5 * i);
    }
}
