/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss.rule;

import com.curioloop.numopt4j.quad.gauss.GaussRule;

/**
 * Standard Gauss-Legendre quadrature rule for
 *   ∫_{−1}^{1} f(x) dx.
 *
 * <p>Three-term recurrence:
 *   diagonal αₖ = 0  (symmetric rule)
 *   off-diagonal βₖ = k / √(4k²−1)
 * Zero-th moment: μ₀ = 2</p>
 *
 */
public final class LegendreRule implements GaussRule {

    public static final LegendreRule INSTANCE = new LegendreRule();

    @Override
    public double zeroMoment() { return 2.0; }

    @Override
    public void fillJacobi(int points, double[] arena, int diag, int offDiag) {
        for (int i = 0; i < points; i++) arena[diag + i] = 0.0;
        for (int i = 1; i < points; i++)
            arena[offDiag + i - 1] = i / Math.sqrt(4.0 * i * i - 1.0);
    }
}
