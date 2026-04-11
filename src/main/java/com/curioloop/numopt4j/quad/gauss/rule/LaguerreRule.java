/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss.rule;

import com.curioloop.numopt4j.quad.gauss.GaussRule;

/**
 * Standard Gauss-Laguerre quadrature rule for
 *   ∫_{0}^{+∞} e^{−x} · f(x) dx.
 *
 * <p>Three-term recurrence:
 *   diagonal αₖ = 2k + 1
 *   off-diagonal βₖ = k
 * Zero-th moment: μ₀ = 1</p>
 *
 */
public final class LaguerreRule implements GaussRule {

    public static final LaguerreRule INSTANCE = new LaguerreRule();

    @Override
    public double zeroMoment() { return 1.0; }

    @Override
    public void fillJacobi(int points, double[] diag, int diagOff, double[] offDiag, int offDiagOff) {
        for (int i = 0; i < points; i++) diag[diagOff + i] = 2.0 * i + 1.0;
        for (int i = 1; i < points; i++) offDiag[offDiagOff + i - 1] = i;
    }
}
