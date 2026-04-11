/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss.rule;

import com.curioloop.numopt4j.quad.gauss.GaussRule;

/**
 * Generalized Gauss-Laguerre quadrature rule for
 *   ∫_{0}^{+∞} x^s · e^{−x} · f(x) dx,  s > −1.
 *
 * <p>When s = 0 this reduces to the standard Gauss-Laguerre rule {@link LaguerreRule}.</p>
 *
 * <p>Three-term recurrence (Golub-Welsch):
 *   diagonal αₖ = 2k + s + 1
 *   off-diagonal βₖ = √(k·(k+s))
 * Zero-th moment: μ₀ = Γ(s+1)</p>
 *
 * <p>Usage — ∫₀^∞ f(x) dx (absorbing the weight):
 * <pre>{@code
 * // ∫₀^∞ f(x) dx = ∫₀^∞ [f(x)·x^{-s}·e^x] · x^s·e^{-x} dx
 * double result = Integrator.weighted()
 *     .function(x -> f(x) * Math.pow(x, -s) * Math.exp(x))
 *     .points(n).rule(new GeneralizedLaguerreRule(s)).integrate();
 * }</pre></p>
 */
public final class GeneralizedLaguerreRule implements GaussRule {

    private final double s;

    /**
     * @param s shape parameter, must be &gt; −1
     */
    public GeneralizedLaguerreRule(double s) {
        if (!Double.isFinite(s)) throw new IllegalArgumentException("s must be finite");
        if (s <= -1.0)           throw new IllegalArgumentException("s must be > -1");
        this.s = s;
    }

    /**
     * Zero-th moment: μ₀ = Γ(s+1).
     */
    @Override
    public double zeroMoment() {
        return Math.exp(GaussRule.logGamma(s + 1.0));
    }

    /**
     * Fills the Jacobi matrix for the generalized Laguerre recurrence:
     *   αₖ = 2k + s + 1
     *   βₖ = √(k·(k+s))
     */
    @Override
    public void fillJacobi(int n, double[] diag, int diagOff, double[] offDiag, int offDiagOff) {
        for (int i = 0; i < n; i++)
            diag[diagOff + i] = 2.0 * i + s + 1.0;
        for (int i = 1; i < n; i++)
            offDiag[offDiagOff + i - 1] = Math.sqrt((double) i * (i + s));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof GeneralizedLaguerreRule)) return false;
        return Double.compare(s, ((GeneralizedLaguerreRule) o).s) == 0;
    }

    @Override
    public int hashCode() { return Double.hashCode(s); }
}
