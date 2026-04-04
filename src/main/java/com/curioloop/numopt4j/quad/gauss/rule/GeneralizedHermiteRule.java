/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss.rule;

import com.curioloop.numopt4j.quad.gauss.GaussRule;

/**
 * Generalized Gauss-Hermite quadrature rule for
 *   ∫_{−∞}^{+∞} |x|^{2s} · e^{−x²} · f(x) dx,  s > −1/2.
 *
 * <p>When s = 0 this reduces to the standard Gauss-Hermite rule {@link HermiteRule}.</p>
 *
 * <p>Three-term recurrence (Golub-Welsch):
 *   diagonal αₖ = 0  (rule is symmetric about 0)
 *   off-diagonal βₖ = √(k/2 + s)      for k odd
 *                   = √(k/2)           for k even
 * When s = 0, both cases collapse to √(k/2) = standard Hermite rule.
 * Zero-th moment: μ₀ = Γ(s + 1/2)</p>
 *
 * <p>Usage — ∫₋∞^∞ f(x) dx (absorbing the weight):
 * <pre>{@code
 * // ∫₋∞^∞ f(x) dx = ∫₋∞^∞ [f(x)·|x|^{-2s}·e^{x²}] · |x|^{2s}·e^{-x²} dx
 * double result = Integrator.weighted()
 *     .function(x -> f(x) * Math.pow(Math.abs(x), -2*s) * Math.exp(x*x))
 *     .points(n).rule(new GeneralizedHermiteRule(s)).integrate();
 * }</pre></p>
 */
public final class GeneralizedHermiteRule implements GaussRule {

    private final double s;

    /**
     * @param s shape parameter, must be &gt; −1/2
     */
    public GeneralizedHermiteRule(double s) {
        if (!Double.isFinite(s)) throw new IllegalArgumentException("s must be finite");
        if (s <= -0.5)           throw new IllegalArgumentException("s must be > -1/2");
        this.s = s;
    }

    /**
     * Zero-th moment: μ₀ = Γ(s + 1/2).
     */
    @Override
    public double zeroMoment() {
        return Math.exp(GaussRule.logGamma(s + 0.5));
    }

    /**
     * Fills the Jacobi matrix for the generalized Hermite recurrence:
     *   αₖ = 0  (symmetric rule)
     *
     * <p>For the off-diagonal entry at position i (1-indexed):
     *   i odd:  βᵢ = √(i/2 + s)   — matches QuantLib GaussHermitePolynomial::beta()
     *   i even: βᵢ = √(i/2)
     * When s=0 both cases reduce to √(i/2), recovering the standard Hermite rule.</p>
     */
    @Override
    public void fillJacobi(int n, double[] arena, int diag, int offDiag) {
        for (int i = 0; i < n; i++)
            arena[diag + i] = 0.0;
        for (int i = 1; i < n; i++) {
            // i is the 1-based index of the off-diagonal entry
            double val = (i % 2 == 1)
                    ? 0.5 * i + s      // odd i: i/2 + s
                    : 0.5 * i;         // even i: i/2
            arena[offDiag + i - 1] = Math.sqrt(val);
        }
    }
}
