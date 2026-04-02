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
 *   off-diagonal βₖ = √(k/2)          for k odd
 *                   = √(k/2 + s)      for k even (k ≥ 2)
 *                   = √s              for k = 1 (first off-diagonal, even index 0→1)
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
     *   β₁ = √s,  βₖ = √(⌊k/2⌋/1 + s·(k mod 2 == 0 ? 1 : 0))
     *
     * <p>More precisely, for the off-diagonal entry at position i (1-indexed):
     *   i odd:  βᵢ = √(i/2)
     *   i even: βᵢ = √(i/2 + s)</p>
     */
    @Override
    public void fillJacobi(int n, double[] arena, int diag, int offDiag) {
        for (int i = 0; i < n; i++)
            arena[diag + i] = 0.0;
        for (int i = 1; i < n; i++) {
            // i is the 1-based index of the off-diagonal entry
            double val = (i % 2 == 1)
                    ? 0.5 * i          // odd i: k/2 where k=i
                    : 0.5 * i + s;     // even i: k/2 + s
            arena[offDiag + i - 1] = Math.sqrt(val);
        }
    }
}
