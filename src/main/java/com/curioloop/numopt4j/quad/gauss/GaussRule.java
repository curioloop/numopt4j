/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss;
import java.util.Objects;

import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.quad.gauss.rule.GeneralizedHermiteRule;
import com.curioloop.numopt4j.quad.gauss.rule.GeneralizedLaguerreRule;
import com.curioloop.numopt4j.quad.gauss.rule.HermiteRule;
import com.curioloop.numopt4j.quad.gauss.rule.JacobiRule;
import com.curioloop.numopt4j.quad.gauss.rule.LaguerreRule;
import com.curioloop.numopt4j.quad.gauss.rule.LegendreRule;

import java.util.Arrays;

/**
 * A Gaussian quadrature rule defined on its own canonical domain and weight function.
 *
 * <p>Implementations generate nodes xᵢ and weights wᵢ such that
 *   ∫ w(x)·f(x) dx ≈ Σᵢ wᵢ·f(xᵢ)
 * is exact for polynomials up to a degree determined by the number of points n.
 * An n-point rule is exact for polynomials of degree ≤ 2n−1.</p>
 *
 * <p>Standard rules (returned by the no-arg factory methods):</p>
 * <ul>
 *   <li>{@link #legendre()} — ∫_{−1}^{1} f(x) dx,  w(x) = 1</li>
 *   <li>{@link #laguerre()} — ∫_{0}^{+∞} e^{−x}·f(x) dx</li>
 *   <li>{@link #hermite()}  — ∫_{−∞}^{+∞} e^{−x²}·f(x) dx</li>
 * </ul>
 *
 * <p>Parameterized rules (returned by the factory methods with arguments):</p>
 * <ul>
 *   <li>{@link #laguerre(double)} — ∫_{0}^{+∞} x^s·e^{−x}·f(x) dx,  s &gt; −1</li>
 *   <li>{@link #hermite(double)}  — ∫_{−∞}^{+∞} |x|^{2s}·e^{−x²}·f(x) dx,  s &gt; −1/2</li>
 *   <li>{@link #jacobi(double, double)} — ∫_{−1}^{1} (1−x)^α·(1+x)^β·f(x) dx,  α,β &gt; −1</li>
 * </ul>
 *
 * <p>All rules are generated via the Golub-Welsch algorithm: nodes are eigenvalues
 * of the symmetric tridiagonal Jacobi matrix J, and weights are wᵢ = μ₀·v₀ᵢ² where
 * μ₀ = ∫ w(x) dx is the zero-th moment and v₀ᵢ is the first component of the
 * i-th normalised eigenvector of J.</p>
 */
public interface GaussRule {

    /** Returns the zero-th moment μ₀ = ∫ w(x) dx of this rule's weight function. */
    double zeroMoment();

    /** Fills the symmetric tridiagonal Jacobi matrix entries for this rule family. */
    void fillJacobi(int points, double[] arena, int diagonalOffset, int offDiagonalOffset);

    /**
     * Generates nodes and weights for the given number of quadrature points
     * via the Golub-Welsch algorithm.
     *
     * <p>Given the three-term recurrence
     *   p_{n+1}(x) = (x − αₙ)·pₙ(x) − βₙ·p_{n−1}(x),
     * the quadrature nodes are the eigenvalues of the symmetric tridiagonal matrix
     *   J = diag(α₀,…,αₙ₋₁) + off-diag(√β₁,…,√βₙ₋₁),
     * and the weights are wᵢ = μ₀·v₀ᵢ² where v₀ᵢ is the first component of the
     * i-th normalised eigenvector.  The eigendecomposition uses LAPACK dsteqr.</p>
     *
     * @param points    number of quadrature points, must be positive
     * @param workspace reusable rule-generation workspace, must not be null
     */
    default void generate(int points, GaussPool workspace) {
        if (points <= 0) throw new IllegalArgumentException("points must be > 0");
        Objects.requireNonNull(workspace, "workspace must not be null");
        workspace.ensure(points);
        double[] arena = workspace.arena();
        int mat = workspace.matrixOffset(), spec = workspace.spectrumOffset();
        int offDiag = workspace.offDiagonalOffset(), nodes = workspace.nodesOffset();
        int weights = workspace.weightsOffset(), work = workspace.workOffset();

        Arrays.fill(arena, mat, mat + points * points, 0.0);
        Arrays.fill(arena, spec, spec + points, 0.0);
        if (points > 1) Arrays.fill(arena, offDiag, offDiag + points - 1, 0.0);
        fillJacobi(points, arena, spec, offDiag);

        int info = BLAS.dsteqr('I', points, arena, spec, arena, offDiag, arena, mat, points, arena, work);
        if (info != 0) throw new ArithmeticException("Quadrature rule generation failed, dsteqr info=" + info);

        double zeroMoment = zeroMoment();
        System.arraycopy(arena, spec, arena, nodes, points);
        for (int j = 0; j < points; j++) {
            double v0 = arena[mat + j];
            arena[weights + j] = zeroMoment * v0 * v0;
        }
    }

    // -----------------------------------------------------------------------
    // Standard rule factory methods
    // -----------------------------------------------------------------------

    /** Returns the standard Gauss-Legendre rule: ∫_{−1}^{1} f(x) dx.  μ₀ = 2. */
    static GaussRule legendre() { return LegendreRule.INSTANCE; }

    /** Returns the standard Gauss-Laguerre rule: ∫_{0}^{+∞} e^{−x} f(x) dx.  μ₀ = 1. */
    static GaussRule laguerre() { return LaguerreRule.INSTANCE; }

    /** Returns the standard Gauss-Hermite rule: ∫_{−∞}^{+∞} e^{−x²} f(x) dx.  μ₀ = √π. */
    static GaussRule hermite()  { return HermiteRule.INSTANCE; }

    // -----------------------------------------------------------------------
    // Parameterized rule factory methods
    // -----------------------------------------------------------------------

    /**
     * Returns a generalized Gauss-Laguerre rule for
     *   ∫_{0}^{+∞} x^s · e^{−x} · f(x) dx,  s > −1.
     * When s = 0 this is equivalent to {@link #laguerre()}.
     */
    static GaussRule laguerre(double s) { return new GeneralizedLaguerreRule(s); }

    /**
     * Returns a generalized Gauss-Hermite rule for
     *   ∫_{−∞}^{+∞} |x|^{2s} · e^{−x²} · f(x) dx,  s > −1/2.
     * When s = 0 this is equivalent to {@link #hermite()}.
     */
    static GaussRule hermite(double s)  { return new GeneralizedHermiteRule(s); }

    /**
     * Returns a Gauss-Jacobi rule for
     *   ∫_{−1}^{1} (1−x)^α · (1+x)^β · f(x) dx,  α,β > −1.
     * Special cases: α=β=0 → Legendre; α=β=−1/2 → Chebyshev 1st kind.
     */
    static GaussRule jacobi(double alpha, double beta) { return new JacobiRule(alpha, beta); }

    /**
     * Returns a Gauss-Chebyshev rule of the first kind for
     *   ∫_{−1}^{1} f(x) / √(1−x²) dx.
     * Equivalent to {@link #jacobi(double, double) jacobi(-0.5, -0.5)}.
     * Zero-th moment: μ₀ = π.
     */
    static GaussRule chebyshev1() { return new JacobiRule(-0.5, -0.5); }

    /**
     * Returns a Gauss-Chebyshev rule of the second kind for
     *   ∫_{−1}^{1} f(x) · √(1−x²) dx.
     * Equivalent to {@link #jacobi(double, double) jacobi(0.5, 0.5)}.
     * Zero-th moment: μ₀ = π/2.
     */
    static GaussRule chebyshev2() { return new JacobiRule(0.5, 0.5); }

    /**
     * Returns a Gauss-Gegenbauer (ultraspherical) rule for
     *   ∫_{−1}^{1} (1−x²)^{λ−1/2} · f(x) dx,  λ > −1/2.
     * Equivalent to {@link #jacobi(double, double) jacobi(λ−0.5, λ−0.5)}.
     * Special cases: λ=0 → Chebyshev 1st kind; λ=1 → Chebyshev 2nd kind; λ=1/2 → Legendre.
     */
    static GaussRule gegenbauer(double lambda) { return new JacobiRule(lambda - 0.5, lambda - 0.5); }

    // -----------------------------------------------------------------------
    // Utility: log-Gamma function (Lanczos approximation)
    // -----------------------------------------------------------------------

    static final double LOG_TWO = Math.log(2.0);

    static final double[] LANCZOS = {
            676.5203681218851, -1259.1392167224028, 771.3234287776531,
            -176.6150291621406, 12.507343278686905, -0.13857109526572012,
            9.984369578019572e-6, 1.5056327351493116e-7
    };

    /**
     * Computes ln Γ(x) via the Lanczos approximation.
     *
     * <p>Used internally by {@link JacobiRule}, {@link GeneralizedLaguerreRule},
     * and {@link GeneralizedHermiteRule} to compute zero-th moments.</p>
     */
    public static double logGamma(double x) {
        if (x < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * x)) - logGamma(1.0 - x);
        double s = x - 1.0, sum = 0.9999999999998099;
        for (int i = 0; i < LANCZOS.length; i++) sum += LANCZOS[i] / (s + i + 1.0);
        double t = s + LANCZOS.length - 0.5;
        return 0.9189385332046727 + (s + 0.5) * Math.log(t) - t + Math.log(sum);
    }

}
