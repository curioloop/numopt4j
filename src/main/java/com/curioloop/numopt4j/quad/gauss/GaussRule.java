/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss;
import java.util.Objects;

import com.curioloop.numopt4j.quad.gauss.rule.GeneralizedHermiteRule;
import com.curioloop.numopt4j.quad.gauss.rule.GeneralizedLaguerreRule;
import com.curioloop.numopt4j.quad.gauss.rule.HermiteRule;
import com.curioloop.numopt4j.quad.gauss.rule.JacobiRule;
import com.curioloop.numopt4j.quad.gauss.rule.LaguerreRule;
import com.curioloop.numopt4j.quad.gauss.rule.LegendreRule;

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

    /** Fills the symmetric tridiagonal Jacobi matrix entries for this rule family.
     * @param points number of quadrature points
     * @param diag   array to receive diagonal entries αₖ (n values at diagOff)
     * @param diagOff starting offset in diag
     * @param offDiag array to receive off-diagonal entries βₖ (n-1 values at offDiagOff)
     * @param offDiagOff starting offset in offDiag
     */
    void fillJacobi(int points, double[] diag, int diagOff, double[] offDiag, int offDiagOff);

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
     * @param workspace reusable rule-generation workspace, must not be null
     */
    default void generate(GaussPool workspace) {
        int n = workspace.points;
        if (n <= 0) throw new IllegalArgumentException("points must be > 0");
        Objects.requireNonNull(workspace, "workspace must not be null");

        // Arena layout: [nodes(n) | weights(n)] — total 2n doubles.
        // During QR iteration we reuse this space:
        //   arena[0..n-1]   → diag (diagonal αₖ, later eigenvalues)
        //   arena[n..2n-1]  → sub  (off-diagonal βₖ, 1-indexed: sub[0] unused)
        // Only ev0 (first row of Q) needs a separate allocation.
        double[] arena = workspace.arena();
        int dOff = workspace.nodesOffset();    // diag lives at arena[0..n-1]
        int sOff = workspace.weightsOffset();  // sub  lives at arena[n..2n-1]

        // fillJacobi writes n diagonal entries at dOff and n-1 off-diagonal entries at sOff+1.
        // sub is 1-indexed: arena[sOff+0] is unused, arena[sOff+k] = βₖ for k=1..n-1.
        fillJacobi(n, arena, dOff, arena, sOff + 1);
        arena[sOff] = 0.0; // sub[0] unused, ensure clean

        double[] ev0 = new double[n];
        ev0[0] = 1.0;  // first row of identity: e₀ = [1, 0, ..., 0]

        // Wilkinson-shift QL iteration (QuantLib TqrEigenDecomposition, first-row-only variant)
        for (int k = n - 1; k >= 1; k--) {
            while (!offDiagIsZero(k, arena, dOff, arena, sOff)) {
                int l = k;
                while (l > 0 && !offDiagIsZero(l, arena, dOff, arena, sOff)) l--;

                // Wilkinson shift: eigenvalue of the trailing 2×2 sub-matrix closest to diag[k]
                double dk  = arena[dOff + k], dk1 = arena[dOff + k - 1], sk = arena[sOff + k];
                double t1 = Math.sqrt(0.25 * (dk * dk + dk1 * dk1) - 0.5 * dk1 * dk + sk * sk);
                double t2 = 0.5 * (dk + dk1);
                double lambda = (Math.abs(t2 + t1 - dk) < Math.abs(t2 - t1 - dk))
                        ? t2 + t1 : t2 - t1;
                double q = arena[dOff + l] - lambda;

                // QR sweep from l+1 to k
                double sine = 1.0, cosine = 1.0, u = 0.0;
                boolean underflow = false;
                for (int i = l + 1; i <= k && !underflow; i++) {
                    double h = cosine * arena[sOff + i];
                    double p = sine   * arena[sOff + i];
                    arena[sOff + i - 1] = Math.sqrt(p * p + q * q);
                    if (arena[sOff + i - 1] != 0.0) {
                        sine   = p / arena[sOff + i - 1];
                        cosine = q / arena[sOff + i - 1];
                        double g = arena[dOff + i - 1] - u;
                        double t = (arena[dOff + i] - g) * sine + 2.0 * cosine * h;
                        u = sine * t;
                        arena[dOff + i - 1] = g + u;
                        q = cosine * t - h;
                        // update only the first row of Q
                        double e = ev0[i - 1];
                        ev0[i - 1] = sine * ev0[i] + cosine * e;
                        ev0[i]     = cosine * ev0[i] - sine * e;
                    } else {
                        arena[dOff + i - 1] -= u;
                        arena[sOff + l] = 0.0;
                        underflow = true;
                    }
                }
                if (!underflow) {
                    arena[dOff + k] -= u;
                    arena[sOff + k]  = q;
                    arena[sOff + l]  = 0.0;
                }
            }
        }

        // arena[dOff..dOff+n-1] now holds eigenvalues (nodes).
        // Compute weights = μ₀·v₀ᵢ² and write into arena[sOff..sOff+n-1].
        double mu0 = zeroMoment();
        for (int j = 0; j < n; j++) {
            arena[sOff + j] = mu0 * ev0[j] * ev0[j];
        }
    }

    /** Convergence check: sub[k] is negligible relative to diag[k-1] and diag[k]. */
    static boolean offDiagIsZero(int k, double[] diag, int dOff, double[] sub, int sOff) {
        double sum = Math.abs(diag[dOff + k - 1]) + Math.abs(diag[dOff + k]);
        return sum == sum + Math.abs(sub[sOff + k]);
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
