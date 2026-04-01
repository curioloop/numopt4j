/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;

import com.curioloop.numopt4j.quad.Quadrature;
import com.curioloop.numopt4j.quad.gauss.JacobiRule;
import com.curioloop.numopt4j.quad.gauss.GaussPool;

import java.util.function.DoubleUnaryOperator;

/**
 * Quadrature algorithms for endpoint-singular integrals on a finite interval.
 *
 * <p>Two strategies are provided:</p>
 * <ul>
 *   <li>{@link #algebraic} — Gauss-Jacobi rule refinement for pure algebraic singularities.
 *       Target: ∫_{a}^{b} (x−a)^α (b−x)^β f(x) dx,  α,β > −1.
 *       Nodes and weights are generated via the Jacobi three-term recurrence
 *       and the point count doubles at each refinement level.</li>
 *   <li>{@link #tanhSinh} — Tanh-sinh (double-exponential) quadrature for integrands
 *       that also carry logarithmic endpoint factors.
 *       Substitution: x = tanh(π/2·sinh(t)), mapping t ∈ (−∞,+∞) to x ∈ (−1,1).
 *       Weight: w(t) = π/2·cosh(t)/cosh²(π/2·sinh(t)).
 *       Achieves doubly-exponential convergence for algebraic/logarithmic singularities.</li>
 * </ul>
 *
 * <p>References:</p>
 * <ul>
 *   <li>Takahasi &amp; Mori, "Double exponential formulas for numerical integration",
 *       RIMS 1974.</li>
 *   <li>Bailey et al., "A comparison of three high-precision quadrature schemes",
 *       Experimental Mathematics 2005.</li>
 * </ul>
 */
final class EndpointSingularQuadrature {

    private EndpointSingularQuadrature() {}

    // -----------------------------------------------------------------------
    // Gauss-Jacobi refinement (algebraic singularities only)
    // -----------------------------------------------------------------------

    private static final int INITIAL_POINTS = 8;

    /**
     * Integrates ∫_{a}^{b} (x−a)^α (b−x)^β f(x) dx via adaptive Gauss-Jacobi quadrature.
     *
     * <p>Affine map: x = c + h·t,  c = (a+b)/2,  h = (b−a)/2,  t ∈ [−1,1]
     *   x − a = h·(1+t)   →  contributes (1+t)^α
     *   b − x = h·(1−t)   →  contributes (1−t)^β
     * So the integral becomes:
     *   h^(α+β+1) · ∫_{−1}^{1} (1−t)^β (1+t)^α f(c+h·t) dt
     * which is exactly the Gauss-Jacobi form with parameters (β, α).
     * The factor h^(α+β+1) is absorbed into {@code scale}.</p>
     *
     * <p>The point count doubles at each refinement level starting from {@value INITIAL_POINTS}.
     * Convergence is declared when |I_n − I_{n/2}| ≤ max(absTol, relTol·|I_n|).</p>
     */
    static Quadrature algebraic(DoubleUnaryOperator f, double min, double max,
                                double alpha, double beta,
                                double absTol, double relTol,
                                int maxRefinements, GaussPool workspace) {
        // Under the affine map x = center + halfSpan*t  (t ∈ [-1,1]):
        //   x - min  = halfSpan*(1+t)   →  contributes (1+t)^alpha
        //   max - x  = halfSpan*(1-t)   →  contributes (1-t)^beta
        // JacobiRule(a,b) generates weights for ∫(1-t)^a (1+t)^b g(t) dt,
        // so JacobiRule(beta, alpha) matches (1-t)^beta (1+t)^alpha.
        // The factor halfSpan^(alpha+beta+1) is absorbed into `scale`.
        JacobiRule rule = new JacobiRule(beta, alpha);
        double halfSpan = 0.5 * (max - min);
        double center   = 0.5 * (min + max);
        double scale    = Math.pow(halfSpan, alpha + beta + 1.0);
        double previous = Double.NaN, bestValue = Double.NaN, bestError = Double.POSITIVE_INFINITY;
        int totalEvaluations = 0;

        for (int level = 0; level < maxRefinements; level++) {
            int points = INITIAL_POINTS << level;
            workspace.ensure(points);
            rule.generate(points, workspace);

            double value = 0.0;
            for (int i = 0; i < points; i++) {
                value += workspace.weightAt(i) * f.applyAsDouble(center + halfSpan * workspace.nodeAt(i));
            }
            value *= scale;
            totalEvaluations += points;

            double error = Double.isNaN(previous) ? Math.abs(value) : Math.abs(value - previous);
            if (error < bestError) { bestValue = value; bestError = error; }
            if (!Double.isNaN(previous) && error <= Math.max(absTol, relTol * Math.abs(value))) {
                return new Quadrature(value, error, Quadrature.Status.CONVERGED, level + 1, totalEvaluations);
            }
            previous = value;
        }
        return new Quadrature(bestValue, bestError, Quadrature.Status.MAX_REFINEMENTS_REACHED, maxRefinements, totalEvaluations);
    }

    // -----------------------------------------------------------------------
    // Tanh-sinh (double-exponential) quadrature (algebraic + logarithmic)
    // -----------------------------------------------------------------------

    private static final double HALF_PI       = 0.5 * Math.PI;
    private static final double MIN_COMPLEMENT = 64.0 * Math.ulp(1.0);
    private static final int    MAX_TERMS      = 4096;
    private static final int    TAIL_STREAK    = 6;

    /**
     * Integrates f on [min, max] via tanh-sinh (double-exponential) quadrature.
     *
     * <p>Substitution: x = c + h·tanh(π/2·sinh(t)),  c = (min+max)/2,  h = (max−min)/2
     * Jacobian: dx/dt = h·π/2·cosh(t)/cosh²(π/2·sinh(t))
     * Euler-Maclaurin sum at step size δ = 2^{−level}:
     *   I ≈ h·δ · Σ_{k} w(k·δ) · [f(c − h·tanh(u_k)) + f(c + h·tanh(u_k))]
     * where u_k = π/2·sinh(k·δ) and w(t) = π/2·cosh(t)/cosh²(π/2·sinh(t)).</p>
     *
     * <p>The abscissa complement xjc = 1/(exp(u)·cosh(u)) ≈ 1 − tanh(u) is used
     * as the stopping criterion: when xjc ≤ 64·ε the abscissa is indistinguishable
     * from the endpoint in double precision.</p>
     *
     * <p>Suitable when f carries logarithmic endpoint factors in addition to algebraic ones.
     * The step size halves at each refinement level: δ = 2^{−level}.</p>
     */
    static Quadrature tanhSinh(DoubleUnaryOperator f, double min, double max,
                               double absTol, double relTol, int maxRefinements) {
        double halfSpan = 0.5 * (max - min);
        double center   = 0.5 * (min + max);
        double previous = Double.NaN, bestValue = Double.NaN, bestError = Double.POSITIVE_INFINITY;
        int evaluations = 0;

        for (int level = 0; level < maxRefinements; level++) {
            double h = Math.scalb(1.0, -level);
            double estimate = 0.0;
            int smallTail = 0;

            for (int k = 0; k < MAX_TERMS; k++) {
                double t     = k * h;
                double sinh  = Math.sinh(t);
                double cosh  = Math.cosh(t);
                double u     = HALF_PI * sinh;
                double tanhU = Math.tanh(u);
                double sechU = 1.0 / Math.cosh(u);
                double weight = HALF_PI * cosh * sechU * sechU;

                if (k == 0) {
                    double value = f.applyAsDouble(center);
                    evaluations++;
                    if (!Double.isFinite(value)) {
                        return new Quadrature(Double.NaN, Double.NaN, Quadrature.Status.ABNORMAL_TERMINATION, level, evaluations);
                    }
                    estimate += weight * value;
                    continue;
                }

                // Stop when the abscissa complement underflows (indistinguishable from endpoint)
                double xjc = sechU / (Math.exp(u) * Math.cosh(u));
                if (!(xjc > MIN_COMPLEMENT)) break;

                double left  = center - halfSpan * tanhU;
                double right = center + halfSpan * tanhU;
                double lv = f.applyAsDouble(left), rv = f.applyAsDouble(right);
                evaluations += 2;
                if (!Double.isFinite(lv) || !Double.isFinite(rv)) {
                    return new Quadrature(Double.NaN, Double.NaN, Quadrature.Status.ABNORMAL_TERMINATION, level, evaluations);
                }

                double term = weight * (lv + rv);
                estimate += term;

                double tailLimit = Math.max(absTol / (halfSpan * h), relTol * Math.abs(estimate)) * 0.01;
                if (Math.abs(term) <= tailLimit) { if (++smallTail >= TAIL_STREAK) break; }
                else smallTail = 0;
            }

            double value = estimate * halfSpan * h;
            double error = Double.isNaN(previous) ? Math.abs(value) : Math.abs(value - previous);
            if (error < bestError) { bestValue = value; bestError = error; }
            if (!Double.isNaN(previous) && error <= Math.max(absTol, relTol * Math.abs(value))) {
                return new Quadrature(value, error, Quadrature.Status.CONVERGED, level + 1, evaluations);
            }
            previous = value;
        }
        return new Quadrature(bestValue, bestError, Quadrature.Status.MAX_REFINEMENTS_REACHED, maxRefinements, evaluations);
    }
}
