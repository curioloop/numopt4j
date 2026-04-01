/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.adapt;

import com.curioloop.numopt4j.quad.Quadrature;

import java.util.function.DoubleUnaryOperator;

/**
 * Adaptive 4-point Gauss-Lobatto quadrature with endpoint reuse.
 *
 * <p>The 4-point Gauss-Lobatto rule on [a, b]:
 *   I ≈ h · [w₁·f(a) + w₂·f(a + h·(1−1/√5)) + w₂·f(a + h·(1+1/√5)) + w₁·f(b)]
 * where h = (b−a)/2, w₁ = 1/6, w₂ = 5/6.
 * Exact for polynomials of degree ≤ 5.</p>
 *
 * <p>Endpoint reuse: when an interval [a, b] is bisected into [a, m] and [m, b],
 * the values f(a), f(m), f(b) are already known from the parent evaluation.
 * Each bisection therefore requires only 2 new function evaluations (the two
 * interior Lobatto nodes of each child), compared to 4 for a fresh 4-point rule.</p>
 *
 * <p>Error estimate: |I_fine − I_coarse| where I_fine = I_left + I_right
 * and I_coarse is the parent interval estimate.</p>
 *
 * <p>Convergence criterion: totalError ≤ max(absTol, relTol·|totalEstimate|)</p>
 */
final class GaussLobattoCore {

    /** 1/√5 — relative position of the two interior Lobatto nodes */
    private static final double INV_SQRT5 = 1.0 / Math.sqrt(5.0);

    /** Endpoint weight w₁ = 1/6 */
    private static final double W1 = 1.0 / 6.0;

    /** Interior weight w₂ = 5/6 */
    private static final double W2 = 5.0 / 6.0;

    private GaussLobattoCore() {}

    static Quadrature integrate(DoubleUnaryOperator f, double min, double max,
                                double absTol, double relTol,
                                int maxSubdivisions, int maxEvaluations,
                                AdaptivePool workspace) {
        AdaptivePool pool = workspace.ensure(maxSubdivisions);

        // Evaluate the 4-point rule on the initial interval
        double fa = f.applyAsDouble(min);
        double fb = f.applyAsDouble(max);
        if (!Double.isFinite(fa) || !Double.isFinite(fb)) {
            return new Quadrature(Double.NaN, Double.NaN,
                    Quadrature.Status.ABNORMAL_TERMINATION, 0, 2);
        }

        int[] evalCount = {2};
        double coarse = lobatto4(f, min, max, fa, fb, evalCount);
        if (!Double.isFinite(coarse)) {
            return new Quadrature(Double.NaN, Double.NaN,
                    Quadrature.Status.ABNORMAL_TERMINATION, 0, evalCount[0]);
        }

        int[] subdivCount = {0};
        double result = refine(f, min, max, fa, fb, coarse,
                absTol, relTol, maxSubdivisions, maxEvaluations,
                evalCount, subdivCount, pool);

        Quadrature.Status status = Double.isNaN(result)
                ? Quadrature.Status.ABNORMAL_TERMINATION
                : (evalCount[0] + 4 > maxEvaluations ? Quadrature.Status.MAX_EVALUATIONS_REACHED
                : (subdivCount[0] >= maxSubdivisions   ? Quadrature.Status.MAX_SUBDIVISIONS_REACHED
                : Quadrature.Status.CONVERGED));

        return new Quadrature(Double.isNaN(result) ? Double.NaN : result,
                Double.isNaN(result) ? Double.NaN : 0.0,
                status, subdivCount[0], evalCount[0]);
    }

    /**
     * Evaluates the 4-point Gauss-Lobatto rule on [a, b], reusing the known
     * endpoint values fa = f(a) and fb = f(b).
     *
     * <p>Only the two interior nodes are evaluated (2 new function calls).
     * Rule: h·[w₁·fa + w₂·f(a+h·(1−1/√5)) + w₂·f(a+h·(1+1/√5)) + w₁·fb]
     * where h = (b−a)/2.</p>
     */
    private static double lobatto4(DoubleUnaryOperator f, double a, double b,
                                   double fa, double fb, int[] evals) {
        double h  = 0.5 * (b - a);
        double n2 = a + h * (1.0 - INV_SQRT5);
        double n3 = a + h * (1.0 + INV_SQRT5);
        double fn2 = f.applyAsDouble(n2);
        double fn3 = f.applyAsDouble(n3);
        evals[0] += 2;
        if (!Double.isFinite(fn2) || !Double.isFinite(fn3)) return Double.NaN;
        return h * (W1 * fa + W2 * fn2 + W2 * fn3 + W1 * fb);
    }

    /**
     * Recursively refines [a, b] by bisection until the error estimate satisfies
     * the tolerance or a resource limit is reached.
     *
     * <p>The midpoint m = (a+b)/2 is evaluated once and reused as the shared
     * endpoint of both child intervals [a, m] and [m, b].</p>
     */
    private static double refine(DoubleUnaryOperator f, double a, double b,
                                 double fa, double fb, double prevEst,
                                 double absTol, double relTol,
                                 int maxSubdivisions, int maxEvaluations,
                                 int[] evals, int[] subdivs, AdaptivePool pool) {
        if (evals[0] >= maxEvaluations || subdivs[0] >= maxSubdivisions) {
            return prevEst;
        }

        double m  = 0.5 * (a + b);
        double fm = f.applyAsDouble(m);
        evals[0]++;
        if (!Double.isFinite(fm)) return Double.NaN;

        int[] e = {0};
        double left  = lobatto4(f, a, m, fa, fm, e);
        double right = lobatto4(f, m, b, fm, fb, e);
        evals[0] += e[0];
        subdivs[0]++;

        if (!Double.isFinite(left) || !Double.isFinite(right)) return Double.NaN;

        double combined = left + right;
        double error    = Math.abs(combined - prevEst);
        double tol      = Math.max(absTol, relTol * Math.abs(combined));

        if (error <= tol) return combined;

        // Bisect both halves
        double l = refine(f, a, m, fa, fm, left,  absTol * 0.5, relTol,
                maxSubdivisions, maxEvaluations, evals, subdivs, pool);
        double r = refine(f, m, b, fm, fb, right, absTol * 0.5, relTol,
                maxSubdivisions, maxEvaluations, evals, subdivs, pool);

        if (!Double.isFinite(l) || !Double.isFinite(r)) return Double.NaN;
        return l + r;
    }
}
