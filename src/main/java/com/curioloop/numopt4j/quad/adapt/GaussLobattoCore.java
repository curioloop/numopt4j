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
 * Each split therefore needs 5 new function evaluations in total: the shared midpoint
 * plus the two interior Lobatto nodes of each child, instead of 8 evaluations for two
 * fresh child rules.</p>
 *
 * <p>Error estimate: |I_fine − I_coarse| where I_fine = I_left + I_right
 * and I_coarse is the parent interval estimate.</p>
 *
 * <p>Convergence criterion: totalError ≤ max(absTol, relTol·|totalEstimate|)</p>
 *
 * <p>absTol scaling: before entering recursion the effective absolute tolerance is
 * scaled by the initial full-interval estimate, matching QuantLib's behaviour:
 *   effectiveAbsTol = max(absTol, relTol · |initialEstimate|)
 * The same absolute budget is then propagated through recursion instead of being
 * halved at every level; otherwise deeply nested sub-intervals collapse toward
 * machine epsilon and spuriously exhaust the subdivision budget.</p>
 */
final class GaussLobattoCore {

    /** 1/√5 — relative position of the two interior Lobatto nodes */
    private static final double INV_SQRT5 = 1.0 / Math.sqrt(5.0);

    /** Endpoint weight w₁ = 1/6 */
    private static final double W1 = 1.0 / 6.0;

    /** Interior weight w₂ = 5/6 */
    private static final double W2 = 5.0 / 6.0;

    private GaussLobattoCore() {}

    static Quadrature.Status integrate(DoubleUnaryOperator f, double min, double max,
                                       double absTol, double relTol,
                                       int maxSubdivisions, int maxEvaluations,
                                       AdaptivePool workspace) {
        AdaptivePool pool = workspace.ensure(maxSubdivisions);

        // Evaluate the 4-point rule on the initial interval
        double fa = f.applyAsDouble(min);
        double fb = f.applyAsDouble(max);
        if (!Double.isFinite(fa) || !Double.isFinite(fb)) {
            pool.resultValue = Double.NaN; pool.resultError = Double.NaN;
            pool.resultIterations = 0;     pool.resultEvaluations = 2;
            return Quadrature.Status.ABNORMAL_TERMINATION;
        }

        pool.resultEvaluations = 2;
        pool.resultIterations = 0;
        pool.resultError = 0.0;
        pool.lobattoRoundOffDetected = false;

        double coarse = lobatto4(f, min, max, fa, fb, pool);
        if (!Double.isFinite(coarse)) {
            pool.resultValue = Double.NaN; pool.resultError = Double.NaN;
            pool.resultIterations = 0;
            return Quadrature.Status.ABNORMAL_TERMINATION;
        }

        double effectiveAbsTol = Math.max(absTol, relTol * Math.abs(coarse));

        double result = refine(f, min, max, fa, fb, coarse,
                effectiveAbsTol, relTol, maxSubdivisions, maxEvaluations, pool);

        Quadrature.Status status = Double.isNaN(result)
                ? Quadrature.Status.ABNORMAL_TERMINATION
            : (pool.resultEvaluations >= maxEvaluations ? Quadrature.Status.MAX_EVALUATIONS_REACHED
                : (pool.resultIterations >= maxSubdivisions   ? Quadrature.Status.MAX_SUBDIVISIONS_REACHED
                : (pool.lobattoRoundOffDetected ? Quadrature.Status.ROUND_OFF_DETECTED
                : Quadrature.Status.CONVERGED)));

        pool.resultValue = Double.isNaN(result) ? Double.NaN : result;
        pool.resultError = Double.isNaN(result) ? Double.NaN : pool.resultError;
        return status;
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
                                   double fa, double fb, AdaptivePool pool) {
        double h  = 0.5 * (b - a);
        double n2 = a + h * (1.0 - INV_SQRT5);
        double n3 = a + h * (1.0 + INV_SQRT5);
        double fn2 = f.applyAsDouble(n2);
        double fn3 = f.applyAsDouble(n3);
        pool.resultEvaluations += 2;
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
                                 AdaptivePool pool) {
        if (pool.resultEvaluations >= maxEvaluations || pool.resultIterations >= maxSubdivisions) {
            pool.resultError += Math.max(absTol, Math.ulp(1.0) * Math.max(1.0, Math.abs(prevEst)));
            return prevEst;
        }

        double m  = 0.5 * (a + b);
        if (!(m > a && m < b)) {
            pool.lobattoRoundOffDetected = true;
            pool.resultError += Math.max(absTol, Math.ulp(1.0) * Math.max(1.0, Math.abs(prevEst)));
            return prevEst;
        }
        double fm = f.applyAsDouble(m);
        pool.resultEvaluations++;
        if (!Double.isFinite(fm)) return Double.NaN;

        double left  = lobatto4(f, a, m, fa, fm, pool);
        double right = lobatto4(f, m, b, fm, fb, pool);
        pool.resultIterations++;

        if (!Double.isFinite(left) || !Double.isFinite(right)) return Double.NaN;

        double combined = left + right;
        double error    = Math.abs(combined - prevEst);
        double tol      = Math.max(absTol, relTol * Math.abs(combined));

        if (error <= tol) {
            pool.resultError += error;
            return combined;
        }

        // Protective check (QuantLib): if the sub-interval contribution is already
        // below numerical precision, further bisection cannot improve accuracy.
        if (Math.abs(combined) == 0.0 || absTol >= Math.abs(combined)) {
            pool.lobattoRoundOffDetected = true;
            pool.resultError += Math.max(error, Math.ulp(1.0) * Math.max(1.0, Math.abs(combined)));
            return combined;
        }

        // Bisect both halves
        double l = refine(f, a, m, fa, fm, left,  absTol, relTol,
            maxSubdivisions, maxEvaluations, pool);
        double r = refine(f, m, b, fm, fb, right, absTol, relTol,
            maxSubdivisions, maxEvaluations, pool);

        if (!Double.isFinite(l) || !Double.isFinite(r)) return Double.NaN;
        return l + r;
    }
}
