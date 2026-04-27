/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;

import com.curioloop.numopt4j.quad.de.DEPool;
import com.curioloop.numopt4j.quad.de.DoubleExponentialCore;
import com.curioloop.numopt4j.quad.Quadrature;

import com.curioloop.numopt4j.quad.gauss.GaussPool;
import com.curioloop.numopt4j.quad.gauss.GaussRule;

import java.util.function.DoubleBinaryOperator;
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
 *   <li>{@link #logarithmic} — Boost-style double-exponential quadrature for integrands
 *       that also carry logarithmic endpoint factors.
 *       A power preconditioning map removes the algebraic endpoint weights before
 *       the DE rule is applied, which stabilizes the near-{@code -1} exponent regime.</li>
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
final class EndpointSingularCore {

    private static final int DE_TOLERANCE_RETRIES = 6;

    private EndpointSingularCore() {}

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
                                int maxRefinements, GaussRule rule, GaussPool workspace) {
        // Under the affine map x = center + halfSpan*t  (t ∈ [-1,1]):
        //   x - min  = halfSpan*(1+t)   →  contributes (1+t)^alpha
        //   max - x  = halfSpan*(1-t)   →  contributes (1-t)^beta
        // JacobiRule(a,b) generates weights for ∫(1-t)^a (1+t)^b g(t) dt,
        // so JacobiRule(beta, alpha) matches (1-t)^beta (1+t)^alpha.
        // The factor halfSpan^(alpha+beta+1) is absorbed into `scale`.
        double halfSpan = 0.5 * (max - min);
        double center   = 0.5 * (min + max);
        double exponent = alpha + beta + 1.0;
        double scale    = powScale(halfSpan, exponent);
        if (!Double.isFinite(scale)) {
            return new Quadrature(Double.NaN, Double.NaN,
                    Quadrature.Status.ABNORMAL_TERMINATION, 0, 0);
        }
        double previous = Double.NaN, bestValue = Double.NaN, bestError = Double.POSITIVE_INFINITY;
        int totalEvaluations = 0;

        for (int level = 0; level < maxRefinements; level++) {
            int points = INITIAL_POINTS << level;
            workspace.ensure(points, rule);

            double value = 0.0;
            double compensation = 0.0;
            for (int i = 0; i < points; i++) {
                double term = workspace.weightAt(i) * f.applyAsDouble(center + halfSpan * workspace.nodeAt(i));
                double adjusted = term - compensation;
                double updated = value + adjusted;
                compensation = (updated - value) - adjusted;
                value = updated;
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
    // Double-exponential quadrature with endpoint power preconditioning
    // -----------------------------------------------------------------------

    static Quadrature logarithmic(DoubleUnaryOperator function, double min, double max,
                                  double alpha, double beta, EndpointOpts opts,
                                  double absTol, double relTol, int maxRefinements,
                                  DEPool workspace) {
        double interval = max - min;
        double leftExponent = alpha + 1.0;
        double rightExponent = beta + 1.0;
        double inverseLeftExponent = 1.0 / leftExponent;
        double inverseRightExponent = 1.0 / rightExponent;
        double logInterval = Math.log(interval);
        double intervalWeightLog = (alpha + beta + 1.0) * logInterval;
        double denominatorExponent = alpha + beta + 2.0;

        DoubleBinaryOperator transformed = (unitPosition, unitComplement) -> {
            double complement = Math.abs(unitComplement);
            double leftUnit;
            double rightUnit;
            if (unitPosition <= 0.5) {
                leftUnit = complement;
                rightUnit = 1.0 - complement;
            } else {
                rightUnit = complement;
                leftUnit = 1.0 - complement;
            }

            double logLeftUnit = Math.log(leftUnit);
            double logRightUnit = Math.log(rightUnit);
            double logA = inverseLeftExponent * logLeftUnit;
            double logB = inverseRightExponent * logRightUnit;
            double maxLog = Math.max(logA, logB);
            double scaledA = Math.exp(logA - maxLog);
            double scaledB = Math.exp(logB - maxLog);
            double scaledSum = scaledA + scaledB;
            double logSum = maxLog + Math.log(scaledSum);
            double leftFraction = scaledA / scaledSum;
            double rightFraction = scaledB / scaledSum;
            double point = leftFraction <= rightFraction
                ? min + interval * leftFraction
                : max - interval * rightFraction;
            double blend = rightUnit * inverseLeftExponent + leftUnit * inverseRightExponent;
            double value = function.applyAsDouble(point) * Math.exp(
                intervalWeightLog + Math.log(blend) - denominatorExponent * logSum
            );
            if (opts.logLeft) {
                value *= logInterval + logA - logSum;
            }
            if (opts.logRight) {
                value *= logInterval + logB - logSum;
            }
            return value;
        };

        Quadrature best = null;
        double tolerance = initialDeTolerance(absTol, relTol);
        int totalEvaluations = 0;
        for (int attempt = 0; attempt < DE_TOLERANCE_RETRIES; attempt++) {
            int[] attemptEvaluations = {0};
            DoubleBinaryOperator counted = (unitPosition, unitComplement) -> {
                attemptEvaluations[0]++;
                return transformed.applyAsDouble(unitPosition, unitComplement);
            };
            Quadrature current;
            try {
                current = DoubleExponentialCore.tanhSinh(
                    counted,
                    0.0,
                    1.0,
                    tolerance,
                    maxRefinements,
                    DoubleExponentialCore.DEFAULT_MIN_COMPLEMENT,
                    workspace
                );
            } catch (ArithmeticException ex) {
                totalEvaluations += attemptEvaluations[0];
                break;
            }
            totalEvaluations += attemptEvaluations[0];
            if (best == null || current.getEstimatedError() <= best.getEstimatedError()) {
                best = current;
            }
            if (meetsTolerance(current, absTol, relTol)) {
                return new Quadrature(
                    current.getValue(),
                    current.getEstimatedError(),
                    Quadrature.Status.CONVERGED,
                    current.getIterations(),
                    totalEvaluations
                );
            }

            double tightened = tolerance * 0.25;
            if (!(tightened > 0.0) || tightened == tolerance) {
                break;
            }
            tolerance = tightened;
        }

        if (best == null) {
            return new Quadrature(Double.NaN, Double.NaN, Quadrature.Status.ABNORMAL_TERMINATION, 0, totalEvaluations);
        }
        Quadrature.Status status = meetsTolerance(best, absTol, relTol)
            ? Quadrature.Status.CONVERGED
            : Quadrature.Status.MAX_REFINEMENTS_REACHED;
        return new Quadrature(best.getValue(), best.getEstimatedError(), status, best.getIterations(), totalEvaluations);
    }

    private static double powScale(double base, double exponent) {
        double logScale = exponent * Math.log(base);
        if (Math.abs(logScale) < 1e-4) {
            return 1.0 + Math.expm1(logScale);
        }
        return Math.exp(logScale);
    }

    private static double initialDeTolerance(double absTol, double relTol) {
        if (relTol > 0.0) {
            return relTol;
        }
        if (absTol > 0.0) {
            return Math.min(absTol, DoubleExponentialCore.DEFAULT_TOLERANCE);
        }
        return DoubleExponentialCore.DEFAULT_TOLERANCE;
    }

    private static boolean meetsTolerance(Quadrature result, double absTol, double relTol) {
        return result.getEstimatedError() <= Math.max(absTol, relTol * Math.abs(result.getValue()));
    }
}
