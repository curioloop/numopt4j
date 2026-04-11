/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;

import com.curioloop.numopt4j.quad.Quadrature;
import com.curioloop.numopt4j.quad.adapt.AdaptivePool;
import com.curioloop.numopt4j.quad.adapt.AdaptiveIntegral;

import java.util.function.DoubleUnaryOperator;

final class OscillatoryCore {

    private static final double PI = Math.PI;
    private static final double CYCLE_TOL_SCALE = 0.1;
    private static final double CYCLE_TOL_DECAY = 0.9;
    private static final double RECIPROCAL_EPS = 64.0 * Math.ulp(1.0);

    private OscillatoryCore() {}

    /**
     * Integrates ∫_{a}^{+∞} f(x)·w(ω·x) dx  where w = cos or sin,
     * using the Longman / QUADPACK cycle-by-cycle strategy with ε-algorithm acceleration.
     *
     * <p>Algorithm (QUADPACK dqawfe):</p>
     * <ol>
     *   <li>Partition [a,+∞) into cycles of width c = (2·⌊|ω|⌋+1)·π/|ω|.
     *       Each cycle contains an integer number of half-periods, so the
     *       partial sums form an alternating series.</li>
     *   <li>Integrate each cycle adaptively with a tightening per-cycle tolerance.</li>
     *   <li>Apply the Levin ε-algorithm (Wynn's epsilon) to the sequence of
     *       partial sums to accelerate convergence of the alternating series.</li>
     *   <li>Accept the best candidate (direct or accelerated) once the error
     *       estimate satisfies max(absTol, relTol·|I|).</li>
     * </ol>
     */

    public static Quadrature integrateUpper(DoubleUnaryOperator f, double min,
                                            double omega, boolean sine,
                                            double absTol, double relTol,
                                            int maxCycles, int maxSubdivisions, int maxEvaluations,
                                            AdaptivePool workspace) {
        DoubleUnaryOperator weighted = x -> f.applyAsDouble(x)
                * (sine ? Math.sin(omega * x) : Math.cos(omega * x));

        double[] partialSums = new double[maxCycles];
        double[] epsilonA = new double[maxCycles];
        double[] epsilonB = new double[maxCycles];
        double cycleWidth = cycleWidth(omega);
        double factor = 1.0;
        double totalValue = 0.0;
        double totalError = 0.0;
        double bestValue = Double.NaN;
        double bestError = Double.POSITIVE_INFINITY;
        int totalIterations = 0;
        int totalEvaluations = 0;
        double left = min;

        for (int cycle = 0; cycle < maxCycles; cycle++) {
            if (totalEvaluations >= maxEvaluations) {
                return resultOrDefault(bestValue, bestError, totalValue, totalError,
                        Quadrature.Status.MAX_EVALUATIONS_REACHED, totalIterations, totalEvaluations);
            }

            double right = left + cycleWidth;
            double cycleAbsTol = cycleTolerance(absTol, relTol, totalValue, factor);
            int cycleEvaluations = Math.max(1, maxEvaluations - totalEvaluations);
            Quadrature.Status cycleStatus = AdaptiveIntegral.integrateSegment(weighted, left, right,
                    cycleAbsTol, 0.0, maxSubdivisions, cycleEvaluations, workspace);

            totalIterations  += workspace.resultIterations();
            totalEvaluations += workspace.resultEvaluations();
            double cycleValue = workspace.resultValue();
            totalValue += cycleValue;
            totalError += workspace.resultError();
            partialSums[cycle] = totalValue;

            // Only abort on hard numerical failures (NaN/Inf), not on limit-reached
            // statuses — those are expected when per-cycle tolerance is tight.
            if (cycleStatus == Quadrature.Status.ABNORMAL_TERMINATION
                    || cycleStatus == Quadrature.Status.ROUND_OFF_DETECTED) {
                return new Quadrature(totalValue, totalError,
                        cycleStatus, totalIterations, totalEvaluations);
            }

            Extrapolation extrapolation = extrapolate(partialSums, cycle + 1, epsilonA, epsilonB);
            double directError = totalError + Math.abs(cycleValue);
            double candidateValue = totalValue;
            double candidateError = directError;

            if (extrapolation.available) {
                double acceleratedError = totalError + extrapolation.error;
                if (acceleratedError < candidateError) {
                    candidateValue = extrapolation.value;
                    candidateError = acceleratedError;
                }
            }

            if (candidateError < bestError) {
                bestValue = candidateValue;
                bestError = candidateError;
            }

            if (cycle >= 2 && candidateError <= tolerance(absTol, relTol, candidateValue)) {
                return new Quadrature(candidateValue, candidateError,
                        Quadrature.Status.CONVERGED, totalIterations, totalEvaluations);
            }

            factor *= CYCLE_TOL_DECAY;
            left = right;
        }

        return resultOrDefault(bestValue, bestError, totalValue, totalError,
                Quadrature.Status.MAX_CYCLES_REACHED, totalIterations, totalEvaluations);
    }

    /**
     * Computes the cycle width for oscillatory integration.
     *
     * <p>Formula (QUADPACK dqawfe):
     *   c = (2·⌊|ω|⌋ + 1) · π / |ω|
     * This is the smallest interval containing an integer number of half-periods
     * and at least one full half-period, ensuring the alternating-series
     * cancellation property used by the ε-algorithm extrapolation.</p>
     *
     * <p>Special cases:
     *   |ω| < 1: ⌊|ω|⌋ = 0, so c = π/|ω|  (one half-period)
     *   |ω| ≥ 1: c ≥ 3π/|ω|  (at least one full period)</p>
     */
    private static double cycleWidth(double omega) {
        double absOmega = Math.abs(omega);
        double multiple = 2.0 * Math.floor(absOmega) + 1.0;
        return multiple * PI / absOmega;
    }

    // Note: the formula above is identical to QUADPACK's
    //   dl = 2*l+1  (where l = int(abs(omega)))
    //   cycle = dl*pi/abs(omega)
    // When omega < 1, floor(omega)=0 so multiple=1 and cycle = pi/|omega| (one half-period).
    // When omega >= 1, multiple >= 3 and the cycle spans at least one full period.

    private static double cycleTolerance(double absTol, double relTol, double estimate, double factor) {
        double base = absTol > 0.0
                ? absTol * CYCLE_TOL_SCALE
                : relTol * Math.max(1.0, Math.abs(estimate)) * CYCLE_TOL_SCALE;
        double scaled = base * factor;
        return scaled > 0.0 ? scaled : CYCLE_TOL_SCALE * Math.ulp(1.0);
    }

    private static double tolerance(double absTol, double relTol, double estimate) {
        return Math.max(absTol, relTol * Math.abs(estimate));
    }

    private static Quadrature resultOrDefault(double bestValue, double bestError,
                                              double totalValue, double totalError,
                                              Quadrature.Status status,
                                              int iterations, int evaluations) {
        if (Double.isFinite(bestValue) && Double.isFinite(bestError)) {
            return new Quadrature(bestValue, bestError, status, iterations, evaluations);
        }
        return new Quadrature(totalValue, totalError, status, iterations, evaluations);
    }

    private static Extrapolation extrapolate(double[] partialSums, int count,
                                             double[] rowA, double[] rowB) {
        if (count < 3) {
            return Extrapolation.unavailable();
        }

        // Wynn epsilon-algorithm on the partial-sum sequence.
        //
        // The epsilon table has rows of decreasing length:
        //   row 0 (length=count):   partialSums[0..count-1]          (even order → candidates)
        //   row 1 (length=count-1): reciprocals of consecutive diffs  (odd order)
        //   row k (length=count-k): computed from rows k-2 and k-1
        //
        // We use a rolling two-row scheme with the two pre-allocated arrays rowA / rowB,
        // avoiding any heap allocation inside this method.
        //
        // rowA holds the "previous-previous" row (prePrevious),
        // rowB holds the "previous" row (previous).
        // After each step we swap roles: the old prePrevious buffer is reused for next.

        // Row 0: copy partialSums into rowA
        System.arraycopy(partialSums, 0, rowA, 0, count);

        // Row 1: reciprocals of consecutive differences → rowB
        int len1 = count - 1;
        for (int i = 0; i < len1; i++) {
            rowB[i] = reciprocal(partialSums[i + 1] - partialSums[i]);
        }

        double bestValue = Double.NaN;
        double bestError = Double.POSITIVE_INFINITY;
        double previousEven = Double.NaN;

        // prePrevious = rowA (row 0), previous = rowB (row 1)
        // We alternate which buffer plays which role to avoid allocation.
        double[] prePrevious = rowA;
        double[] previous    = rowB;

        for (int order = 2, length = count - 2; length > 0; order++, length--) {
            // Reuse the prePrevious buffer for the next row (it's no longer needed after this step)
            double[] next = prePrevious;
            for (int i = 0; i < length; i++) {
                double r = reciprocal(previous[i + 1] - previous[i]);
                next[i] = Double.isFinite(r) ? prePrevious[i + 1] + r : Double.NaN;
            }

            if ((order & 1) == 0) {
                double candidate = next[0];
                if (Double.isFinite(candidate)) {
                    double reference = Double.isFinite(previousEven) ? previousEven : partialSums[count - 1];
                    double candidateError = Math.abs(candidate - reference);
                    if (candidateError < bestError) {
                        bestValue = candidate;
                        bestError = candidateError;
                    }
                    previousEven = candidate;
                }
            }

            prePrevious = previous;
            previous    = next;
        }

        if (!Double.isFinite(bestValue)) {
            return Extrapolation.unavailable();
        }
        return new Extrapolation(bestValue, Math.max(bestError, RECIPROCAL_EPS), true);
    }

    private static double reciprocal(double value) {
        if (!Double.isFinite(value) || Math.abs(value) <= RECIPROCAL_EPS) {
            return Double.NaN;
        }
        return 1.0 / value;
    }

    private static final class Extrapolation {
        final double value;
        final double error;
        final boolean available;

        private static final Extrapolation UNAVAILABLE = new Extrapolation(Double.NaN, Double.NaN, false);

        Extrapolation(double value, double error, boolean available) {
            this.value = value;
            this.error = error;
            this.available = available;
        }

        static Extrapolation unavailable() {
            return UNAVAILABLE;
        }
    }
}