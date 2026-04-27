/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;

import com.curioloop.numopt4j.quad.Quadrature;
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
                                            OscillatoryPool workspace) {
        workspace.ensureSeries(maxCycles);
        DoubleUnaryOperator weighted = workspace.weightedIntegrand().configure(f, omega, sine);
        double[] scratch = workspace.seriesArena();
        int partialSumsOffset = 0;
        int epsilonAOffset = maxCycles;
        int epsilonBOffset = maxCycles * 2;
        int extrapolationOffset = maxCycles * 3;
        double cycleWidth = cycleWidth(omega);
        double factor = 1.0;
        double totalValue = 0.0;
        double totalCompensation = 0.0;
        double totalError = 0.0;
        double totalVariation = 0.0;
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
            double cycleAbsTol = cycleTolerance(absTol, relTol, totalValue, totalVariation, factor);
            int cycleEvaluations = Math.max(1, maxEvaluations - totalEvaluations);
            Quadrature.Status cycleStatus = AdaptiveIntegral.integrateSegment(weighted, left, right,
                    cycleAbsTol, 0.0, maxSubdivisions, cycleEvaluations, workspace);

            totalIterations  += workspace.resultIterations();
            totalEvaluations += workspace.resultEvaluations();
            double cycleValue = workspace.resultValue();
            double adjusted = cycleValue - totalCompensation;
            double updated = totalValue + adjusted;
            totalCompensation = (updated - totalValue) - adjusted;
            totalValue = updated;
            totalError += workspace.resultError();
            totalVariation += Math.abs(cycleValue);
            scratch[partialSumsOffset + cycle] = totalValue;

            // Only abort on hard numerical failures (NaN/Inf), not on limit-reached
            // statuses — those are expected when per-cycle tolerance is tight.
            if (cycleStatus == Quadrature.Status.ABNORMAL_TERMINATION
                    || cycleStatus == Quadrature.Status.ROUND_OFF_DETECTED) {
                return new Quadrature(totalValue, totalError,
                        cycleStatus, totalIterations, totalEvaluations);
            }

            double directError = totalError + Math.abs(cycleValue);
            double candidateValue = totalValue;
            double candidateError = directError;

            if (extrapolate(scratch, partialSumsOffset, cycle + 1,
                    epsilonAOffset, epsilonBOffset, extrapolationOffset)) {
                double acceleratedError = totalError + scratch[extrapolationOffset + 1];
                if (acceleratedError < candidateError) {
                    candidateValue = scratch[extrapolationOffset];
                    candidateError = acceleratedError;
                }
            }

            if (candidateError < bestError) {
                bestValue = candidateValue;
                bestError = candidateError;
            }

            if (cycle >= 2 && candidateError <= tolerance(absTol, relTol, candidateValue, totalVariation)) {
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

    private static double cycleTolerance(double absTol, double relTol,
                                         double estimate, double l1Norm, double factor) {
        double base = Math.max(absTol,
                relTol * Math.max(1.0, Math.max(Math.abs(estimate), l1Norm))) * CYCLE_TOL_SCALE;
        double scaled = base * factor;
        return scaled > 0.0 ? scaled : CYCLE_TOL_SCALE * Math.ulp(1.0);
    }

    private static double tolerance(double absTol, double relTol,
                                    double estimate, double l1Norm) {
        return Math.max(absTol, relTol * Math.max(Math.abs(estimate), l1Norm));
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

    private static boolean extrapolate(double[] scratch, int partialSumsOffset, int count,
                                       int rowAOffset, int rowBOffset,
                                       int outOffset) {
        if (count < 3) {
            return false;
        }

        // Wynn epsilon-algorithm on the partial-sum sequence.
        //
        // The epsilon table has rows of decreasing length:
        //   row 0 (length=count):   partialSums[0..count-1]          (even order → candidates)
        //   row 1 (length=count-1): reciprocals of consecutive diffs  (odd order)
        //   row k (length=count-k): computed from rows k-2 and k-1
        //
        // We use a rolling two-row scheme with two regions inside the pre-allocated arena,
        // avoiding any heap allocation inside this method.
        //
        // rowAOffset holds the "previous-previous" row (prePrevious),
        // rowBOffset holds the "previous" row (previous).
        // After each step we swap roles: the old prePrevious buffer is reused for next.

        // Row 0: copy partial sums into row A.
        System.arraycopy(scratch, partialSumsOffset, scratch, rowAOffset, count);

        // Row 1: reciprocals of consecutive differences → row B.
        int len1 = count - 1;
        for (int i = 0; i < len1; i++) {
            double current = scratch[partialSumsOffset + i];
            double next = scratch[partialSumsOffset + i + 1];
            double scale = Math.max(Math.abs(next), Math.abs(current));
            scratch[rowBOffset + i] = reciprocal(next - current, scale);
        }

        double bestValue = Double.NaN;
        double bestError = Double.POSITIVE_INFINITY;
        double previousEven = Double.NaN;

        // prePrevious = row A (row 0), previous = row B (row 1)
        // We alternate which region plays which role to avoid allocation.
        int prePreviousOffset = rowAOffset;
        int previousOffset = rowBOffset;

        for (int order = 2, length = count - 2; length > 0; order++, length--) {
            // Reuse the prePrevious region for the next row (it's no longer needed after this step).
            int nextOffset = prePreviousOffset;
            for (int i = 0; i < length; i++) {
                double previous = scratch[previousOffset + i];
                double nextPrevious = scratch[previousOffset + i + 1];
                double scale = Math.max(Math.abs(nextPrevious), Math.abs(previous));
                double r = reciprocal(nextPrevious - previous, scale);
                scratch[nextOffset + i] = Double.isFinite(r)
                        ? scratch[prePreviousOffset + i + 1] + r
                        : Double.NaN;
            }

            if ((order & 1) == 0) {
                double candidate = scratch[nextOffset];
                if (Double.isFinite(candidate)) {
                    double reference = Double.isFinite(previousEven)
                            ? previousEven
                            : scratch[partialSumsOffset + count - 1];
                    double candidateError = Math.abs(candidate - reference);
                    if (candidateError < bestError) {
                        bestValue = candidate;
                        bestError = candidateError;
                    }
                    previousEven = candidate;
                }
            }

            prePreviousOffset = previousOffset;
            previousOffset = nextOffset;
        }

        if (!Double.isFinite(bestValue)) {
            return false;
        }
        double errorFloor = RECIPROCAL_EPS * Math.max(1.0, Math.abs(bestValue));
        scratch[outOffset] = bestValue;
        scratch[outOffset + 1] = Math.max(bestError, errorFloor);
        return true;
    }

    private static double reciprocal(double value, double scale) {
        if (!Double.isFinite(value)) {
            return Double.NaN;
        }
        double threshold = RECIPROCAL_EPS * Math.max(1.0, scale);
        if (Math.abs(value) <= threshold) {
            return Double.NaN;
        }
        return 1.0 / value;
    }

}