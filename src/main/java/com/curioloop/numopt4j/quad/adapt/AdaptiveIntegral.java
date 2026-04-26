/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.adapt;
import java.util.Objects;

import com.curioloop.numopt4j.quad.Checks;
import com.curioloop.numopt4j.quad.Integral;
import com.curioloop.numopt4j.quad.Quadrature;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

/**
 * Builder for adaptive Gauss-Kronrod (GK15) quadrature on a finite interval.
 *
 * <p>Subdivides [min, max] adaptively, always bisecting the sub-interval with the
 * largest local error estimate, until the global error satisfies
 * {@code max(absTol, relTol·|I|)} or a resource limit is reached.</p>
 *
 * <p>Optional breakpoints split the domain into segments before adaptation begins,
 * which is useful for integrands with known discontinuities.</p>
 *
 * <p>Minimum required setters: {@code .function()}, {@code .bounds()}, {@code .tolerances()}.</p>
 */
public class AdaptiveIntegral implements Integral<Quadrature, AdaptivePool> {

    private DoubleUnaryOperator function;
    private double min = Double.NaN;
    private double max = Double.NaN;
    private double absTol = 1e-10;
    private double relTol = 1e-10;
    private int maxSubdivisions = Checks.DEFAULT_MAX_SUBDIVISIONS;
    private int maxEvaluations = Checks.DEFAULT_MAX_EVALUATIONS;
    private double[] breakpoints;
    private AdaptiveRule rule = AdaptiveRule.GK15;
    private transient AdaptivePool workspace;

    public AdaptiveIntegral() {}

    public AdaptiveIntegral function(DoubleUnaryOperator function) {
        this.function = function;
        return this;
    }

    /** Sets the integration interval. Both bounds must be finite with {@code min < max}. */
    public AdaptiveIntegral bounds(double min, double max) {
        this.min = min;
        this.max = max;
        return this;
    }

    /**
     * Sets absolute and relative tolerances.
     * Convergence is declared when {@code totalError ≤ max(absTol, relTol·|I|)}.
     * At least one must be positive.
     */
    public AdaptiveIntegral tolerances(double absTol, double relTol) {
        if (absTol < 0.0) throw new IllegalArgumentException("absTol must be >= 0");
        if (relTol < 0.0) throw new IllegalArgumentException("relTol must be >= 0");
        this.absTol = absTol;
        this.relTol = relTol;
        return this;
    }

    /** Sets the maximum number of adaptive sub-intervals (default {@value Checks#DEFAULT_MAX_SUBDIVISIONS}). */
    public AdaptiveIntegral maxSubdivisions(int maxSubdivisions) {
        if (maxSubdivisions <= 0) throw new IllegalArgumentException("maxSubdivisions must be > 0");
        this.maxSubdivisions = maxSubdivisions;
        return this;
    }

    /** Sets the maximum total function evaluations (default {@value Checks#DEFAULT_MAX_EVALUATIONS}). */
    public AdaptiveIntegral maxEvaluations(int maxEvaluations) {
        if (maxEvaluations <= 0) throw new IllegalArgumentException("maxEvaluations must be > 0");
        this.maxEvaluations = maxEvaluations;
        return this;
    }

    /**
     * Sets interior breakpoints that split the domain before adaptation.
     * Each point must be strictly inside {@code (min, max)}.
     * Useful for integrands with known discontinuities or sharp features.
     */
    public AdaptiveIntegral breakpoints(double... breakpoints) {
        this.breakpoints = breakpoints == null ? null : Arrays.copyOf(breakpoints, breakpoints.length);
        return this;
    }

    /**
     * Sets the underlying quadrature rule (default {@link AdaptiveRule#GK15}).
     *
     * <p>Use {@link AdaptiveRule#GAUSS_LOBATTO} when function evaluations are expensive:
     * the 4-point Gauss-Lobatto rule reuses endpoint values across bisections,
     * requiring only 2 new evaluations per subdivision instead of 15.</p>
     */
    public AdaptiveIntegral rule(AdaptiveRule rule) {
        Objects.requireNonNull(rule, "rule must not be null");
        this.rule = rule;
        return this;
    }

    @Override
    public Quadrature integrate(AdaptivePool workspace) {
        Checks.validateFunction(function);
        Checks.validateFiniteInterval(min, max);
        Checks.validateTolerances(absTol, relTol);
        Checks.validateAdaptiveLimits(maxSubdivisions, maxEvaluations);
        if (workspace == null) {
            if (this.workspace == null) this.workspace = new AdaptivePool();
            workspace = this.workspace;
        }
        AdaptivePool pool = workspace.ensure(maxSubdivisions);
        if (rule == AdaptiveRule.GAUSS_LOBATTO) {
            Quadrature.Status status = GaussLobattoCore.integrate(function, min, max,
                    absTol, relTol, maxSubdivisions, maxEvaluations, pool);
            return new Quadrature(pool.resultValue, pool.resultError, status,
                    pool.resultIterations, pool.resultEvaluations);
        }
        return integrate(function, min, max, breakpoints, absTol, relTol, maxSubdivisions, maxEvaluations, pool);
    }

    /**
     * Executes adaptive GK15 quadrature, splitting at any provided breakpoints.
     *
    * <p>When breakpoints are present the absolute tolerance is consumed as a running
    * global budget across segments, while the relative tolerance remains unchanged.
    * Integration stops early and returns the partial result if any segment fails.</p>
     *
     * @param f              integrand
     * @param min            lower bound
     * @param max            upper bound
     * @param points         interior breakpoints (may be null or empty)
     * @param absTol         absolute tolerance
     * @param relTol         relative tolerance
     * @param maxSubdivisions maximum sub-intervals per segment
     * @param maxEvaluations  maximum function evaluations across all segments
     * @param pool           reusable workspace
     */
    public static Quadrature integrate(DoubleUnaryOperator f, double min, double max,
                                       double[] points, double absTol, double relTol,
                                       int maxSubdivisions, int maxEvaluations,
                                       AdaptivePool pool) {
        double[] internalPoints = Checks.validateBreakpoints(points, min, max);
        if (internalPoints.length == 0) {
            Quadrature.Status status = GK15Core.integrate(f, min, max, absTol, relTol, maxSubdivisions, maxEvaluations, pool);
            return new Quadrature(pool.resultValue, pool.resultError, status,
                    pool.resultIterations, pool.resultEvaluations);
        }

        double totalValue = 0.0, totalError = 0.0;
        int totalIterations = 0, totalEvaluations = 0;
        int remainingEvaluations = maxEvaluations;
        int segmentCount = internalPoints.length + 1;
        double remainingAbsTol = absTol;
        double left = min;

        for (int i = 0; i < segmentCount; i++) {
            double right = i == internalPoints.length ? max : internalPoints[i];
            int segmentsLeft = segmentCount - i;
            double segmentAbsTol = absTol > 0.0 ? remainingAbsTol / segmentsLeft : 0.0;
            Quadrature.Status status = GK15Core.integrate(
                    f, left, right, segmentAbsTol, relTol,
                    Math.max(1, maxSubdivisions), Math.max(1, remainingEvaluations), pool);

            totalValue      += pool.resultValue;
            totalError      += pool.resultError;
            totalIterations += pool.resultIterations;
            totalEvaluations += pool.resultEvaluations;
            remainingEvaluations  -= pool.resultEvaluations;
            if (absTol > 0.0) {
                remainingAbsTol = Math.max(0.0, remainingAbsTol - Math.max(0.0, pool.resultError));
            }

            if (!status.isConverged()) {
                return new Quadrature(totalValue, totalError, status, totalIterations, totalEvaluations);
            }
            left = right;
        }
        Quadrature.Status status = totalError <= tolerance(absTol, relTol, totalValue)
                ? Quadrature.Status.CONVERGED
                : Quadrature.Status.ROUND_OFF_DETECTED;
        return new Quadrature(totalValue, totalError, status, totalIterations, totalEvaluations);
    }

    /**
     * Integrates a single segment [min, max] via GK15, writing results into pool.
     * Returns the status; callers read value/error/iterations/evaluations from pool.
     */
    public static Quadrature.Status integrateSegment(DoubleUnaryOperator f, double min, double max,
                                                     double absTol, double relTol,
                                                     int maxSubdivisions, int maxEvaluations,
                                                     AdaptivePool pool) {
        return GK15Core.integrate(f, min, max, absTol, relTol, maxSubdivisions, maxEvaluations, pool);
    }

    private static double tolerance(double absTol, double relTol, double estimate) {
        return Math.max(absTol, relTol * Math.abs(estimate));
    }
}
