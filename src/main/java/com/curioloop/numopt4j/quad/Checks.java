/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad;
import java.util.Objects;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

/**
 * Shared validation utilities and default limit constants for quadrature builders.
 *
 * <p>All methods throw {@link IllegalArgumentException} (or {@link IllegalStateException}
 * where appropriate) with a descriptive message when a constraint is violated.</p>
 */
public final class Checks {

    /** ULP guard band used to reject principal-value poles that are too close to an endpoint. */
    private static final double PRINCIPAL_VALUE_BOUNDARY_GAP_ULPS = 128.0;

    /** Default maximum number of oscillation cycles for {@link special.OscillatoryIntegral}. */
    public static final int DEFAULT_MAX_CYCLES = 64;

    /** Default maximum number of refinement levels for {@link special.EndpointSingularIntegral}. */
    public static final int DEFAULT_MAX_REFINEMENTS = 12;

    /** Default maximum number of adaptive subdivisions for {@link adapt.AdaptiveIntegral}. */
    public static final int DEFAULT_MAX_SUBDIVISIONS = 256;

    /** Default maximum number of function evaluations for adaptive quadrature. */
    public static final int DEFAULT_MAX_EVALUATIONS = 16384;

    private Checks() {}

    /** Throws if {@code f} is null. */
    public static void validateFunction(DoubleUnaryOperator f) {
        Objects.requireNonNull(f, "function must not be null");
    }

    /** Throws if {@code points} is not positive. */
    public static void validatePoints(int points) {
        if (points <= 0) {
            throw new IllegalArgumentException("points must be > 0");
        }
    }

    /** Throws if {@code [min, max]} is not a valid finite interval with {@code min < max}. */
    public static void validateFiniteInterval(double min, double max) {
        if (!Double.isFinite(min) || !Double.isFinite(max)) {
            throw new IllegalArgumentException("fixed quadrature requires finite bounds");
        }
        if (!(min < max)) {
            throw new IllegalArgumentException("min must be < max");
        }
    }

    /** Throws if either tolerance is negative, or both are zero. */
    public static void validateTolerances(double absTol, double relTol) {
        if (absTol < 0.0) {
            throw new IllegalArgumentException("absTol must be >= 0");
        }
        if (relTol < 0.0) {
            throw new IllegalArgumentException("relTol must be >= 0");
        }
        if (absTol == 0.0 && relTol == 0.0) {
            throw new IllegalArgumentException("absTol and relTol must not both be zero");
        }
    }

    /** Throws if either adaptive limit is not positive. */
    public static void validateAdaptiveLimits(int maxSubdivisions, int maxEvaluations) {
        if (maxSubdivisions <= 0) {
            throw new IllegalArgumentException("maxSubdivisions must be > 0");
        }
        if (maxEvaluations <= 0) {
            throw new IllegalArgumentException("maxEvaluations must be > 0");
        }
    }

    /** Throws if {@code omega} is not finite. */
    public static void validateFrequency(double omega) {
        if (!Double.isFinite(omega)) {
            throw new IllegalArgumentException("omega must be finite");
        }
    }

    /** Throws if either exponent is not finite or not greater than −1. */
    public static void validateEndpointExponents(double alpha, double beta) {
        if (!Double.isFinite(alpha) || !Double.isFinite(beta)) {
            throw new IllegalArgumentException("alpha and beta must be finite");
        }
        if (!(alpha > -1.0)) {
            throw new IllegalArgumentException("alpha must be > -1");
        }
        if (!(beta > -1.0)) {
            throw new IllegalArgumentException("beta must be > -1");
        }
    }

    /** Throws if {@code pole} is not finite or coincides with an integration bound. */
    public static void validatePole(double pole, double min, double max) {
        if (!Double.isFinite(pole)) {
            throw new IllegalArgumentException("pole must be finite");
        }
        if (pole == min || pole == max) {
            throw new IllegalArgumentException("pole must not coincide with an integration bound");
        }
        if (pole > min && pole < max) {
            double span = Math.abs(max - min);
            double scale = Math.max(1.0,
                    Math.max(Math.abs(min), Math.max(Math.abs(max), Math.abs(pole))));
            double minGap = Math.max(
                    PRINCIPAL_VALUE_BOUNDARY_GAP_ULPS * Math.ulp(scale),
                    PRINCIPAL_VALUE_BOUNDARY_GAP_ULPS * Math.ulp(1.0) * span);
            if (Math.min(pole - min, max - pole) <= minGap) {
                throw new IllegalArgumentException(
                        "pole is too close to an integration bound for stable principal value regularization");
            }
        }
    }

    /** Throws if {@code maxCycles} is not positive. */
    public static void validateMaxCycles(int maxCycles) {
        if (maxCycles <= 0) {
            throw new IllegalArgumentException("maxCycles must be > 0");
        }
    }

    /** Throws if {@code maxRefinements} is not positive. */
    public static void validateMaxRefinements(int maxRefinements) {
        if (maxRefinements <= 0) {
            throw new IllegalArgumentException("maxRefinements must be > 0");
        }
    }

    /**
     * Validates, deduplicates, and sorts a breakpoint array.
     *
     * @param points raw breakpoints (may be null or empty)
     * @param min    lower integration bound (exclusive)
     * @param max    upper integration bound (exclusive)
     * @return sorted, deduplicated array of breakpoints strictly inside {@code (min, max)}
     * @throws IllegalArgumentException if any breakpoint is non-finite or outside {@code (min, max)}
     */
    public static double[] validateBreakpoints(double[] points, double min, double max) {
        if (points == null || points.length == 0) {
            return new double[0];
        }

        double[] sorted = Arrays.copyOf(points, points.length);
        Arrays.sort(sorted);
        double[] unique = new double[sorted.length];
        int size = 0;
        for (double point : sorted) {
            if (!Double.isFinite(point)) {
                throw new IllegalArgumentException("breakpoints must be finite");
            }
            if (!(point > min && point < max)) {
                throw new IllegalArgumentException("breakpoints must lie strictly inside (min, max)");
            }
            if (size == 0 || point > unique[size - 1]) {
                unique[size++] = point;
            }
        }
        return Arrays.copyOf(unique, size);
    }
}