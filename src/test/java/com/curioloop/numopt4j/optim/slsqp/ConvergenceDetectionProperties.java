/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for Convergence Detection (checkStop and checkConv).
 *
 * **Property 11: Convergence Detection Correctness**
 */
package com.curioloop.numopt4j.optim.slsqp;

import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.optim.OptimizationStatus;
import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for Convergence Detection (checkStop and checkConv).
 *
 * <p>Tests the following properties:</p>
 * <ul>
 *   <li>Property 11.1: Returns false when constraint violation >= tol</li>
 *   <li>Property 11.2: Returns false when badQP is true</li>
 *   <li>Property 11.3: Returns false when f is NaN</li>
 *   <li>Property 11.4: Returns true when |f - f0| < tol and constraints satisfied</li>
 *   <li>Property 11.5: Returns true when ||s||_2 < tol and constraints satisfied</li>
 *   <li>Property 11.6: Returns true when |f| < fEvalTol (if fEvalTol >= 0)</li>
 *   <li>Property 11.7: Returns true when |f - f0| < fDiffTol (if fDiffTol >= 0)</li>
 *   <li>Property 11.8: Returns true when ||x - x0||_2 < xDiffTol (if xDiffTol >= 0)</li>
 *   <li>Property 11.9: Constraint violation is computed correctly (L1 norm)</li>
 *   <li>Property 11.10: Deterministic behavior - same input produces same output</li>
 * </ul>
 */
@Tag("Feature: slsqp-java-rewrite, Property 11: Convergence Detection Correctness")
class ConvergenceDetectionProperties {

    private static final double EPSILON = 1e-15;
    // Use a fixed tolerance for tests to avoid jqwik decimal scale issues
    private static final double TOL = 0.001;


    // ========================================================================
    // Helper Methods
    // ========================================================================

    private double[] generateVector(int n, java.util.Random rand) {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            v[i] = rand.nextDouble() * 4 - 2;
        }
        return v;
    }

    private double[] generateSmallVector(int n, double maxNorm, java.util.Random rand) {
        double[] v = new double[n];
        double scale = maxNorm / Math.sqrt(n);
        for (int i = 0; i < n; i++) {
            v[i] = (rand.nextDouble() * 2 - 1) * scale;
        }
        return v;
    }

    private double norm2(double[] v) {
        double sum = 0;
        for (double x : v) {
            sum += x * x;
        }
        return Math.sqrt(sum);
    }

    private double computeConstraintViolation(double[] c, int m, int meq) {
        double vio = 0;
        for (int j = 0; j < m; j++) {
            double h1 = (j < meq) ? c[j] : 0.0;
            vio += Math.max(-c[j], h1);
        }
        return vio;
    }

    @Provide
    Arbitrary<Long> randomSeed() {
        return Arbitraries.longs().between(1, Long.MAX_VALUE);
    }


    // ========================================================================
    // Property 11.1: Returns false when constraint violation >= tol
    // ========================================================================

    /**
     *
     * When constraint violation is >= tolerance, checkStop should return false.
     */
    @Property(tries = 100)
    void returnsFalseWhenConstraintViolationExceedsTolerance(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = TOL + rand.nextDouble() * TOL;
        double f = 1.0;
        double f0 = 1.0;
        double[] s = new double[n];
        double[] x = generateVector(n, rand);
        double[] x0 = x.clone();
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNull(result, "Should return 0 when vio >= tol");
    }

    /**
     *
     * When constraint violation equals tolerance, checkStop should return false.
     */
    @Property(tries = 100)
    void returnsFalseWhenConstraintViolationEqualsTolerance(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = TOL;
        double f = 1.0;
        double f0 = 1.0;
        double[] s = new double[n];
        double[] x = generateVector(n, rand);
        double[] x0 = x.clone();
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNull(result, "Should return 0 when vio == tol");
    }


    // ========================================================================
    // Property 11.2: Returns false when badQP is true
    // ========================================================================

    /**
     *
     * When badQP is true, checkStop should return false.
     */
    @Property(tries = 100)
    void returnsFalseWhenBadQPIsTrue(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = 0;
        double f = 1.0;
        double f0 = 1.0;
        double[] s = new double[n];
        double[] x = generateVector(n, rand);
        double[] x0 = x.clone();
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, true, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNull(result, "Should return 0 when badQP is true");
    }

    /**
     *
     * When badQP is false and criteria met, should return true.
     */
    @Property(tries = 100)
    void returnsTrueWhenBadQPIsFalseAndCriteriaMet(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = TOL / 2;
        double f = 1.0;
        double f0 = 1.0;  // |f - f0| = 0 < TOL
        double[] s = new double[n];  // ||s|| = 0 < TOL
        double[] x = generateVector(n, rand);
        double[] x0 = x.clone();
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNotNull(result, "Should return non-zero when badQP is false and criteria met");
    }


    // ========================================================================
    // Property 11.3: Returns false when f is NaN
    // ========================================================================

    /**
     *
     * When f is NaN, checkStop should return false.
     */
    @Property(tries = 100)
    void returnsFalseWhenFunctionValueIsNaN(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = 0;
        double f = Double.NaN;
        double f0 = 1.0;
        double[] s = new double[n];
        double[] x = generateVector(n, rand);
        double[] x0 = x.clone();
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNull(result, "Should return 0 when f is NaN");
    }

    /**
     *
     * When f is finite and criteria met, should return true.
     */
    @Property(tries = 100)
    void checksOtherCriteriaWhenFunctionValueIsFinite(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @DoubleRange(min = -100, max = 100) double f,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(Double.isFinite(f));
        java.util.Random rand = new java.util.Random(seed);
        double vio = 0;
        double f0 = f;  // |f - f0| = 0 < TOL
        double[] s = new double[n];
        double[] x = generateVector(n, rand);
        double[] x0 = x.clone();
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNotNull(result, "Should return non-zero when f is finite and criteria met");
    }


    // ========================================================================
    // Property 11.4: Returns true when |f - f0| < tol and constraints satisfied
    // ========================================================================

    /**
     *
     * When vio < tol and |f - f0| < tol, should return true.
     */
    @Property(tries = 100)
    void returnsTrueWhenFunctionChangeSmall(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @DoubleRange(min = -100, max = 100) double f,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(Double.isFinite(f));
        java.util.Random rand = new java.util.Random(seed);
        double vio = TOL / 2;
        double fDiff = (rand.nextDouble() * 2 - 1) * TOL * 0.5;
        double f0 = f - fDiff;

        // Large step so step norm criterion is not met
        double[] s = generateVector(n, rand);
        for (int i = 0; i < n; i++) s[i] *= 10;

        double[] x = generateVector(n, rand);
        double[] x0 = generateVector(n, rand);
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNotNull(result, "Should return non-zero when |f - f0| < tol");
    }

    /**
     *
     * When |f - f0| >= tol and ||s|| >= tol, should return false.
     */
    @Property(tries = 100)
    void returnsFalseWhenFunctionChangeLarge(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = TOL / 2;
        double f = 1.0;
        double f0 = f + TOL * 2;

        double[] s = generateVector(n, rand);
        for (int i = 0; i < n; i++) s[i] *= 10;

        double[] x = generateVector(n, rand);
        double[] x0 = generateVector(n, rand);
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNull(result, "Should return 0 when |f - f0| >= tol");
    }


    // ========================================================================
    // Property 11.5: Returns true when ||s||_2 < tol and constraints satisfied
    // ========================================================================

    /**
     *
     * When vio < tol and ||s||_2 < tol, should return true.
     */
    @Property(tries = 100)
    void returnsTrueWhenStepNormSmall(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = TOL / 2;
        double f = 1.0;
        double f0 = f + TOL * 2;  // |f - f0| >= tol

        double[] s = generateSmallVector(n, TOL * 0.5, rand);
        double stepNorm = BLAS.dnrm2(n, s, 0, 1);
        Assume.that(stepNorm < TOL);

        double[] x = generateVector(n, rand);
        double[] x0 = generateVector(n, rand);
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNotNull(result, "Should return non-zero when ||s||_2 < tol");
    }

    /**
     *
     * When ||s||_2 >= tol and |f - f0| >= tol, should return false.
     */
    @Property(tries = 100)
    void returnsFalseWhenStepNormLarge(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = TOL / 2;
        double f = 1.0;
        double f0 = f + TOL * 2;

        double[] s = generateVector(n, rand);
        double stepNorm = BLAS.dnrm2(n, s, 0, 1);
        Assume.that(stepNorm >= TOL);

        double[] x = generateVector(n, rand);
        double[] x0 = generateVector(n, rand);
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNull(result, "Should return 0 when ||s||_2 >= tol");
    }


    // ========================================================================
    // Property 11.6: Returns true when |f| < fEvalTol (if fEvalTol >= 0)
    // ========================================================================

    /**
     *
     * When fEvalTol >= 0 and |f| < fEvalTol, should return true.
     */
    @Property(tries = 100)
    void returnsTrueWhenFunctionValueSmall(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double fEvalTol = 0.01;
        double vio = TOL / 2;
        double f = (rand.nextDouble() * 2 - 1) * fEvalTol * 0.5;
        double f0 = f + TOL * 2;

        double[] s = generateVector(n, rand);
        for (int i = 0; i < n; i++) s[i] *= 10;

        double[] x = generateVector(n, rand);
        double[] x0 = generateVector(n, rand);
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, fEvalTol, -1, -1, x, 0, x0, 0, u, 0);

        assertNotNull(result, "Should return non-zero when |f| < fEvalTol");
    }

    /**
     *
     * When fEvalTol < 0, the criterion should be disabled.
     */
    @Property(tries = 100)
    void fEvalTolDisabledWhenNegative(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = TOL / 2;
        double f = 1e-10;  // Very small |f|
        double f0 = f + TOL * 2;

        double[] s = generateVector(n, rand);
        for (int i = 0; i < n; i++) s[i] *= 10;

        double[] x = generateVector(n, rand);
        double[] x0 = generateVector(n, rand);
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNull(result, "Should return 0 when fEvalTol disabled");
    }


    // ========================================================================
    // Property 11.7: Returns true when |f - f0| < fDiffTol (if fDiffTol >= 0)
    // ========================================================================

    /**
     *
     * When fDiffTol >= 0 and |f - f0| < fDiffTol, should return true.
     */
    @Property(tries = 100)
    void returnsTrueWhenFunctionDifferenceSmallWithFDiffTol(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double fDiffTol = 0.01;
        double vio = TOL / 2;
        double f = 100.0;
        // Make |f - f0| < fDiffTol but >= TOL
        double fDiff = TOL + (fDiffTol - TOL) * 0.5;
        Assume.that(fDiff < fDiffTol && fDiff >= TOL);
        double f0 = f + fDiff;

        double[] s = generateVector(n, rand);
        for (int i = 0; i < n; i++) s[i] *= 10;

        double[] x = generateVector(n, rand);
        double[] x0 = generateVector(n, rand);
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, fDiffTol, -1, x, 0, x0, 0, u, 0);

        assertNotNull(result, "Should return non-zero when |f - f0| < fDiffTol");
    }

    /**
     *
     * When fDiffTol < 0, the criterion should be disabled.
     */
    @Property(tries = 100)
    void fDiffTolDisabledWhenNegative(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = TOL / 2;
        double f = 100.0;
        double f0 = f + TOL * 2;

        double[] s = generateVector(n, rand);
        for (int i = 0; i < n; i++) s[i] *= 10;

        double[] x = generateVector(n, rand);
        double[] x0 = generateVector(n, rand);
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNull(result, "Should return 0 when fDiffTol disabled");
    }


    // ========================================================================
    // Property 11.8: Returns true when ||x - x0||_2 < xDiffTol (if xDiffTol >= 0)
    // ========================================================================

    /**
     *
     * When xDiffTol >= 0 and ||x - x0||_2 < xDiffTol, should return true.
     */
    @Property(tries = 100)
    void returnsTrueWhenVariableDifferenceSmall(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double xDiffTol = 0.01;
        double vio = TOL / 2;
        double f = 100.0;
        double f0 = f + TOL * 2;

        double[] s = generateVector(n, rand);
        for (int i = 0; i < n; i++) s[i] *= 10;

        double[] x = generateVector(n, rand);
        double[] xDiff = generateSmallVector(n, xDiffTol * 0.5, rand);
        double[] x0 = new double[n];
        for (int i = 0; i < n; i++) x0[i] = x[i] - xDiff[i];
        double[] u = new double[n];

        double xNormDiff = norm2(xDiff);
        Assume.that(xNormDiff < xDiffTol);

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, xDiffTol, x, 0, x0, 0, u, 0);

        assertNotNull(result, "Should return non-zero when ||x - x0||_2 < xDiffTol");
    }

    /**
     *
     * When xDiffTol < 0, the criterion should be disabled.
     */
    @Property(tries = 100)
    void xDiffTolDisabledWhenNegative(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = TOL / 2;
        double f = 100.0;
        double f0 = f + TOL * 2;

        double[] s = generateVector(n, rand);
        for (int i = 0; i < n; i++) s[i] *= 10;

        double[] x = generateVector(n, rand);
        double[] x0 = x.clone();
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNull(result, "Should return 0 when xDiffTol disabled");
    }


    // ========================================================================
    // Property 11.9: Constraint violation is computed correctly (L1 norm)
    // ========================================================================

    /**
     *
     * checkConv should correctly compute constraint violation as L1 norm.
     */
    @Property(tries = 100)
    void constraintViolationComputedCorrectly(
            @ForAll @IntRange(min = 1, max = 5) int meq,
            @ForAll @IntRange(min = 1, max = 5) int mineq,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        int m = meq + mineq;
        int n = 5;

        double[] c = new double[m];
        for (int j = 0; j < m; j++) {
            c[j] = rand.nextDouble() * 4 - 2;
        }

        double expectedVio = computeConstraintViolation(c, m, meq);

        double f = 1.0;
        double f0 = 1.0;
        double[] s = new double[n];
        double[] x = new double[n];
        double[] x0 = new double[n];
        double[] u = new double[n];
        double[] vioOut = new double[1];

        SLSQPCore.checkConv(c, 0, m, meq, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0, vioOut);

        assertEquals(expectedVio, vioOut[0], EPSILON,
                "Constraint violation should be computed correctly");
    }

    /**
     *
     * Satisfied equality constraints (c[j] = 0) should have zero violation.
     */
    @Property(tries = 100)
    void satisfiedEqualityConstraintsHaveZeroViolation(
            @ForAll @IntRange(min = 1, max = 5) int meq
    ) {
        int m = meq;
        int n = 5;
        double[] c = new double[m];  // All zeros

        double f = 1.0;
        double f0 = 1.0;
        double[] s = new double[n];
        double[] x = new double[n];
        double[] x0 = new double[n];
        double[] u = new double[n];
        double[] vioOut = new double[1];

        SLSQPCore.checkConv(c, 0, m, meq, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0, vioOut);

        assertEquals(0.0, vioOut[0], EPSILON,
                "Satisfied equality constraints should have zero violation");
    }

    /**
     *
     * Satisfied inequality constraints (c[j] >= 0) should have zero violation.
     */
    @Property(tries = 100)
    void satisfiedInequalityConstraintsHaveZeroViolation(
            @ForAll @IntRange(min = 1, max = 5) int mineq,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        int meq = 0;
        int m = mineq;
        int n = 5;

        double[] c = new double[m];
        for (int j = 0; j < m; j++) {
            c[j] = rand.nextDouble() * 2;  // [0, 2]
        }

        double f = 1.0;
        double f0 = 1.0;
        double[] s = new double[n];
        double[] x = new double[n];
        double[] x0 = new double[n];
        double[] u = new double[n];
        double[] vioOut = new double[1];

        SLSQPCore.checkConv(c, 0, m, meq, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0, vioOut);

        assertEquals(0.0, vioOut[0], EPSILON,
                "Satisfied inequality constraints should have zero violation");
    }


    /**
     *
     * Violated equality constraints should contribute |c[j]|.
     */
    @Property(tries = 100)
    void violatedEqualityConstraintsContributeAbsoluteValue(
            @ForAll @IntRange(min = 1, max = 5) int meq,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        int m = meq;
        int n = 5;

        double[] c = new double[m];
        double expectedVio = 0.0;
        for (int j = 0; j < m; j++) {
            c[j] = rand.nextDouble() * 4 - 2;
            expectedVio += Math.abs(c[j]);
        }

        double f = 1.0;
        double f0 = 1.0;
        double[] s = new double[n];
        double[] x = new double[n];
        double[] x0 = new double[n];
        double[] u = new double[n];
        double[] vioOut = new double[1];

        SLSQPCore.checkConv(c, 0, m, meq, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0, vioOut);

        assertEquals(expectedVio, vioOut[0], EPSILON,
                "Violated equality constraints should contribute |c[j]|");
    }

    /**
     *
     * Violated inequality constraints (c[j] < 0) should contribute -c[j].
     */
    @Property(tries = 100)
    void violatedInequalityConstraintsContributeNegativeValue(
            @ForAll @IntRange(min = 1, max = 5) int mineq,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        int meq = 0;
        int m = mineq;
        int n = 5;

        double[] c = new double[m];
        double expectedVio = 0.0;
        for (int j = 0; j < m; j++) {
            c[j] = -(rand.nextDouble() * 2 + 0.1);  // [-2.1, -0.1]
            expectedVio += -c[j];
        }

        double f = 1.0;
        double f0 = 1.0;
        double[] s = new double[n];
        double[] x = new double[n];
        double[] x0 = new double[n];
        double[] u = new double[n];
        double[] vioOut = new double[1];

        SLSQPCore.checkConv(c, 0, m, meq, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0, vioOut);

        assertEquals(expectedVio, vioOut[0], EPSILON,
                "Violated inequality constraints should contribute -c[j]");
    }


    // ========================================================================
    // Property 11.10: Deterministic behavior - same input produces same output
    // ========================================================================

    /**
     *
     * checkStop should be deterministic.
     */
    @Property(tries = 100)
    void checkStopIsDeterministic(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @DoubleRange(min = 0, max = 0.01) double vio,
            @ForAll @DoubleRange(min = -100, max = 100) double f,
            @ForAll @DoubleRange(min = -100, max = 100) double f0,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(Double.isFinite(f) && Double.isFinite(f0));
        java.util.Random rand = new java.util.Random(seed);

        double[] s = generateVector(n, rand);
        double[] x = generateVector(n, new java.util.Random(seed + 1));
        double[] x0 = generateVector(n, new java.util.Random(seed + 2));
        double[] u1 = new double[n];
        double[] u2 = new double[n];

        OptimizationStatus result1 = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s.clone(), 0, n, -1, -1, -1, x.clone(), 0, x0.clone(), 0, u1, 0);

        OptimizationStatus result2 = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s.clone(), 0, n, -1, -1, -1, x.clone(), 0, x0.clone(), 0, u2, 0);

        assertEquals(result1, result2, "checkStop should be deterministic");
    }

    /**
     *
     * checkConv should be deterministic.
     */
    @Property(tries = 100)
    void checkConvIsDeterministic(
            @ForAll @IntRange(min = 1, max = 5) int meq,
            @ForAll @IntRange(min = 1, max = 5) int mineq,
            @ForAll @DoubleRange(min = -100, max = 100) double f,
            @ForAll @DoubleRange(min = -100, max = 100) double f0,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(Double.isFinite(f) && Double.isFinite(f0));
        java.util.Random rand = new java.util.Random(seed);
        int m = meq + mineq;
        int n = 5;

        double[] c = new double[m];
        for (int j = 0; j < m; j++) {
            c[j] = rand.nextDouble() * 4 - 2;
        }

        double[] s = generateVector(n, new java.util.Random(seed + 1));
        double[] x = generateVector(n, new java.util.Random(seed + 2));
        double[] x0 = generateVector(n, new java.util.Random(seed + 3));
        double[] u1 = new double[n];
        double[] u2 = new double[n];
        double[] vioOut1 = new double[1];
        double[] vioOut2 = new double[1];

        OptimizationStatus result1 = SLSQPCore.checkConv(c.clone(), 0, m, meq, TOL, false, f, f0,
                s.clone(), 0, n, -1, -1, -1, x.clone(), 0, x0.clone(), 0, u1, 0, vioOut1);

        OptimizationStatus result2 = SLSQPCore.checkConv(c.clone(), 0, m, meq, TOL, false, f, f0,
                s.clone(), 0, n, -1, -1, -1, x.clone(), 0, x0.clone(), 0, u2, 0, vioOut2);

        assertEquals(result1, result2, "checkConv should be deterministic");
        assertEquals(vioOut1[0], vioOut2[0], EPSILON, "Violation output should be deterministic");
    }


    // ========================================================================
    // Additional Edge Case Tests
    // ========================================================================

    /**
     *
     * When all criteria are met, should return true.
     */
    @Property(tries = 100)
    void returnsTrueWhenAllCriteriaMet(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = 0;
        double f = 0;
        double f0 = 0;
        double[] s = new double[n];
        double[] x = generateVector(n, rand);
        double[] x0 = x.clone();
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, TOL, TOL, TOL, x, 0, x0, 0, u, 0);

        assertNotNull(result, "Should return non-zero when all criteria met");
    }

    /**
     *
     * When no criteria are met, should return false.
     */
    @Property(tries = 100)
    void returnsFalseWhenNoCriteriaMet(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        double vio = TOL * 2;
        double f = 100.0;
        double f0 = f + TOL * 2;

        double[] s = generateVector(n, rand);
        for (int i = 0; i < n; i++) s[i] *= 10;

        double[] x = generateVector(n, rand);
        double[] x0 = generateVector(n, rand);
        double[] u = new double[n];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        assertNull(result, "Should return 0 when no criteria met");
    }

    /**
     *
     * checkStop should handle n = 1 correctly.
     */
    @Property(tries = 50)
    void checkStopHandlesSingleDimension(
            @ForAll @DoubleRange(min = -100, max = 100) double f,
            @ForAll @DoubleRange(min = -100, max = 100) double f0,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(Double.isFinite(f) && Double.isFinite(f0));
        java.util.Random rand = new java.util.Random(seed);
        double vio = rand.nextDouble() * TOL * 0.5;  // vio < TOL
        int n = 1;
        double[] s = {0.0};
        double[] x = {1.0};
        double[] x0 = {1.0};
        double[] u = new double[1];

        OptimizationStatus result = SLSQPCore.checkStop(vio, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0);

        if (vio < TOL && !Double.isNaN(f) && Math.abs(f - f0) < TOL) {
            assertNotNull(result, "Should return non-zero for n=1 when criteria met");
        }
    }

    /**
     *
     * checkConv should handle m = 0 (no constraints) correctly.
     */
    @Property(tries = 50)
    void checkConvHandlesNoConstraints(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        int m = 0;
        int meq = 0;
        double[] c = new double[0];

        double f = 1.0;
        double f0 = 1.0;
        double[] s = new double[n];
        double[] x = generateVector(n, rand);
        double[] x0 = x.clone();
        double[] u = new double[n];
        double[] vioOut = new double[1];

        OptimizationStatus result = SLSQPCore.checkConv(c, 0, m, meq, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0, vioOut);

        assertEquals(0.0, vioOut[0], EPSILON, "No constraints should have zero violation");
        assertNotNull(result, "Should return non-zero when no constraints and function converged");
    }

    /**
     *
     * checkConv should handle offset correctly.
     */
    @Property(tries = 50)
    void checkConvHandlesOffset(
            @ForAll @IntRange(min = 1, max = 3) int meq,
            @ForAll @IntRange(min = 1, max = 3) int mineq,
            @ForAll @IntRange(min = 1, max = 5) int offset,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);
        int m = meq + mineq;
        int n = 5;

        double[] c = new double[offset + m];
        for (int j = 0; j < m; j++) {
            c[offset + j] = rand.nextDouble() * 4 - 2;
        }

        double expectedVio = computeConstraintViolation(
                java.util.Arrays.copyOfRange(c, offset, offset + m), m, meq);

        double f = 1.0;
        double f0 = 1.0;
        double[] s = new double[n];
        double[] x = new double[n];
        double[] x0 = new double[n];
        double[] u = new double[n];
        double[] vioOut = new double[1];

        SLSQPCore.checkConv(c, offset, m, meq, TOL, false, f, f0,
                s, 0, n, -1, -1, -1, x, 0, x0, 0, u, 0, vioOut);

        assertEquals(expectedVio, vioOut[0], EPSILON, "checkConv should handle offset correctly");
    }
}
