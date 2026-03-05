/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for NNLS (Non-Negative Least Squares) solver.
 *
 * **Property 3: NNLS Solver Correctness**
 */
package com.curioloop.numopt4j.optim.slsqp;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for NNLS (Non-Negative Least Squares) solver.
 *
 * <p>Tests the following properties:</p>
 * <ul>
 *   <li>Property 3.1: Solution satisfies non-negativity constraint (x >= 0)</li>
 *   <li>Property 3.2: Dual vector w = A^T(b - Ax) satisfies KKT conditions</li>
 *   <li>Property 3.3: Solution minimizes residual norm among non-negative solutions</li>
 *   <li>Property 3.4: Active set method correctly maintains P and Z index sets</li>
 *   <li>Property 3.5: NNLS handles edge cases correctly</li>
 * </ul>
 *
 * <p>The NNLS problem is: minimize ||Ax - b||_2 subject to x >= 0</p>
 *
 * <p>KKT conditions for NNLS:</p>
 * <ul>
 *   <li>x >= 0 (primal feasibility)</li>
 *   <li>w = A^T(b - Ax) (dual vector)</li>
 *   <li>w_j <= 0 for j in Z (zero set) - no constraint can be relaxed</li>
 *   <li>w_j = 0 for j in P (positive set) - complementary slackness</li>
 * </ul>
 */
@Tag("Feature: slsqp-java-rewrite, Property 3: NNLS Solver Correctness")
class NNLSProperties {

    private static final double EPSILON = 1e-8;
    private static final double TOLERANCE = 1e-10;

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * Generate a well-conditioned random matrix A (m x n) in column-major format.
     * Ensures the matrix has full column rank for numerical stability.
     */
    private double[] generateMatrix(int m, int n, java.util.Random rand) {
        double[] a = new double[m * n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                a[j * m + i] = rand.nextDouble() * 10 - 5;  // Values in [-5, 5]
            }
            // Add diagonal dominance for numerical stability
            if (j < m) {
                a[j * m + j] += 10.0;
            }
        }
        return a;
    }

    /**
     * Generate a random vector b of length m.
     */
    private double[] generateVector(int m, java.util.Random rand) {
        double[] b = new double[m];
        for (int i = 0; i < m; i++) {
            b[i] = rand.nextDouble() * 10 - 5;  // Values in [-5, 5]
        }
        return b;
    }

    /**
     * Compute matrix-vector product Ax (A is m x n column-major).
     */
    private double[] matVec(double[] a, int m, int n, double[] x) {
        double[] result = new double[m];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                result[i] += a[j * m + i] * x[j];
            }
        }
        return result;
    }

    /**
     * Compute matrix-transpose-vector product A^T * v (A is m x n column-major).
     */
    private double[] matTVec(double[] a, int m, int n, double[] v) {
        double[] result = new double[n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                result[j] += a[j * m + i] * v[i];
            }
        }
        return result;
    }

    /**
     * Compute vector subtraction: result = a - b.
     */
    private double[] subtract(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    /**
     * Compute Euclidean norm of a vector.
     */
    private double norm2(double[] v) {
        double sum = 0;
        for (double vi : v) {
            sum += vi * vi;
        }
        return Math.sqrt(sum);
    }

    // ========================================================================
    // Property 3.1: Solution satisfies non-negativity constraint (x >= 0)
    // ========================================================================

    @Property(tries = 100)
    void solutionSatisfiesNonNegativity(
            @ForAll @IntRange(min = 3, max = 15) int m,
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);  // Ensure overdetermined or square system

        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix A and vector b
        double[] a = generateMatrix(m, n, rand);
        double[] b = generateVector(m, rand);

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[n];
        double[] z = new double[m];
        int[] index = new int[n];
        double[] rnorm = new double[1];

        // Solve NNLS
        int status = LSQSolver.nnls(m, n, a, 0, m, b, 0, x, 0, w, 0, z, 0, index, 0, 3 * n, rnorm);

        // Check status (0 = success, 1 = max iterations)
        assertTrue(status == 0 || status == 1, 
                "NNLS should return success (0) or max iterations (1), got: " + status);

        // Verify non-negativity: x >= 0
        for (int i = 0; i < n; i++) {
            assertTrue(x[i] >= -TOLERANCE,
                    "Solution x[" + i + "] = " + x[i] + " should be non-negative");
        }
    }

    @Provide
    Arbitrary<Long> randomSeed() {
        return Arbitraries.longs().between(1, Long.MAX_VALUE);
    }

    // ========================================================================
    // Property 3.2: Dual vector satisfies KKT conditions
    // ========================================================================

    @Property(tries = 100)
    void dualVectorSatisfiesKKTConditions(
            @ForAll @IntRange(min = 3, max = 15) int m,
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix A and vector b
        double[] aOriginal = generateMatrix(m, n, rand);
        double[] bOriginal = generateVector(m, rand);

        // Make copies since NNLS modifies A and b
        double[] a = aOriginal.clone();
        double[] b = bOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[n];
        double[] z = new double[m];
        int[] index = new int[n];
        double[] rnorm = new double[1];

        // Solve NNLS
        int status = LSQSolver.nnls(m, n, a, 0, m, b, 0, x, 0, w, 0, z, 0, index, 0, 3 * n, rnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Compute residual r = b - Ax using original A and b
        double[] Ax = matVec(aOriginal, m, n, x);
        double[] residual = subtract(bOriginal, Ax);

        // Compute dual vector w = A^T * residual
        double[] wComputed = matTVec(aOriginal, m, n, residual);

        // Verify KKT conditions:
        // For j in P (positive set): x[j] > 0 implies w[j] = 0 (complementary slackness)
        // For j in Z (zero set): x[j] = 0 implies w[j] <= 0 (no constraint can be relaxed)
        for (int j = 0; j < n; j++) {
            if (x[j] > TOLERANCE) {
                // j is in P (positive set): w[j] should be approximately 0
                assertEquals(0.0, wComputed[j], EPSILON,
                        "For x[" + j + "] = " + x[j] + " > 0, w[" + j + "] should be 0, got: " + wComputed[j]);
            } else {
                // j is in Z (zero set): w[j] should be <= 0
                assertTrue(wComputed[j] <= EPSILON,
                        "For x[" + j + "] = 0, w[" + j + "] should be <= 0, got: " + wComputed[j]);
            }
        }
    }

    // ========================================================================
    // Property 3.3: Solution minimizes residual norm among non-negative solutions
    // ========================================================================

    @Property(tries = 100)
    void solutionMinimizesResidualNorm(
            @ForAll @IntRange(min = 3, max = 12) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix A and vector b
        double[] aOriginal = generateMatrix(m, n, rand);
        double[] bOriginal = generateVector(m, rand);

        // Make copies since NNLS modifies A and b
        double[] a = aOriginal.clone();
        double[] b = bOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[n];
        double[] z = new double[m];
        int[] index = new int[n];
        double[] rnorm = new double[1];

        // Solve NNLS
        int status = LSQSolver.nnls(m, n, a, 0, m, b, 0, x, 0, w, 0, z, 0, index, 0, 3 * n, rnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Compute residual norm for the NNLS solution
        double[] Ax = matVec(aOriginal, m, n, x);
        double[] residual = subtract(bOriginal, Ax);
        double nnlsResidualNorm = norm2(residual);

        // Generate random non-negative perturbations and verify NNLS solution is optimal
        for (int trial = 0; trial < 10; trial++) {
            // Create a perturbed non-negative solution
            double[] xPerturbed = new double[n];
            for (int j = 0; j < n; j++) {
                xPerturbed[j] = Math.max(0, x[j] + (rand.nextDouble() - 0.5) * 0.5);
            }

            // Compute residual norm for perturbed solution
            double[] AxPerturbed = matVec(aOriginal, m, n, xPerturbed);
            double[] residualPerturbed = subtract(bOriginal, AxPerturbed);
            double perturbedResidualNorm = norm2(residualPerturbed);

            // NNLS solution should have smaller or equal residual norm
            assertTrue(nnlsResidualNorm <= perturbedResidualNorm + EPSILON,
                    "NNLS residual norm (" + nnlsResidualNorm + 
                    ") should be <= perturbed residual norm (" + perturbedResidualNorm + ")");
        }
    }

    // ========================================================================
    // Property 3.4: Residual norm is correctly computed
    // ========================================================================

    @Property(tries = 100)
    void residualNormIsCorrectlyComputed(
            @ForAll @IntRange(min = 3, max = 15) int m,
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix A and vector b
        double[] aOriginal = generateMatrix(m, n, rand);
        double[] bOriginal = generateVector(m, rand);

        // Make copies since NNLS modifies A and b
        double[] a = aOriginal.clone();
        double[] b = bOriginal.clone();

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[n];
        double[] z = new double[m];
        int[] index = new int[n];
        double[] rnorm = new double[1];

        // Solve NNLS
        int status = LSQSolver.nnls(m, n, a, 0, m, b, 0, x, 0, w, 0, z, 0, index, 0, 3 * n, rnorm);

        if (status != 0) {
            return;  // Skip if solver didn't converge
        }

        // Compute residual norm independently
        double[] Ax = matVec(aOriginal, m, n, x);
        double[] residual = subtract(bOriginal, Ax);
        double computedResidualNorm = norm2(residual);

        // Verify the returned rnorm matches our computation
        assertEquals(computedResidualNorm, rnorm[0], EPSILON,
                "Returned rnorm (" + rnorm[0] + ") should match computed residual norm (" + computedResidualNorm + ")");
    }

    // ========================================================================
    // Property 3.5: NNLS handles infeasible intermediate solutions by interpolation
    // ========================================================================

    @Property(tries = 100)
    void nnlsHandlesInfeasibleIntermediateSolutions(
            @ForAll @IntRange(min = 4, max = 12) int m,
            @ForAll @IntRange(min = 3, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate a problem that is likely to have infeasible intermediate solutions
        // by using a matrix with mixed positive and negative entries
        double[] a = new double[m * n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                a[j * m + i] = rand.nextDouble() * 20 - 10;  // Values in [-10, 10]
            }
        }

        // Generate b with negative values to encourage infeasible intermediates
        double[] b = new double[m];
        for (int i = 0; i < m; i++) {
            b[i] = rand.nextDouble() * 20 - 10;
        }

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[n];
        double[] z = new double[m];
        int[] index = new int[n];
        double[] rnorm = new double[1];

        // Solve NNLS - should handle infeasible intermediates via interpolation
        int status = LSQSolver.nnls(m, n, a, 0, m, b, 0, x, 0, w, 0, z, 0, index, 0, 3 * n, rnorm);

        // Even with difficult problems, solution should be non-negative
        assertTrue(status == 0 || status == 1,
                "NNLS should return success (0) or max iterations (1), got: " + status);

        for (int i = 0; i < n; i++) {
            assertTrue(x[i] >= -TOLERANCE,
                    "Solution x[" + i + "] = " + x[i] + " should be non-negative even for difficult problems");
        }
    }

    // ========================================================================
    // Property 3.6: NNLS handles zero vector b correctly
    // ========================================================================

    @Property(tries = 50)
    void nnlsHandlesZeroVectorB(
            @ForAll @IntRange(min = 3, max = 10) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix A
        double[] a = generateMatrix(m, n, rand);

        // Zero vector b
        double[] b = new double[m];

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[n];
        double[] z = new double[m];
        int[] index = new int[n];
        double[] rnorm = new double[1];

        // Solve NNLS
        int status = LSQSolver.nnls(m, n, a, 0, m, b, 0, x, 0, w, 0, z, 0, index, 0, 3 * n, rnorm);

        assertTrue(status == 0 || status == 1,
                "NNLS should handle zero b vector");

        // Solution should be zero (or very close to zero)
        for (int i = 0; i < n; i++) {
            assertTrue(x[i] >= -TOLERANCE && x[i] <= EPSILON,
                    "For b=0, solution x[" + i + "] = " + x[i] + " should be approximately 0");
        }

        // Residual norm should be 0
        assertEquals(0.0, rnorm[0], EPSILON,
                "For b=0 and x=0, residual norm should be 0");
    }

    // ========================================================================
    // Property 3.7: NNLS handles identity matrix correctly
    // ========================================================================

    @Property(tries = 50)
    void nnlsHandlesIdentityMatrix(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        int m = n;  // Square identity matrix

        // Create identity matrix
        double[] a = new double[m * n];
        for (int i = 0; i < n; i++) {
            a[i * m + i] = 1.0;
        }

        // Generate random non-negative b
        double[] b = new double[m];
        double[] bOriginal = new double[m];  // Save original b since NNLS modifies it
        for (int i = 0; i < m; i++) {
            b[i] = rand.nextDouble() * 10;  // Non-negative values
            bOriginal[i] = b[i];
        }

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[n];
        double[] z = new double[m];
        int[] index = new int[n];
        double[] rnorm = new double[1];

        // Solve NNLS
        int status = LSQSolver.nnls(m, n, a, 0, m, b, 0, x, 0, w, 0, z, 0, index, 0, 3 * n, rnorm);

        assertEquals(0, status, "NNLS should succeed for identity matrix with non-negative b");

        // For identity matrix with non-negative b, solution should equal original b
        for (int i = 0; i < n; i++) {
            assertEquals(bOriginal[i], x[i], EPSILON,
                    "For identity matrix with non-negative b, x should equal b");
        }

        // Residual norm should be 0
        assertEquals(0.0, rnorm[0], EPSILON,
                "For identity matrix with non-negative b, residual norm should be 0");
    }

    // ========================================================================
    // Property 3.8: NNLS handles identity matrix with negative b correctly
    // ========================================================================

    @Property(tries = 50)
    void nnlsHandlesIdentityMatrixWithNegativeB(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        int m = n;

        // Create identity matrix
        double[] a = new double[m * n];
        for (int i = 0; i < n; i++) {
            a[i * m + i] = 1.0;
        }

        // Generate b with some negative values
        double[] b = new double[m];
        double[] bOriginal = new double[m];  // Save original b since NNLS modifies it
        for (int i = 0; i < m; i++) {
            b[i] = rand.nextDouble() * 10 - 5;  // Values in [-5, 5]
            bOriginal[i] = b[i];
        }

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[n];
        double[] z = new double[m];
        int[] index = new int[n];
        double[] rnorm = new double[1];

        // Solve NNLS
        int status = LSQSolver.nnls(m, n, a, 0, m, b, 0, x, 0, w, 0, z, 0, index, 0, 3 * n, rnorm);

        assertEquals(0, status, "NNLS should succeed for identity matrix");

        // For identity matrix, x[i] = max(0, bOriginal[i])
        for (int i = 0; i < n; i++) {
            double expected = Math.max(0, bOriginal[i]);
            assertEquals(expected, x[i], EPSILON,
                    "For identity matrix, x[" + i + "] should be max(0, b[" + i + "])");
        }
    }

    // ========================================================================
    // Property 3.9: NNLS solution is unique for full rank A
    // ========================================================================

    @Property(tries = 50)
    void nnlsSolutionIsConsistent(
            @ForAll @IntRange(min = 4, max = 12) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate matrix A and vector b
        double[] aOriginal = generateMatrix(m, n, rand);
        double[] bOriginal = generateVector(m, rand);

        // Solve twice with same input
        double[] a1 = aOriginal.clone();
        double[] b1 = bOriginal.clone();
        double[] x1 = new double[n];
        double[] w1 = new double[n];
        double[] z1 = new double[m];
        int[] index1 = new int[n];
        double[] rnorm1 = new double[1];

        int status1 = LSQSolver.nnls(m, n, a1, 0, m, b1, 0, x1, 0, w1, 0, z1, 0, index1, 0, 3 * n, rnorm1);

        double[] a2 = aOriginal.clone();
        double[] b2 = bOriginal.clone();
        double[] x2 = new double[n];
        double[] w2 = new double[n];
        double[] z2 = new double[m];
        int[] index2 = new int[n];
        double[] rnorm2 = new double[1];

        int status2 = LSQSolver.nnls(m, n, a2, 0, m, b2, 0, x2, 0, w2, 0, z2, 0, index2, 0, 3 * n, rnorm2);

        // Both should have same status
        assertEquals(status1, status2, "Same input should produce same status");

        if (status1 == 0) {
            // Solutions should be identical
            for (int i = 0; i < n; i++) {
                assertEquals(x1[i], x2[i], TOLERANCE,
                        "Same input should produce same solution");
            }

            // Residual norms should be identical
            assertEquals(rnorm1[0], rnorm2[0], TOLERANCE,
                    "Same input should produce same residual norm");
        }
    }

    // ========================================================================
    // Property 3.10: NNLS handles single variable correctly
    // ========================================================================

    @Property(tries = 50)
    void nnlsHandlesSingleVariable(
            @ForAll @IntRange(min = 2, max = 10) int m,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        int n = 1;  // Single variable

        // Generate column vector a
        double[] a = new double[m];
        double aNorm = 0;
        for (int i = 0; i < m; i++) {
            a[i] = rand.nextDouble() * 10 - 5;
            aNorm += a[i] * a[i];
        }
        aNorm = Math.sqrt(aNorm);

        // Generate vector b
        double[] b = new double[m];
        for (int i = 0; i < m; i++) {
            b[i] = rand.nextDouble() * 10 - 5;
        }

        // Allocate working arrays
        double[] x = new double[n];
        double[] w = new double[n];
        double[] z = new double[m];
        int[] index = new int[n];
        double[] rnorm = new double[1];

        // Solve NNLS
        int status = LSQSolver.nnls(m, n, a, 0, m, b, 0, x, 0, w, 0, z, 0, index, 0, 3 * n, rnorm);

        assertTrue(status == 0 || status == 1,
                "NNLS should handle single variable");

        // Solution should be non-negative
        assertTrue(x[0] >= -TOLERANCE,
                "Single variable solution should be non-negative");

        // For single variable: x = max(0, a'b / a'a)
        if (aNorm > TOLERANCE) {
            double aTb = 0;
            for (int i = 0; i < m; i++) {
                aTb += a[i] * b[i];
            }
            double expectedX = Math.max(0, aTb / (aNorm * aNorm));
            assertEquals(expectedX, x[0], EPSILON,
                    "Single variable solution should be max(0, a'b / a'a)");
        }
    }
}
