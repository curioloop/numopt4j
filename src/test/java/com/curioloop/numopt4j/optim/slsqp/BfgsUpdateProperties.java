/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for BFGS Update (LDL^T factorization update).
 *
 * **Property 10: BFGS Update LDL^T Form**
 */
package com.curioloop.numopt4j.optim.slsqp;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for BFGS Update (LDL^T factorization update).
 *
 * <p>Tests the following properties:</p>
 * <ul>
 *   <li>Property 10.1: Reset initializes to identity matrix (L = I, D = I)</li>
 *   <li>Property 10.2: Positive update (σ > 0) maintains positive definiteness</li>
 *   <li>Property 10.3: Negative update (σ < 0) handles potential indefiniteness</li>
 *   <li>Property 10.4: Zero update (σ = 0) leaves matrix unchanged</li>
 *   <li>Property 10.5: Update is equivalent to A' = A + σzz^T</li>
 *   <li>Property 10.6: Diagonal elements remain positive after positive update</li>
 *   <li>Property 10.7: Deterministic behavior - same input produces same output</li>
 * </ul>
 *
 * <p>The BFGS update maintains the Hessian approximation in LDL^T form:</p>
 * <ul>
 *   <li>L is a lower triangular matrix with unit diagonal elements</li>
 *   <li>D is a diagonal matrix with positive diagonal elements</li>
 *   <li>B = LDL^T is symmetric positive definite</li>
 * </ul>
 *
 * <p>The composite transformation method updates the LDL^T factors directly:</p>
 * <pre>
 *   A' = A + σzz^T = L'D'L'^T
 * </pre>
 */
@Tag("Feature: slsqp-java-rewrite, Property 10: BFGS Update LDL^T Form")
class BfgsUpdateProperties {

    private static final double EPSILON = 1e-10;
    private static final double TOLERANCE = 1e-8;

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * Compute the size of packed LDL^T storage: n*(n+1)/2
     */
    private int packedSize(int n) {
        return n * (n + 1) / 2;
    }

    /**
     * Generate a positive definite LDL^T factorization in packed form.
     * 
     * Layout: [D_0, L_10, L_20, ..., L_(n-1)0, D_1, L_21, L_31, ..., D_(n-1)]
     * where D_i are diagonal elements and L_ij are lower triangular elements.
     */
    private double[] generatePositiveDefiniteLDLT(int n, java.util.Random rand) {
        int nl = packedSize(n);
        double[] l = new double[nl];

        int idx = 0;
        for (int j = 0; j < n; j++) {
            // D_jj: positive diagonal element (ensure positive definiteness)
            l[idx++] = rand.nextDouble() * 4 + 1.0;  // Values in [1, 5]

            // L_ij for i > j: off-diagonal elements (small values for stability)
            for (int i = j + 1; i < n; i++) {
                l[idx++] = rand.nextDouble() * 0.4 - 0.2;  // Small values in [-0.2, 0.2]
            }
        }

        return l;
    }

    /**
     * Generate a random vector.
     */
    private double[] generateVector(int n, java.util.Random rand) {
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            z[i] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
        }
        return z;
    }

    /**
     * Extract the full matrix A = LDL^T from packed storage.
     * Returns an n x n matrix in row-major format.
     */
    private double[][] extractFullMatrix(double[] l, int n) {
        // First extract L and D
        double[][] L = new double[n][n];
        double[] D = new double[n];

        int idx = 0;
        for (int j = 0; j < n; j++) {
            D[j] = l[idx++];
            L[j][j] = 1.0;  // Unit diagonal
            for (int i = j + 1; i < n; i++) {
                L[i][j] = l[idx++];
            }
        }

        // Compute A = LDL^T
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += L[i][k] * D[k] * L[j][k];
                }
                A[i][j] = sum;
            }
        }

        return A;
    }

    /**
     * Extract diagonal elements D from packed LDL^T storage.
     */
    private double[] extractDiagonal(double[] l, int n) {
        double[] D = new double[n];
        int idx = 0;
        for (int j = 0; j < n; j++) {
            D[j] = l[idx];
            idx += n - j;  // Skip to next diagonal
        }
        return D;
    }

    /**
     * Compute the outer product update: A' = A + sigma * z * z^T
     */
    private double[][] outerProductUpdate(double[][] A, double[] z, double sigma) {
        int n = z.length;
        double[][] result = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = A[i][j] + sigma * z[i] * z[j];
            }
        }
        return result;
    }

    /**
     * Check if a matrix is symmetric.
     */
    private boolean isSymmetric(double[][] A, double tol) {
        int n = A.length;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (Math.abs(A[i][j] - A[j][i]) > tol) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Check if a matrix is positive definite using Cholesky decomposition attempt.
     * Returns true if all diagonal elements of the Cholesky factor are positive.
     */
    private boolean isPositiveDefinite(double[][] A) {
        int n = A.length;
        double[][] L = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0;
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }
                if (i == j) {
                    double val = A[i][i] - sum;
                    if (val <= 0) {
                        return false;
                    }
                    L[i][j] = Math.sqrt(val);
                } else {
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }
        return true;
    }

    /**
     * Compute the Frobenius norm of the difference between two matrices.
     */
    private double matrixDifferenceNorm(double[][] A, double[][] B) {
        int n = A.length;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double diff = A[i][j] - B[i][j];
                sum += diff * diff;
            }
        }
        return Math.sqrt(sum);
    }

    @Provide
    Arbitrary<Long> randomSeed() {
        return Arbitraries.longs().between(1, Long.MAX_VALUE);
    }

    // ========================================================================
    // Property 10.1: Reset initializes to identity matrix (L = I, D = I)
    // ========================================================================

    /**
     *
     * After reset, the LDL^T factorization should represent the identity matrix:
     * - L = I (identity matrix, all off-diagonal elements are zero)
     * - D = I (identity matrix, all diagonal elements are one)
     */
    @Property(tries = 100)
    void resetInitializesToIdentityMatrix(
            @ForAll @IntRange(min = 1, max = 20) int n
    ) {
        int nl = packedSize(n);
        double[] l = new double[nl];

        // Fill with random values first
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < nl; i++) {
            l[i] = rand.nextDouble() * 100 - 50;
        }

        // Reset to identity
        BfgsUpdate.reset(n, l, 0);

        // Extract full matrix and verify it's identity
        double[][] A = extractFullMatrix(l, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, A[i][j], EPSILON,
                        "After reset, A[" + i + "][" + j + "] should be " + expected);
            }
        }
    }

    /**
     *
     * After reset, diagonal elements should all be 1.0.
     */
    @Property(tries = 100)
    void resetSetsDiagonalToOne(
            @ForAll @IntRange(min = 1, max = 20) int n
    ) {
        int nl = packedSize(n);
        double[] l = new double[nl];

        // Fill with random values first
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < nl; i++) {
            l[i] = rand.nextDouble() * 100 - 50;
        }

        // Reset to identity
        BfgsUpdate.reset(n, l, 0);

        // Extract diagonal elements
        double[] D = extractDiagonal(l, n);

        for (int i = 0; i < n; i++) {
            assertEquals(1.0, D[i], EPSILON,
                    "After reset, D[" + i + "] should be 1.0");
        }
    }

    /**
     *
     * After reset, off-diagonal elements should all be 0.0.
     */
    @Property(tries = 100)
    void resetSetsOffDiagonalToZero(
            @ForAll @IntRange(min = 2, max = 20) int n
    ) {
        int nl = packedSize(n);
        double[] l = new double[nl];

        // Fill with random values first
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < nl; i++) {
            l[i] = rand.nextDouble() * 100 - 50;
        }

        // Reset to identity
        BfgsUpdate.reset(n, l, 0);

        // Check off-diagonal elements are zero
        int idx = 0;
        for (int j = 0; j < n; j++) {
            idx++;  // Skip diagonal
            for (int i = j + 1; i < n; i++) {
                assertEquals(0.0, l[idx], EPSILON,
                        "After reset, L[" + i + "][" + j + "] should be 0.0");
                idx++;
            }
        }
    }

    // ========================================================================
    // Property 10.2: Positive update (σ > 0) maintains positive definiteness
    // ========================================================================

    /**
     *
     * For a positive definite matrix A and σ > 0, the updated matrix
     * A' = A + σzz^T should remain positive definite.
     */
    @Property(tries = 100)
    void positiveUpdateMaintainsPositiveDefiniteness(
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll @DoubleRange(min = 0.1, max = 2.0) double sigma,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate positive definite LDL^T
        double[] l = generatePositiveDefiniteLDLT(n, rand);
        double[] z = generateVector(n, rand);
        double[] w = new double[n];  // Working array for negative updates

        // Verify initial matrix is positive definite
        double[][] A_before = extractFullMatrix(l, n);
        Assume.that(isPositiveDefinite(A_before));

        // Apply positive update
        BfgsUpdate.compositeT(n, l, 0, z.clone(), 0, sigma, w, 0);

        // Extract updated matrix
        double[][] A_after = extractFullMatrix(l, n);

        // Verify positive definiteness is maintained
        assertTrue(isPositiveDefinite(A_after),
                "Positive update (σ > 0) should maintain positive definiteness");
    }

    /**
     *
     * After a positive update, the matrix should remain symmetric.
     */
    @Property(tries = 100)
    void positiveUpdateMaintainsSymmetry(
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll @DoubleRange(min = 0.1, max = 2.0) double sigma,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        double[] l = generatePositiveDefiniteLDLT(n, rand);
        double[] z = generateVector(n, rand);
        double[] w = new double[n];

        // Apply positive update
        BfgsUpdate.compositeT(n, l, 0, z.clone(), 0, sigma, w, 0);

        // Extract updated matrix
        double[][] A_after = extractFullMatrix(l, n);

        // Verify symmetry
        assertTrue(isSymmetric(A_after, TOLERANCE),
                "Updated matrix should remain symmetric");
    }

    // ========================================================================
    // Property 10.3: Negative update (σ < 0) handles potential indefiniteness
    // ========================================================================

    /**
     *
     * For a negative update (σ < 0), the algorithm should handle the update
     * without crashing, even if the result might not be positive definite.
     * The algorithm uses an auxiliary vector w to handle negative updates.
     */
    @Property(tries = 100)
    void negativeUpdateHandlesIndefiniteness(
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll @DoubleRange(min = -0.5, max = -0.01) double sigma,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate positive definite LDL^T with larger diagonal for stability
        double[] l = generatePositiveDefiniteLDLT(n, rand);
        double[] z = generateVector(n, rand);
        double[] w = new double[n];  // Required for negative updates

        // Apply negative update - should not throw
        double[] zCopy = z.clone();
        BfgsUpdate.compositeT(n, l, 0, zCopy, 0, sigma, w, 0);

        // Extract updated matrix
        double[][] A_after = extractFullMatrix(l, n);

        // Verify symmetry is maintained even for negative updates
        assertTrue(isSymmetric(A_after, TOLERANCE),
                "Negative update should maintain symmetry");
    }

    /**
     *
     * For a small negative update on a strongly positive definite matrix,
     * the result should still be positive definite.
     */
    @Property(tries = 100)
    void smallNegativeUpdateOnStronglyPositiveDefiniteMatrix(
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Generate strongly positive definite matrix (large diagonal)
        int nl = packedSize(n);
        double[] l = new double[nl];
        int idx = 0;
        for (int j = 0; j < n; j++) {
            l[idx++] = 10.0 + rand.nextDouble() * 5;  // Large diagonal [10, 15]
            for (int i = j + 1; i < n; i++) {
                l[idx++] = rand.nextDouble() * 0.1 - 0.05;  // Very small off-diagonal
            }
        }

        // Small z vector
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            z[i] = rand.nextDouble() * 0.5 - 0.25;  // Small values [-0.25, 0.25]
        }

        double[] w = new double[n];

        // Verify initial matrix is positive definite
        double[][] A_before = extractFullMatrix(l, n);
        Assume.that(isPositiveDefinite(A_before));

        // Apply small negative update
        double sigma = -0.01;
        BfgsUpdate.compositeT(n, l, 0, z.clone(), 0, sigma, w, 0);

        // Extract updated matrix
        double[][] A_after = extractFullMatrix(l, n);

        // For small negative update on strongly positive definite matrix,
        // result should still be positive definite
        assertTrue(isPositiveDefinite(A_after),
                "Small negative update on strongly positive definite matrix should remain positive definite");
    }

    // ========================================================================
    // Property 10.4: Zero update (σ = 0) leaves matrix unchanged
    // ========================================================================

    /**
     *
     * When σ = 0, the update A' = A + 0*zz^T = A should leave the matrix unchanged.
     */
    @Property(tries = 100)
    void zeroUpdateLeavesMatrixUnchanged(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        double[] l = generatePositiveDefiniteLDLT(n, rand);
        double[] lOriginal = l.clone();
        double[] z = generateVector(n, rand);
        double[] w = new double[n];

        // Apply zero update
        BfgsUpdate.compositeT(n, l, 0, z.clone(), 0, 0.0, w, 0);

        // Matrix should be unchanged
        for (int i = 0; i < l.length; i++) {
            assertEquals(lOriginal[i], l[i], EPSILON,
                    "Zero update should leave l[" + i + "] unchanged");
        }
    }

    /**
     *
     * When σ = 0, the full matrix A should be identical before and after.
     */
    @Property(tries = 100)
    void zeroUpdatePreservesFullMatrix(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        double[] l = generatePositiveDefiniteLDLT(n, rand);
        double[][] A_before = extractFullMatrix(l, n);
        double[] z = generateVector(n, rand);
        double[] w = new double[n];

        // Apply zero update
        BfgsUpdate.compositeT(n, l, 0, z.clone(), 0, 0.0, w, 0);

        double[][] A_after = extractFullMatrix(l, n);

        // Matrices should be identical
        double diff = matrixDifferenceNorm(A_before, A_after);
        assertEquals(0.0, diff, EPSILON,
                "Zero update should preserve the full matrix");
    }

    // ========================================================================
    // Property 10.5: Update is equivalent to A' = A + σzz^T
    // ========================================================================

    /**
     *
     * The composite transformation update should be mathematically equivalent
     * to the rank-1 update A' = A + σzz^T.
     */
    @Property(tries = 100)
    void updateIsEquivalentToRankOneUpdate(
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @DoubleRange(min = 0.1, max = 2.0) double sigma,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        double[] l = generatePositiveDefiniteLDLT(n, rand);
        double[] z = generateVector(n, rand);
        double[] w = new double[n];

        // Extract matrix before update
        double[][] A_before = extractFullMatrix(l, n);

        // Compute expected result using direct outer product update
        double[][] A_expected = outerProductUpdate(A_before, z, sigma);

        // Apply composite transformation update
        BfgsUpdate.compositeT(n, l, 0, z.clone(), 0, sigma, w, 0);

        // Extract matrix after update
        double[][] A_after = extractFullMatrix(l, n);

        // Compare with expected result
        double diff = matrixDifferenceNorm(A_expected, A_after);
        assertTrue(diff < TOLERANCE,
                "Composite transformation should be equivalent to A' = A + σzz^T, diff = " + diff);
    }

    /**
     *
     * For identity matrix (after reset), the update should give I + σzz^T.
     */
    @Property(tries = 100)
    void updateOnIdentityMatrixGivesExpectedResult(
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll @DoubleRange(min = 0.1, max = 2.0) double sigma,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        int nl = packedSize(n);
        double[] l = new double[nl];
        double[] z = generateVector(n, rand);
        double[] w = new double[n];

        // Reset to identity
        BfgsUpdate.reset(n, l, 0);

        // Compute expected: I + σzz^T
        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++) {
            I[i][i] = 1.0;
        }
        double[][] A_expected = outerProductUpdate(I, z, sigma);

        // Apply update
        BfgsUpdate.compositeT(n, l, 0, z.clone(), 0, sigma, w, 0);

        // Extract result
        double[][] A_after = extractFullMatrix(l, n);

        // Compare
        double diff = matrixDifferenceNorm(A_expected, A_after);
        assertTrue(diff < TOLERANCE,
                "Update on identity should give I + σzz^T, diff = " + diff);
    }

    // ========================================================================
    // Property 10.6: Diagonal elements remain positive after positive update
    // ========================================================================

    /**
     *
     * For a positive update (σ > 0) on a positive definite matrix,
     * the diagonal elements of D should remain positive.
     */
    @Property(tries = 100)
    void diagonalElementsRemainPositiveAfterPositiveUpdate(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @DoubleRange(min = 0.1, max = 2.0) double sigma,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        double[] l = generatePositiveDefiniteLDLT(n, rand);
        double[] z = generateVector(n, rand);
        double[] w = new double[n];

        // Verify initial diagonal elements are positive
        double[] D_before = extractDiagonal(l, n);
        for (int i = 0; i < n; i++) {
            Assume.that(D_before[i] > 0);
        }

        // Apply positive update
        BfgsUpdate.compositeT(n, l, 0, z.clone(), 0, sigma, w, 0);

        // Extract diagonal elements after update
        double[] D_after = extractDiagonal(l, n);

        // All diagonal elements should remain positive
        for (int i = 0; i < n; i++) {
            assertTrue(D_after[i] > 0,
                    "Diagonal element D[" + i + "] = " + D_after[i] + 
                    " should remain positive after positive update");
        }
    }

    /**
     *
     * After reset, all diagonal elements should be exactly 1.0.
     */
    @Property(tries = 100)
    void diagonalElementsAreOneAfterReset(
            @ForAll @IntRange(min = 1, max = 20) int n
    ) {
        int nl = packedSize(n);
        double[] l = new double[nl];

        // Fill with random values
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < nl; i++) {
            l[i] = rand.nextDouble() * 100;
        }

        // Reset
        BfgsUpdate.reset(n, l, 0);

        // Check diagonal elements
        double[] D = extractDiagonal(l, n);
        for (int i = 0; i < n; i++) {
            assertEquals(1.0, D[i], EPSILON,
                    "After reset, D[" + i + "] should be 1.0");
        }
    }

    // ========================================================================
    // Property 10.7: Deterministic behavior - same input produces same output
    // ========================================================================

    /**
     *
     * Running the update twice with the same input should produce identical results.
     */
    @Property(tries = 100)
    void updateIsDeterministic(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @DoubleRange(min = -1.0, max = 2.0) double sigma,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        double[] l1 = generatePositiveDefiniteLDLT(n, rand);
        double[] l2 = l1.clone();
        
        // Reset random to get same z vector
        rand = new java.util.Random(seed + 1);
        double[] z1 = generateVector(n, rand);
        rand = new java.util.Random(seed + 1);
        double[] z2 = generateVector(n, rand);
        
        double[] w1 = new double[n];
        double[] w2 = new double[n];

        // Apply update to both
        BfgsUpdate.compositeT(n, l1, 0, z1, 0, sigma, w1, 0);
        BfgsUpdate.compositeT(n, l2, 0, z2, 0, sigma, w2, 0);

        // Results should be identical
        for (int i = 0; i < l1.length; i++) {
            assertEquals(l1[i], l2[i], EPSILON,
                    "Deterministic: l[" + i + "] should be identical");
        }
    }

    /**
     *
     * Reset should always produce the same result regardless of initial state.
     */
    @Property(tries = 100)
    void resetIsDeterministic(
            @ForAll @IntRange(min = 1, max = 20) int n,
            @ForAll("randomSeed") long seed
    ) {
        int nl = packedSize(n);
        
        // Create two arrays with different initial values
        java.util.Random rand1 = new java.util.Random(seed);
        java.util.Random rand2 = new java.util.Random(seed + 12345);
        
        double[] l1 = new double[nl];
        double[] l2 = new double[nl];
        
        for (int i = 0; i < nl; i++) {
            l1[i] = rand1.nextDouble() * 100;
            l2[i] = rand2.nextDouble() * 100;
        }

        // Reset both
        BfgsUpdate.reset(n, l1, 0);
        BfgsUpdate.reset(n, l2, 0);

        // Results should be identical
        for (int i = 0; i < nl; i++) {
            assertEquals(l1[i], l2[i], EPSILON,
                    "Reset should produce identical results regardless of initial state");
        }
    }

    // ========================================================================
    // Property 10.8: Edge cases
    // ========================================================================

    /**
     *
     * Reset should handle n = 1 correctly (single element matrix).
     */
    @Property(tries = 50)
    void resetHandlesSingleElementMatrix() {
        double[] l = new double[1];
        l[0] = 999.0;

        BfgsUpdate.reset(1, l, 0);

        assertEquals(1.0, l[0], EPSILON,
                "Reset should set single element to 1.0");
    }

    /**
     *
     * Update should handle n = 1 correctly.
     */
    @Property(tries = 50)
    void updateHandlesSingleElementMatrix(
            @ForAll @DoubleRange(min = 0.1, max = 2.0) double sigma,
            @ForAll @DoubleRange(min = -2.0, max = 2.0) double z
    ) {
        double[] l = new double[1];
        l[0] = 1.0;  // Start with identity

        double[] zArr = {z};
        double[] w = new double[1];

        // Expected: A' = 1 + sigma * z^2
        double expected = 1.0 + sigma * z * z;

        BfgsUpdate.compositeT(1, l, 0, zArr, 0, sigma, w, 0);

        assertEquals(expected, l[0], TOLERANCE,
                "Single element update should give 1 + σz²");
    }

    /**
     *
     * Update with zero vector z should leave matrix unchanged.
     */
    @Property(tries = 50)
    void updateWithZeroVectorLeavesMatrixUnchanged(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @DoubleRange(min = 0.1, max = 2.0) double sigma,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        double[] l = generatePositiveDefiniteLDLT(n, rand);
        double[] lOriginal = l.clone();
        double[] z = new double[n];  // Zero vector
        double[] w = new double[n];

        BfgsUpdate.compositeT(n, l, 0, z, 0, sigma, w, 0);

        // Matrix should be unchanged (since zz^T = 0)
        for (int i = 0; i < l.length; i++) {
            assertEquals(lOriginal[i], l[i], EPSILON,
                    "Update with zero vector should leave l[" + i + "] unchanged");
        }
    }

    /**
     *
     * Reset should handle n = 0 gracefully (no-op).
     */
    @Property(tries = 10)
    void resetHandlesZeroDimension() {
        double[] l = new double[10];
        for (int i = 0; i < 10; i++) {
            l[i] = 999.0;
        }

        // Should not crash
        BfgsUpdate.reset(0, l, 0);

        // Array should be unchanged
        for (int i = 0; i < 10; i++) {
            assertEquals(999.0, l[i], EPSILON,
                    "Reset with n=0 should not modify array");
        }
    }

    /**
     *
     * Update should handle n = 0 gracefully (no-op).
     */
    @Property(tries = 10)
    void updateHandlesZeroDimension() {
        double[] l = new double[10];
        for (int i = 0; i < 10; i++) {
            l[i] = 999.0;
        }
        double[] z = new double[10];
        double[] w = new double[10];

        // Should not crash
        BfgsUpdate.compositeT(0, l, 0, z, 0, 1.0, w, 0);

        // Array should be unchanged
        for (int i = 0; i < 10; i++) {
            assertEquals(999.0, l[i], EPSILON,
                    "Update with n=0 should not modify array");
        }
    }

    // ========================================================================
    // Property 10.9: Offset handling
    // ========================================================================

    /**
     *
     * Reset should correctly handle non-zero offset.
     */
    @Property(tries = 50)
    void resetHandlesOffset(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 1, max = 5) int offset
    ) {
        int nl = packedSize(n);
        double[] l = new double[nl + offset + 5];

        // Fill with marker values
        for (int i = 0; i < l.length; i++) {
            l[i] = 999.0;
        }

        // Reset with offset
        BfgsUpdate.reset(n, l, offset);

        // Check values before offset are unchanged
        for (int i = 0; i < offset; i++) {
            assertEquals(999.0, l[i], EPSILON,
                    "Values before offset should be unchanged");
        }

        // Check values after the LDL^T region are unchanged
        for (int i = offset + nl; i < l.length; i++) {
            assertEquals(999.0, l[i], EPSILON,
                    "Values after LDL^T region should be unchanged");
        }

        // Check diagonal elements are 1.0
        int idx = offset;
        for (int j = 0; j < n; j++) {
            assertEquals(1.0, l[idx], EPSILON,
                    "Diagonal element D[" + j + "] should be 1.0");
            idx += n - j;
        }
    }

    /**
     *
     * Update should correctly handle non-zero offsets for l and z.
     */
    @Property(tries = 50)
    void updateHandlesOffsets(
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @IntRange(min = 1, max = 3) int lOffset,
            @ForAll @IntRange(min = 1, max = 3) int zOffset,
            @ForAll @DoubleRange(min = 0.1, max = 1.0) double sigma,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        int nl = packedSize(n);
        double[] l = new double[nl + lOffset + 5];
        double[] z = new double[n + zOffset + 5];
        double[] w = new double[n + 5];

        // Fill with marker values
        for (int i = 0; i < l.length; i++) {
            l[i] = 888.0;
        }
        for (int i = 0; i < z.length; i++) {
            z[i] = 777.0;
        }

        // Generate LDL^T at offset
        double[] lTemp = generatePositiveDefiniteLDLT(n, rand);
        System.arraycopy(lTemp, 0, l, lOffset, nl);

        // Generate z at offset
        double[] zTemp = generateVector(n, rand);
        System.arraycopy(zTemp, 0, z, zOffset, n);

        // Apply update with offsets
        BfgsUpdate.compositeT(n, l, lOffset, z, zOffset, sigma, w, 0);

        // Check values before l offset are unchanged
        for (int i = 0; i < lOffset; i++) {
            assertEquals(888.0, l[i], EPSILON,
                    "Values before l offset should be unchanged");
        }

        // Check values after l region are unchanged
        for (int i = lOffset + nl; i < l.length; i++) {
            assertEquals(888.0, l[i], EPSILON,
                    "Values after l region should be unchanged");
        }

        // Check values before z offset are unchanged
        for (int i = 0; i < zOffset; i++) {
            assertEquals(777.0, z[i], EPSILON,
                    "Values before z offset should be unchanged");
        }

        // Check values after z region are unchanged
        for (int i = zOffset + n; i < z.length; i++) {
            assertEquals(777.0, z[i], EPSILON,
                    "Values after z region should be unchanged");
        }
    }

    // ========================================================================
    // Property 10.10: Consecutive updates
    // ========================================================================

    /**
     *
     * Multiple consecutive positive updates should maintain positive definiteness.
     */
    @Property(tries = 50)
    void consecutivePositiveUpdatesMaintainPositiveDefiniteness(
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @IntRange(min = 2, max = 5) int numUpdates,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        int nl = packedSize(n);
        double[] l = new double[nl];
        double[] w = new double[n];

        // Start with identity
        BfgsUpdate.reset(n, l, 0);

        // Apply multiple positive updates
        for (int k = 0; k < numUpdates; k++) {
            double[] z = generateVector(n, rand);
            double sigma = rand.nextDouble() * 0.5 + 0.1;  // σ in [0.1, 0.6]

            BfgsUpdate.compositeT(n, l, 0, z.clone(), 0, sigma, w, 0);

            // Verify positive definiteness after each update
            double[][] A = extractFullMatrix(l, n);
            assertTrue(isPositiveDefinite(A),
                    "Matrix should remain positive definite after update " + (k + 1));
        }
    }

    /**
     *
     * Two updates with opposite signs should approximately cancel out.
     * A + σzz^T - σzz^T ≈ A
     */
    @Property(tries = 50)
    void oppositeUpdatesApproximatelyCancel(
            @ForAll @IntRange(min = 2, max = 6) int n,
            @ForAll @DoubleRange(min = 0.1, max = 0.5) double sigma,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        double[] l = generatePositiveDefiniteLDLT(n, rand);
        double[][] A_original = extractFullMatrix(l, n);
        
        // Ensure original is positive definite
        Assume.that(isPositiveDefinite(A_original));

        double[] z = generateVector(n, rand);
        double[] w = new double[n];

        // Apply positive update
        BfgsUpdate.compositeT(n, l, 0, z.clone(), 0, sigma, w, 0);

        // Apply negative update with same z
        // Note: z is modified by compositeT, so we need to use original z
        rand = new java.util.Random(seed);
        generatePositiveDefiniteLDLT(n, rand);  // Skip to same state
        double[] z2 = generateVector(n, rand);
        
        BfgsUpdate.compositeT(n, l, 0, z2.clone(), 0, -sigma, w, 0);

        // Result should be approximately equal to original
        double[][] A_final = extractFullMatrix(l, n);
        double diff = matrixDifferenceNorm(A_original, A_final);

        // Due to numerical precision, we use a relaxed tolerance
        assertTrue(diff < 1e-6,
                "Opposite updates should approximately cancel, diff = " + diff);
    }
}
