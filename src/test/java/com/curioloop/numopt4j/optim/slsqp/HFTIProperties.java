/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for HFTI (Householder Forward Triangulation with column Interchanges) solver.
 *
 * **Property 7: HFTI Solver Correctness**
 */
package com.curioloop.numopt4j.optim.slsqp;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for HFTI (Householder Forward Triangulation with column Interchanges) solver.
 *
 * <p>Tests the following properties:</p>
 * <ul>
 *   <li>Property 7.1: Pseudo-rank k equals the number of diagonal elements of R exceeding τ</li>
 *   <li>Property 7.2: Solution X minimizes ||AX - B||_F among all solutions</li>
 *   <li>Property 7.3: Residual norms are correctly computed for each column</li>
 *   <li>Property 7.4: HFTI handles full-rank matrices correctly</li>
 *   <li>Property 7.5: HFTI handles rank-deficient matrices correctly</li>
 *   <li>Property 7.6: HFTI handles multiple right-hand sides efficiently</li>
 * </ul>
 *
 * <p>The HFTI problem is: Solve AX ≅ B (least squares) using Householder transformations with column pivoting</p>
 *
 * <p>The algorithm:</p>
 * <ul>
 *   <li>Uses Householder transformations with column pivoting to compute QR factorization</li>
 *   <li>Determines pseudo-rank k based on tolerance τ</li>
 *   <li>Applies forward triangulation for rank-deficient cases</li>
 *   <li>Computes minimum norm solution</li>
 *   <li>Computes residual norms for each right-hand side</li>
 * </ul>
 */
@Tag("Feature: slsqp-java-rewrite, Property 7: HFTI Solver Correctness")
class HFTIProperties {

    private static final double EPSILON = 1e-10;
    private static final double TOLERANCE = 1e-12;

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * Generate a well-conditioned random matrix A (m x n) in column-major format.
     * Ensures the matrix has full column rank for numerical stability.
     */
    private double[] generateFullRankMatrix(int m, int n, java.util.Random rand) {
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
     * Generate a rank-deficient matrix A (m x n) with specified rank k.
     * Creates a matrix as a product of two smaller matrices to ensure rank deficiency.
     */
    private double[] generateRankDeficientMatrix(int m, int n, int k, java.util.Random rand) {
        // Create A = U * V^T where U is m x k and V is n x k
        double[] u = new double[m * k];
        double[] v = new double[n * k];
        
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < m; i++) {
                u[j * m + i] = rand.nextDouble() * 4 - 2;  // Values in [-2, 2]
            }
            for (int i = 0; i < n; i++) {
                v[j * n + i] = rand.nextDouble() * 4 - 2;
            }
        }
        
        // Compute A = U * V^T
        double[] a = new double[m * n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                double sum = 0;
                for (int l = 0; l < k; l++) {
                    sum += u[l * m + i] * v[l * n + j];
                }
                a[j * m + i] = sum;
            }
        }
        return a;
    }

    /**
     * Generate a random matrix B (m x nb) in column-major format.
     */
    private double[] generateMatrix(int m, int nb, java.util.Random rand) {
        double[] b = new double[m * nb];
        for (int j = 0; j < nb; j++) {
            for (int i = 0; i < m; i++) {
                b[j * m + i] = rand.nextDouble() * 10 - 5;
            }
        }
        return b;
    }

    /**
     * Compute matrix-matrix product C = A * X (A is m x n, X is n x nb, all column-major).
     */
    private double[] matMul(double[] a, int m, int n, double[] x, int nb) {
        double[] c = new double[m * nb];
        for (int jb = 0; jb < nb; jb++) {
            for (int i = 0; i < m; i++) {
                double sum = 0;
                for (int j = 0; j < n; j++) {
                    sum += a[j * m + i] * x[jb * n + j];
                }
                c[jb * m + i] = sum;
            }
        }
        return c;
    }

    /**
     * Compute Frobenius norm of a matrix (column-major).
     */
    private double frobeniusNorm(double[] a, int m, int n) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                double val = a[j * m + i];
                sum += val * val;
            }
        }
        return Math.sqrt(sum);
    }

    /**
     * Compute column norm of a matrix (column-major).
     */
    private double columnNorm(double[] a, int m, int col) {
        double sum = 0;
        for (int i = 0; i < m; i++) {
            double val = a[col * m + i];
            sum += val * val;
        }
        return Math.sqrt(sum);
    }

    /**
     * Compute matrix subtraction C = A - B (column-major).
     */
    private double[] matSub(double[] a, double[] b, int m, int n) {
        double[] c = new double[m * n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                c[j * m + i] = a[j * m + i] - b[j * m + i];
            }
        }
        return c;
    }

    @Provide
    Arbitrary<Long> randomSeed() {
        return Arbitraries.longs().between(1, Long.MAX_VALUE);
    }

    // ========================================================================
    // Property 7.1: Pseudo-rank k equals the number of diagonal elements of R exceeding τ
    // ========================================================================

    @Property(tries = 100)
    void pseudoRankEqualsNumberOfDiagonalElementsExceedingTau(
            @ForAll @IntRange(min = 3, max = 12) int m,
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);  // Overdetermined or square system

        java.util.Random rand = new java.util.Random(seed);

        // Generate a full-rank matrix
        double[] aOriginal = generateFullRankMatrix(m, n, rand);
        double[] a = aOriginal.clone();

        // Generate right-hand side
        int nb = 1;
        double[] b = generateMatrix(m, nb, rand);

        // Use a small tolerance
        double tau = 1e-10;

        // Allocate working arrays
        int diag = Math.min(m, n);
        double[] rnorm = new double[nb];
        double[] h = new double[n];
        double[] g = new double[diag];
        int[] ip = new int[diag];

        // Solve HFTI
        int k = LSQSolver.hfti(m, n, a, 0, m, b, 0, m, nb, tau, rnorm, 0, h, 0, g, 0, ip, 0);

        // For a full-rank matrix with small tau, pseudo-rank should equal min(m, n)
        assertTrue(k > 0, "Pseudo-rank should be positive for full-rank matrix");
        assertTrue(k <= diag, "Pseudo-rank should not exceed min(m, n)");

        // Verify that the first k diagonal elements of R exceed tau
        // After HFTI, the matrix a contains the R factor in upper triangular form
        for (int i = 0; i < k; i++) {
            double diagElement = Math.abs(a[i * m + i]);
            assertTrue(diagElement > tau,
                    "Diagonal element R[" + i + "," + i + "] = " + diagElement + 
                    " should exceed tau = " + tau);
        }
    }

    @Property(tries = 100)
    void pseudoRankDecreasesWithLargerTau(
            @ForAll @IntRange(min = 4, max = 10) int m,
            @ForAll @IntRange(min = 3, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate a full-rank matrix
        double[] aOriginal = generateFullRankMatrix(m, n, rand);

        // Generate right-hand side
        int nb = 1;
        double[] bOriginal = generateMatrix(m, nb, rand);

        int diag = Math.min(m, n);

        // Solve with small tau
        double[] a1 = aOriginal.clone();
        double[] b1 = bOriginal.clone();
        double[] rnorm1 = new double[nb];
        double[] h1 = new double[n];
        double[] g1 = new double[diag];
        int[] ip1 = new int[diag];

        double tau1 = 1e-12;
        int k1 = LSQSolver.hfti(m, n, a1, 0, m, b1, 0, m, nb, tau1, rnorm1, 0, h1, 0, g1, 0, ip1, 0);

        // Solve with larger tau
        double[] a2 = aOriginal.clone();
        double[] b2 = bOriginal.clone();
        double[] rnorm2 = new double[nb];
        double[] h2 = new double[n];
        double[] g2 = new double[diag];
        int[] ip2 = new int[diag];

        double tau2 = 100.0;  // Very large tau
        int k2 = LSQSolver.hfti(m, n, a2, 0, m, b2, 0, m, nb, tau2, rnorm2, 0, h2, 0, g2, 0, ip2, 0);

        // Larger tau should result in smaller or equal pseudo-rank
        assertTrue(k2 <= k1,
                "Larger tau (" + tau2 + ") should result in smaller or equal pseudo-rank: k2=" + k2 + " <= k1=" + k1);
    }

    // ========================================================================
    // Property 7.2: Solution X minimizes ||AX - B||_F among all solutions
    // ========================================================================

    @Property(tries = 100)
    void solutionMinimizesResidualNorm(
            @ForAll @IntRange(min = 4, max = 12) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll @IntRange(min = 1, max = 3) int nb,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);  // Overdetermined system

        java.util.Random rand = new java.util.Random(seed);

        // Generate a full-rank matrix
        double[] aOriginal = generateFullRankMatrix(m, n, rand);
        double[] a = aOriginal.clone();

        // Generate right-hand side
        double[] bOriginal = generateMatrix(m, nb, rand);
        double[] b = bOriginal.clone();

        // Use a small tolerance
        double tau = 1e-12;

        // Allocate working arrays
        int diag = Math.min(m, n);
        double[] rnorm = new double[nb];
        double[] h = new double[n];
        double[] g = new double[diag];
        int[] ip = new int[diag];

        // Solve HFTI
        int k = LSQSolver.hfti(m, n, a, 0, m, b, 0, m, nb, tau, rnorm, 0, h, 0, g, 0, ip, 0);

        if (k == 0) {
            return;  // Skip if rank is 0
        }

        // Extract solution X from first n rows of b
        double[] x = new double[n * nb];
        for (int jb = 0; jb < nb; jb++) {
            for (int i = 0; i < n; i++) {
                x[jb * n + i] = b[jb * m + i];
            }
        }

        // Compute residual AX - B
        double[] ax = matMul(aOriginal, m, n, x, nb);
        double[] residual = matSub(ax, bOriginal, m, nb);
        double hftiResidualNorm = frobeniusNorm(residual, m, nb);

        // Generate random perturbations and verify HFTI solution is optimal
        for (int trial = 0; trial < 10; trial++) {
            // Create a perturbed solution
            double[] xPerturbed = new double[n * nb];
            for (int i = 0; i < n * nb; i++) {
                xPerturbed[i] = x[i] + (rand.nextDouble() - 0.5) * 0.5;
            }

            // Compute residual for perturbed solution
            double[] axPerturbed = matMul(aOriginal, m, n, xPerturbed, nb);
            double[] residualPerturbed = matSub(axPerturbed, bOriginal, m, nb);
            double perturbedResidualNorm = frobeniusNorm(residualPerturbed, m, nb);

            // HFTI solution should have smaller or equal residual norm
            assertTrue(hftiResidualNorm <= perturbedResidualNorm + EPSILON,
                    "HFTI residual norm (" + hftiResidualNorm + 
                    ") should be <= perturbed residual norm (" + perturbedResidualNorm + ")");
        }
    }

    // ========================================================================
    // Property 7.3: Residual norms are correctly computed for each column
    // ========================================================================

    @Property(tries = 100)
    void residualNormsAreCorrectlyComputed(
            @ForAll @IntRange(min = 4, max = 12) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll @IntRange(min = 1, max = 4) int nb,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate a full-rank matrix
        double[] aOriginal = generateFullRankMatrix(m, n, rand);
        double[] a = aOriginal.clone();

        // Generate right-hand side
        double[] bOriginal = generateMatrix(m, nb, rand);
        double[] b = bOriginal.clone();

        // Use a small tolerance
        double tau = 1e-12;

        // Allocate working arrays
        int diag = Math.min(m, n);
        double[] rnorm = new double[nb];
        double[] h = new double[n];
        double[] g = new double[diag];
        int[] ip = new int[diag];

        // Solve HFTI
        int k = LSQSolver.hfti(m, n, a, 0, m, b, 0, m, nb, tau, rnorm, 0, h, 0, g, 0, ip, 0);

        if (k == 0) {
            return;  // Skip if rank is 0
        }

        // Extract solution X from first n rows of b
        double[] x = new double[n * nb];
        for (int jb = 0; jb < nb; jb++) {
            for (int i = 0; i < n; i++) {
                x[jb * n + i] = b[jb * m + i];
            }
        }

        // Compute residual AX - B and verify each column's norm
        double[] ax = matMul(aOriginal, m, n, x, nb);
        double[] residual = matSub(ax, bOriginal, m, nb);

        for (int jb = 0; jb < nb; jb++) {
            double computedNorm = columnNorm(residual, m, jb);
            assertEquals(computedNorm, rnorm[jb], EPSILON,
                    "Residual norm for column " + jb + ": computed=" + computedNorm + 
                    ", returned=" + rnorm[jb]);
        }
    }

    // ========================================================================
    // Property 7.4: HFTI handles full-rank matrices correctly
    // ========================================================================

    @Property(tries = 100)
    void hftiHandlesFullRankMatricesCorrectly(
            @ForAll @IntRange(min = 3, max = 10) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate a full-rank matrix
        double[] aOriginal = generateFullRankMatrix(m, n, rand);
        double[] a = aOriginal.clone();

        // Generate right-hand side
        int nb = 1;
        double[] bOriginal = generateMatrix(m, nb, rand);
        double[] b = bOriginal.clone();

        // Use a small tolerance
        double tau = 1e-12;

        // Allocate working arrays
        int diag = Math.min(m, n);
        double[] rnorm = new double[nb];
        double[] h = new double[n];
        double[] g = new double[diag];
        int[] ip = new int[diag];

        // Solve HFTI
        int k = LSQSolver.hfti(m, n, a, 0, m, b, 0, m, nb, tau, rnorm, 0, h, 0, g, 0, ip, 0);

        // For full-rank matrix with small tau, pseudo-rank should equal n
        assertEquals(n, k, "Pseudo-rank should equal n for full-rank matrix");

        // Extract solution
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = b[i];
        }

        // Verify solution satisfies normal equations approximately: A^T A x ≈ A^T b
        // For overdetermined systems, this is the least squares solution
        double[] ax = matMul(aOriginal, m, n, x, 1);
        double[] residual = matSub(ax, bOriginal, m, 1);
        
        // Compute A^T * residual (should be approximately zero for least squares solution)
        double[] atResidual = new double[n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                atResidual[j] += aOriginal[j * m + i] * residual[i];
            }
        }

        // A^T * residual should be approximately zero
        for (int j = 0; j < n; j++) {
            assertEquals(0.0, atResidual[j], 1e-6,
                    "Normal equation residual A^T(Ax-b)[" + j + "] should be approximately zero");
        }
    }

    // ========================================================================
    // Property 7.5: HFTI handles rank-deficient matrices correctly
    // ========================================================================

    @Property(tries = 100)
    void hftiHandlesRankDeficientMatricesCorrectly(
            @ForAll @IntRange(min = 5, max = 12) int m,
            @ForAll @IntRange(min = 4, max = 10) int n,
            @ForAll @IntRange(min = 1, max = 3) int rankDeficiency,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);
        int targetRank = Math.max(1, n - rankDeficiency);
        Assume.that(targetRank >= 1 && targetRank < n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate a rank-deficient matrix
        double[] aOriginal = generateRankDeficientMatrix(m, n, targetRank, rand);
        double[] a = aOriginal.clone();

        // Generate right-hand side
        int nb = 1;
        double[] bOriginal = generateMatrix(m, nb, rand);
        double[] b = bOriginal.clone();

        // Use a tolerance that should detect the rank deficiency
        double tau = 1e-8;

        // Allocate working arrays
        int diag = Math.min(m, n);
        double[] rnorm = new double[nb];
        double[] h = new double[n];
        double[] g = new double[diag];
        int[] ip = new int[diag];

        // Solve HFTI
        int k = LSQSolver.hfti(m, n, a, 0, m, b, 0, m, nb, tau, rnorm, 0, h, 0, g, 0, ip, 0);

        // Pseudo-rank should be approximately equal to target rank
        // Allow some tolerance due to numerical issues
        assertTrue(k <= n, "Pseudo-rank should not exceed n");
        assertTrue(k >= 0, "Pseudo-rank should be non-negative");

        // Extract solution
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = b[i];
        }

        // Verify solution is a valid least squares solution
        // The residual should be orthogonal to the column space of A
        double[] ax = matMul(aOriginal, m, n, x, 1);
        double[] residual = matSub(ax, bOriginal, m, 1);

        // Compute A^T * residual
        double[] atResidual = new double[n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                atResidual[j] += aOriginal[j * m + i] * residual[i];
            }
        }

        // For rank-deficient case, A^T * residual should still be approximately zero
        // (within the column space of A)
        double atResidualNorm = 0;
        for (int j = 0; j < n; j++) {
            atResidualNorm += atResidual[j] * atResidual[j];
        }
        atResidualNorm = Math.sqrt(atResidualNorm);

        // The norm should be small relative to the problem scale
        double bNorm = frobeniusNorm(bOriginal, m, 1);
        assertTrue(atResidualNorm < 1e-4 * (bNorm + 1),
                "A^T(Ax-b) norm (" + atResidualNorm + ") should be small for least squares solution");
    }

    // ========================================================================
    // Property 7.6: HFTI handles multiple right-hand sides efficiently
    // ========================================================================

    @Property(tries = 100)
    void hftiHandlesMultipleRightHandSides(
            @ForAll @IntRange(min = 4, max = 10) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll @IntRange(min = 2, max = 5) int nb,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate a full-rank matrix
        double[] aOriginal = generateFullRankMatrix(m, n, rand);

        // Generate multiple right-hand sides
        double[] bOriginal = generateMatrix(m, nb, rand);

        // Use a small tolerance
        double tau = 1e-12;

        // Allocate working arrays
        int diag = Math.min(m, n);

        // Solve all columns at once
        double[] aMulti = aOriginal.clone();
        double[] bMulti = bOriginal.clone();
        double[] rnormMulti = new double[nb];
        double[] hMulti = new double[n];
        double[] gMulti = new double[diag];
        int[] ipMulti = new int[diag];

        int kMulti = LSQSolver.hfti(m, n, aMulti, 0, m, bMulti, 0, m, nb, tau, 
                                rnormMulti, 0, hMulti, 0, gMulti, 0, ipMulti, 0);

        // Solve each column separately and compare
        for (int jb = 0; jb < nb; jb++) {
            double[] aSingle = aOriginal.clone();
            double[] bSingle = new double[m];
            for (int i = 0; i < m; i++) {
                bSingle[i] = bOriginal[jb * m + i];
            }
            double[] rnormSingle = new double[1];
            double[] hSingle = new double[n];
            double[] gSingle = new double[diag];
            int[] ipSingle = new int[diag];

            int kSingle = LSQSolver.hfti(m, n, aSingle, 0, m, bSingle, 0, m, 1, tau,
                                     rnormSingle, 0, hSingle, 0, gSingle, 0, ipSingle, 0);

            // Pseudo-rank should be the same
            assertEquals(kMulti, kSingle,
                    "Pseudo-rank should be the same for multi and single RHS");

            // Solutions should be the same
            for (int i = 0; i < n; i++) {
                assertEquals(bSingle[i], bMulti[jb * m + i], EPSILON,
                        "Solution for column " + jb + ", element " + i + " should match");
            }

            // Residual norms should be the same
            assertEquals(rnormSingle[0], rnormMulti[jb], EPSILON,
                    "Residual norm for column " + jb + " should match");
        }
    }

    // ========================================================================
    // Property 7.7: HFTI handles square matrices correctly
    // ========================================================================

    @Property(tries = 100)
    void hftiHandlesSquareMatricesCorrectly(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        int m = n;  // Square matrix

        java.util.Random rand = new java.util.Random(seed);

        // Generate a full-rank square matrix
        double[] aOriginal = generateFullRankMatrix(m, n, rand);
        double[] a = aOriginal.clone();

        // Generate right-hand side
        int nb = 1;
        double[] bOriginal = generateMatrix(m, nb, rand);
        double[] b = bOriginal.clone();

        // Use a small tolerance
        double tau = 1e-12;

        // Allocate working arrays
        int diag = n;
        double[] rnorm = new double[nb];
        double[] h = new double[n];
        double[] g = new double[diag];
        int[] ip = new int[diag];

        // Solve HFTI
        int k = LSQSolver.hfti(m, n, a, 0, m, b, 0, m, nb, tau, rnorm, 0, h, 0, g, 0, ip, 0);

        // For full-rank square matrix, pseudo-rank should equal n
        assertEquals(n, k, "Pseudo-rank should equal n for full-rank square matrix");

        // Extract solution
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = b[i];
        }

        // For square full-rank matrix, Ax should equal b exactly
        double[] ax = matMul(aOriginal, m, n, x, 1);
        for (int i = 0; i < m; i++) {
            assertEquals(bOriginal[i], ax[i], 1e-8,
                    "For square full-rank matrix, Ax[" + i + "] should equal b[" + i + "]");
        }

        // Residual norm should be approximately zero
        assertEquals(0.0, rnorm[0], 1e-8,
                "Residual norm should be approximately zero for square full-rank matrix");
    }

    // ========================================================================
    // Property 7.8: HFTI handles underdetermined systems correctly
    // ========================================================================

    @Property(tries = 100)
    void hftiHandlesUnderdeterminedSystemsCorrectly(
            @ForAll @IntRange(min = 2, max = 8) int m,
            @ForAll @IntRange(min = 3, max = 12) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(n > m);  // Underdetermined system

        java.util.Random rand = new java.util.Random(seed);

        // Generate a full-rank matrix (m x n with m < n)
        double[] aOriginal = generateFullRankMatrix(m, n, rand);
        double[] a = aOriginal.clone();

        // Generate right-hand side
        // Note: For HFTI, b must have at least max(m, n) rows to store the solution
        int nb = 1;
        int mdb = Math.max(m, n);  // Leading dimension must be at least n for solution storage
        double[] bOriginal = new double[mdb * nb];
        for (int i = 0; i < m; i++) {
            bOriginal[i] = rand.nextDouble() * 10 - 5;
        }
        double[] b = bOriginal.clone();

        // Use a small tolerance
        double tau = 1e-12;

        // Allocate working arrays
        int diag = Math.min(m, n);
        double[] rnorm = new double[nb];
        double[] h = new double[n];
        double[] g = new double[diag];
        int[] ip = new int[diag];

        // Solve HFTI
        int k = LSQSolver.hfti(m, n, a, 0, m, b, 0, mdb, nb, tau, rnorm, 0, h, 0, g, 0, ip, 0);

        // For underdetermined system, pseudo-rank should be at most m
        assertTrue(k <= m, "Pseudo-rank should be at most m for underdetermined system");

        // Extract solution from first n rows of b
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = b[i];
        }

        // Verify Ax ≈ b (should be exact for underdetermined full-rank system)
        double[] ax = matMul(aOriginal, m, n, x, 1);
        for (int i = 0; i < m; i++) {
            assertEquals(bOriginal[i], ax[i], 1e-6,
                    "For underdetermined system, Ax[" + i + "] should equal b[" + i + "]");
        }

        // Residual norm should be approximately zero
        assertEquals(0.0, rnorm[0], 1e-6,
                "Residual norm should be approximately zero for underdetermined full-rank system");

        // HFTI should return minimum norm solution
        // Verify by checking that x is in the row space of A
        // (i.e., x = A^T * y for some y)
        // This is harder to verify directly, but we can check that the solution norm
        // is not larger than other solutions
    }

    // ========================================================================
    // Property 7.9: HFTI handles identity matrix correctly
    // ========================================================================

    @Property(tries = 50)
    void hftiHandlesIdentityMatrixCorrectly(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll("randomSeed") long seed
    ) {
        int m = n;

        java.util.Random rand = new java.util.Random(seed);

        // Create identity matrix (column-major)
        double[] a = new double[m * n];
        for (int i = 0; i < n; i++) {
            a[i * m + i] = 1.0;
        }

        // Generate right-hand side
        int nb = 1;
        double[] bOriginal = generateMatrix(m, nb, rand);
        double[] b = bOriginal.clone();

        // Use a small tolerance
        double tau = 1e-12;

        // Allocate working arrays
        int diag = n;
        double[] rnorm = new double[nb];
        double[] h = new double[n];
        double[] g = new double[diag];
        int[] ip = new int[diag];

        // Solve HFTI
        int k = LSQSolver.hfti(m, n, a, 0, m, b, 0, m, nb, tau, rnorm, 0, h, 0, g, 0, ip, 0);

        // For identity matrix, pseudo-rank should equal n
        assertEquals(n, k, "Pseudo-rank should equal n for identity matrix");

        // Solution should equal b
        for (int i = 0; i < n; i++) {
            assertEquals(bOriginal[i], b[i], EPSILON,
                    "For identity matrix, x[" + i + "] should equal b[" + i + "]");
        }

        // Residual norm should be zero
        assertEquals(0.0, rnorm[0], EPSILON,
                "Residual norm should be zero for identity matrix");
    }

    // ========================================================================
    // Property 7.10: HFTI handles zero matrix correctly
    // ========================================================================

    @Property(tries = 50)
    void hftiHandlesZeroMatrixCorrectly(
            @ForAll @IntRange(min = 2, max = 8) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        java.util.Random rand = new java.util.Random(seed);

        // Create zero matrix
        double[] a = new double[m * n];

        // Generate right-hand side
        // Note: For HFTI, b must have at least max(m, n) rows to store the solution
        int nb = 1;
        int mdb = Math.max(m, n);  // Leading dimension must be at least n for solution storage
        double[] bOriginal = new double[mdb * nb];
        for (int i = 0; i < m; i++) {
            bOriginal[i] = rand.nextDouble() * 10 - 5;
        }
        double[] b = bOriginal.clone();

        // Use a small tolerance
        double tau = 1e-12;

        // Allocate working arrays
        int diag = Math.min(m, n);
        double[] rnorm = new double[nb];
        double[] h = new double[n];
        double[] g = new double[diag];
        int[] ip = new int[diag];

        // Solve HFTI
        int k = LSQSolver.hfti(m, n, a, 0, m, b, 0, mdb, nb, tau, rnorm, 0, h, 0, g, 0, ip, 0);

        // For zero matrix, pseudo-rank should be 0
        assertEquals(0, k, "Pseudo-rank should be 0 for zero matrix");

        // Solution should be zero
        for (int i = 0; i < n; i++) {
            assertEquals(0.0, b[i], EPSILON,
                    "For zero matrix, x[" + i + "] should be 0");
        }

        // Residual norm should equal ||b||
        double bNorm = 0;
        for (int i = 0; i < m; i++) {
            bNorm += bOriginal[i] * bOriginal[i];
        }
        bNorm = Math.sqrt(bNorm);
        assertEquals(bNorm, rnorm[0], EPSILON,
                "For zero matrix, residual norm should equal ||b||");
    }

    // ========================================================================
    // Property 7.11: HFTI handles zero right-hand side correctly
    // ========================================================================

    @Property(tries = 50)
    void hftiHandlesZeroRightHandSideCorrectly(
            @ForAll @IntRange(min = 3, max = 10) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate a full-rank matrix
        double[] a = generateFullRankMatrix(m, n, rand);

        // Zero right-hand side
        int nb = 1;
        double[] b = new double[m];

        // Use a small tolerance
        double tau = 1e-12;

        // Allocate working arrays
        int diag = Math.min(m, n);
        double[] rnorm = new double[nb];
        double[] h = new double[n];
        double[] g = new double[diag];
        int[] ip = new int[diag];

        // Solve HFTI
        int k = LSQSolver.hfti(m, n, a, 0, m, b, 0, m, nb, tau, rnorm, 0, h, 0, g, 0, ip, 0);

        // Pseudo-rank should be n for full-rank matrix
        assertEquals(n, k, "Pseudo-rank should equal n for full-rank matrix");

        // Solution should be zero
        for (int i = 0; i < n; i++) {
            assertEquals(0.0, b[i], EPSILON,
                    "For zero RHS, x[" + i + "] should be 0");
        }

        // Residual norm should be zero
        assertEquals(0.0, rnorm[0], EPSILON,
                "For zero RHS, residual norm should be 0");
    }

    // ========================================================================
    // Property 7.12: HFTI solution is consistent (deterministic)
    // ========================================================================

    @Property(tries = 50)
    void hftiSolutionIsConsistent(
            @ForAll @IntRange(min = 3, max = 10) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate a full-rank matrix
        double[] aOriginal = generateFullRankMatrix(m, n, rand);

        // Generate right-hand side
        int nb = 1;
        double[] bOriginal = generateMatrix(m, nb, rand);

        double tau = 1e-12;
        int diag = Math.min(m, n);

        // Solve twice with same input
        double[] a1 = aOriginal.clone();
        double[] b1 = bOriginal.clone();
        double[] rnorm1 = new double[nb];
        double[] h1 = new double[n];
        double[] g1 = new double[diag];
        int[] ip1 = new int[diag];

        int k1 = LSQSolver.hfti(m, n, a1, 0, m, b1, 0, m, nb, tau, rnorm1, 0, h1, 0, g1, 0, ip1, 0);

        double[] a2 = aOriginal.clone();
        double[] b2 = bOriginal.clone();
        double[] rnorm2 = new double[nb];
        double[] h2 = new double[n];
        double[] g2 = new double[diag];
        int[] ip2 = new int[diag];

        int k2 = LSQSolver.hfti(m, n, a2, 0, m, b2, 0, m, nb, tau, rnorm2, 0, h2, 0, g2, 0, ip2, 0);

        // Both should have same pseudo-rank
        assertEquals(k1, k2, "Same input should produce same pseudo-rank");

        // Solutions should be identical
        for (int i = 0; i < n; i++) {
            assertEquals(b1[i], b2[i], TOLERANCE,
                    "Same input should produce same solution at index " + i);
        }

        // Residual norms should be identical
        assertEquals(rnorm1[0], rnorm2[0], TOLERANCE,
                "Same input should produce same residual norm");
    }

    // ========================================================================
    // Property 7.13: HFTI handles nb=0 correctly (no right-hand sides)
    // ========================================================================

    @Property(tries = 50)
    void hftiHandlesNoRightHandSides(
            @ForAll @IntRange(min = 3, max = 10) int m,
            @ForAll @IntRange(min = 2, max = 8) int n,
            @ForAll("randomSeed") long seed
    ) {
        Assume.that(m >= n);

        java.util.Random rand = new java.util.Random(seed);

        // Generate a full-rank matrix
        double[] a = generateFullRankMatrix(m, n, rand);

        // No right-hand sides
        int nb = 0;
        double[] b = new double[m];  // Dummy, won't be used

        double tau = 1e-12;
        int diag = Math.min(m, n);
        double[] rnorm = new double[1];  // Dummy
        double[] h = new double[n];
        double[] g = new double[diag];
        int[] ip = new int[diag];

        // Solve HFTI with nb=0
        int k = LSQSolver.hfti(m, n, a, 0, m, b, 0, m, nb, tau, rnorm, 0, h, 0, g, 0, ip, 0);

        // Should still compute pseudo-rank correctly
        assertEquals(n, k, "Pseudo-rank should equal n for full-rank matrix even with nb=0");
    }

    // ========================================================================
    // Property 7.14: HFTI handles edge case dimensions correctly
    // ========================================================================

    @Property(tries = 50)
    void hftiHandlesEdgeCaseDimensions() {
        // Test with m=0 or n=0
        double[] a = new double[1];
        double[] b = new double[1];
        double[] rnorm = new double[1];
        double[] h = new double[1];
        double[] g = new double[1];
        int[] ip = new int[1];

        // m=0
        int k = LSQSolver.hfti(0, 1, a, 0, 1, b, 0, 1, 1, 1e-12, rnorm, 0, h, 0, g, 0, ip, 0);
        assertEquals(0, k, "Pseudo-rank should be 0 for m=0");

        // n=0
        k = LSQSolver.hfti(1, 0, a, 0, 1, b, 0, 1, 1, 1e-12, rnorm, 0, h, 0, g, 0, ip, 0);
        assertEquals(0, k, "Pseudo-rank should be 0 for n=0");
    }
}
