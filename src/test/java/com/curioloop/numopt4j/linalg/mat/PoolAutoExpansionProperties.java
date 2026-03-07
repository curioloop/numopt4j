/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for Pool auto-expansion (Property 11).
 *
 *
 */
class PoolAutoExpansionProperties {

    private static final double MACHINE_EPSILON = 2.22e-16;

    // ========================================================================
    // Helper utilities
    // ========================================================================

    /** Create a random m×n matrix with values in [-1, 1] */
    private static double[] randomMatrix(int m, int n, long seed) {
        double[] A = new double[m * n];
        java.util.Random rng = new java.util.Random(seed);
        for (int i = 0; i < A.length; i++) {
            A[i] = rng.nextDouble() * 2.0 - 1.0;
        }
        return A;
    }

    /** Create a random symmetric n×n matrix */
    private static double[] randomSymmetricMatrix(int n, long seed) {
        double[] A = new double[n * n];
        java.util.Random rng = new java.util.Random(seed);
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double v = rng.nextDouble() * 2.0 - 1.0;
                A[i * n + j] = v;
                A[j * n + i] = v;
            }
        }
        return A;
    }

    /** Frobenius norm of a matrix */
    private static double frobeniusNorm(double[] A) {
        double sum = 0.0;
        for (double v : A) sum += v * v;
        return Math.sqrt(sum);
    }


    // ========================================================================
    // Property 11: SVD Pool auto-expansion
    // ========================================================================

    /**
     * Property 11 (SVD): Initialize Pool with small m/n, then call SVD.decompose with larger m/n.
     * The Pool should auto-expand via ensure(), not throw ArrayIndexOutOfBoundsException,
     * and produce numerically correct results.
     *
     */
    @Property(tries = 100)
    void svdPoolAutoExpansion(
            @ForAll @IntRange(min = 1, max = 5) int smallM,
            @ForAll @IntRange(min = 1, max = 5) int smallN,
            @ForAll @IntRange(min = 6, max = 15) int bigM,
            @ForAll @IntRange(min = 6, max = 15) int bigN,
            @ForAll long seed
    ) {
        // Initialize Pool with small dimensions
        SVD.Pool pool = new SVD.Pool();
        pool.ensureS(Math.min(smallM, smallN));
        int smallUSize  = smallM * Math.min(smallM, smallN);
        int smallVTSize = Math.min(smallM, smallN) * smallN;
        pool.ensureUV(Math.max(smallUSize, smallVTSize));

        // Verify pool is too small for big dimensions
        int bigMinMN = Math.min(bigM, bigN);
        boolean sOk  = pool.s  != null && pool.s.length  >= bigMinMN;
        boolean uvOk = pool.UV != null && pool.UV.length >= Math.max(bigM * bigMinMN, bigMinMN * bigN);
        assertFalse(sOk && uvOk,
                "Pool initialized with small dims should not be compatible with big dims");

        // Now call decompose with big dimensions — should auto-expand, not throw
        double[] A = randomMatrix(bigM, bigN, seed);
        double[] ACopy = A.clone();

        SVD svdPool;
        try {
            svdPool = SVD.decompose(A, bigM, bigN, SVD.SVD_ALL, pool);
        } catch (ArrayIndexOutOfBoundsException e) {
            fail("SVD Pool should auto-expand, not throw ArrayIndexOutOfBoundsException: " + e.getMessage());
            return;
        }

        assertTrue(svdPool.ok(), "SVD with auto-expanded Pool should succeed");

        // Compare with non-Pool result for numerical correctness
        SVD svdRef = SVD.decompose(ACopy, bigM, bigN, SVD.SVD_ALL, null);
        assertTrue(svdRef.ok(), "Reference SVD should succeed");

        int minMN = Math.min(bigM, bigN);
        double normA = frobeniusNorm(ACopy);
        double tol = 10.0 * minMN * MACHINE_EPSILON * normA + 1e-12;

        // Singular values should match (up to sign/ordering)
        double[] sPool = svdPool.singularValues();
        double[] sRef = svdRef.singularValues();
        for (int i = 0; i < minMN; i++) {
            assertEquals(sRef[i], sPool[i], tol,
                    "Singular value " + i + " should match reference");
        }
    }

    // ========================================================================
    // Property 11: Eigen Pool auto-expansion (symmetric)
    // ========================================================================

    /**
     * Property 11 (Eigen symmetric): Initialize Pool with small n, then call Eigen.decompose
     * with larger n. The Pool should auto-expand, not throw ArrayIndexOutOfBoundsException,
     * and produce numerically correct results.
     *
     */
    @Property(tries = 100)
    void eigenSymmetricPoolAutoExpansion(
            @ForAll @IntRange(min = 1, max = 3) int smallN,
            @ForAll @IntRange(min = 4, max = 12) int bigN,
            @ForAll long seed
    ) {
        // Initialize Pool with small n
        Eigen.Pool pool = new Eigen.Pool();
        pool.ensure(smallN, true, Eigen.EIGEN_RIGHT);

        // Verify pool is too small for big n
        assertFalse(pool.isCompatible(bigN, true, Eigen.EIGEN_RIGHT),
                "Pool initialized with small n should not be compatible with big n");

        // Now call decompose with big n — should auto-expand, not throw
        double[] A = randomSymmetricMatrix(bigN, seed);
        double[] ACopy = A.clone();

        Eigen eigenPool;
        try {
            eigenPool = Eigen.decompose(A, bigN, true, 'L', Eigen.EIGEN_RIGHT, pool);
        } catch (ArrayIndexOutOfBoundsException e) {
            fail("Eigen Pool should auto-expand, not throw ArrayIndexOutOfBoundsException: " + e.getMessage());
            return;
        }

        assertTrue(eigenPool.ok(), "Eigen with auto-expanded Pool should succeed");

        // Compare with non-Pool result for numerical correctness
        Eigen eigenRef = Eigen.decompose(ACopy, bigN, true, 'L', Eigen.EIGEN_RIGHT, null);
        assertTrue(eigenRef.ok(), "Reference Eigen should succeed");

        double normA = frobeniusNorm(ACopy);
        double tol = 10.0 * bigN * MACHINE_EPSILON * normA + 1e-12;

        double[] wrPool = eigenPool.eigenvalues();
        double[] wrRef = eigenRef.eigenvalues();
        for (int i = 0; i < bigN; i++) {
            assertEquals(wrRef[i], wrPool[i], tol,
                    "Eigenvalue " + i + " should match reference");
        }
    }

    // ========================================================================
    // Property 11: Eigen Pool auto-expansion (general)
    // ========================================================================

    /**
     * Property 11 (Eigen general): Initialize Pool with small n, then call Eigen.decompose
     * (general) with larger n. The Pool should auto-expand, not throw ArrayIndexOutOfBoundsException,
     * and produce numerically correct results.
     *
     */
    @Property(tries = 100)
    void eigenGeneralPoolAutoExpansion(
            @ForAll @IntRange(min = 1, max = 3) int smallN,
            @ForAll @IntRange(min = 4, max = 10) int bigN,
            @ForAll long seed
    ) {
        // Initialize Pool with small n
        Eigen.Pool pool = new Eigen.Pool();
        pool.ensure(smallN, false, Eigen.EIGEN_RIGHT);

        // Verify pool is too small for big n
        assertFalse(pool.isCompatible(bigN, false, Eigen.EIGEN_RIGHT),
                "Pool initialized with small n should not be compatible with big n");

        // Now call decompose with big n — should auto-expand, not throw
        double[] A = randomMatrix(bigN, bigN, seed);
        double[] ACopy = A.clone();

        Eigen eigenPool;
        try {
            eigenPool = Eigen.decompose(A, bigN, false, '\0', Eigen.EIGEN_RIGHT, pool);
        } catch (ArrayIndexOutOfBoundsException e) {
            fail("Eigen Pool should auto-expand, not throw ArrayIndexOutOfBoundsException: " + e.getMessage());
            return;
        }

        assertTrue(eigenPool.ok(), "Eigen general with auto-expanded Pool should succeed");

        // Compare with non-Pool result: eigenvalues should match (sorted by real part)
        Eigen eigenRef = Eigen.decompose(ACopy, bigN, false, '\0', Eigen.EIGEN_RIGHT, null);
        assertTrue(eigenRef.ok(), "Reference Eigen general should succeed");

        double normA = frobeniusNorm(ACopy);
        double tol = 10.0 * bigN * MACHINE_EPSILON * normA + 1e-10;

        // Sort eigenvalues by real part for comparison
        double[] wrPool = eigenPool.eigenvalues().clone();
        double[] wiPool = eigenPool.eigenvaluesImag().clone();
        double[] wrRef = eigenRef.eigenvalues().clone();
        double[] wiRef = eigenRef.eigenvaluesImag().clone();

        sortEigenvalues(wrPool, wiPool);
        sortEigenvalues(wrRef, wiRef);

        for (int i = 0; i < bigN; i++) {
            assertEquals(wrRef[i], wrPool[i], tol,
                    "Real part of eigenvalue " + i + " should match reference");
            assertEquals(wiRef[i], wiPool[i], tol,
                    "Imaginary part of eigenvalue " + i + " should match reference");
        }
    }

    /** Sort eigenvalues by (real, imag) for comparison */
    private static void sortEigenvalues(double[] wr, double[] wi) {
        int n = wr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (wr[j] < wr[i] || (wr[j] == wr[i] && wi[j] < wi[i])) {
                    double tmp = wr[i]; wr[i] = wr[j]; wr[j] = tmp;
                    tmp = wi[i]; wi[i] = wi[j]; wi[j] = tmp;
                }
            }
        }
    }

    // ========================================================================
    // Property 11: Schur Pool auto-expansion
    // ========================================================================

    /**
     * Property 11 (Schur): Initialize Pool with small n, then call Schur.decompose with larger n.
     * The Pool should auto-expand, not throw ArrayIndexOutOfBoundsException,
     * and produce numerically correct results.
     *
     */
    @Property(tries = 100)
    void schurPoolAutoExpansion(
            @ForAll @IntRange(min = 1, max = 3) int smallN,
            @ForAll @IntRange(min = 4, max = 10) int bigN,
            @ForAll long seed
    ) {
        // Initialize Pool with small n
        Schur.Pool pool = new Schur.Pool();
        pool.ensure(smallN, true);

        // Verify pool is too small for big n
        assertFalse(pool.isCompatible(bigN, true),
                "Pool initialized with small n should not be compatible with big n");

        // Now call decompose with big n — should auto-expand, not throw
        double[] A = randomMatrix(bigN, bigN, seed);
        double[] ACopy = A.clone();

        Schur schurPool;
        try {
            schurPool = Schur.decompose(A, bigN, true, null, pool);
        } catch (ArrayIndexOutOfBoundsException e) {
            fail("Schur Pool should auto-expand, not throw ArrayIndexOutOfBoundsException: " + e.getMessage());
            return;
        }

        assertTrue(schurPool.ok(), "Schur with auto-expanded Pool should succeed");

        // Compare eigenvalues with non-Pool result
        Schur schurRef = Schur.decompose(ACopy, bigN, true, null, null);
        assertTrue(schurRef.ok(), "Reference Schur should succeed");

        double normA = frobeniusNorm(ACopy);
        double tol = 10.0 * bigN * MACHINE_EPSILON * normA + 1e-10;

        // Sort eigenvalues for comparison
        double[] wrPool = schurPool.eigenvalues().clone();
        double[] wiPool = schurPool.eigenvaluesImag().clone();
        double[] wrRef = schurRef.eigenvalues().clone();
        double[] wiRef = schurRef.eigenvaluesImag().clone();

        sortEigenvalues(wrPool, wiPool);
        sortEigenvalues(wrRef, wiRef);

        for (int i = 0; i < bigN; i++) {
            assertEquals(wrRef[i], wrPool[i], tol,
                    "Real part of eigenvalue " + i + " should match reference");
            assertEquals(wiRef[i], wiPool[i], tol,
                    "Imaginary part of eigenvalue " + i + " should match reference");
        }
    }
}
