/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Dlaqr23 (aggressive early deflation window).
 *
 * Covers:
 *  - D1: 1×1 window convergence check (converged vs. not converged)
 *  - D2: deflation loop ilst update uses trexcWork[1] (ilst_final)
 *  - D3: sorting loop ok flag and i update use correct return value / trexcWork[1]
 *  - Integration via Dhseqr on various matrix sizes
 */
class Dlaqr23Test {

    private static final double TOL = 1e-10;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /** Build a jw×jw identity-initialized V, jw×jw zero T, and call dlaqr23 directly. */
    private int[] callDlaqr23(double[] h, int n, int ktop, int kbot, int nw) {
        int jw = nw;
        double[] v   = new double[jw * jw];
        double[] t   = new double[jw * jw];
        double[] wv  = new double[jw * jw];
        double[] sr  = new double[kbot + 1];
        double[] si  = new double[kbot + 1];
        double[] work = new double[Math.max(1, 4 * jw + 100)];

        long ret = Dlaqr.dlaqr23(
                false, false, n, ktop, kbot, nw,
                h, 0, n,
                0, n - 1,
                null, 0, n,
                sr, 0, si, 0,
                v, 0, jw,
                jw, t, 0, jw,
                jw, wv, 0, jw,
                work, 0, work.length, 0);
        return new int[]{(int)(ret >>> 32), (int) ret};
    }

    // -----------------------------------------------------------------------
    // D1: 1×1 deflation window convergence check
    // -----------------------------------------------------------------------

    /**
     * When nw=1 and the spike entry s is tiny relative to the diagonal,
     * the eigenvalue should be marked as converged: ns=0, nd=1.
     */
    @Test
    void test1x1WindowConverged() {
        // 3×3 upper Hessenberg, window at kbot=2, nw=1
        // kwtop = kbot - nw + 1 = 2, so s = h[2*3+1] (subdiagonal of window)
        // Make s very small so convergence check passes
        double[] h = {
            2, 1, 0,
            0, 3, 1,
            0, 1e-20, 5   // h[2*3+1] = 1e-20 (spike s)
        };
        int[] result = callDlaqr23(h, 3, 0, 2, 1);
        // s=1e-20 is tiny → converged → ns=0, nd=1
        assertEquals(0, result[0], "ns should be 0 (converged)");
        assertEquals(1, result[1], "nd should be 1 (deflated)");
    }

    /**
     * When nw=1 and the spike entry s is large, the eigenvalue is NOT converged:
     * ns=1, nd=0.
     */
    @Test
    void test1x1WindowNotConverged() {
        // Large spike s = 10.0
        double[] h = {
            2, 1, 0,
            0, 3, 1,
            0, 10.0, 5   // h[2*3+1] = 10.0 (spike s)
        };
        int[] result = callDlaqr23(h, 3, 0, 2, 1);
        // s=10.0 is large → not converged → ns=1, nd=0
        assertEquals(1, result[0], "ns should be 1 (not converged)");
        assertEquals(0, result[1], "nd should be 0 (not deflated)");
    }

    /**
     * When kwtop == ktop (no spike, s=0), the 1×1 window is always converged.
     */
    @Test
    void test1x1WindowNoSpike() {
        // kwtop == ktop means s=0, always converged
        double[] h = { 7.0 };
        int[] result = callDlaqr23(h, 1, 0, 0, 1);
        assertEquals(0, result[0], "ns should be 0 (s=0, always converged)");
        assertEquals(1, result[1], "nd should be 1");
    }

    // -----------------------------------------------------------------------
    // D2 / D3: deflation and sorting via integration tests through Dhseqr
    // -----------------------------------------------------------------------

    /**
     * Diagonal matrix: all eigenvalues should be found exactly.
     * Exercises the deflation path (all entries deflate immediately).
     */
    @Test
    void testDiagonalMatrixEigenvalues() {
        int n = 6;
        double[] H = new double[n * n];
        double[] diag = {6, 5, 4, 3, 2, 1};
        for (int i = 0; i < n; i++) H[i * n + i] = diag[i];

        double[] wr = new double[n], wi = new double[n];
        double[] work = new double[200];
        int info = Dhseqr.dhseqr('E', 'N', n, 0, n - 1, H, n, wr, wi, null, n, work, 0, work.length);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) {
            assertEquals(diag[i], wr[i], TOL);
            assertEquals(0.0, wi[i], TOL);
        }
    }

    /**
     * Upper triangular matrix: eigenvalues are the diagonal entries.
     * Exercises deflation with non-zero upper triangle.
     */
    @Test
    void testUpperTriangularEigenvalues() {
        int n = 5;
        double[] H = new double[n * n];
        double[] diag = {5, 4, 3, 2, 1};
        for (int i = 0; i < n; i++) {
            H[i * n + i] = diag[i];
            for (int j = i + 1; j < n; j++) H[i * n + j] = (i + 1) * 0.5;
        }

        double[] wr = new double[n], wi = new double[n];
        double[] work = new double[200];
        int info = Dhseqr.dhseqr('E', 'N', n, 0, n - 1, H, n, wr, wi, null, n, work, 0, work.length);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) assertEquals(diag[i], wr[i], TOL);
    }

    /**
     * Known 2×2 matrix with real eigenvalues: [[4,1],[0,2]] → eigenvalues 4, 2.
     */
    @Test
    void test2x2RealEigenvalues() {
        double[] H = { 4, 1, 0, 2 };
        double[] wr = new double[2], wi = new double[2];
        double[] work = new double[20];
        int info = Dhseqr.dhseqr('E', 'N', 2, 0, 1, H, 2, wr, wi, null, 2, work, 0, work.length);

        assertEquals(0, info);
        // eigenvalues are 4 and 2 (order may vary)
        double sum = wr[0] + wr[1];
        double prod = wr[0] * wr[1];
        assertEquals(6.0, sum, TOL, "trace = 6");
        assertEquals(8.0, prod, TOL, "det = 8");
        assertEquals(0.0, wi[0], TOL);
        assertEquals(0.0, wi[1], TOL);
    }

    /**
     * Known 2×2 matrix with complex eigenvalues: [[0,-1],[1,0]] → eigenvalues ±i.
     */
    @Test
    void test2x2ComplexEigenvalues() {
        double[] H = { 0, -1, 1, 0 };
        double[] wr = new double[2], wi = new double[2];
        double[] work = new double[20];
        int info = Dhseqr.dhseqr('E', 'N', 2, 0, 1, H, 2, wr, wi, null, 2, work, 0, work.length);

        assertEquals(0, info);
        assertEquals(0.0, wr[0] + wr[1], TOL, "real parts sum to 0");
        assertEquals(1.0, Math.abs(wi[0]), TOL, "imaginary part magnitude = 1");
        assertEquals(1.0, Math.abs(wi[1]), TOL, "imaginary part magnitude = 1");
        assertEquals(-wi[0], wi[1], TOL, "complex conjugate pair");
    }

    /**
     * Trace preservation: sum of eigenvalues equals trace of H.
     * Uses a medium Hessenberg matrix to exercise the full AED path.
     */
    @Test
    void testTracePreservationMediumMatrix() {
        int n = 10;
        double[] H = buildHessenberg(n, 42);
        double trace = 0;
        for (int i = 0; i < n; i++) trace += H[i * n + i];

        double[] wr = new double[n], wi = new double[n];
        double[] work = new double[500];
        int info = Dhseqr.dhseqr('E', 'N', n, 0, n - 1, H, n, wr, wi, null, n, work, 0, work.length);

        assertEquals(0, info);
        double eigenSum = 0;
        for (double v : wr) eigenSum += v;
        assertEquals(trace, eigenSum, 1e-8, "trace must equal sum of real parts of eigenvalues");
    }

    /**
     * Large matrix (n=30) exercises the recursive AED path (recur>0 in Dlaqr04).
     * Verifies all eigenvalues are finite and info=0.
     */
    @Test
    void testLargeMatrixAED() {
        int n = 30;
        double[] H = buildHessenberg(n, 7);

        double[] wr = new double[n], wi = new double[n];
        double[] work = new double[2000];
        int info = Dhseqr.dhseqr('E', 'N', n, 0, n - 1, H, n, wr, wi, null, n, work, 0, work.length);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) assertTrue(Double.isFinite(wr[i]));
        for (int i = 0; i < n; i++) assertTrue(Double.isFinite(wi[i]));
    }

    /**
     * Schur form computation: verifies Z is orthogonal (Z^T Z ≈ I).
     */
    @Test
    void testSchurFormOrthogonality() {
        int n = 8;
        double[] H = buildHessenberg(n, 13);
        double[] Z = new double[n * n];
        for (int i = 0; i < n; i++) Z[i * n + i] = 1.0; // identity

        double[] wr = new double[n], wi = new double[n];
        double[] work = new double[500];
        int info = Dhseqr.dhseqr('S', 'I', n, 0, n - 1, H, n, wr, wi, Z, n, work, 0, work.length);

        assertEquals(0, info);
        // Check Z^T Z ≈ I
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dot = 0;
                for (int k = 0; k < n; k++) dot += Z[k * n + i] * Z[k * n + j];
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, dot, 1e-10, "Z^T Z[" + i + "," + j + "]");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    /** Build a deterministic upper Hessenberg matrix of size n×n. */
    private static double[] buildHessenberg(int n, long seed) {
        double[] H = new double[n * n];
        java.util.Random rng = new java.util.Random(seed);
        for (int i = 0; i < n; i++) {
            for (int j = Math.max(0, i - 1); j < n; j++) {
                H[i * n + j] = rng.nextDouble() * 10 - 5;
            }
        }
        return H;
    }
}
