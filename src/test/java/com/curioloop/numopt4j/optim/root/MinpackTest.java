package com.curioloop.numopt4j.optim.root;

import net.jqwik.api.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link Minpack} QR factorization methods.
 */
class MinpackTest {

    // ---- Unit tests ----

    @Test
    void qrfacKnown2x2() {
        // A = [[2, 1], [0, 2]] (column-major: [2,0,1,2])
        int m = 2, n = 2, lda = 2;
        double[] a = {2.0, 0.0, 1.0, 2.0};
        int[] ipvt = new int[n];
        double[] rdiag = new double[n];
        double[] acnorm = new double[n];
        double[] wa = new double[n];
        Minpack.qrfac(m, n, a, lda, true, ipvt, n, rdiag, 0, acnorm, 0, wa, 0);
        // |rdiag[0]| >= |rdiag[1]|
        assertTrue(Math.abs(rdiag[0]) >= Math.abs(rdiag[1]) - 1e-12);
    }

    @Test
    void qformIdentityForN0() {
        // m=2, n=0: Q should be identity
        int m = 2;
        double[] q = {1.0, 0.0, 0.0, 1.0}; // identity column-major
        double[] wa = new double[m];
        Minpack.qform(m, 0, q, m, wa, 0);
        // Q should remain identity
        assertEquals(1.0, q[0], 1e-14);
        assertEquals(0.0, q[1], 1e-14);
        assertEquals(0.0, q[2], 1e-14);
        assertEquals(1.0, q[3], 1e-14);
    }

    @Test
    void qrsolv1x1() {
        // R = [[3]], diag = [2], qtb = [6] => solve [3;2]*x = [6;0] => x = 6/sqrt(13)
        int n = 1;
        double[] r = {3.0};
        int[] ipvt = {0};
        double[] diag = {2.0};
        double[] qtb = {6.0};
        double[] x = new double[1];
        double[] sdiag = new double[1];
        double[] wa = new double[1];
        Minpack.qrsolv(n, r, 0, n, ipvt, diag, 0, qtb, 0, x, sdiag, 0, wa, 0);
        // x should minimize ||[3;2]*x - [6;0]||
        double expected = 3.0 * 6.0 / (9.0 + 4.0); // normal equations: (R^T R + D^T D) x = R^T qtb
        assertEquals(expected, x[0], 1e-12);
    }

    // Property: qrfac pivot columns non-increasing
    @Property(tries = 200)
    void qrfacPivotNonIncreasing(@ForAll("fullRankMatrices") double[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        int lda = m;
        double[] a = toColumnMajor(mat, m, n);
        int[] ipvt = new int[n];
        double[] rdiag = new double[n];
        double[] acnorm = new double[n];
        double[] wa = new double[n];
        Minpack.qrfac(m, n, a, lda, true, ipvt, n, rdiag, 0, acnorm, 0, wa, 0);
        for (int j = 0; j < n - 1; j++) {
            assertTrue(Math.abs(rdiag[j]) >= Math.abs(rdiag[j + 1]) - 1e-10,
                    "rdiag not non-increasing at j=" + j + ": " + rdiag[j] + " < " + rdiag[j + 1]);
        }
    }

    // Property: qform orthogonality
    @Property(tries = 100)
    void qformOrthogonality(@ForAll("fullRankMatrices") double[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        int lda = m;
        double[] a = toColumnMajor(mat, m, n);
        int[] ipvt = new int[n];
        double[] rdiag = new double[n];
        double[] acnorm = new double[n];
        double[] wa = new double[n];
        Minpack.qrfac(m, n, a, lda, true, ipvt, n, rdiag, 0, acnorm, 0, wa, 0);

        // Accumulate Q (m x m)
        double[] q = new double[m * m];
        // Copy lower trapezoidal part of a into q
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                q[i + m * j] = a[i + lda * j];
            }
        }
        // Fill remaining columns with identity
        for (int j = n; j < m; j++) {
            q[j + m * j] = 1.0;
        }
        double[] waQ = new double[m];
        Minpack.qform(m, n, q, m, waQ, 0);

        // Check Q^T * Q ≈ I
        double tol = n * Minpack.dpmpar(1) * 100;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double dot = 0.0;
                for (int k = 0; k < m; k++) {
                    dot += q[k + m * i] * q[k + m * j];
                }
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, dot, tol,
                        "Q^T*Q[" + i + "," + j + "] = " + dot + " expected " + expected);
            }
        }
    }

    // Property: qrsolv least-squares solution
    @Property(tries = 100)
    void qrsolvLeastSquares(@ForAll("qrsolvInputs") Object[] inputs) {
        double[] diag = (double[]) inputs[0];
        double[] qtb  = (double[]) inputs[1];
        int n = diag.length;
        if (n == 0) return;
        double[] r = new double[n * n];
        for (int j = 0; j < n; j++) r[j + n * j] = diag[j] + 1.0;
        int[] ipvt = new int[n];
        for (int j = 0; j < n; j++) ipvt[j] = j;
        double[] x = new double[n];
        double[] sdiag = new double[n];
        double[] wa = new double[n];
        Minpack.qrsolv(n, r, 0, n, ipvt, diag, 0, qtb, 0, x, sdiag, 0, wa, 0);
        for (double xi : x) {
            assertFalse(Double.isNaN(xi), "x contains NaN");
            assertFalse(Double.isInfinite(xi), "x contains Infinity");
        }
    }

    // ---- Helpers ----

    private static double[] toColumnMajor(double[][] mat, int m, int n) {
        double[] a = new double[m * n];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                a[i + m * j] = mat[i][j];
        return a;
    }

    @Provide
    Arbitrary<double[][]> fullRankMatrices() {
        // Generate m x n matrices (m >= n) with entries in [-10, 10]
        // Add diagonal dominance to ensure full rank
        return Arbitraries.integers().between(2, 5).flatMap(n ->
            Arbitraries.integers().between(n, n + 3).flatMap(m ->
                Arbitraries.doubles().between(-5.0, 5.0)
                    .array(double[].class).ofSize(m * n)
                    .map(flat -> {
                        double[][] mat = new double[m][n];
                        for (int j = 0; j < n; j++)
                            for (int i = 0; i < m; i++)
                                mat[i][j] = flat[i + m * j];
                        // Add diagonal dominance
                        for (int j = 0; j < n; j++)
                            mat[j][j] += (mat[j][j] >= 0 ? 1 : -1) * 10.0;
                        return mat;
                    })
            )
        );
    }

    @Provide
    Arbitrary<Object[]> qrsolvInputs() {
        return Arbitraries.integers().between(1, 4).flatMap(n ->
            Arbitraries.doubles().between(0.1, 5.0).array(double[].class).ofSize(n).flatMap(diag ->
                Arbitraries.doubles().between(-10.0, 10.0).array(double[].class).ofSize(n)
                    .map(qtb -> new Object[]{diag, qtb})
            )
        );
    }

    @Provide
    Arbitrary<double[]> smallPositiveDiag() {
        return Arbitraries.doubles().between(0.1, 5.0)
                .array(double[].class).ofMinSize(1).ofMaxSize(4);
    }

    @Provide
    Arbitrary<double[]> smallQtb() {
        return Arbitraries.doubles().between(-10.0, 10.0)
                .array(double[].class).ofMinSize(1).ofMaxSize(4);
    }
}
