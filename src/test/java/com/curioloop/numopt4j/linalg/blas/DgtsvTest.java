/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DgtsvTest {

    private static final double TOL = 1e-14;

    /** Compute A*X - B in place (B := A*X - B), where A is tridiagonal. */
    private static void dlagtm(int n, int nrhs,
                                double[] dl, double[] d, double[] du,
                                double[] X, int ldx,
                                double[] B, int ldb) {
        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < n; i++) {
                double ax = d[i] * X[i * ldx + j];
                if (i > 0)   ax += dl[i - 1] * X[(i - 1) * ldx + j];
                if (i < n-1) ax += du[i]     * X[(i + 1) * ldx + j];
                B[i * ldb + j] = ax - B[i * ldb + j];
            }
        }
    }

    /** 1-norm of a general matrix (max column sum). */
    private static double dlange1(int m, int n, double[] A, int lda) {
        double norm = 0;
        for (int j = 0; j < n; j++) {
            double col = 0;
            for (int i = 0; i < m; i++) col += Math.abs(A[i * lda + j]);
            norm = Math.max(norm, col);
        }
        return norm;
    }

    /** 1-norm of a tridiagonal matrix (max column sum). */
    private static double dlangt1(int n, double[] dl, double[] d, double[] du) {
        if (n == 0) return 0;
        if (n == 1) return Math.abs(d[0]);
        double norm = Math.max(Math.abs(d[0]) + Math.abs(dl[0]),
                               Math.abs(d[n-1]) + Math.abs(du[n-2]));
        for (int i = 1; i < n-1; i++) {
            norm = Math.max(norm, Math.abs(du[i-1]) + Math.abs(d[i]) + Math.abs(dl[i]));
        }
        return norm;
    }

    private void dgtsvTest(long seed, int n, int nrhs, int ldb) {
        Random rnd = new Random(seed);

        if (n == 0) {
            assertTrue(Dptsv.dgtsv(0, nrhs, new double[0], 0, new double[0], 0,
                    new double[0], 0, new double[0], 0, ldb));
            return;
        }

        double[] d  = new double[n]; for (int i = 0; i < n; i++) d[i]  = rnd.nextDouble() * 2 - 1;
        double[] dl = n > 1 ? new double[n-1] : new double[0];
        double[] du = n > 1 ? new double[n-1] : new double[0];
        for (int i = 0; i < n-1; i++) { dl[i] = rnd.nextDouble() * 2 - 1; du[i] = rnd.nextDouble() * 2 - 1; }

        double[] dlCopy = dl.clone(), dCopy = d.clone(), duCopy = du.clone();

        // Generate random B (the right-hand side)
        double[] B = new double[n * ldb];
        for (int i = 0; i < B.length; i++) B[i] = rnd.nextDouble() * 2 - 1;
        double[] Bcopy = B.clone();

        boolean ok = Dptsv.dgtsv(n, nrhs, dl, 0, d, 0, du, 0, B, 0, ldb);
        if (!ok) return; // singular matrix, skip residual check

        // Compute A*X - B_orig, store in Bcopy
        dlagtm(n, nrhs, dlCopy, dCopy, duCopy, B, ldb, Bcopy, ldb);

        double anorm = dlangt1(n, dlCopy, dCopy, duCopy);
        double xnorm = dlange1(n, nrhs, B, ldb);
        double rnorm = dlange1(n, nrhs, Bcopy, ldb);

        double resid = (anorm == 0 || xnorm == 0) ? rnorm : rnorm / anorm / xnorm;
        assertTrue(resid <= TOL, "n=" + n + " nrhs=" + nrhs + " ldb=" + ldb + " resid=" + resid);
    }

    @Test
    void testRandom() {
        int[] ns    = {0, 1, 2, 3, 4, 5, 10, 25, 50};
        int[] nrhss = {0, 1, 2, 3, 4, 10};
        long seed = 1L;
        for (int n : ns) {
            for (int nrhs : nrhss) {
                for (int ldb : new int[]{Math.max(1, nrhs), nrhs + 3}) {
                    dgtsvTest(seed++, n, nrhs, ldb);
                }
            }
        }
    }

    @Test
    void testSingular() {
        // A = [1 1; 1 1]: after elimination d[1] = 0
        double[] dl = {1}, d = {1, 1}, du = {1};
        double[] B = {1, 1};
        assertFalse(Dptsv.dgtsv(2, 1, dl, 0, d, 0, du, 0, B, 0, 1));
    }
}
