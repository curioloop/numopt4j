/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DormbrTest {

    private static final double TOL = 1e-12;

    @Test
    void testDormbrApplyQLeft() {
        int m = 4, n = 3, k = 3;
        double[] A = new double[m * k];
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < A.length; i++) {
            A[i] = rand.nextDouble() * 2 - 1;
        }

        double[] d = new double[Math.min(m, k)];
        double[] e = new double[Math.min(m, k)];
        double[] tauQ = new double[Math.min(m, k)];
        double[] tauP = new double[Math.min(m, k)];
        double[] work = new double[100];

        Dgebrd.dgebrd(m, k, A, 0, k, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0, work.length);

        double[] C = new double[m * n];
        for (int i = 0; i < C.length; i++) {
            C[i] = rand.nextDouble() * 2 - 1;
        }

        int info = Dormbr.dormbr('Q', BLAS.Side.Left, BLAS.Trans.NoTrans, m, n, k, A, 0, k, tauQ, 0, C, 0, n, work, 0, work.length);

        assertEquals(0, info);
        assertNotNull(C);
    }

    @Test
    void testDormbrWorkspaceQuery() {
        int m = 4, n = 3, k = 3;
        double[] A = new double[m * k];
        double[] tauQ = new double[Math.min(m, k)];
        double[] C = new double[m * n];
        double[] work = new double[1];

        int info = Dormbr.dormbr('Q', BLAS.Side.Left, BLAS.Trans.NoTrans, m, n, k, A, 0, k, tauQ, 0, C, 0, n, work, 0, -1);

        assertEquals(0, info);
        assertTrue(work[0] > 0);
    }

    @Test
    void testDormbrEmpty() {
        double[] work = new double[1];
        int info = Dormbr.dormbr('Q', BLAS.Side.Left, BLAS.Trans.NoTrans, 0, 0, 0, new double[0], 0, 1, new double[0], 0, new double[0], 0, 1, work, 0, 1);
        assertEquals(0, info);
    }

    @Test
    void testDormqrBlockedBasic() {
        int m = 5, n = 4, k = 4;
        double[] A = new double[m * k];
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < A.length; i++) {
            A[i] = rand.nextDouble() * 2 - 1;
        }

        double[] tau = new double[k];
        double[] work = new double[100];

        Dgeqr.dgeqr2(m, k, A, 0, k, tau, 0, work, 0);

        double[] C = new double[m * n];
        for (int i = 0; i < C.length; i++) {
            C[i] = rand.nextDouble() * 2 - 1;
        }

        int info = Dormbr.dormqrBlocked(BLAS.Side.Left, BLAS.Trans.NoTrans, m, n, k, A, 0, k, tau, 0, C, 0, n, work, 0, work.length);

        assertEquals(0, info);
    }
}
