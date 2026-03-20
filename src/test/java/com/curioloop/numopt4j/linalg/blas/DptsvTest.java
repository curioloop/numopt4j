/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

class DptsvTest {

    private static final double TOL = 1e-15;

    private void dptsvTest(long seed, int n, int nrhs, int ldb) {
        Random rnd = new Random(seed);
        double[][] mat = TridiagTestHelper.newRandomSymTridiag(n, rnd);
        double[] d = mat[0], e = mat[1];

        double[] xWant = new double[n * ldb];
        for (int i = 0; i < xWant.length; i++) xWant[i] = rnd.nextDouble() * 2 - 1;

        double[] B = new double[n * ldb];
        TridiagTestHelper.dstmm(n, nrhs, d, e, xWant, ldb, B, ldb);

        boolean ok = Dptsv.dptsv(n, nrhs, d, 0, e, 0, B, 0, ldb);
        assertTrue(ok, "n=" + n + ": Dptsv failed");

        double resid = TridiagTestHelper.residual(n, nrhs, B, xWant, ldb);
        assertTrue(resid <= TOL, "n=" + n + " nrhs=" + nrhs + " resid=" + resid);
    }

    @Test
    void testRandom() {
        int[] ns    = {0, 1, 2, 3, 4, 5, 10, 20, 50, 51, 52, 53, 54, 100};
        int[] nrhss = {0, 1, 2, 3, 4, 5, 10, 20, 50};
        long seed = 1L;
        for (int n : ns)
            for (int nrhs : nrhss)
                for (int ldb : new int[]{Math.max(1, nrhs), nrhs + 3})
                    dptsvTest(seed++, n, nrhs, ldb);
    }

    @Test
    void testNotPositiveDefinite() {
        double[] d = {4, -1, 4}, e = {1, 1}, B = {1, 1, 1};
        assertFalse(Dptsv.dptsv(3, 1, d, 0, e, 0, B, 0, 1));
    }

    @Test
    void testEmpty() {
        assertTrue(Dptsv.dptsv(0, 1, new double[0], 0, new double[0], 0, new double[0], 0, 1));
    }
}
