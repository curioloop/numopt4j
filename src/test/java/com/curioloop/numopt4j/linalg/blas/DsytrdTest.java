/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DsytrdTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasicUpper() {
        double[] A = {
            4, 1, 2,
            0, 3, 1,
            0, 0, 2
        };
        int n = 3;
        double[] d = new double[n];
        double[] e = new double[n - 1];
        double[] tau = new double[n - 1];
        double[] work = new double[100];

        Dsytrd.dsytrd(BLAS.Uplo.Upper, n, A, n, d, 0, e, 0, tau, 0, work, work.length);

        for (int i = 0; i < n; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
        for (int i = 0; i < n - 1; i++) {
            assertTrue(Double.isFinite(e[i]), "e[" + i + "] should be finite");
        }
    }

    @Test
    void testBasicLower() {
        double[] A = {
            4, 0, 0,
            1, 3, 0,
            2, 1, 2
        };
        int n = 3;
        double[] d = new double[n];
        double[] e = new double[n - 1];
        double[] tau = new double[n - 1];
        double[] work = new double[100];

        Dsytrd.dsytrd(BLAS.Uplo.Lower, n, A, n, d, 0, e, 0, tau, 0, work, work.length);

        assertEquals(4, d[0], TOL);
    }

    @Test
    void testWorkspaceQuery() {
        double[] A = new double[9];
        double[] d = new double[3];
        double[] e = new double[2];
        double[] tau = new double[2];
        double[] work = new double[1];

        Dsytrd.dsytrd(BLAS.Uplo.Upper, 3, A, 3, d, 0, e, 0, tau, 0, work, -1);

        assertTrue(work[0] > 0);
    }

    @Test
    void testEmpty() {
        Dsytrd.dsytrd(BLAS.Uplo.Upper, 0, new double[0], 0, new double[0], 0, new double[0], 0, new double[0], 0, new double[1], 0);
    }

    @Test
    void testSingleElement() {
        double[] A = {5};
        double[] d = new double[1];
        double[] e = new double[0];
        double[] tau = new double[0];
        double[] work = new double[10];

        Dsytrd.dsytrd(BLAS.Uplo.Upper, 1, A, 1, d, 0, e, 0, tau, 0, work, work.length);

        assertEquals(5, d[0], TOL);
    }
}
