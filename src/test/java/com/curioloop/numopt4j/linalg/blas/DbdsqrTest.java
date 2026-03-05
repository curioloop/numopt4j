/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DbdsqrTest {

    private static final double TOL = 1e-10;

    private static int workSize(int n) {
        return 4 * n + 11;
    }

    @Test
    void testBasicUpper() {
        int n = 2;
        double[] d = {4, 3};
        double[] e = {1};
        double[] work = new double[workSize(n)];

        boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, 0, 0, 0,
                d, 0, e, 0,
                null, 0, 0,
                null, 0, 0,
                null, 0, 0,
                work, 0);

        assertTrue(ok);
        assertTrue(d[0] >= d[1]);
    }

    @Test
    void testBasicLower() {
        int n = 2;
        double[] d = {4, 3};
        double[] e = {1};
        double[] work = new double[workSize(n)];

        boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Lower, n, 0, 0, 0,
                d, 0, e, 0,
                null, 0, 0,
                null, 0, 0,
                null, 0, 0,
                work, 0);

        assertTrue(ok);
    }

    @Test
    void testWithVT() {
        int n = 2;
        double[] d = {4, 3};
        double[] e = {1};
        double[] vt = {
            1, 0,
            0, 1
        };
        double[] work = new double[workSize(n)];

        boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, n, 0, 0,
                d, 0, e, 0,
                vt, 0, n,
                null, 0, 0,
                null, 0, 0,
                work, 0);

        assertTrue(ok);
    }

    @Test
    void testWithU() {
        int n = 2;
        double[] d = {4, 3};
        double[] e = {1};
        double[] u = {
            1, 0,
            0, 1
        };
        double[] work = new double[workSize(n)];

        boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, 0, n, 0,
                d, 0, e, 0,
                null, 0, 0,
                u, 0, n,
                null, 0, 0,
                work, 0);

        assertTrue(ok);
    }

    @Test
    void testSingleElement() {
        double[] d = {5};
        double[] work = new double[workSize(1)];

        boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, 1, 0, 0, 0,
                d, 0, new double[0], 0,
                null, 0, 0,
                null, 0, 0,
                null, 0, 0,
                work, 0);

        assertTrue(ok);
        assertEquals(5, d[0], TOL);
    }

    @Test
    void testEmpty() {
        boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, 0, 0, 0, 0,
                new double[0], 0, new double[0], 0,
                null, 0, 0,
                null, 0, 0,
                null, 0, 0,
                new double[0], 0);
        assertTrue(ok);
    }
}
