/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dgebd2Test {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        int m = 4, n = 5;
        double[] A = {
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20
        };
        int minmn = Math.min(m, n);
        double[] d = new double[minmn];
        double[] e = new double[minmn - 1];
        double[] tauQ = new double[minmn];
        double[] tauP = new double[minmn];
        double[] work = new double[Math.max(m, n)];

        Dgebd2.dgebd2(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertTrue(Double.isFinite(d[0]));
        assertTrue(Double.isFinite(tauQ[0]));
    }

    @Test
    void testSquare() {
        int n = 3;
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] d = new double[n];
        double[] e = new double[n - 1];
        double[] tauQ = new double[n];
        double[] tauP = new double[n];
        double[] work = new double[n];

        Dgebd2.dgebd2(n, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertTrue(Double.isFinite(d[0]));
    }

    @Test
    void testEmpty() {
        Dgebd2.dgebd2(0, 0, new double[0], 0, 0,
                new double[0], 0, new double[0], 0,
                new double[0], 0, new double[0], 0,
                new double[0], 0);
    }

    @Test
    void testSingleElement() {
        double[] A = {5};
        double[] d = new double[1];
        double[] e = new double[0];
        double[] tauQ = new double[1];
        double[] tauP = new double[1];
        double[] work = new double[1];

        Dgebd2.dgebd2(1, 1, A, 0, 1, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertEquals(5, d[0], TOL);
    }

    @Test
    void testTwoByTwo() {
        double[] A = {
            4, 3,
            6, 5
        };
        int m = 2, n = 2;
        double[] d = new double[2];
        double[] e = new double[1];
        double[] tauQ = new double[2];
        double[] tauP = new double[2];
        double[] work = new double[2];

        Dgebd2.dgebd2(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        assertTrue(Double.isFinite(d[0]));
        assertTrue(Double.isFinite(d[1]));
        assertTrue(Double.isFinite(e[0]));
    }

    @Test
    void testTallMatrix() {
        int m = 5, n = 3;
        double[] A = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            A[i] = Math.random();
        }
        int minmn = Math.min(m, n);
        double[] d = new double[minmn];
        double[] e = new double[minmn - 1];
        double[] tauQ = new double[minmn];
        double[] tauP = new double[minmn];
        double[] work = new double[Math.max(m, n)];

        Dgebd2.dgebd2(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        for (int i = 0; i < minmn; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
    }

    @Test
    void testWideMatrix() {
        int m = 3, n = 5;
        double[] A = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            A[i] = Math.random();
        }
        int minmn = Math.min(m, n);
        double[] d = new double[minmn];
        double[] e = new double[minmn - 1];
        double[] tauQ = new double[minmn];
        double[] tauP = new double[minmn];
        double[] work = new double[Math.max(m, n)];

        Dgebd2.dgebd2(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        for (int i = 0; i < minmn; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
    }

    @Test
    void testIdentityMatrix() {
        int n = 4;
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            A[i * n + i] = 1.0;
        }
        double[] d = new double[n];
        double[] e = new double[n - 1];
        double[] tauQ = new double[n];
        double[] tauP = new double[n];
        double[] work = new double[n];

        Dgebd2.dgebd2(n, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        for (int i = 0; i < n; i++) {
            assertEquals(1.0, Math.abs(d[i]), TOL, "d[" + i + "] should be 1 or -1");
        }
        for (int i = 0; i < n - 1; i++) {
            assertEquals(0.0, e[i], TOL, "e[" + i + "] should be 0");
        }
    }

    @Test
    void testRandomMatrix() {
        int m = 6, n = 6;
        double[] A = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            A[i] = Math.random() * 10 - 5;
        }
        int minmn = Math.min(m, n);
        double[] d = new double[minmn];
        double[] e = new double[minmn - 1];
        double[] tauQ = new double[minmn];
        double[] tauP = new double[minmn];
        double[] work = new double[Math.max(m, n)];

        Dgebd2.dgebd2(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        for (int i = 0; i < minmn; i++) {
            assertTrue(Double.isFinite(d[i]));
        }
        for (int i = 0; i < minmn - 1; i++) {
            assertTrue(Double.isFinite(e[i]));
        }
    }

    @Test
    void testLargeMatrix() {
        int m = 20, n = 15;
        double[] A = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            A[i] = Math.random();
        }
        int minmn = Math.min(m, n);
        double[] d = new double[minmn];
        double[] e = new double[minmn - 1];
        double[] tauQ = new double[minmn];
        double[] tauP = new double[minmn];
        double[] work = new double[Math.max(m, n)];

        Dgebd2.dgebd2(m, n, A, 0, n, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);

        for (int i = 0; i < minmn; i++) {
            assertTrue(Double.isFinite(d[i]), "d[" + i + "] should be finite");
        }
    }
}
