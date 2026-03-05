/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DtrsenTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasicReorder() {
        double[] T = {
            0.7995, -0.1144, 0.0060, 0.0336,
            0.0000, -0.0994, 0.2478, 0.3474,
            0.0000, -0.6483, -0.0994, 0.2026,
            0.0000, 0.0000, 0.0000, -0.1007
        };

        double[] Q = {
            0.6551, 0.1037, 0.3450, 0.6641,
            0.5236, -0.5807, -0.6141, -0.1068,
            -0.5362, -0.3073, -0.2935, 0.7293,
            0.0956, 0.7467, -0.6463, 0.1249
        };

        int n = 4;
        boolean[] selects = {true, false, false, true};
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];
        int[] iwork = new int[10];

        boolean ok = Dtrsen.dtrsen(Dtrsen.BOTH_COND, true, selects, n,
                T, 0, n, Q, 0, n, wr, wi, work, work.length, iwork, iwork.length);

        assertTrue(ok);
        assertEquals(2, iwork[1]);
    }

    @Test
    void testNoReorder() {
        double[] T = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        int n = 3;

        boolean[] selects = {true, true, true};
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];
        int[] iwork = new int[10];

        boolean ok = Dtrsen.dtrsen(Dtrsen.NO_COND, false, selects, n,
                T, 0, n, null, 0, n, wr, wi, work, work.length, iwork, iwork.length);

        assertTrue(ok);
        assertEquals(3, iwork[1]);
        assertEquals(1, wr[0], TOL);
        assertEquals(2, wr[1], TOL);
        assertEquals(3, wr[2], TOL);
    }

    @Test
    void testSelectNone() {
        double[] T = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        int n = 3;

        boolean[] selects = {false, false, false};
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];
        int[] iwork = new int[10];

        boolean ok = Dtrsen.dtrsen(Dtrsen.NO_COND, false, selects, n,
                T, 0, n, null, 0, n, wr, wi, work, work.length, iwork, iwork.length);

        assertTrue(ok);
        assertEquals(0, iwork[1]);
    }

    @Test
    void testComplexEigenvalues() {
        double[] T = {
            0, -1, 0,
            1, 0, 0,
            0, 0, 2
        };
        int n = 3;

        boolean[] selects = {true, true, false};
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];
        int[] iwork = new int[10];

        boolean ok = Dtrsen.dtrsen(Dtrsen.NO_COND, false, selects, n,
                T, 0, n, null, 0, n, wr, wi, work, work.length, iwork, iwork.length);

        assertTrue(ok);
        assertEquals(2, iwork[1]);
    }

    @Test
    void testSubspaceCondition() {
        double[] T = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        int n = 3;

        boolean[] selects = {true, false, false};
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];
        int[] iwork = new int[10];

        boolean ok = Dtrsen.dtrsen(Dtrsen.SUBSPACE_COND, false, selects, n,
                T, 0, n, null, 0, n, wr, wi, work, work.length, iwork, iwork.length);

        assertTrue(ok);
        assertEquals(1, iwork[1]);
        assertEquals(1.0, work[1], TOL); // s
    }

    @Test
    void testEigenvalCondition() {
        double[] T = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        int n = 3;

        boolean[] selects = {true, false, false};
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];
        int[] iwork = new int[10];

        boolean ok = Dtrsen.dtrsen(Dtrsen.EIGENVAL_COND, false, selects, n,
                T, 0, n, null, 0, n, wr, wi, work, work.length, iwork, iwork.length);

        assertTrue(ok);
        assertEquals(1, iwork[1]);
        assertTrue(work[2] > 0); // sep
    }

    @Test
    void testWorkspaceQuery() {
        double[] T = new double[4];
        double[] wr = new double[2];
        double[] wi = new double[2];
        double[] work = new double[3];
        int[] iwork = new int[2];
        boolean[] selects = {true, false};

        boolean ok = Dtrsen.dtrsen(Dtrsen.NO_COND, false, selects, 2,
                T, 0, 2, null, 0, 2, wr, wi, work, -1, iwork, -1);

        assertTrue(ok);
        assertTrue(work[0] > 0);
        assertTrue(iwork[0] > 0);
    }

    @Test
    void testZeroMatrix() {
        double[] T = new double[4];
        int n = 2;

        boolean[] selects = {true, false};
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];
        int[] iwork = new int[10];

        boolean ok = Dtrsen.dtrsen(Dtrsen.NO_COND, false, selects, n,
                T, 0, n, null, 0, n, wr, wi, work, work.length, iwork, iwork.length);

        assertTrue(ok);
        assertEquals(1, iwork[1]);
        assertEquals(0, wr[0], TOL);
        assertEquals(0, wr[1], TOL);
    }

    @Test
    void testWithQUpdate() {
        double[] T = {
            2, 1, 0,
            0, 1, 1,
            0, 0, 0
        };

        double[] Q = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        int n = 3;

        boolean[] selects = {true, false, true};
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];
        int[] iwork = new int[10];

        boolean ok = Dtrsen.dtrsen(Dtrsen.NO_COND, true, selects, n,
                T, 0, n, Q, 0, n, wr, wi, work, work.length, iwork, iwork.length);

        assertTrue(ok);
        assertEquals(2, iwork[1]);
    }
}
