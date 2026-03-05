/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DsyevTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasicUpper() {
        double[] A = {
            4, 1, 1,
            0, 3, 1,
            0, 0, 2
        };
        int n = 3;
        double[] w = new double[n];
        double[] work = new double[100];

        int info = Dsyev.dsyev('V', 'U', n, A, n, w, 0, work, 0, work.length);

        assertEquals(0, info);

        assertTrue(w[0] <= w[1]);
        assertTrue(w[1] <= w[2]);
    }

    @Test
    void testBasicLower() {
        double[] A = {
            4, 0, 0,
            1, 3, 0,
            1, 1, 2
        };
        int n = 3;
        double[] w = new double[n];
        double[] work = new double[100];

        int info = Dsyev.dsyev('V', 'L', n, A, n, w, 0, work, 0, work.length);

        assertEquals(0, info);

        assertTrue(w[0] <= w[1]);
        assertTrue(w[1] <= w[2]);
    }

    @Test
    void testEigenvaluesOnly() {
        double[] A = {
            4, 1, 1,
            0, 3, 1,
            0, 0, 2
        };
        int n = 3;
        double[] w = new double[n];
        double[] work = new double[100];

        int info = Dsyev.dsyev('N', 'U', n, A, n, w, 0, work, 0, work.length);

        assertEquals(0, info);

        assertTrue(w[0] <= w[1]);
        assertTrue(w[1] <= w[2]);
    }

    @Test
    void testDiagonal() {
        double[] A = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        int n = 3;
        double[] w = new double[n];
        double[] work = new double[100];

        int info = Dsyev.dsyev('N', 'U', n, A, n, w, 0, work, 0, work.length);

        assertEquals(0, info);
        assertTrue(w[0] <= w[1]);
        assertTrue(w[1] <= w[2]);
    }

    @Test
    void testWorkspaceQuery() {
        double[] A = new double[9];
        double[] w = new double[3];
        double[] work = new double[1];

        int info = Dsyev.dsyev('V', 'U', 3, A, 3, w, 0, work, 0, -1);

        assertEquals(0, info);
        assertTrue(work[0] > 0);
    }

    @Test
    void testEmpty() {
        int info = Dsyev.dsyev('V', 'U', 0, new double[0], 0, new double[0], 0, new double[1], 0, 0);
        assertEquals(0, info);
    }

    @Test
    void testSingleElement() {
        double[] A = {5};
        double[] w = new double[1];
        double[] work = new double[10];

        int info = Dsyev.dsyev('V', 'U', 1, A, 1, w, 0, work, 0, work.length);

        assertEquals(0, info);
        assertEquals(5, w[0], TOL);
    }
}
