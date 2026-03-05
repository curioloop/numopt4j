/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dlasq1Test {

    private static final double TOL = 1e-10;

    @Test
    void testSingleElement() {
        double[] d = {5};
        double[] e = new double[1];
        double[] work = new double[10];

        int info = Dlasq1.dlasq1(1, d, 0, e, 0, work, 0);

        assertEquals(0, info);
        assertEquals(5, d[0], TOL);
    }

    @Test
    void testEmpty() {
        double[] d = new double[0];
        double[] e = new double[0];
        double[] work = new double[1];

        int info = Dlasq1.dlasq1(0, d, 0, e, 0, work, 0);

        assertEquals(0, info);
    }

    @Test
    void testTwoElements() {
        double[] d = {4, 3};
        double[] e = {1};
        double[] work = new double[10];

        int info = Dlasq1.dlasq1(2, d, 0, e, 0, work, 0);

        assertEquals(0, info);
        assertTrue(d[0] >= d[1], "Singular values should be sorted in descending order");
    }

    @Test
    void testDiagonalMatrix() {
        int n = 5;
        double[] d = {5, 4, 3, 2, 1};
        double[] e = {0, 0, 0, 0};
        double[] work = new double[4 * n];

        int info = Dlasq1.dlasq1(n, d, 0, e, 0, work, 0);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) {
            assertEquals(n - i, d[i], TOL, "Diagonal element " + i);
        }
    }

    @Test
    void testIdentityMatrix() {
        int n = 4;
        double[] d = {1, 1, 1, 1};
        double[] e = {0, 0, 0};
        double[] work = new double[4 * n];

        int info = Dlasq1.dlasq1(n, d, 0, e, 0, work, 0);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) {
            assertEquals(1.0, d[i], TOL);
        }
    }

    @Test
    void testNegativeDiagonal() {
        double[] d = {-5};
        double[] e = new double[1];
        double[] work = new double[10];

        int info = Dlasq1.dlasq1(1, d, 0, e, 0, work, 0);

        assertEquals(0, info);
        assertEquals(5, d[0], TOL);
    }
}
