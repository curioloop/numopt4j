/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DgeevTest {

    private static final double TOL = 1e-10;

    @Test
    void testDiagonal() {
        double[] A = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        int n = 3;
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vl = new double[n * n];
        double[] vr = new double[n * n];
        double[] work = new double[100];

        int info = Dgeev.dgeev(false, false, n, A, n, wr, wi, vl, n, vr, n, work, 0, work.length);

        assertEquals(0, info);
    }

    @Test
    void testEigenvaluesOnly() {
        double[] A = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 6
        };
        int n = 3;
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];

        int info = Dgeev.dgeev(false, false, n, A, n, wr, wi, null, 1, null, 1, work, 0, work.length);

        assertEquals(0, info);
    }

    @Test
    void testSingleElement() {
        double[] A = {5};
        double[] wr = new double[1];
        double[] wi = new double[1];
        double[] work = new double[10];

        int info = Dgeev.dgeev(false, false, 1, A, 1, wr, wi, null, 1, null, 1, work, 0, work.length);

        assertEquals(0, info);
        assertEquals(5, wr[0], TOL);
        assertEquals(0, wi[0], TOL);
    }
}
