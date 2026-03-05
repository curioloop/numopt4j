/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DsteqrTest {

    private static final double TOL = 1e-10;

    @Test
    void testEigenvaluesOnly() {
        int n = 3;
        double[] d = {4, 3, 2};
        double[] e = {1, 1};
        double[] work = new double[2 * n];

        int info = Dsteqr.dsteqr('N', n, d, 0, e, 0, null, 0, n, work, 0);

        assertEquals(0, info);
    }

    @Test
    void testWithVectors() {
        int n = 3;
        double[] d = {4, 3, 2};
        double[] e = {1, 1};
        double[] Z = new double[n * n];
        double[] work = new double[2 * n];

        int info = Dsteqr.dsteqr('I', n, d, 0, e, 0, Z, 0, n, work, 0);

        assertEquals(0, info);
    }

    @Test
    void testEmpty() {
        int info = Dsteqr.dsteqr('N', 0, new double[0], 0, new double[0], 0, null, 0, 0, new double[1], 0);
        assertEquals(0, info);
    }

    @Test
    void testSingleElement() {
        double[] d = {5};
        double[] e = new double[0];
        double[] work = new double[10];

        int info = Dsteqr.dsteqr('N', 1, d, 0, e, 0, null, 0, 1, work, 0);

        assertEquals(0, info);
        assertEquals(5, d[0], TOL);
    }
}
