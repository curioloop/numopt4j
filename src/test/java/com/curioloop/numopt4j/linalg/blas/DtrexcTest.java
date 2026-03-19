/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DtrexcTest {

    private static final double TOL = 1e-13;

    private static int workSize(int n) {
        return Math.max(n, 32);
    }

    @Test
    void testBasicReorder() {
        double[] T = {
            1, 2, 0,
            0, 3, 4,
            0, 0, 5
        };
        int n = 3;
        double[] work = new double[workSize(n)];

        boolean ok = Dtrexc.dtrexc(false, n, T, 0, n, null, 0, n, 2, 0, work, 0);

        assertTrue(ok);
    }

    @Test
    void testWithQ() {
        double[] T = {
            1, 2, 0,
            0, 3, 4,
            0, 0, 5
        };
        double[] Q = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        int n = 3;
        double[] work = new double[workSize(n)];

        boolean ok = Dtrexc.dtrexc(true, n, T, 0, n, Q, 0, n, 2, 0, work, 0);

        assertTrue(ok);
    }

    @Test
    void testNoMove() {
        double[] T = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        double[] TCopy = T.clone();
        int n = 3;
        double[] work = new double[workSize(n)];

        boolean ok = Dtrexc.dtrexc(false, n, T, 0, n, null, 0, n, 1, 1, work, 0);

        assertTrue(ok);
    }

    @Test
    void testEmpty() {
        double[] work = new double[32];
        boolean ok = Dtrexc.dtrexc(false, 0, new double[0], 0, 0, null, 0, 0, 0, 0, work, 0);
        assertTrue(ok);
    }

    @Test
    void testSingleElement() {
        double[] T = {5};
        double[] work = new double[workSize(1)];

        boolean ok = Dtrexc.dtrexc(false, 1, T, 0, 1, null, 0, 1, 0, 0, work, 0);

        assertTrue(ok);
    }
}
