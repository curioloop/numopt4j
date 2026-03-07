/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DorghrTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        int n = 3;
        double[] tau = {0.5, 0.3};
        double[] work = new double[100];

        Dgees.dorghr(n, 0, n - 1, A, n, tau, 0, work, 0, work.length);

        assertTrue(Double.isFinite(A[0]));
    }

    @Test
    void testEmpty() {
        Dgees.dorghr(0, 0, -1, new double[0], 0, new double[0], 0, new double[1], 0, 0);
    }

    @Test
    void testSingleElement() {
        double[] A = {5};
        double[] tau = new double[1];
        double[] work = new double[10];

        Dgees.dorghr(1, 0, 0, A, 1, tau, 0, work, 0, work.length);

        assertEquals(5, A[0], TOL);
    }
}
