/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DorgqlTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        int m = 3, n = 2, k = 2;
        double[] tau = {0.5, 0.3};
        double[] work = new double[100];

        Dorgql.dorgql(m, n, k, A, 0, m, tau, 0, work, 0, work.length);

        assertTrue(Double.isFinite(A[0]));
    }

    @Test
    void testEmpty() {
        Dorgql.dorgql(0, 0, 0, new double[0], 0, 0, new double[0], 0, new double[1], 0, 0);
    }

    @Test
    void testSingleElement() {
        double[] A = {5};
        double[] tau = new double[1];
        double[] work = new double[10];

        Dorgql.dorgql(1, 1, 1, A, 1, 0, tau, 0, work, 0, work.length);

        assertTrue(Double.isFinite(A[0]));
    }
}
