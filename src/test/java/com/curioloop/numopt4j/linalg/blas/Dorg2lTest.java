/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dorg2lTest {

    @Test
    void testBasic() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        int m = 3, n = 2, k = 2;
        double[] tau = {0.5, 0.3};
        double[] work = new double[n];

        Dorgql.dorg2l(m, n, k, A, 0, m, tau, 0, work, 0);

        assertTrue(Double.isFinite(A[0]));
    }

    @Test
    void testEmpty() {
        Dorgql.dorg2l(0, 0, 0, new double[0], 0, 0, new double[0], 0, new double[0], 0);
    }
}
