/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DgerTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        double[] x = {1, 2};
        double[] y = {1, 2, 3};

        Dger.dger(2, 3, 1.0, x, 0, 1, y, 0, 1, A, 0, 3);

        assertEquals(2.0, A[0], TOL);
        assertEquals(4.0, A[1], TOL);
        assertEquals(6.0, A[2], TOL);
        assertEquals(6.0, A[3], TOL);
        assertEquals(9.0, A[4], TOL);
        assertEquals(12.0, A[5], TOL);
    }

    @Test
    void testZeroAlpha() {
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        double[] x = {1, 2};
        double[] y = {1, 2, 3};

        Dger.dger(2, 3, 0.0, x, 0, 1, y, 0, 1, A, 0, 3);

        assertEquals(1.0, A[0], TOL);
        assertEquals(2.0, A[1], TOL);
        assertEquals(3.0, A[2], TOL);
    }

    @Test
    void testEmpty() {
        Dger.dger(0, 0, 1.0, new double[0], 0, 1, new double[0], 0, 1, new double[0], 0, 0);
    }
}
