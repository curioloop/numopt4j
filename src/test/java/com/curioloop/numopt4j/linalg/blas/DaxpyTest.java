/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DaxpyTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] x = {1, 2, 3};
        double[] y = {4, 5, 6};

        Daxpy.daxpy(3, 2.0, x, 0, 1, y, 0, 1);

        assertEquals(6.0, y[0], TOL);
        assertEquals(9.0, y[1], TOL);
        assertEquals(12.0, y[2], TOL);
    }

    @Test
    void testZeroAlpha() {
        double[] x = {1, 2, 3};
        double[] y = {4, 5, 6};

        Daxpy.daxpy(3, 0.0, x, 0, 1, y, 0, 1);

        assertEquals(4.0, y[0], TOL);
        assertEquals(5.0, y[1], TOL);
        assertEquals(6.0, y[2], TOL);
    }

    @Test
    void testSingleElement() {
        double[] x = {5};
        double[] y = {3};

        Daxpy.daxpy(1, 2.0, x, 0, 1, y, 0, 1);

        assertEquals(13.0, y[0], TOL);
    }

    @Test
    void testEmpty() {
        Daxpy.daxpy(0, 2.0, new double[0], 0, 1, new double[0], 0, 1);
    }
}
