/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DscalTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] x = {1, 2, 3};

        Dscal.dscal(3, 2.0, x, 0, 1);

        assertEquals(2.0, x[0], TOL);
        assertEquals(4.0, x[1], TOL);
        assertEquals(6.0, x[2], TOL);
    }

    @Test
    void testZero() {
        double[] x = {1, 2, 3};

        Dscal.dscal(3, 0.0, x, 0, 1);

        assertEquals(0.0, x[0], TOL);
        assertEquals(0.0, x[1], TOL);
        assertEquals(0.0, x[2], TOL);
    }

    @Test
    void testSingleElement() {
        double[] x = {5};

        Dscal.dscal(1, 3.0, x, 0, 1);

        assertEquals(15.0, x[0], TOL);
    }

    @Test
    void testEmpty() {
        Dscal.dscal(0, 2.0, new double[0], 0, 1);
    }
}
