/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DrotTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] x = {3, 4};
        double[] y = {5, 6};

        Drot.drot(2, x, 0, 1, y, 0, 1, 0.6, 0.8);

        assertTrue(Double.isFinite(x[0]));
        assertTrue(Double.isFinite(x[1]));
        assertTrue(Double.isFinite(y[0]));
        assertTrue(Double.isFinite(y[1]));
    }

    @Test
    void testIdentity() {
        double[] x = {3, 4};
        double[] y = {5, 6};

        Drot.drot(2, x, 0, 1, y, 0, 1, 1.0, 0.0);

        assertEquals(3.0, x[0], TOL);
        assertEquals(4.0, x[1], TOL);
        assertEquals(5.0, y[0], TOL);
        assertEquals(6.0, y[1], TOL);
    }

    @Test
    void testEmpty() {
        Drot.drot(0, new double[0], 0, 1, new double[0], 0, 1, 0.6, 0.8);
    }
}
