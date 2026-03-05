/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DdotTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] x = {1, 2, 3};
        double[] y = {4, 5, 6};
        double dot = Ddot.ddot(3, x, 0, 1, y, 0, 1);
        assertEquals(32.0, dot, TOL);
    }

    @Test
    void testZero() {
        double[] x = {0, 0, 0};
        double[] y = {1, 2, 3};
        double dot = Ddot.ddot(3, x, 0, 1, y, 0, 1);
        assertEquals(0.0, dot, TOL);
    }

    @Test
    void testSingleElement() {
        double[] x = {5};
        double[] y = {3};
        double dot = Ddot.ddot(1, x, 0, 1, y, 0, 1);
        assertEquals(15.0, dot, TOL);
    }

    @Test
    void testEmpty() {
        double dot = Ddot.ddot(0, new double[0], 0, 1, new double[0], 0, 1);
        assertEquals(0.0, dot, TOL);
    }
}
