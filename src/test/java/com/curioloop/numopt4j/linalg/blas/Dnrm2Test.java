/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dnrm2Test {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] x = {3, 4};
        double norm = Dnrm2.dnrm2(2, x, 0, 1);
        assertEquals(5.0, norm, TOL);
    }

    @Test
    void testZero() {
        double[] x = {0, 0, 0};
        double norm = Dnrm2.dnrm2(3, x, 0, 1);
        assertEquals(0.0, norm, TOL);
    }

    @Test
    void testSingleElement() {
        double[] x = {5};
        double norm = Dnrm2.dnrm2(1, x, 0, 1);
        assertEquals(5.0, norm, TOL);
    }

    @Test
    void testEmpty() {
        double norm = Dnrm2.dnrm2(0, new double[0], 0, 1);
        assertEquals(0.0, norm, TOL);
    }

    @Test
    void testNegative() {
        double[] x = {-3, -4};
        double norm = Dnrm2.dnrm2(2, x, 0, 1);
        assertEquals(5.0, norm, TOL);
    }
}
