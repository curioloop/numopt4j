/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DlanstTest {

    private static final double TOL = 1e-10;

    @Test
    void testMaxNorm() {
        double[] d = {1, 4, 2};
        double[] e = {3, 5};

        double norm = Dlanst.dlanst('M', 3, d, 0, e, 0);

        assertEquals(5.0, norm, TOL);
    }

    @Test
    void testFrobeniusNorm() {
        double[] d = {1, 2, 3};
        double[] e = {4, 5};

        double norm = Dlanst.dlanst('F', 3, d, 0, e, 0);

        assertTrue(norm > 0);
    }

    @Test
    void testEmpty() {
        double norm = Dlanst.dlanst('M', 0, new double[0], 0, new double[0], 0);
        assertEquals(0.0, norm, TOL);
    }

    @Test
    void testSingleElement() {
        double[] d = {5};
        double[] e = new double[0];

        double norm = Dlanst.dlanst('M', 1, d, 0, e, 0);

        assertEquals(5.0, norm, TOL);
    }
}
