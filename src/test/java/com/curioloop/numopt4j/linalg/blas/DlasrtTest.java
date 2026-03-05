/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DlasrtTest {

    private static final double TOL = 1e-10;

    @Test
    void testSortIncreasing() {
        double[] d = {3, 1, 4, 1, 5};

        Dlasrt.dlasrt('I', 5, d, 0);

        assertEquals(1.0, d[0], TOL);
        assertEquals(1.0, d[1], TOL);
        assertEquals(3.0, d[2], TOL);
        assertEquals(4.0, d[3], TOL);
        assertEquals(5.0, d[4], TOL);
    }

    @Test
    void testSortDecreasing() {
        double[] d = {3, 1, 4, 1, 5};

        Dlasrt.dlasrt('D', 5, d, 0);

        assertEquals(5.0, d[0], TOL);
        assertEquals(4.0, d[1], TOL);
        assertEquals(3.0, d[2], TOL);
        assertEquals(1.0, d[3], TOL);
        assertEquals(1.0, d[4], TOL);
    }

    @Test
    void testEmpty() {
        Dlasrt.dlasrt('I', 0, new double[0], 0);
    }

    @Test
    void testSingleElement() {
        double[] d = {5};

        Dlasrt.dlasrt('I', 1, d, 0);

        assertEquals(5.0, d[0], TOL);
    }
}
