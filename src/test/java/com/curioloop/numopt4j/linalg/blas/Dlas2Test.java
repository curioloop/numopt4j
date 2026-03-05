/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dlas2Test {

    private static final double TOL = 1e-12;

    @Test
    void testDlas2Basic() {
        double[] out = new double[2];
        Dlas2.dlas2(3, 2, 5, out, 0);

        assertTrue(out[0] >= 0);
        assertTrue(out[1] >= out[0]);
    }

    @Test
    void testDlas2Zero() {
        double[] out = new double[2];
        Dlas2.dlas2(0, 5, 0, out, 0);

        assertEquals(0, out[0], TOL);
        assertEquals(5, out[1], TOL);
    }
}
