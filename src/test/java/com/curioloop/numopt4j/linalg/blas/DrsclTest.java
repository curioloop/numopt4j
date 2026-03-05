/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DrsclTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] x = {2, 4, 6};

        Drscl.drscl(3, 2.0, x, 0, 1);

        assertEquals(1.0, x[0], TOL);
        assertEquals(2.0, x[1], TOL);
        assertEquals(3.0, x[2], TOL);
    }

    @Test
    void testEmpty() {
        Drscl.drscl(0, 2.0, new double[0], 0, 1);
    }
}
