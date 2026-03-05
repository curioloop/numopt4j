/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dlaln2Test {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double smin = 1e-10;
        double[] a = {2};
        double[] b = {1};
        double[] x = new double[6];

        boolean ok = Dlaln2.dlaln2(false, 1, 1, smin, 1, a, 0, 1, 1, 1, b, 0, 1, 1, 0, x, 0, 1);

        assertTrue(ok);
    }
}
