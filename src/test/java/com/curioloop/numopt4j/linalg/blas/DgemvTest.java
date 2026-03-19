/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DgemvTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        double[] x = {1, 2, 3};
        double[] y = {0, 0};

        Dgemv.dgemv(BLAS.Trans.NoTrans, 2, 3, 1.0, A, 0, 3, x, 0, 1, 0.0, y, 0, 1);

        assertEquals(14.0, y[0], TOL);
        assertEquals(32.0, y[1], TOL);
    }
}
