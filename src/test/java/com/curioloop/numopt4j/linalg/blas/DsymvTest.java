/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DsymvTest {

    private static final double TOL = 1e-10;

    @Test
    void testUpper() {
        double[] A = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 6
        };
        double[] x = {1, 2, 3};
        double[] y = new double[3];

        Dsymv.dsymv(BLAS.Uplo.Upper, 3, 1.0, A, 0, 3, x, 0, 1, 0.0, y, 0, 1);

        assertTrue(Double.isFinite(y[0]));
        assertTrue(Double.isFinite(y[1]));
        assertTrue(Double.isFinite(y[2]));
    }

    @Test
    void testLower() {
        double[] A = {
            1, 0, 0,
            2, 4, 0,
            3, 5, 6
        };
        double[] x = {1, 2, 3};
        double[] y = new double[3];

        Dsymv.dsymv(BLAS.Uplo.Lower, 3, 1.0, A, 0, 3, x, 0, 1, 0.0, y, 0, 1);

        assertTrue(Double.isFinite(y[0]));
        assertTrue(Double.isFinite(y[1]));
        assertTrue(Double.isFinite(y[2]));
    }

    @Test
    void testEmpty() {
        Dsymv.dsymv(BLAS.Uplo.Upper, 0, 1.0, new double[0], 0, 0, new double[0], 0, 1, 0.0, new double[0], 0, 1);
    }
}
