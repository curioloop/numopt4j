/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DlatrsTest {

    private static final double TOL = 1e-10;

    @Test
    void testUpper() {
        double[] A = {
            2, 1, 0,
            0, 3, 1,
            0, 0, 4
        };
        double[] x = {3, 7, 4};
        double[] cnorm = new double[3];

        double scale = Dlatrs.dlatrs(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, false, 3, A, 3, x, 0, cnorm, 0);

        assertTrue(scale > 0);
        assertTrue(Double.isFinite(x[0]));
        assertTrue(Double.isFinite(x[1]));
        assertTrue(Double.isFinite(x[2]));
    }

    @Test
    void testLower() {
        double[] A = {
            2, 0, 0,
            1, 3, 0,
            0, 1, 4
        };
        double[] x = {3, 7, 4};
        double[] cnorm = new double[3];

        double scale = Dlatrs.dlatrs(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, false, 3, A, 3, x, 0, cnorm, 0);

        assertTrue(scale > 0);
    }

    @Test
    void testEmpty() {
        double scale = Dlatrs.dlatrs(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, false, 0, new double[0], 0, new double[0], 0, new double[0], 0);
        assertEquals(1.0, scale, TOL);
    }

    @Test
    void testSingleElement() {
        double[] A = {5};
        double[] x = {3};
        double[] cnorm = new double[1];

        double scale = Dlatrs.dlatrs(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, false, 1, A, 1, x, 0, cnorm, 0);

        assertEquals(1.0, scale, TOL);
        assertEquals(0.6, x[0], TOL);
    }
}
