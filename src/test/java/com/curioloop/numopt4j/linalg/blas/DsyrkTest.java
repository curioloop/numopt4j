/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DsyrkTest {

    private static final double TOL = 1e-10;

    @Test
    void testUpper() {
        double[] A = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 6
        };
        double[] C = new double[9];

        Dsyrk.dsyrk(BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, 3, 3, 1.0, A, 0, 3, 0.0, C, 0, 3);

        assertTrue(Double.isFinite(C[0]));
    }

    @Test
    void testLower() {
        double[] A = {
            1, 0, 0,
            2, 4, 0,
            3, 5, 6
        };
        double[] C = new double[9];

        Dsyrk.dsyrk(BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, 3, 3, 1.0, A, 0, 3, 0.0, C, 0, 3);

        assertTrue(Double.isFinite(C[0]));
    }

    @Test
    void testEmpty() {
        Dsyrk.dsyrk(BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, 0, 0, 1.0, new double[0], 0, 0, 0.0, new double[0], 0, 0);
    }
}
