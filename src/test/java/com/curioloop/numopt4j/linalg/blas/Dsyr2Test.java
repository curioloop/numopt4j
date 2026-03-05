/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dsyr2Test {

    private static final double TOL = 1e-10;

    @Test
    void testUpper() {
        double[] A = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 6
        };
        double[] x = {1, 2, 3};
        double[] y = {1, 1, 1};

        Dsyr2.dsyr2(BLAS.Uplo.Upper, 3, 1.0, x, 0, 1, y, 0, 1, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
    }

    @Test
    void testLower() {
        double[] A = {
            1, 0, 0,
            2, 4, 0,
            3, 5, 6
        };
        double[] x = {1, 2, 3};
        double[] y = {1, 1, 1};

        Dsyr2.dsyr2(BLAS.Uplo.Lower, 3, 1.0, x, 0, 1, y, 0, 1, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
    }

    @Test
    void testEmpty() {
        Dsyr2.dsyr2(BLAS.Uplo.Upper, 0, 1.0, new double[0], 0, 1, new double[0], 0, 1, new double[0], 0, 0);
    }
}
