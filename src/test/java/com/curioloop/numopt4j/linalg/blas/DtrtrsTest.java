/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DtrtrsTest {

    private static final double TOL = 1e-10;

    @Test
    void testUpperNonUnit() {
        double[] A = {
            2, 1, 0,
            0, 3, 1,
            0, 0, 4
        };
        double[] B = {3, 7, 4, 0, 0, 0, 0, 0, 0};

        boolean ok = Dtrtrs.dtrtrs(BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit, 3, 1, A, 0, 3, B, 0, 3);

        assertTrue(ok);
    }

    @Test
    void testLowerNonUnit() {
        double[] A = {
            2, 0, 0,
            1, 3, 0,
            0, 1, 4
        };
        double[] B = {2, 4, 8, 0, 0, 0, 0, 0, 0};

        boolean ok = Dtrtrs.dtrtrs(BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit, 3, 1, A, 0, 3, B, 0, 3);

        assertTrue(ok);
    }

    @Test
    void testEmpty() {
        boolean ok = Dtrtrs.dtrtrs(BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit, 0, 0, new double[0], 0, 0, new double[0], 0, 0);
        assertTrue(ok);
    }
}
