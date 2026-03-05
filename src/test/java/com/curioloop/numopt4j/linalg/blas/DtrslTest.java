/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DtrslTest {

    @Test
    void testUpper() {
        double[] A = {
            2, 1, 0,
            0, 3, 1,
            0, 0, 4
        };
        double[] b = {3, 7, 4};
        int n = 3;

        int info = Dtrsl.dtrsl(A, 0, n, n, b, 0, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans);

        assertEquals(0, info);
    }

    @Test
    void testLower() {
        double[] A = {
            2, 0, 0,
            1, 3, 0,
            0, 1, 4
        };
        double[] b = {2, 4, 8};
        int n = 3;

        int info = Dtrsl.dtrsl(A, 0, n, n, b, 0, BLAS.Uplo.Lower, BLAS.Transpose.NoTrans);

        assertEquals(0, info);
    }

    @Test
    void testEmpty() {
        int info = Dtrsl.dtrsl(new double[0], 0, 0, 0, new double[0], 0, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans);
        assertEquals(0, info);
    }
}
