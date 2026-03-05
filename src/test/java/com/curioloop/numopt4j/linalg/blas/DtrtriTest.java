/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DtrtriTest {

    private static final double TOL = 1e-10;

    @Test
    void testUpperNonUnit() {
        double[] A = {
            4, 2, 1,
            0, 3, 1,
            0, 0, 2
        };
        int n = 3;

        int info = Dtrtri.dtrtri(BLAS.Uplo.Upper, BLAS.Diag.NonUnit, n, A, 0, n);

        assertEquals(0, info);

        double[] expected = {
            0.25, -0.16666666666666666, -0.041666666666666664,
            0, 0.3333333333333333, -0.16666666666666666,
            0, 0, 0.5
        };

        for (int i = 0; i < n * n; i++) {
            assertEquals(expected[i], A[i], TOL);
        }
    }

    @Test
    void testLowerNonUnit() {
        double[] A = {
            2, 0, 0,
            1, 3, 0,
            1, 2, 4
        };
        int n = 3;

        int info = Dtrtri.dtrtri(BLAS.Uplo.Lower, BLAS.Diag.NonUnit, n, A, 0, n);

        assertEquals(0, info);

        assertEquals(0.5, A[0], TOL);
        assertEquals(0.3333333333333333, A[4], TOL);
        assertEquals(0.25, A[8], TOL);
    }

    @Test
    void testUnitDiagonal() {
        double[] A = {
            1, 2, 3,
            0, 1, 4,
            0, 0, 1
        };
        int n = 3;

        int info = Dtrtri.dtrtri(BLAS.Uplo.Upper, BLAS.Diag.Unit, n, A, 0, n);

        assertEquals(0, info);

        assertEquals(1, A[0], TOL);
        assertEquals(-2, A[1], TOL);
        assertEquals(5, A[2], TOL);
        assertEquals(1, A[4], TOL);
        assertEquals(-4, A[5], TOL);
        assertEquals(1, A[8], TOL);
    }

    @Test
    void testIdentity() {
        double[] A = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        int n = 3;

        int info = Dtrtri.dtrtri(BLAS.Uplo.Upper, BLAS.Diag.NonUnit, n, A, 0, n);

        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            assertEquals(1, A[i * n + i], TOL);
        }
    }

    @Test
    void testSingular() {
        double[] A = {
            0, 1,
            0, 0
        };
        int n = 2;

        int info = Dtrtri.dtrtri(BLAS.Uplo.Upper, BLAS.Diag.NonUnit, n, A, 0, n);

        assertNotEquals(0, info);
    }

    @Test
    void testEmpty() {
        int info = Dtrtri.dtrtri(BLAS.Uplo.Upper, BLAS.Diag.NonUnit, 0, new double[0], 0, 0);
        assertEquals(0, info);
    }
}
