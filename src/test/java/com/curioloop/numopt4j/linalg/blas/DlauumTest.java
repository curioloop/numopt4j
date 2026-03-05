/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DlauumTest {

    private static final double TOL = 1e-10;

    @Test
    void testUpper() {
        double[] A = {
            2, 1, 0,
            0, 3, 1,
            0, 0, 2
        };
        int n = 3;

        Dlauum.dlauum(BLAS.Uplo.Upper, n, A, 0, n);

        assertTrue(Double.isFinite(A[0]));
    }

    @Test
    void testLower() {
        double[] A = {
            2, 0, 0,
            1, 3, 0,
            0, 1, 2
        };
        int n = 3;

        Dlauum.dlauum(BLAS.Uplo.Lower, n, A, 0, n);

        assertTrue(Double.isFinite(A[0]));
    }

    @Test
    void testEmpty() {
        Dlauum.dlauum(BLAS.Uplo.Upper, 0, new double[0], 0, 0);
    }

    @Test
    void testSingleElement() {
        double[] A = {5};

        Dlauum.dlauum(BLAS.Uplo.Upper, 1, A, 0, 1);

        assertEquals(25, A[0], TOL);
    }

    @Test
    void testIdentity() {
        double[] A = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        int n = 3;

        Dlauum.dlauum(BLAS.Uplo.Upper, n, A, 0, n);

        assertEquals(1, A[0], TOL);
        assertEquals(1, A[4], TOL);
        assertEquals(1, A[8], TOL);
    }

    @Test
    void testUpperLarge() {
        int n = 10;
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                A[i * n + j] = Math.random();
                A[j * n + i] = A[i * n + j];
            }
        }

        Dlauum.dlauum(BLAS.Uplo.Upper, n, A, 0, n);

        for (int i = 0; i < n; i++) {
            assertTrue(Double.isFinite(A[i * n + i]), "Diagonal element should be finite");
        }
    }

    @Test
    void testLowerLarge() {
        int n = 10;
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                A[i * n + j] = Math.random();
                A[j * n + i] = A[i * n + j];
            }
        }

        Dlauum.dlauum(BLAS.Uplo.Lower, n, A, 0, n);

        for (int i = 0; i < n; i++) {
            assertTrue(Double.isFinite(A[i * n + i]), "Diagonal element should be finite");
        }
    }

    @Test
    void testDiagonalUpper() {
        int n = 5;
        double[] A = new double[n * n];
        double[] diag = {1, 2, 3, 4, 5};
        for (int i = 0; i < n; i++) {
            A[i * n + i] = diag[i];
        }

        Dlauum.dlauum(BLAS.Uplo.Upper, n, A, 0, n);

        for (int i = 0; i < n; i++) {
            assertEquals(diag[i] * diag[i], A[i * n + i], TOL);
        }
    }

    @Test
    void testDiagonalLower() {
        int n = 5;
        double[] A = new double[n * n];
        double[] diag = {1, 2, 3, 4, 5};
        for (int i = 0; i < n; i++) {
            A[i * n + i] = diag[i];
        }

        Dlauum.dlauum(BLAS.Uplo.Lower, n, A, 0, n);

        for (int i = 0; i < n; i++) {
            assertEquals(diag[i] * diag[i], A[i * n + i], TOL);
        }
    }
}
