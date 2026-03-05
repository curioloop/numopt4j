/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dsytf2Test {

    private static final double TOL = 1e-10;

    @Test
    void testEmpty() {
        int[] ipiv = new int[0];
        double[] work = new double[0];
        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Lower, 0, new double[0], 0, 1, ipiv, 0, work);
        assertTrue(ok);
    }

    @Test
    void testSingleElement() {
        double[] A = {4.0};
        int[] ipiv = new int[1];
        double[] work = new double[1];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Lower, 1, A, 0, 1, ipiv, 0, work);

        assertTrue(ok);
        assertEquals(1, ipiv[0]);
        assertEquals(4.0, A[0], TOL);
    }

    @Test
    void testLowerBasic() {
        double[] A = {
            4, 0,
            2, 3
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work);

        assertTrue(ok);
    }

    @Test
    void testUpperBasic() {
        double[] A = {
            4, 2,
            0, 3
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work);

        assertTrue(ok);
    }

    @Test
    void testIndefiniteMatrix() {
        double[] A = {
            1, 0,
            0, -1
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work);

        assertTrue(ok);
    }

    @Test
    void testSingularMatrix() {
        double[] A = {
            0, 0,
            0, 0
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work);

        assertTrue(ok);
        assertEquals(1, ipiv[0]);
        assertEquals(2, ipiv[1]);
    }

    @Test
    void testPositiveDefiniteLower() {
        double[] A = {
            4, 0, 0,
            2, 5, 0,
            2, 3, 6
        };
        double[] Aorig = A.clone();
        int n = 3;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work);
        assertTrue(ok);

        verifySolveLower(Aorig, A.clone(), ipiv, n);
    }

    @Test
    void testPositiveDefiniteUpper() {
        double[] A = {
            4, 2, 2,
            0, 5, 3,
            0, 0, 6
        };
        double[] Aorig = A.clone();
        int n = 3;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work);
        assertTrue(ok);

        verifySolveUpper(Aorig, A.clone(), ipiv, n);
    }

    @Test
    void testIndefiniteLower() {
        double[] A = {
            1, 0, 0,
            2, 1, 0,
            3, 2, 1
        };
        double[] Aorig = A.clone();
        int n = 3;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work);
        assertTrue(ok);

        verifySolveLower(Aorig, A.clone(), ipiv, n);
    }

    @Test
    void testIndefiniteUpper() {
        double[] A = {
            1, 2, 3,
            0, 1, 2,
            0, 0, 1
        };
        double[] Aorig = A.clone();
        int n = 3;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work);
        assertTrue(ok);

        verifySolveUpper(Aorig, A.clone(), ipiv, n);
    }

    @Test
    void test2x2BlockPivot() {
        double[] A = {
            1, 2,
            2, 1
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work);
        assertTrue(ok);

        assertTrue(ipiv[0] < 0);
    }

    @Test
    void testLargeMatrixLower() {
        int n = 50;
        java.util.Random rand = new java.util.Random(42);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                A[i * n + j] = rand.nextDouble();
                A[j * n + i] = A[i * n + j];
            }
        }
        for (int i = 0; i < n; i++) {
            A[i * n + i] += n;
        }
        double[] Aorig = A.clone();
        int[] ipiv = new int[n];
        double[] work = new double[n];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work);
        assertTrue(ok);

        verifySolveLower(Aorig, A.clone(), ipiv, n);
    }

    @Test
    void testLargeMatrixUpper() {
        int n = 50;
        java.util.Random rand = new java.util.Random(42);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                A[i * n + j] = rand.nextDouble();
                A[j * n + i] = A[i * n + j];
            }
        }
        for (int i = 0; i < n; i++) {
            A[i * n + i] += n;
        }
        double[] Aorig = A.clone();
        int[] ipiv = new int[n];
        double[] work = new double[n];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work);
        assertTrue(ok);

        verifySolveUpper(Aorig, A.clone(), ipiv, n);
    }

    @Test
    void testInvalidUplo() {
        // With enum types, invalid uplo values are prevented at compile time
        // This test verifies that negative n still returns false
        double[] A = {1.0};
        int[] ipiv = new int[1];
        double[] work = new double[1];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Lower, -1, A, 0, 1, ipiv, 0, work);
        assertFalse(ok);
    }

    @Test
    void testNegativeN() {
        double[] A = {1.0};
        int[] ipiv = new int[1];
        double[] work = new double[1];

        boolean ok = Dsytrf.dsytf2(BLAS.Uplo.Lower, -1, A, 0, 1, ipiv, 0, work);
        assertFalse(ok);
    }

    private void verifySolveLower(double[] Aorig, double[] A, int[] ipiv, int n) {
        double[] b = new double[n];
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            b[i] = i + 1;
            x[i] = b[i];
        }

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, x, 0, 1);

        double[] Ax = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double aij = (i >= j) ? Aorig[i * n + j] : Aorig[j * n + i];
                Ax[i] += aij * x[j];
            }
        }

        for (int i = 0; i < n; i++) {
            assertEquals(b[i], Ax[i], 1e-8);
        }
    }

    private void verifySolveUpper(double[] Aorig, double[] A, int[] ipiv, int n) {
        double[] b = new double[n];
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            b[i] = i + 1;
            x[i] = b[i];
        }

        Dsytrs.dsytrs(BLAS.Uplo.Upper, n, 1, A, 0, n, ipiv, 0, x, 0, 1);

        double[] Ax = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double aij = (i <= j) ? Aorig[i * n + j] : Aorig[j * n + i];
                Ax[i] += aij * x[j];
            }
        }

        for (int i = 0; i < n; i++) {
            assertEquals(b[i], Ax[i], 1e-8);
        }
    }
}
