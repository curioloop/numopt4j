/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

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
        double[] expected = y.clone();

        dsymvScalar(BLAS.Uplo.Upper, 3, 1.0, A, 0, 3, x, 0, 1, 0.0, expected, 0, 1);
        Dsymv.dsymv(BLAS.Uplo.Upper, 3, 1.0, A, 0, 3, x, 0, 1, 0.0, y, 0, 1);

        assertArrayEquals(expected, y, TOL);
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
        double[] expected = y.clone();

        dsymvScalar(BLAS.Uplo.Lower, 3, 1.0, A, 0, 3, x, 0, 1, 0.0, expected, 0, 1);
        Dsymv.dsymv(BLAS.Uplo.Lower, 3, 1.0, A, 0, 3, x, 0, 1, 0.0, y, 0, 1);

        assertArrayEquals(expected, y, TOL);
    }

    @Test
    void testEmpty() {
        Dsymv.dsymv(BLAS.Uplo.Upper, 0, 1.0, new double[0], 0, 0, new double[0], 0, 1, 0.0, new double[0], 0, 1);
    }

    @Test
    void testUpperStridedYParity() {
        double[] A = {
            2, -1, 3, 4,
            0, 5, -2, 6,
            0, 0, 7, 1,
            0, 0, 0, 8
        };
        double[] x = {1, 2, 3, 4};
        double[] expected = {0.5, 99, -1.5, 77, 2.0, 55, -2.5, 33};
        double[] actual = expected.clone();

        dsymvScalar(BLAS.Uplo.Upper, 4, -0.25, A, 0, 4, x, 0, 1, 1.0, expected, 0, 2);
        Dsymv.dsymv(BLAS.Uplo.Upper, 4, -0.25, A, 0, 4, x, 0, 1, 1.0, actual, 0, 2);

        assertArrayEquals(expected, actual, TOL);
    }

    @Test
    void testLowerStridedXYParity() {
        int n = 5;
        int lda = 5;
        double[] A = {
            4, 0, 0, 0, 0,
            1, 5, 0, 0, 0,
            -2, 3, 6, 0, 0,
            7, -1, 2, 8, 0,
            4, 5, -3, 1, 9
        };
        double[] x = new double[1 + (n - 1) * 3];
        double[] yExpected = new double[1 + (n - 1) * 2];
        double[] yActual = new double[yExpected.length];
        for (int i = 0; i < n; i++) {
            x[i * 3] = i + 1.0;
            yExpected[i * 2] = 0.25 * (i + 1);
        }
        System.arraycopy(yExpected, 0, yActual, 0, yExpected.length);

        dsymvScalar(BLAS.Uplo.Lower, n, 0.5, A, 0, lda, x, 0, 3, -1.0, yExpected, 0, 2);
        Dsymv.dsymv(BLAS.Uplo.Lower, n, 0.5, A, 0, lda, x, 0, 3, -1.0, yActual, 0, 2);

        assertArrayEquals(yExpected, yActual, TOL);
    }

    @Test
    void testUpperLargeStridedXParity() {
        verifyRandomParity(BLAS.Uplo.Upper, 96, 96, 0.75, -0.5, 96, 1, 1e-9);
    }

    @Test
    void testLowerLargeStridedXParity() {
        verifyRandomParity(BLAS.Uplo.Lower, 128, 128, -1.25, 0.25, 128, 1, 1e-9);
    }

    private static void verifyRandomParity(BLAS.Uplo uplo, int n, int incX, double alpha, double beta,
                                           int seedOffset, int incY, double tolerance) {
        Random random = new Random(1234L + seedOffset + uplo.ordinal());
        double[] A = new double[n * n];
        double[] x = new double[1 + (n - 1) * incX];
        double[] expected = new double[1 + (n - 1) * incY];
        double[] actual = new double[expected.length];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = random.nextDouble() - 0.5;
            }
            x[i * incX] = random.nextDouble() - 0.5;
            expected[i * incY] = random.nextDouble() - 0.5;
        }
        System.arraycopy(expected, 0, actual, 0, expected.length);

        dsymvScalar(uplo, n, alpha, A, 0, n, x, 0, incX, beta, expected, 0, incY);
        Dsymv.dsymv(uplo, n, alpha, A, 0, n, x, 0, incX, beta, actual, 0, incY);

        assertArrayEquals(expected, actual, tolerance);
    }

    private static void dsymvScalar(BLAS.Uplo uplo, int n, double alpha,
                                    double[] A, int aOff, int lda,
                                    double[] x, int xOff, int incX,
                                    double beta, double[] y, int yOff, int incY) {
        if (n <= 0) {
            return;
        }

        for (int i = 0; i < n; i++) {
            y[yOff + i * incY] *= beta;
        }
        if (alpha == 0.0) {
            return;
        }

        boolean upper = uplo == BLAS.Uplo.Upper;
        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i * incX];
            double temp1 = alpha * xi;
            int rowOff = aOff + i * lda;
            double sum = upper ? xi * A[rowOff + i] : 0.0;

            if (upper) {
                for (int j = i + 1; j < n; j++) {
                    double aij = A[rowOff + j];
                    sum = Math.fma(x[xOff + j * incX], aij, sum);
                    y[yOff + j * incY] = Math.fma(temp1, aij, y[yOff + j * incY]);
                }
            } else {
                for (int j = 0; j < i; j++) {
                    double aij = A[rowOff + j];
                    sum = Math.fma(x[xOff + j * incX], aij, sum);
                    y[yOff + j * incY] = Math.fma(temp1, aij, y[yOff + j * incY]);
                }
                sum = Math.fma(xi, A[rowOff + i], sum);
            }

            y[yOff + i * incY] = Math.fma(alpha, sum, y[yOff + i * incY]);
        }
    }
}
