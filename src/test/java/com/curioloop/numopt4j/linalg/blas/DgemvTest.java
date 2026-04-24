/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DgemvTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasicNoTrans() {
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        double[] x = {1, 2, 3};
        double[] y = {0, 0};

        Dgemv.dgemv(BLAS.Trans.NoTrans, 2, 3, 1.0, A, 0, 3, x, 0, 1, 0.0, y, 0, 1);

        assertEquals(14.0, y[0], TOL);
        assertEquals(32.0, y[1], TOL);
    }

    @Test
    void testBasicTrans() {
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        double[] x = {1, 2};
        double[] y = {0, 0, 0};

        Dgemv.dgemv(BLAS.Trans.Trans, 2, 3, 1.0, A, 0, 3, x, 0, 1, 0.0, y, 0, 1);

        assertEquals(9.0, y[0], TOL);
        assertEquals(12.0, y[1], TOL);
        assertEquals(15.0, y[2], TOL);
    }

    @Test
    void testNoTransWithBetaScaling() {
        double[] A = {
            1, 2,
            3, 4
        };
        double[] x = {2, -1};
        double[] y = {10, 20};

        Dgemv.dgemv(BLAS.Trans.NoTrans, 2, 2, 1.5, A, 0, 2, x, 0, 1, -0.5, y, 0, 1);

        assertEquals(-5.0, y[0], TOL);
        assertEquals(-7.0, y[1], TOL);
    }

    @Test
    void testNoTransStride() {
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        double[] x = {1, 99, 2, 99, 3};
        double[] y = {10, 99, 20, 99};

        Dgemv.dgemv(BLAS.Trans.NoTrans, 2, 3, 1.0, A, 0, 3, x, 0, 2, 1.0, y, 0, 2);

        assertEquals(24.0, y[0], TOL);
        assertEquals(52.0, y[2], TOL);
    }

    @Test
    void testTransStride() {
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        double[] x = {1, 99, 2};
        double[] y = {0, 99, 0, 99, 0};

        Dgemv.dgemv(BLAS.Trans.Trans, 2, 3, 1.0, A, 0, 3, x, 0, 2, 0.0, y, 0, 2);

        assertEquals(9.0, y[0], TOL);
        assertEquals(12.0, y[2], TOL);
        assertEquals(15.0, y[4], TOL);
    }

    @Test
    void testNoTransContiguousXStridedYLarge() {
        int m = 96;
        int n = 64;
        int lda = 128;
        int incy = lda;
        Random random = new Random(20260422L);

        double[] A = new double[m * lda];
        double[] x = new double[n];
        double[] expected = new double[(m - 1) * incy + 1];
        double[] actual = new double[expected.length];

        fillRandom(random, A);
        fillRandom(random, x);
        fillRandom(random, expected);
        System.arraycopy(expected, 0, actual, 0, expected.length);

        dgemvScalar(BLAS.Trans.NoTrans, m, n, -0.75, A, 0, lda, x, 0, 1, 0.25, expected, 0, incy);
        Dgemv.dgemv(BLAS.Trans.NoTrans, m, n, -0.75, A, 0, lda, x, 0, 1, 0.25, actual, 0, incy);

        assertVectorClose(expected, actual);
    }

    @Test
    void testTransStridedXContiguousYLarge() {
        int m = 96;
        int n = 128;
        int lda = 256;
        int incx = lda;
        Random random = new Random(20260423L);

        double[] A = new double[m * lda];
        double[] x = new double[(m - 1) * incx + 1];
        double[] expected = new double[n];
        double[] actual = new double[n];

        fillRandom(random, A);
        fillRandom(random, x);
        fillRandom(random, expected);
        System.arraycopy(expected, 0, actual, 0, expected.length);

        dgemvScalar(BLAS.Trans.Trans, m, n, 0.5, A, 0, lda, x, 0, incx, -0.25, expected, 0, 1);
        Dgemv.dgemv(BLAS.Trans.Trans, m, n, 0.5, A, 0, lda, x, 0, incx, -0.25, actual, 0, 1);

        assertVectorClose(expected, actual);
    }

    @Test
    void testNoTransContiguousXPairedFallbackParity() {
        Random random = new Random(20260424L);
        verifyNoTransParity(random, 65, 32, 64, 1);
        verifyNoTransParity(random, 96, 48, 256, 256);
        verifyNoTransParity(random, 129, 64, 256, 256);
    }

    @Test
    void testTransPairedLeafParity() {
        Random random = new Random(20260425L);
        verifyTransParity(random, 17, 16, 16, 1);
        verifyTransParity(random, 31, 32, 256, 256);
        verifyTransParity(random, 31, 48, 256, 256);
        verifyTransParity(random, 31, 63, 63, 1);
    }

    @Test
    void testRandomContiguousParity() {
        Random random = new Random(12345L);
        int[][] shapes = {
            {8, 8},
            {31, 17},
            {64, 64},
            {96, 128},
            {128, 96}
        };

        for (BLAS.Trans trans : new BLAS.Trans[]{BLAS.Trans.NoTrans, BLAS.Trans.Trans}) {
            for (int[] shape : shapes) {
                int m = shape[0];
                int n = shape[1];
                double[] A = new double[m * n];
                double[] x = new double[trans == BLAS.Trans.NoTrans ? n : m];
                double[] expected = new double[trans == BLAS.Trans.NoTrans ? m : n];
                double[] actual = new double[expected.length];

                fillRandom(random, A);
                fillRandom(random, x);
                fillRandom(random, expected);
                System.arraycopy(expected, 0, actual, 0, expected.length);

                double alpha = random.nextDouble() * 2.0 - 1.0;
                double beta = random.nextDouble() * 2.0 - 1.0;

                dgemvScalar(trans, m, n, alpha, A, 0, n, x, 0, 1, beta, expected, 0, 1);
                Dgemv.dgemv(trans, m, n, alpha, A, 0, n, x, 0, 1, beta, actual, 0, 1);

                assertVectorClose(expected, actual);
            }
        }
    }

    private static void fillRandom(Random random, double[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = random.nextDouble() - 0.5;
        }
    }

    private static void verifyNoTransParity(Random random, int m, int n, int lda, int incy) {
        double[] A = new double[m * lda];
        double[] x = new double[n];
        double[] expected = new double[(m - 1) * incy + 1];
        double[] actual = new double[expected.length];

        fillRandom(random, A);
        fillRandom(random, x);
        fillRandom(random, expected);
        System.arraycopy(expected, 0, actual, 0, expected.length);

        double alpha = random.nextDouble() * 2.0 - 1.0;
        double beta = random.nextDouble() * 2.0 - 1.0;

        dgemvScalar(BLAS.Trans.NoTrans, m, n, alpha, A, 0, lda, x, 0, 1, beta, expected, 0, incy);
        Dgemv.dgemv(BLAS.Trans.NoTrans, m, n, alpha, A, 0, lda, x, 0, 1, beta, actual, 0, incy);

        assertVectorClose(expected, actual);
    }

    private static void verifyTransParity(Random random, int m, int n, int lda, int incx) {
        double[] A = new double[m * lda];
        double[] x = new double[(m - 1) * incx + 1];
        double[] expected = new double[n];
        double[] actual = new double[n];

        fillRandom(random, A);
        fillRandom(random, x);
        fillRandom(random, expected);
        System.arraycopy(expected, 0, actual, 0, expected.length);

        double alpha = random.nextDouble() * 2.0 - 1.0;
        double beta = random.nextDouble() * 2.0 - 1.0;

        dgemvScalar(BLAS.Trans.Trans, m, n, alpha, A, 0, lda, x, 0, incx, beta, expected, 0, 1);
        Dgemv.dgemv(BLAS.Trans.Trans, m, n, alpha, A, 0, lda, x, 0, incx, beta, actual, 0, 1);

        assertVectorClose(expected, actual);
    }

    private static void assertVectorClose(double[] expected, double[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            double tolerance = Math.max(1e-12, Math.abs(expected[i]) * 1e-11);
            assertEquals(expected[i], actual[i], tolerance, "mismatch at index " + i);
        }
    }

    private static void dgemvScalar(BLAS.Trans trans, int m, int n, double alpha,
                                    double[] A, int aOff, int lda,
                                    double[] x, int xOff, int incX,
                                    double beta, double[] y, int yOff, int incY) {
        boolean noTrans = trans == BLAS.Trans.NoTrans;
        int lenY = noTrans ? m : n;
        if (lenY <= 0) {
            return;
        }

        if (beta != 1.0) {
            if (beta == 0.0) {
                for (int i = 0; i < lenY; i++) {
                    y[yOff + i * incY] = 0.0;
                }
            } else {
                for (int i = 0; i < lenY; i++) {
                    y[yOff + i * incY] *= beta;
                }
            }
        }

        if (alpha == 0.0) {
            return;
        }

        if (noTrans) {
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                int rowOff = aOff + i * lda;
                for (int j = 0; j < n; j++) {
                    sum = Math.fma(A[rowOff + j], x[xOff + j * incX], sum);
                }
                y[yOff + i * incY] = Math.fma(alpha, sum, y[yOff + i * incY]);
            }
        } else {
            for (int i = 0; i < m; i++) {
                double axi = alpha * x[xOff + i * incX];
                int rowOff = aOff + i * lda;
                for (int j = 0; j < n; j++) {
                    int yi = yOff + j * incY;
                    y[yi] = Math.fma(A[rowOff + j], axi, y[yi]);
                }
            }
        }
    }
}
