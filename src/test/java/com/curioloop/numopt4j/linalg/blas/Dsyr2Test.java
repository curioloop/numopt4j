/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class Dsyr2Test {

    private static final double TOL = 1e-10;

    @Test
    void testUpper() {
        double[] template = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 6
        };
        double[] x = {1, 2, 3};
        double[] y = {1, 1, 1};
        double[] expected = template.clone();
        double[] current = template.clone();

        dsyr2Scalar(BLAS.Uplo.Upper, 3, 1.0, x, 0, 1, y, 0, 1, expected, 0, 3);
        Dsyr2.dsyr2(BLAS.Uplo.Upper, 3, 1.0, x, 0, 1, y, 0, 1, current, 0, 3);

        assertArrayClose(expected, current, TOL);
    }

    @Test
    void testLower() {
        double[] template = {
            1, 0, 0,
            2, 4, 0,
            3, 5, 6
        };
        double[] x = {1, 2, 3};
        double[] y = {1, 1, 1};
        double[] expected = template.clone();
        double[] current = template.clone();

        dsyr2Scalar(BLAS.Uplo.Lower, 3, 1.0, x, 0, 1, y, 0, 1, expected, 0, 3);
        Dsyr2.dsyr2(BLAS.Uplo.Lower, 3, 1.0, x, 0, 1, y, 0, 1, current, 0, 3);

        assertArrayClose(expected, current, TOL);
    }

    @Test
    void testEmpty() {
        Dsyr2.dsyr2(BLAS.Uplo.Upper, 0, 1.0, new double[0], 0, 1, new double[0], 0, 1, new double[0], 0, 0);
    }

    @Test
    void testUpperStridedParity() {
        verifyRandomParity(BLAS.Uplo.Upper, 9, 11, 11, 2, 20260422L);
    }

    @Test
    void testLowerStridedParity() {
        verifyRandomParity(BLAS.Uplo.Lower, 9, 11, 11, 2, 20260423L);
    }

    @Test
    void testUpperContiguousLargeParity() {
        verifyRandomParity(BLAS.Uplo.Upper, 128, 128, 1, 1, 20260426L);
    }

    @Test
    void testLowerContiguousLargeParity() {
        verifyRandomParity(BLAS.Uplo.Lower, 128, 128, 1, 1, 20260427L);
    }

    @Test
    void testUpperAliasingHouseholderParity() {
        verifyAliasingParity(BLAS.Uplo.Upper, 32, 32, 0, 1, 32, 20260424L);
    }

    @Test
    void testLowerAliasingHouseholderParity() {
        verifyAliasingParity(BLAS.Uplo.Lower, 32, 33, 34, 33, 33, 20260425L);
    }

    private static void verifyRandomParity(BLAS.Uplo uplo, int n, int lda, int incX, int incY, long seed) {
        Random random = new Random(seed);
        double[] template = new double[n * lda];
        double[] x = new double[1 + (n - 1) * incX];
        double[] y = new double[1 + (n - 1) * incY];
        double[] expected = new double[template.length];
        double[] current = new double[template.length];

        fillRandom(random, template);
        fillRandom(random, x);
        fillRandom(random, y);
        System.arraycopy(template, 0, expected, 0, template.length);
        System.arraycopy(template, 0, current, 0, template.length);

        dsyr2Scalar(uplo, n, -0.75, x, 0, incX, y, 0, incY, expected, 0, lda);
        Dsyr2.dsyr2(uplo, n, -0.75, x, 0, incX, y, 0, incY, current, 0, lda);

        assertArrayClose(expected, current, TOL);
    }

    private static void verifyAliasingParity(BLAS.Uplo uplo, int n, int lda, int aOff, int xOff, int incX, long seed) {
        Random random = new Random(seed);
        int totalRows = uplo == BLAS.Uplo.Upper ? n : n + 1;
        double[] template = new double[totalRows * lda];
        double[] y = new double[n];
        double[] expected = new double[template.length];
        double[] current = new double[template.length];

        fillRandom(random, template);
        fillRandom(random, y);
        System.arraycopy(template, 0, expected, 0, template.length);
        System.arraycopy(template, 0, current, 0, template.length);

        dsyr2Scalar(uplo, n, -1.0, expected, xOff, incX, y, 0, 1, expected, aOff, lda);
        Dsyr2.dsyr2(uplo, n, -1.0, current, xOff, incX, y, 0, 1, current, aOff, lda);

        assertArrayClose(expected, current, TOL);
    }

    private static void fillRandom(Random random, double[] values) {
        for (int i = 0; i < values.length; i++) {
            values[i] = random.nextDouble() - 0.5;
        }
    }

    private static void dsyr2Scalar(BLAS.Uplo uplo, int n, double alpha,
                                    double[] x, int xOff, int incX,
                                    double[] y, int yOff, int incY,
                                    double[] a, int aOff, int lda) {
        if (n <= 0 || alpha == 0.0) {
            return;
        }

        boolean upper = uplo == BLAS.Uplo.Upper;
        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i * incX];
            double yi = y[yOff + i * incY];
            double temp1 = alpha * xi;
            double temp2 = alpha * yi;
            int rowOff = aOff + i * lda;
            int start = upper ? i : 0;
            int end = upper ? n : i + 1;

            for (int j = start; j < end; j++) {
                double updated = a[rowOff + j];
                updated = Math.fma(temp1, y[yOff + j * incY], updated);
                updated = Math.fma(temp2, x[xOff + j * incX], updated);
                a[rowOff + j] = updated;
            }
        }
    }

    private static void assertArrayClose(double[] expected, double[] actual, double tolerance) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tolerance, "Mismatch at index " + i);
        }
    }
}
