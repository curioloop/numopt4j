/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DgerTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        double[] x = {1, 2};
        double[] y = {1, 2, 3};

        Dger.dger(2, 3, 1.0, x, 0, 1, y, 0, 1, A, 0, 3);

        assertEquals(2.0, A[0], TOL);
        assertEquals(4.0, A[1], TOL);
        assertEquals(6.0, A[2], TOL);
        assertEquals(6.0, A[3], TOL);
        assertEquals(9.0, A[4], TOL);
        assertEquals(12.0, A[5], TOL);
    }

    @Test
    void testZeroAlpha() {
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        double[] x = {1, 2};
        double[] y = {1, 2, 3};

        Dger.dger(2, 3, 0.0, x, 0, 1, y, 0, 1, A, 0, 3);

        assertEquals(1.0, A[0], TOL);
        assertEquals(2.0, A[1], TOL);
        assertEquals(3.0, A[2], TOL);
    }

    @Test
    void testEmpty() {
        Dger.dger(0, 0, 1.0, new double[0], 0, 1, new double[0], 0, 1, new double[0], 0, 0);
    }

    @Test
    void testRandomParityVariousStrides() {
        int[][] shapes = {
            {1, 1},
            {5, 7},
            {17, 9}
        };
        int[][] increments = {
            {1, 1},
            {3, 1},
            {1, 2},
            {-2, 1},
            {1, -3},
            {-2, -3}
        };
        double[] alphas = {0.75, -1.25};

        for (int[] shape : shapes) {
            for (int[] stride : increments) {
                for (double alpha : alphas) {
                    verifyParity(shape[0], shape[1], stride[0], stride[1], alpha,
                        20260423L + shape[0] * 31L + shape[1] * 17L + stride[0] * 13L + stride[1] * 7L);
                }
            }
        }
    }

    private static void verifyParity(int m, int n, int incX, int incY, double alpha, long seed) {
        int lda = n + 2;
        double[] expected = new double[m * lda];
        double[] actual = new double[m * lda];
        double[] x = new double[1 + (m - 1) * Math.abs(incX)];
        double[] y = new double[1 + (n - 1) * Math.abs(incY)];
        Random random = new Random(seed);

        fillUsedVector(random, x, 0, incX, m);
        fillUsedVector(random, y, 0, incY, n);
        fillDense(random, expected);
        System.arraycopy(expected, 0, actual, 0, expected.length);

        BlasTestSupport.scalarDger(m, n, alpha, x, 0, incX, y, 0, incY, expected, 0, lda);
        Dger.dger(m, n, alpha, x, 0, incX, y, 0, incY, actual, 0, lda);

        assertMatrixClose(expected, actual, 1e-12);
    }

    private static void fillUsedVector(Random random, double[] vector, int off, int inc, int n) {
        int index = off + (inc < 0 ? (n - 1) * (-inc) : 0);
        for (int i = 0; i < n; i++) {
            vector[index] = (random.nextDouble() - 0.5) * 2.0;
            index += inc;
        }
    }

    private static void fillDense(Random random, double[] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = random.nextDouble() - 0.5;
        }
    }

    private static void assertMatrixClose(double[] expected, double[] actual, double tolerance) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            final int index = i;
            double diff = Math.abs(expected[i] - actual[i]);
            double scale = Math.max(1.0, Math.max(Math.abs(expected[i]), Math.abs(actual[i])));
            assertTrue(diff <= tolerance * scale,
                () -> "Mismatch at index " + index + ": expected=" + expected[index] + ", actual=" + actual[index]);
        }
    }
}
