/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DrotTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] x = {3, 4};
        double[] y = {5, 6};

        Drot.drot(2, x, 0, 1, y, 0, 1, 0.6, 0.8);

        assertTrue(Double.isFinite(x[0]));
        assertTrue(Double.isFinite(x[1]));
        assertTrue(Double.isFinite(y[0]));
        assertTrue(Double.isFinite(y[1]));
    }

    @Test
    void testIdentity() {
        double[] x = {3, 4};
        double[] y = {5, 6};

        Drot.drot(2, x, 0, 1, y, 0, 1, 1.0, 0.0);

        assertEquals(3.0, x[0], TOL);
        assertEquals(4.0, x[1], TOL);
        assertEquals(5.0, y[0], TOL);
        assertEquals(6.0, y[1], TOL);
    }

    @Test
    void testEmpty() {
        Drot.drot(0, new double[0], 0, 1, new double[0], 0, 1, 0.6, 0.8);
    }

    @Test
    void testRandomParityVariousStrides() {
        int[] sizes = {1, 5, 17};
        int[][] increments = {
            {1, 1},
            {3, 3},
            {1, 4},
            {4, 1},
            {-2, -2},
            {-2, 1},
            {1, -3}
        };
        double[][] rotations = {
            {0.8, 0.6},
            {-0.6, 0.8}
        };

        for (int n : sizes) {
            for (int[] inc : increments) {
                for (double[] rotation : rotations) {
                    verifyParity(n, inc[0], inc[1], rotation[0], rotation[1],
                        20260423L + n * 31L + inc[0] * 13L + inc[1] * 7L);
                }
            }
        }
    }

    private static void verifyParity(int n, int incX, int incY, double c, double s, long seed) {
        double[] expectedX = new double[1 + (n - 1) * Math.abs(incX)];
        double[] expectedY = new double[1 + (n - 1) * Math.abs(incY)];
        double[] actualX = new double[expectedX.length];
        double[] actualY = new double[expectedY.length];
        Random random = new Random(seed);

        fillUsedVector(random, expectedX, 0, incX, n);
        fillUsedVector(random, expectedY, 0, incY, n);
        System.arraycopy(expectedX, 0, actualX, 0, expectedX.length);
        System.arraycopy(expectedY, 0, actualY, 0, expectedY.length);

        BlasTestSupport.scalarDrot(n, expectedX, 0, incX, expectedY, 0, incY, c, s);
        Drot.drot(n, actualX, 0, incX, actualY, 0, incY, c, s);

        assertArrayClose(expectedX, actualX, 1e-12);
        assertArrayClose(expectedY, actualY, 1e-12);
    }

    private static void fillUsedVector(Random random, double[] vector, int off, int inc, int n) {
        int index = off + (inc < 0 ? (n - 1) * (-inc) : 0);
        for (int i = 0; i < n; i++) {
            vector[index] = (random.nextDouble() - 0.5) * 2.0;
            index += inc;
        }
    }

    private static void assertArrayClose(double[] expected, double[] actual, double tolerance) {
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
