/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DaxpyTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] x = {1, 2, 3};
        double[] y = {4, 5, 6};

        Daxpy.daxpy(3, 2.0, x, 0, 1, y, 0, 1);

        assertEquals(6.0, y[0], TOL);
        assertEquals(9.0, y[1], TOL);
        assertEquals(12.0, y[2], TOL);
    }

    @Test
    void testZeroAlpha() {
        double[] x = {1, 2, 3};
        double[] y = {4, 5, 6};

        Daxpy.daxpy(3, 0.0, x, 0, 1, y, 0, 1);

        assertEquals(4.0, y[0], TOL);
        assertEquals(5.0, y[1], TOL);
        assertEquals(6.0, y[2], TOL);
    }

    @Test
    void testSingleElement() {
        double[] x = {5};
        double[] y = {3};

        Daxpy.daxpy(1, 2.0, x, 0, 1, y, 0, 1);

        assertEquals(13.0, y[0], TOL);
    }

    @Test
    void testEmpty() {
        Daxpy.daxpy(0, 2.0, new double[0], 0, 1, new double[0], 0, 1);
    }

    @Test
    void testLargeContiguousMatchesReference() {
        int n = 4096;
        double alpha = -0.75;
        double[] x = randomVector(n, 42);
        double[] actual = randomVector(n, 84);
        double[] expected = Arrays.copyOf(actual, actual.length);

        daxpyReference(n, alpha, x, 0, 1, expected, 0, 1);
        Daxpy.daxpy(n, alpha, x, 0, 1, actual, 0, 1);

        assertArrayEquals(expected, actual, TOL);
    }

    @Test
    void testStridedMixedIncrements() {
        int n = 5;
        double alpha = 1.25;
        double[] x = {1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 4.0, 9.0, 5.0};
        double[] actual = {10.0, 8.0, 8.0, 20.0, 8.0, 30.0, 8.0, 40.0, 8.0, 50.0, 8.0, 60.0, 8.0};
        double[] expected = Arrays.copyOf(actual, actual.length);

        daxpyReference(n, alpha, x, 0, 2, expected, 0, 3);
        Daxpy.daxpy(n, alpha, x, 0, 2, actual, 0, 3);

        assertArrayEquals(expected, actual, TOL);
    }

    @Test
    void testSameArrayNonOverlappingViews() {
        double[] actual = new double[32];
        for (int i = 0; i < 16; i++) {
            actual[i] = i + 1.0;
            actual[16 + i] = 100.0 + i;
        }
        double[] expected = Arrays.copyOf(actual, actual.length);

        daxpyReference(16, -0.5, expected, 0, 1, expected, 16, 1);
        Daxpy.daxpy(16, -0.5, actual, 0, 1, actual, 16, 1);

        assertArrayEquals(expected, actual, TOL);
    }

    private double[] randomVector(int n, long seed) {
        Random random = new Random(seed);
        double[] vector = new double[n];
        for (int i = 0; i < n; i++) {
            vector[i] = random.nextDouble() - 0.5;
        }
        return vector;
    }

    private void daxpyReference(int n, double alpha,
                                double[] x, int xOff, int incX,
                                double[] y, int yOff, int incY) {
        int xi = xOff;
        int yi = yOff;
        for (int i = 0; i < n; i++) {
            y[yi] = Math.fma(alpha, x[xi], y[yi]);
            xi += incX;
            yi += incY;
        }
    }
}
