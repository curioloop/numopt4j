/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DdotTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] x = {1, 2, 3};
        double[] y = {4, 5, 6};
        double dot = Ddot.ddot(3, x, 0, 1, y, 0, 1);
        assertEquals(32.0, dot, TOL);
    }

    @Test
    void testZero() {
        double[] x = {0, 0, 0};
        double[] y = {1, 2, 3};
        double dot = Ddot.ddot(3, x, 0, 1, y, 0, 1);
        assertEquals(0.0, dot, TOL);
    }

    @Test
    void testSingleElement() {
        double[] x = {5};
        double[] y = {3};
        double dot = Ddot.ddot(1, x, 0, 1, y, 0, 1);
        assertEquals(15.0, dot, TOL);
    }

    @Test
    void testEmpty() {
        double dot = Ddot.ddot(0, new double[0], 0, 1, new double[0], 0, 1);
        assertEquals(0.0, dot, TOL);
    }

    @Test
    void testLargeContiguousMatchesReference() {
        int n = 4096;
        double[] x = randomVector(n, 42);
        double[] y = randomVector(n, 84);

        double expected = ddotReference(n, x, 0, 1, y, 0, 1);
        double actual = Ddot.ddot(n, x, 0, 1, y, 0, 1);

        assertEquals(expected, actual, 0.0);
    }

    @Test
    void testStridedMixedIncrementsMatchesReference() {
        int n = 257;
        double[] x = randomVector(1 + (n - 1) * 2, 42);
        double[] y = randomVector(1 + (n - 1) * 3, 84);

        double expected = ddotReference(n, x, 0, 2, y, 0, 3);
        double actual = Ddot.ddot(n, x, 0, 2, y, 0, 3);

        assertEquals(expected, actual, 0.0);
    }

    @Test
    void testAlternatingCancellationMatchesReference() {
        int n = 4096;
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double magnitude = (i % 7 == 0) ? 1.0e16 : (i % 11) + 1.0;
            x[i] = (i & 1) == 0 ? magnitude : -magnitude;
            y[i] = 1.0 / ((i % 11) + 1.0);
        }

        double expected = ddotReference(n, x, 0, 1, y, 0, 1);
        double actual = Ddot.ddot(n, x, 0, 1, y, 0, 1);

        assertEquals(expected, actual, 0.0);
    }

    private double[] randomVector(int n, long seed) {
        Random random = new Random(seed);
        double[] vector = new double[n];
        for (int i = 0; i < n; i++) {
            vector[i] = random.nextDouble() - 0.5;
        }
        return vector;
    }

    private double ddotReference(int n,
                                 double[] x, int xOff, int incX,
                                 double[] y, int yOff, int incY) {
        if (n <= 0) {
            return 0.0;
        }
        if (incX == 1 && incY == 1) {
            return ddotContiguousReference(x, xOff, y, yOff, n);
        }
        return ddotStridedReference(x, xOff, incX, y, yOff, incY, n);
    }

    private double ddotContiguousReference(double[] x, int xOff, double[] y, int yOff, int n) {
        if (n <= Ddot.THRESHOLD) {
            double s0 = 0.0;
            double s1 = 0.0;
            double s2 = 0.0;
            double s3 = 0.0;
            int k = 0;
            for (; k + 3 < n; k += 4) {
                s0 = Math.fma(x[xOff + k], y[yOff + k], s0);
                s1 = Math.fma(x[xOff + k + 1], y[yOff + k + 1], s1);
                s2 = Math.fma(x[xOff + k + 2], y[yOff + k + 2], s2);
                s3 = Math.fma(x[xOff + k + 3], y[yOff + k + 3], s3);
            }
            for (; k < n; k++) {
                s0 = Math.fma(x[xOff + k], y[yOff + k], s0);
            }
            return (s0 + s1) + (s2 + s3);
        }

        int mid = n / 2;
        return ddotContiguousReference(x, xOff, y, yOff, mid)
            + ddotContiguousReference(x, xOff + mid, y, yOff + mid, n - mid);
    }

    private double ddotStridedReference(double[] x, int xOff, int incX,
                                        double[] y, int yOff, int incY, int n) {
        if (n <= Ddot.THRESHOLD) {
            double s0 = 0.0;
            double s1 = 0.0;
            double s2 = 0.0;
            double s3 = 0.0;
            int xi = xOff;
            int yi = yOff;
            int k = 0;
            for (; k + 3 < n; k += 4) {
                s0 = Math.fma(x[xi], y[yi], s0);
                xi += incX;
                yi += incY;
                s1 = Math.fma(x[xi], y[yi], s1);
                xi += incX;
                yi += incY;
                s2 = Math.fma(x[xi], y[yi], s2);
                xi += incX;
                yi += incY;
                s3 = Math.fma(x[xi], y[yi], s3);
                xi += incX;
                yi += incY;
            }
            for (; k < n; k++) {
                s0 = Math.fma(x[xi], y[yi], s0);
                xi += incX;
                yi += incY;
            }
            return (s0 + s1) + (s2 + s3);
        }

        int mid = n / 2;
        return ddotStridedReference(x, xOff, incX, y, yOff, incY, mid)
            + ddotStridedReference(x, xOff + mid * incX, incX, y, yOff + mid * incY, incY, n - mid);
    }
}
