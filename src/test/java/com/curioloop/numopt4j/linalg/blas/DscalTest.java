/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DscalTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] x = {1, 2, 3};

        Dscal.dscal(3, 2.0, x, 0, 1);

        assertEquals(2.0, x[0], TOL);
        assertEquals(4.0, x[1], TOL);
        assertEquals(6.0, x[2], TOL);
    }

    @Test
    void testZero() {
        double[] x = {1, 2, 3};

        Dscal.dscal(3, 0.0, x, 0, 1);

        assertEquals(0.0, x[0], TOL);
        assertEquals(0.0, x[1], TOL);
        assertEquals(0.0, x[2], TOL);
    }

    @Test
    void testSingleElement() {
        double[] x = {5};

        Dscal.dscal(1, 3.0, x, 0, 1);

        assertEquals(15.0, x[0], TOL);
    }

    @Test
    void testEmpty() {
        Dscal.dscal(0, 2.0, new double[0], 0, 1);
    }

    @Test
    void testLargeContiguousMatchesReference() {
        int n = 4096;
        double alpha = -1.5;
        double[] actual = randomVector(n, 42);
        double[] expected = Arrays.copyOf(actual, actual.length);

        dscalReference(n, alpha, expected, 0, 1);
        Dscal.dscal(n, alpha, actual, 0, 1);

        assertArrayEquals(expected, actual, TOL);
    }

    @Test
    void testStrideThreeMatchesReference() {
        int n = 4;
        double[] actual = {1.0, 9.0, 9.0, 2.0, 9.0, 9.0, 3.0, 9.0, 9.0, 4.0};
        double[] expected = Arrays.copyOf(actual, actual.length);

        dscalReference(n, -0.25, expected, 0, 3);
        Dscal.dscal(n, -0.25, actual, 0, 3);

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

    private void dscalReference(int n, double alpha, double[] x, int xOff, int incX) {
        int xi = xOff;
        for (int i = 0; i < n; i++) {
            x[xi] *= alpha;
            xi += incX;
        }
    }
}
