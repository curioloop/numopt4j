/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Dasum routine.
 */
class DasumTest {

    private static final double EPSILON = 1e-14;

    @Test
    @DisplayName("dasum: basic L1 norm")
    void testBasic() {
        double[] x = {3.0, -4.0, 5.0};
        double result = Dasum.dasum(3, x, 0, 1);
        assertEquals(12.0, result, EPSILON);
    }

    @Test
    @DisplayName("dasum: all positive")
    void testAllPositive() {
        double[] x = {1.0, 2.0, 3.0, 4.0};
        double result = Dasum.dasum(4, x, 0, 1);
        assertEquals(10.0, result, EPSILON);
    }

    @Test
    @DisplayName("dasum: all negative")
    void testAllNegative() {
        double[] x = {-1.0, -2.0, -3.0};
        double result = Dasum.dasum(3, x, 0, 1);
        assertEquals(6.0, result, EPSILON);
    }

    @Test
    @DisplayName("dasum: with zeros")
    void testWithZeros() {
        double[] x = {0.0, -5.0, 0.0, 3.0};
        double result = Dasum.dasum(4, x, 0, 1);
        assertEquals(8.0, result, EPSILON);
    }

    @Test
    @DisplayName("dasum: strided access")
    void testStrided() {
        double[] x = {1.0, 0.0, -2.0, 0.0, 3.0};
        double result = Dasum.dasum(3, x, 0, 2);
        assertEquals(6.0, result, EPSILON);
    }

    @Test
    @DisplayName("dasum: with offset")
    void testWithOffset() {
        double[] x = {0.0, 0.0, 1.0, -2.0, 3.0};
        double result = Dasum.dasum(3, x, 2, 1);
        assertEquals(6.0, result, EPSILON);
    }

    @Test
    @DisplayName("dasum: n=0 returns 0")
    void testZeroN() {
        double[] x = {1.0, 2.0, 3.0};
        double result = Dasum.dasum(0, x, 0, 1);
        assertEquals(0.0, result, EPSILON);
    }

    @Test
    @DisplayName("dasum: n=1")
    void testOneN() {
        double[] x = {-7.0};
        double result = Dasum.dasum(1, x, 0, 1);
        assertEquals(7.0, result, EPSILON);
    }

    @Test
    @DisplayName("dasum: large values")
    void testLargeValues() {
        double[] x = {1e100, -1e100};
        double result = Dasum.dasum(2, x, 0, 1);
        assertEquals(2e100, result, 1e86);
    }

    @Test
    @DisplayName("dasum: small values")
    void testSmallValues() {
        double[] x = {1e-100, -1e-100};
        double result = Dasum.dasum(2, x, 0, 1);
        assertEquals(2e-100, result, 1e-114);
    }

    @Test
    @DisplayName("dasum: mixed magnitudes")
    void testMixedMagnitudes() {
        double[] x = {1e100, 1.0, -1e-100};
        double result = Dasum.dasum(3, x, 0, 1);
        assertEquals(1e100 + 1.0 + 1e-100, result, 1e86);
    }

    @Test
    @DisplayName("dasum: verify BLAS interface")
    void testBLASInterface() {
        double[] x = {1.0, -2.0, 3.0, -4.0, 5.0};
        double result = BLAS.dasum(5, x, 0, 1);
        assertEquals(15.0, result, EPSILON);
    }

    @Test
    @DisplayName("dasum: long vector")
    void testLongVector() {
        int n = 1000;
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = (i % 2 == 0) ? 1.0 : -1.0;
        }
        double result = Dasum.dasum(n, x, 0, 1);
        assertEquals(n, result, EPSILON);
    }
}
