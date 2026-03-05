/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dlae2Test {

    private static final double TOL = 1e-10;

    @Test
    void testBasicCases() {
        double[][] tests = {
            {-10, 5, 3},
            {3, 5, -10},
            {0, 3, 0},
            {1, 3, 1},
            {1, -3, 1},
            {5, 0, 3},
            {3, 0, -5},
            {1, 3, 1.02},
            {1.02, 3, 1},
            {1, -3, -9}
        };

        for (double[] test : tests) {
            double a = test[0];
            double b = test[1];
            double c = test[2];

            double[] out = new double[2];
            Dlae2.dlae2(a, b, c, out, 0);
            double rt1 = out[0];
            double rt2 = out[1];

            double a1 = a - rt1;
            double c1 = c - rt1;
            double det1 = a1 * c1 - b * b;
            assertEquals(0, det1, TOL, "First eigenvalue mismatch for a=" + a + ", b=" + b + ", c=" + c);

            double a2 = a - rt2;
            double c2 = c - rt2;
            double det2 = a2 * c2 - b * b;
            assertEquals(0, det2, TOL, "Second eigenvalue mismatch for a=" + a + ", b=" + b + ", c=" + c);
        }
    }

    @Test
    void testOrdering() {
        double[] out = new double[2];
        Dlae2.dlae2(5, 0, 3, out, 0);
        assertTrue(out[0] >= out[1], "rt1 should be >= rt2");
    }
}
