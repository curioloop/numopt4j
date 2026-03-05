/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dlanv2Test {

    private static final double TOL = 1e-14;

    @Test
    void testUpperTriangular() {
        double[][] tests = {
            {3, 2, 0, 5},
            {-1, 4, 0, 2},
            {0.5, -0.3, 0, 1.2},
            {10, 5, 0, -3},
            {0, 1, 0, 0}
        };

        for (double[] test : tests) {
            dlanv2Test(test[0], test[1], test[2], test[3]);
        }
    }

    @Test
    void testLowerTriangular() {
        double[][] tests = {
            {3, 0, 2, 5},
            {-1, 0, 4, 2},
            {0.5, 0, -0.3, 1.2},
            {10, 0, 5, -3}
        };

        for (double[] test : tests) {
            dlanv2Test(test[0], test[1], test[2], test[3]);
        }
    }

    @Test
    void testStandardSchur() {
        double[][] tests = {
            {3, 2, -2, 3},
            {-1, 4, -4, -1},
            {0.5, -0.3, 0.3, 0.5}
        };

        for (double[] test : tests) {
            dlanv2Test(test[0], test[1], test[2], test[3]);
        }
    }

    @Test
    void testGeneral() {
        double[][] tests = {
            {1, 2, 3, 4},
            {-5, 3, 1, 2},
            {0.1, 0.2, 0.3, 0.4},
            {10, -5, 3, 7},
            {0, 1, -1, 0}
        };

        for (double[] test : tests) {
            dlanv2Test(test[0], test[1], test[2], test[3]);
        }
    }

    private void dlanv2Test(double a, double b, double c, double d) {
        double[] out = new double[10];
        Dlanv2.dlanv2(a, b, c, d, out, 0);

        double aa = out[0];
        double bb = out[1];
        double cc = out[2];
        double dd = out[3];
        double rt1r = out[4];
        double rt1i = out[5];
        double rt2r = out[6];
        double rt2i = out[7];
        double cs = out[8];
        double sn = out[9];

        if (cc == 0) {
            assertEquals(0, rt1i, TOL, "Expected real eigenvalues");
            assertEquals(0, rt2i, TOL, "Expected real eigenvalues");
        } else {
            assertEquals(aa, dd, TOL, "Diagonal elements should be equal for complex eigenvalues");
            assertTrue(bb * cc < 0, "Non-diagonal elements should have opposite signs");
            double im = Math.sqrt(-bb * cc);
            assertTrue(Math.abs(rt1i - im) < TOL || Math.abs(rt1i + im) < TOL, 
                "Unexpected imaginary part of eigenvalue");
        }

        assertTrue(Math.abs(rt1r - aa) < TOL || Math.abs(rt1r - dd) < TOL,
            "Real part of eigenvalue should match diagonal");
        assertTrue(Math.abs(rt2r - aa) < TOL || Math.abs(rt2r - dd) < TOL,
            "Real part of eigenvalue should match diagonal");

        assertEquals(1.0, Math.hypot(cs, sn), TOL, "Columns of orthogonal matrix should have unit norm");
    }
}
