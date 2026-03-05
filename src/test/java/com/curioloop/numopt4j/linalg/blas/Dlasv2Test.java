/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dlasv2Test {

    private static final double TOL = 1e-12;

    @Test
    void testBasicCases() {
        double[][] tests = {
            {3, 2, 5},
            {-1, 4, 2},
            {0.5, -0.3, 1.2},
            {10, 5, -3},
            {0, 1, 0},
            {1, 0, 1},
            {0, 0, 0}
        };

        for (double[] test : tests) {
            dlasv2Test(test[0], test[1], test[2]);
        }
    }

    @Test
    void testRandomCases() {
        java.util.Random rnd = new java.util.Random(42);
        for (int i = 0; i < 50; i++) {
            double f = rnd.nextGaussian();
            double g = rnd.nextGaussian();
            double h = rnd.nextGaussian();
            dlasv2Test(f, g, h);
        }
    }

    private void dlasv2Test(double f, double g, double h) {
        double[] out = new double[6];
        Dlas2.dlasv2(f, g, h, out, 0);

        double ssmin = out[0];
        double ssmax = out[1];
        double snr = out[2];
        double csr = out[3];
        double snl = out[4];
        double csl = out[5];

        double tmp11 = csl * f;
        double tmp12 = csl * g + snl * h;
        double tmp21 = -snl * f;
        double tmp22 = -snl * g + csl * h;

        double ans11 = tmp11 * csr + tmp12 * snr;
        double ans12 = tmp11 * -snr + tmp12 * csr;
        double ans21 = tmp21 * csr + tmp22 * snr;
        double ans22 = tmp21 * -snr + tmp22 * csr;

        assertEquals(ssmax, ans11, TOL, "SVD mismatch for f=" + f + ", g=" + g + ", h=" + h);
        assertEquals(0, ans12, TOL, "SVD mismatch for f=" + f + ", g=" + g + ", h=" + h);
        assertEquals(0, ans21, TOL, "SVD mismatch for f=" + f + ", g=" + g + ", h=" + h);
        assertEquals(ssmin, ans22, TOL, "SVD mismatch for f=" + f + ", g=" + g + ", h=" + h);
    }
}
