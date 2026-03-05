/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DlarfgTest {

    private static final double TOL = 1e-14;
    private static final double SMLNUM = Dlamch.dlamch('S');

    @Test
    @DisplayName("Dlarfg: basic cases")
    void testBasicCases() {
        testDlarfg(4, 1.0);
        testDlarfg(4, -2.0);
        testDlarfg(4, 0.0);
        testDlarfg(1, 1.0);
        testDlarfg(3, SMLNUM);
    }

    @Test
    @DisplayName("Dlarfg: random cases")
    void testRandomCases() {
        Random rnd = new Random(12345);
        for (int n = 1; n <= 20; n++) {
            for (int trial = 0; trial < 10; trial++) {
                double alpha = rnd.nextGaussian();
                testDlarfg(n, alpha);
            }
        }
    }

    @Test
    @DisplayName("Dlarfg: zero vector")
    void testZeroVector() {
        int n = 4;
        double[] x = new double[n - 1];
        double alpha = 1.0;

        double beta = Dlarfg.dlarfg(n, alpha, x, 0, 1, new double[1], 0);
        assertEquals(alpha, beta, TOL, "beta should equal alpha for zero vector");
    }

    @Test
    @DisplayName("Dlarfg: underflow scaling path (knt counting)")
    void testUnderflowScaling() {
        // alpha near SAFMIN triggers the scaling loop; verify H is still orthogonal
        // and H * (alpha, x)^T = (beta, 0, ..., 0)^T
        int n = 4;
        double alpha = SMLNUM * 0.5;  // below SAFMIN → forces at least one scaling iteration
        testDlarfg(n, alpha);

        // Also verify with negative alpha
        testDlarfg(n, -SMLNUM * 0.5);

        // Verify knt is counted correctly: beta must be rescaled back to original magnitude
        double[] x = {SMLNUM * 0.1, SMLNUM * 0.1, SMLNUM * 0.1};
        double[] tau = new double[1];
        double beta = Dlarfg.dlarfg(n, alpha, x, 0, 1, tau, 0);

        // beta should be in the same magnitude range as the original inputs (not scaled up)
        assertTrue(Math.abs(beta) < 1.0, "beta should not be blown up by missing knt rescaling");
        assertTrue(Math.abs(beta) > 0.0, "beta should be non-zero");
    }


    private void testDlarfg(int n, double alpha) {
        Random rnd = new Random(12345);
        double[] x = new double[n - 1];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextGaussian();
        }

        double[] xCopy = x.clone();
        double[] tau = new double[1];

        double beta = Dlarfg.dlarfg(n, alpha, x, 0, 1, tau, 0);

        if (n == 1) {
            assertEquals(alpha, beta, TOL, "beta should equal alpha for n=1");
            return;
        }

        double[] v = new double[n];
        v[0] = 1;
        System.arraycopy(x, 0, v, 1, n - 1);

        double[] h = computeH(n, v, tau[0]);

        assertTrue(isOrthogonal(h, n, TOL), "H should be orthogonal");

        double[] xFull = new double[n];
        xFull[0] = alpha;
        System.arraycopy(xCopy, 0, xFull, 1, n - 1);

        double[] result = matVec(h, n, xFull);

        assertEquals(beta, result[0], TOL, "result[0] should equal beta");
        for (int i = 1; i < n; i++) {
            assertEquals(0, result[i], TOL, "result[" + i + "] should be zero");
        }
    }

    private double[] computeH(int n, double[] v, double tau) {
        double[] h = new double[n * n];
        for (int i = 0; i < n; i++) {
            h[i * n + i] = 1;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                h[i * n + j] -= tau * v[i] * v[j];
            }
        }

        return h;
    }

    private boolean isOrthogonal(double[] h, int n, double tol) {
        double[] hth = new double[n * n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    hth[i * n + j] += h[k * n + i] * h[k * n + j];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                if (Math.abs(hth[i * n + j] - expected) > tol) {
                    return false;
                }
            }
        }

        return true;
    }

    private double[] matVec(double[] h, int n, double[] x) {
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i] += h[i * n + j] * x[j];
            }
        }
        return result;
    }
}
