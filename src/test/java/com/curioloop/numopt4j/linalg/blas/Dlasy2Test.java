/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class Dlasy2Test {

    private static final double TOL = 1e-10;
    private static final double BIGNUM = 1e10;

    @Test
    @DisplayName("Dlasy2: 1x1 case")
    void test1x1() {
        Random rnd = new Random(12345);
        for (int i = 0; i < 100; i++) {
            testDlasy2Simple(false, false, 1, 1, rnd);
            testDlasy2Simple(true, false, 1, 1, rnd);
            testDlasy2Simple(false, true, 1, 1, rnd);
            testDlasy2Simple(true, true, 1, 1, rnd);
        }
    }

    @Test
    @DisplayName("Dlasy2: 1x2 case")
    void test1x2() {
        Random rnd = new Random(12345);
        for (int i = 0; i < 100; i++) {
            testDlasy2Simple(false, false, 1, 2, rnd);
            testDlasy2Simple(true, false, 1, 2, rnd);
            testDlasy2Simple(false, true, 1, 2, rnd);
            testDlasy2Simple(true, true, 1, 2, rnd);
        }
    }

    @Test
    @DisplayName("Dlasy2: 2x1 case")
    void test2x1() {
        Random rnd = new Random(12345);
        for (int i = 0; i < 100; i++) {
            testDlasy2Simple(false, false, 2, 1, rnd);
            testDlasy2Simple(true, false, 2, 1, rnd);
            testDlasy2Simple(false, true, 2, 1, rnd);
            testDlasy2Simple(true, true, 2, 1, rnd);
        }
    }

    @Test
    @DisplayName("Dlasy2: 2x2 case")
    void test2x2() {
        Random rnd = new Random(12345);
        for (int i = 0; i < 100; i++) {
            testDlasy2Simple(false, false, 2, 2, rnd);
            testDlasy2Simple(true, false, 2, 2, rnd);
            testDlasy2Simple(false, true, 2, 2, rnd);
            testDlasy2Simple(true, true, 2, 2, rnd);
        }
    }

    private void testDlasy2Simple(boolean tranl, boolean tranr, int n1, int n2, Random rnd) {
        int ldtl = Math.max(1, n1);
        int ldtr = Math.max(1, n2);
        int ldb = Math.max(1, n2);
        int ldx = Math.max(1, n2);

        double[] tl = new double[ldtl * n1];
        double[] tr = new double[ldtr * n2];
        double[] b = new double[ldb * n1];
        double[] x = new double[Math.max(ldx * n1, 1) + 6];

        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n1; j++) {
                tl[i * ldtl + j] = rnd.nextGaussian();
            }
        }
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n2; j++) {
                tr[i * ldtr + j] = rnd.nextGaussian();
            }
        }
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                b[i * ldb + j] = rnd.nextGaussian();
            }
        }

        boolean ok = Dlasy2.dlasy2(tranl, tranr, 1, n1, n2,
            tl, 0, ldtl, tr, 0, ldtr, b, 0, ldb, x, 0, ldx);

        double scale = x[4];
        assertTrue(scale > 0 && scale <= 1, "scale should be in (0,1], got " + scale);

        if (!ok) return;

        double resid = computeResidual(tranl, tranr, 1, n1, n2, tl, ldtl, tr, ldtr, b, ldb, x, ldx, scale);
        assertTrue(resid < TOL, "residual too large: " + resid);
    }

    private double computeResidual(boolean tranl, boolean tranr, int isgn, int n1, int n2,
                                  double[] tl, int ldtl, double[] tr, int ldtr,
                                  double[] b, int ldb, double[] x, int ldx, double scale) {
        double maxDiff = 0;
        double maxX = 0;

        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                double sum = 0;

                for (int k = 0; k < n1; k++) {
                    double tl_ik = tranl ? tl[k * ldtl + i] : tl[i * ldtl + k];
                    sum += tl_ik * x[k * ldx + j];
                }

                for (int k = 0; k < n2; k++) {
                    double tr_kj = tranr ? tr[j * ldtr + k] : tr[k * ldtr + j];
                    sum += isgn * x[i * ldx + k] * tr_kj;
                }

                double diff = Math.abs(sum - scale * b[i * ldb + j]);
                maxDiff = Math.max(maxDiff, diff);
            }
        }

        for (int i = 0; i < n1; i++) {
            double rowSum = 0;
            for (int j = 0; j < n2; j++) {
                rowSum += Math.abs(x[i * ldx + j]);
            }
            maxX = Math.max(maxX, rowSum);
        }

        return maxDiff / Math.max(maxX, 1e-300);
    }
}
