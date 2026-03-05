/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DlarfxTest {

    private static final double TOL = 1e-13;

    @Test
    @DisplayName("Dlarfx: left side, small matrices")
    void testLeftSideSmall() {
        Random rnd = new Random(12345);
        for (int m = 1; m < 12; m++) {
            for (int n = 1; n < 12; n++) {
                for (int extra = 0; extra <= 11; extra += 5) {
                    for (int cas = 0; cas < 3; cas++) {
                        testDlarfx(true, m, n, extra, rnd);
                    }
                }
            }
        }
    }

    @Test
    @DisplayName("Dlarfx: right side, small matrices")
    void testRightSideSmall() {
        Random rnd = new Random(12346);
        for (int m = 1; m < 12; m++) {
            for (int n = 1; n < 12; n++) {
                for (int extra = 0; extra <= 11; extra += 5) {
                    for (int cas = 0; cas < 3; cas++) {
                        testDlarfx(false, m, n, extra, rnd);
                    }
                }
            }
        }
    }

    @Test
    @DisplayName("Dlarfx: large matrices (calls Dlarf)")
    void testLargeMatrices() {
        Random rnd = new Random(12347);
        int[] sizes = {11, 15, 20};
        for (int size : sizes) {
            testDlarfx(true, size, 5, 0, rnd);
            testDlarfx(false, 5, size, 0, rnd);
        }
    }

    @Test
    @DisplayName("Dlarfx: zero tau")
    void testZeroTau() {
        Random rnd = new Random(12348);
        int m = 5, n = 4, ldc = n + 2;
        
        double[] v = randomSlice(m, rnd);
        double[] c = randomMatrix(m, n, ldc, rnd);
        double[] cOrig = c.clone();
        double[] work = new double[n];
        
        Dlarfx.dlarfx(true, m, n, v, 0, 0.0, c, 0, ldc, work, 0);
        
        assertArrayEquals(cOrig, c, TOL, "C should be unchanged when tau is zero");
    }

    @Test
    @DisplayName("Dlarfx: zero dimensions")
    void testZeroDimensions() {
        Random rnd = new Random(12349);
        double[] v = randomSlice(5, rnd);
        double[] c = randomMatrix(5, 5, 5, rnd);
        double[] work = new double[5];
        
        Dlarfx.dlarfx(true, 0, 5, v, 0, 1.0, c, 0, 5, work, 0);
        Dlarfx.dlarfx(true, 5, 0, v, 0, 1.0, c, 0, 5, work, 0);
    }

    private void testDlarfx(boolean left, int m, int n, int extra, Random rnd) {
        int vLen = left ? m : n;
        double[] v = randomSlice(vLen, rnd);
        double tau = rnd.nextGaussian();
        int ldc = n + extra;
        double[] c = randomMatrix(m, n, ldc, rnd);
        double[] cCopy = c.clone();
        
        double[] h = constructH(v, tau, vLen, extra);
        int hSize = vLen + extra;
        
        double[] cWant = new double[m * ldc];
        fillNaN(cWant);
        
        if (left) {
            matMul(h, m, m, hSize, c, ldc, n, cWant, ldc);
        } else {
            matMul(c, m, n, ldc, h, hSize, n, cWant, ldc);
        }
        
        double[] work = null;
        if (vLen > 10) {
            work = new double[left ? n : m];
        }
        
        Dlarfx.dlarfx(left, m, n, v, 0, tau, c, 0, ldc, work, 0);
        
        String prefix = String.format("Case left=%b, m=%d, n=%d, extra=%d", left, m, n, extra);
        assertMatrixClose(cWant, c, m, n, ldc, TOL, prefix);
    }

    private double[] randomSlice(int n, Random rnd) {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextGaussian();
        }
        return v;
    }

    private double[] randomMatrix(int rows, int cols, int ldc, Random rnd) {
        double[] a = new double[rows * ldc];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                a[i * ldc + j] = rnd.nextGaussian();
            }
        }
        return a;
    }

    private double[] constructH(double[] v, double tau, int n, int extra) {
        int size = n + extra;
        double[] h = new double[size * size];
        for (int i = 0; i < size; i++) {
            h[i * size + i] = 1.0;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                h[i * size + j] -= tau * v[i] * v[j];
            }
        }
        return h;
    }

    private void fillNaN(double[] a) {
        for (int i = 0; i < a.length; i++) {
            a[i] = Double.NaN;
        }
    }

    private void matMul(double[] a, int m, int k, int lda, double[] b, int ldb, int n, double[] c, int ldc) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int l = 0; l < k; l++) {
                    sum += a[i * lda + l] * b[l * ldb + j];
                }
                c[i * ldc + j] = sum;
            }
        }
    }

    private void assertMatrixClose(double[] expected, double[] actual, int rows, int cols, int ldc, double tol, String prefix) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double e = expected[i * ldc + j];
                double a = actual[i * ldc + j];
                if (Double.isNaN(e)) continue;
                double diff = Math.abs(e - a);
                if (diff > tol) {
                    fail(String.format("%s: mismatch at (%d,%d): expected %g, got %g, diff %g",
                        prefix, i, j, e, a, diff));
                }
            }
        }
    }
}
