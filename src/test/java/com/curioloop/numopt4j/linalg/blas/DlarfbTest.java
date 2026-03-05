/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DlarfbTest {

    private static final double TOL = 1e-10;

    @Nested
    @DisplayName("ColumnWise Storage")
    class ColumnWiseTests {

        @Nested
        @DisplayName("Forward Direction")
        class ForwardTests {

            @Test
            @DisplayName("Left + NoTrans")
            void testLeftNoTrans() {
                testDlarfb('L', 'N', 'F', 'C', 6, 4, 3);
            }

            @Test
            @DisplayName("Left + Trans")
            void testLeftTrans() {
                testDlarfb('L', 'T', 'F', 'C', 6, 4, 3);
            }

            @Test
            @DisplayName("Right + NoTrans")
            void testRightNoTrans() {
                testDlarfb('R', 'N', 'F', 'C', 4, 6, 3);
            }

            @Test
            @DisplayName("Right + Trans")
            void testRightTrans() {
                testDlarfb('R', 'T', 'F', 'C', 4, 6, 3);
            }
        }

        @Nested
        @DisplayName("Backward Direction")
        class BackwardTests {

            @Test
            @DisplayName("Left + NoTrans")
            void testLeftNoTrans() {
                testDlarfb('L', 'N', 'B', 'C', 6, 4, 3);
            }

            @Test
            @DisplayName("Left + Trans")
            void testLeftTrans() {
                testDlarfb('L', 'T', 'B', 'C', 6, 4, 3);
            }

            @Test
            @DisplayName("Right + NoTrans")
            void testRightNoTrans() {
                testDlarfb('R', 'N', 'B', 'C', 4, 6, 3);
            }

            @Test
            @DisplayName("Right + Trans")
            void testRightTrans() {
                testDlarfb('R', 'T', 'B', 'C', 4, 6, 3);
            }
        }
    }

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("Single reflector")
        void testSingleReflector() {
            testDlarfb('L', 'N', 'F', 'C', 5, 3, 1);
        }

        @Test
        @DisplayName("k equals m (Left side)")
        void testKEqualsM() {
            testDlarfb('L', 'N', 'F', 'C', 4, 5, 4);
        }

        @Test
        @DisplayName("k equals n (Right side)")
        void testKEqualsN() {
            testDlarfb('R', 'N', 'F', 'C', 5, 4, 4);
        }
    }

    @Nested
    @DisplayName("RowWise Storage")
    class RowWiseTests {

        @Nested
        @DisplayName("Forward Direction")
        class ForwardTests {

            @Test
            @DisplayName("Left + NoTrans")
            void testLeftNoTrans() {
                testDlarfbRowWise('L', 'N', 'F', 6, 4, 3);
            }

            @Test
            @DisplayName("Left + Trans")
            void testLeftTrans() {
                testDlarfbRowWise('L', 'T', 'F', 6, 4, 3);
            }

            @Test
            @DisplayName("Right + NoTrans")
            void testRightNoTrans() {
                testDlarfbRowWise('R', 'N', 'F', 4, 6, 3);
            }

            @Test
            @DisplayName("Right + Trans")
            void testRightTrans() {
                testDlarfbRowWise('R', 'T', 'F', 4, 6, 3);
            }
        }

        @Nested
        @DisplayName("Backward Direction")
        class BackwardTests {

            @Test
            @DisplayName("Left + NoTrans")
            void testLeftNoTrans() {
                testDlarfbRowWise('L', 'N', 'B', 6, 4, 3);
            }

            @Test
            @DisplayName("Left + Trans")
            void testLeftTrans() {
                testDlarfbRowWise('L', 'T', 'B', 6, 4, 3);
            }

            @Test
            @DisplayName("Right + NoTrans")
            void testRightNoTrans() {
                testDlarfbRowWise('R', 'N', 'B', 4, 6, 3);
            }

            @Test
            @DisplayName("Right + Trans")
            void testRightTrans() {
                testDlarfbRowWise('R', 'T', 'B', 4, 6, 3);
            }
        }
    }

    private static void testDlarfb(char side, char trans, char direct, char storev, int m, int n, int k) {
        Random rnd = new Random(12345);
        boolean left = side == 'L';
        boolean forward = direct == 'F';

        int ldc = n;
        int nv = left ? m : n;
        int ldv = k;
        int ldt = k;
        int ldwork = k;

        double[] tau = new double[k];
        for (int i = 0; i < k; i++) {
            tau[i] = 0.5 + rnd.nextDouble() * 0.5;
        }

        double[] v;
        if (forward) {
            v = generateVForwardColWise(nv, k, rnd);
        } else {
            v = generateVBackwardColWise(nv, k, rnd);
        }

        double[] t = new double[k * ldt];
        Dlarft.dlarft(direct, storev, nv, k, v, 0, ldv, tau, 0, t, 0, ldt);

        double[] c = randomMatrix(m, n, rnd);
        double[] cCopy = c.clone();

        int workSize = left ? n * k : m * k;
        double[] work = new double[workSize];
        Dlarfb.dlarfb(side, trans, direct, storev, m, n, k, v, 0, ldv, t, 0, ldt, c, 0, ldc, work, 0, ldwork);

        double[] h;
        if (forward) {
            h = constructHFromVColWise(tau, v, nv, k);
        } else {
            h = constructHBackwardFromVColWise(tau, v, nv, k);
        }

        double[] hFinal;
        if (trans == 'T') {
            hFinal = transpose(h, nv, nv);
        } else {
            hFinal = h;
        }

        double[] expected;
        if (left) {
            expected = matMul(hFinal, nv, nv, cCopy, m, n);
        } else {
            expected = matMul(cCopy, m, n, hFinal, nv, nv);
        }

        String msg = String.format("side=%c, trans=%c, direct=%c, m=%d, n=%d, k=%d", side, trans, direct, m, n, k);
        assertMatrixClose(expected, c, m, n, TOL, msg);
    }

    private static double[] randomMatrix(int rows, int cols, Random rnd) {
        double[] a = new double[rows * cols];
        for (int i = 0; i < a.length; i++) {
            a[i] = rnd.nextDouble();
        }
        return a;
    }

    private static double[] generateVForwardColWise(int m, int k, Random rnd) {
        double[] v = new double[m * k];
        for (int col = 0; col < k; col++) {
            for (int row = 0; row < col; row++) {
                v[row * k + col] = 0;
            }
            v[col * k + col] = 1;
            for (int row = col + 1; row < m; row++) {
                v[row * k + col] = rnd.nextDouble();
            }
        }
        return v;
    }

    private static double[] generateVBackwardColWise(int m, int k, Random rnd) {
        double[] v = new double[m * k];
        for (int col = 0; col < k; col++) {
            for (int row = 0; row < m - k + col; row++) {
                v[row * k + col] = rnd.nextDouble();
            }
            v[(m - k + col) * k + col] = 1;
            for (int row = m - k + col + 1; row < m; row++) {
                v[row * k + col] = 0;
            }
        }
        return v;
    }

    private static double[] constructHFromVColWise(double[] tau, double[] v, int m, int k) {
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            h[i * m + i] = 1;
        }
        
        for (int i = 0; i < k; i++) {
            double[] vec = new double[m];
            for (int j = 0; j < m; j++) {
                vec[j] = v[j * k + i];
            }
            applyHouseholderLeft(h, m, tau[i], vec);
        }
        return h;
    }

    private static double[] constructHBackwardFromVColWise(double[] tau, double[] v, int m, int k) {
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            h[i * m + i] = 1;
        }
        
        for (int i = k - 1; i >= 0; i--) {
            double[] vec = new double[m];
            for (int j = 0; j < m; j++) {
                vec[j] = v[j * k + i];
            }
            applyHouseholderLeft(h, m, tau[i], vec);
        }
        return h;
    }

    private static void applyHouseholderLeft(double[] h, int m, double tau, double[] vec) {
        double[] hi = new double[m * m];
        for (int j = 0; j < m; j++) {
            hi[j * m + j] = 1;
        }
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < m; col++) {
                hi[row * m + col] -= tau * vec[row] * vec[col];
            }
        }
        
        double[] hCopy = h.clone();
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < m; col++) {
                h[row * m + col] = 0;
                for (int l = 0; l < m; l++) {
                    h[row * m + col] += hCopy[row * m + l] * hi[l * m + col];
                }
            }
        }
    }

    private static void applyHouseholderRight(double[] h, int m, double tau, double[] vec) {
        double[] hi = new double[m * m];
        for (int j = 0; j < m; j++) {
            hi[j * m + j] = 1;
        }
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < m; col++) {
                hi[row * m + col] -= tau * vec[row] * vec[col];
            }
        }
        
        double[] hCopy = h.clone();
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < m; col++) {
                h[row * m + col] = 0;
                for (int l = 0; l < m; l++) {
                    h[row * m + col] += hi[row * m + l] * hCopy[l * m + col];
                }
            }
        }
    }

    private static double[] transpose(double[] a, int rows, int cols) {
        double[] result = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j * rows + i] = a[i * cols + j];
            }
        }
        return result;
    }

    private static double[] matMul(double[] a, int aRows, int aCols, double[] b, int bRows, int bCols) {
        double[] c = new double[aRows * bCols];
        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < bCols; j++) {
                double sum = 0;
                for (int k = 0; k < aCols; k++) {
                    sum += a[i * aCols + k] * b[k * bCols + j];
                }
                c[i * bCols + j] = sum;
            }
        }
        return c;
    }

    private static void assertMatrixClose(double[] expected, double[] actual, 
                                          int rows, int cols, double tol, String msg) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double diff = Math.abs(expected[i * cols + j] - actual[i * cols + j]);
                if (diff > tol) {
                    fail(String.format("%s: mismatch at (%d,%d): expected %g, got %g, diff %g",
                        msg, i, j, expected[i * cols + j], actual[i * cols + j], diff));
                }
            }
        }
    }

    private static void testDlarfbRowWise(char side, char trans, char direct, int m, int n, int k) {
        Random rnd = new Random(12345);
        boolean left = side == 'L';
        boolean forward = direct == 'F';

        int ldc = n;
        int nv = left ? m : n;
        int ldv = nv;
        int ldt = k;
        int ldwork = k;

        double[] tau = new double[k];
        for (int i = 0; i < k; i++) {
            tau[i] = 0.5 + rnd.nextDouble() * 0.5;
        }

        double[] v;
        if (forward) {
            v = generateVForwardRowWise(nv, k, rnd);
        } else {
            v = generateVBackwardRowWise(nv, k, rnd);
        }

        double[] t = new double[k * ldt];
        Dlarft.dlarft(direct, 'R', nv, k, v, 0, ldv, tau, 0, t, 0, ldt);

        double[] c = randomMatrix(m, n, rnd);
        double[] cCopy = c.clone();

        int workSize = left ? n * k : m * k;
        double[] work = new double[workSize];
        Dlarfb.dlarfb(side, trans, direct, 'R', m, n, k, v, 0, ldv, t, 0, ldt, c, 0, ldc, work, 0, ldwork);

        double[] h;
        if (forward) {
            h = constructHFromVRowWise(tau, v, nv, k);
        } else {
            h = constructHBackwardFromVRowWise(tau, v, nv, k);
        }

        double[] hFinal;
        if (trans == 'T') {
            hFinal = transpose(h, nv, nv);
        } else {
            hFinal = h;
        }

        double[] expected;
        if (left) {
            expected = matMul(hFinal, nv, nv, cCopy, m, n);
        } else {
            expected = matMul(cCopy, m, n, hFinal, nv, nv);
        }

        String msg = String.format("side=%c, trans=%c, direct=%c, storev=R, m=%d, n=%d, k=%d", side, trans, direct, m, n, k);
        assertMatrixClose(expected, c, m, n, TOL, msg);
    }

    private static double[] generateVForwardRowWise(int m, int k, Random rnd) {
        double[] v = new double[k * m];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < i; j++) {
                v[i * m + j] = 0;
            }
            v[i * m + i] = 1;
            for (int j = i + 1; j < m; j++) {
                v[i * m + j] = rnd.nextDouble();
            }
        }
        return v;
    }

    private static double[] generateVBackwardRowWise(int m, int k, Random rnd) {
        double[] v = new double[k * m];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < m - k + i; j++) {
                v[i * m + j] = rnd.nextDouble();
            }
            v[i * m + (m - k + i)] = 1;
            for (int j = m - k + i + 1; j < m; j++) {
                v[i * m + j] = 0;
            }
        }
        return v;
    }

    private static double[] constructHFromVRowWise(double[] tau, double[] v, int m, int k) {
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            h[i * m + i] = 1;
        }
        
        for (int i = 0; i < k; i++) {
            double[] vec = new double[m];
            for (int j = 0; j < m; j++) {
                vec[j] = v[i * m + j];
            }
            applyHouseholderLeft(h, m, tau[i], vec);
        }
        return h;
    }

    private static double[] constructHBackwardFromVRowWise(double[] tau, double[] v, int m, int k) {
        double[] h = new double[m * m];
        for (int i = 0; i < m; i++) {
            h[i * m + i] = 1;
        }
        
        for (int i = k - 1; i >= 0; i--) {
            double[] vec = new double[m];
            for (int j = 0; j < m; j++) {
                vec[j] = v[i * m + j];
            }
            applyHouseholderLeft(h, m, tau[i], vec);
        }
        return h;
    }
}
