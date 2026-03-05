/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static java.lang.Math.*;
import static org.junit.jupiter.api.Assertions.*;

class DgelqTest {

    private static final double TOL = 1e-10;

    @Nested
    @DisplayName("DGELQF Blocked vs Unblocked")
    class DgelqfComparisonTests {

        @Test
        @DisplayName("Square matrix")
        void testSquareMatrix() {
            testDgelqfBlockedVsUnblocked(50, 50);
        }

        @Test
        @DisplayName("Wide matrix")
        void testWideMatrix() {
            testDgelqfBlockedVsUnblocked(30, 100);
        }

        @Test
        @DisplayName("Tall matrix")
        void testTallMatrix() {
            testDgelqfBlockedVsUnblocked(100, 30);
        }

        @Test
        @DisplayName("Large matrix")
        void testLargeMatrix() {
            testDgelqfBlockedVsUnblocked(200, 200);
        }

        private void testDgelqfBlockedVsUnblocked(int m, int n) {
            Random rnd = new Random(12345);
            int k = min(m, n);
            int lda = n;

            double[] original = randomMatrix(m, n, rnd);
            double[] tauBlocked = new double[k];
            double[] tauUnblocked = new double[k];

            int nb = Ilaenv.ilaenv(1, "DGELQF", " ", m, n, -1, -1);
            int lworkBlocked = m * nb + nb * nb;
            double[] workBlocked = new double[lworkBlocked];
            int lworkUnblocked = m;
            double[] workUnblocked = new double[lworkUnblocked];

            double[] aBlocked = original.clone();
            double[] aUnblocked = original.clone();

            Dgelq.dgelqf(m, n, aBlocked, 0, lda, tauBlocked, 0, workBlocked, 0, lworkBlocked);
            Dgelq.dgelq2(m, n, aUnblocked, 0, lda, tauUnblocked, 0, workUnblocked, 0);

            assertMatrixClose(aBlocked, aUnblocked, m, n, TOL, 
                String.format("A matrix mismatch for m=%d, n=%d", m, n));
            assertArrayClose(tauBlocked, tauUnblocked, TOL,
                String.format("tau mismatch for m=%d, n=%d", m, n));
        }
    }

    @Nested
    @DisplayName("DGELQF Correctness")
    class DgelqfCorrectnessTests {

        @Test
        @DisplayName("LQ decomposition: A = L * Q")
        void testLQDecomposition() {
            testLQDecomposition(50, 50);
            testLQDecomposition(30, 100);
            testLQDecomposition(100, 30);
        }

        private void testLQDecomposition(int m, int n) {
            Random rnd = new Random(12345);
            int k = min(m, n);
            int lda = n;

            double[] original = randomMatrix(m, n, rnd);
            double[] a = original.clone();
            double[] tau = new double[k];

            int nb = Ilaenv.ilaenv(1, "DGELQF", " ", m, n, -1, -1);
            int lwork = m * nb + nb * nb;
            double[] work = new double[lwork];

            Dgelq.dgelqf(m, n, a, 0, lda, tau, 0, work, 0, lwork);

            double[] l = extractL(a, m, n);
            double[] q = generateQ(a, m, n, k, tau);

            double[] aReconstructed = matMul(l, m, n, q, n, n);

            assertMatrixClose(original, aReconstructed, m, n, TOL,
                String.format("A != L*Q for m=%d, n=%d", m, n));
        }

        @Test
        @DisplayName("Q orthogonality: Q * Q^T = I")
        void testQOrthogonality() {
            testQOrthogonality(50, 50);
            testQOrthogonality(30, 100);
            testQOrthogonality(100, 30);
        }

        private void testQOrthogonality(int m, int n) {
            Random rnd = new Random(12345);
            int k = min(m, n);
            int lda = n;

            double[] a = randomMatrix(m, n, rnd);
            double[] tau = new double[k];

            int nb = Ilaenv.ilaenv(1, "DGELQF", " ", m, n, -1, -1);
            int lwork = m * nb + nb * nb;
            double[] work = new double[lwork];

            Dgelq.dgelqf(m, n, a, 0, lda, tau, 0, work, 0, lwork);

            double[] q = generateQ(a, m, n, k, tau);

            double[] qqt = matMul(q, n, n, transpose(q, n, n), n, n);

            double[] identity = new double[n * n];
            for (int i = 0; i < n; i++) {
                identity[i * n + i] = 1;
            }

            assertMatrixClose(identity, qqt, n, n, TOL,
                String.format("Q*Q^T != I for m=%d, n=%d", m, n));
        }
    }

    @Nested
    @DisplayName("DORGLQ Blocked vs Unblocked")
    class DorglqComparisonTests {

        @Test
        @DisplayName("Square matrix")
        void testSquareMatrix() {
            testDorglqBlockedVsUnblocked(50, 50);
        }

        @Test
        @DisplayName("Wide matrix")
        void testWideMatrix() {
            testDorglqBlockedVsUnblocked(30, 100);
        }

        @Test
        @DisplayName("Tall matrix")
        void testTallMatrix() {
            testDorglqBlockedVsUnblocked(100, 30);
        }

        private void testDorglqBlockedVsUnblocked(int m, int n) {
            Random rnd = new Random(12345);
            int k = min(m, n);
            int lda = n;

            double[] original = randomMatrix(m, n, rnd);
            double[] tau = new double[k];

            int nbLq = Ilaenv.ilaenv(1, "DGELQF", " ", m, n, -1, -1);
            int lworkLq = m * nbLq + nbLq * nbLq;
            double[] workLq = new double[lworkLq];
            Dgelq.dgelqf(m, n, original, 0, lda, tau, 0, workLq, 0, lworkLq);

            int nb = Ilaenv.ilaenv(1, "DORGLQ", " ", m, n, k, -1);
            int lworkBlocked = m * nb + nb * nb;
            double[] workBlocked = new double[lworkBlocked];
            int lworkUnblocked = m;
            double[] workUnblocked = new double[lworkUnblocked];

            double[] qBlocked = original.clone();
            double[] qUnblocked = original.clone();

            Dgelq.dorglq(m, n, k, qBlocked, 0, lda, tau, 0, workBlocked, 0, lworkBlocked);
            Dgelq.dorgl2(m, n, k, qUnblocked, 0, lda, tau, 0, workUnblocked, 0);

            assertMatrixClose(qBlocked, qUnblocked, m, n, TOL,
                String.format("Q matrix mismatch for m=%d, n=%d", m, n));
        }
    }

    @Nested
    @DisplayName("DORGLQ Correctness")
    class DorglqCorrectnessTests {

        @Test
        @DisplayName("Q orthogonality")
        void testQOrthogonality() {
            testDorglqOrthogonality(50, 50);
            testDorglqOrthogonality(30, 100);
        }

        private void testDorglqOrthogonality(int m, int n) {
            Random rnd = new Random(12345);
            int k = min(m, n);
            int lda = n;

            double[] a = randomMatrix(m, n, rnd);
            double[] tau = new double[k];

            int nbLq = Ilaenv.ilaenv(1, "DGELQF", " ", m, n, -1, -1);
            int lworkLq = m * nbLq + nbLq * nbLq;
            double[] workLq = new double[lworkLq];
            Dgelq.dgelqf(m, n, a, 0, lda, tau, 0, workLq, 0, lworkLq);

            int nb = Ilaenv.ilaenv(1, "DORGLQ", " ", m, n, k, -1);
            int lwork = m * nb + nb * nb;
            double[] work = new double[lwork];

            Dgelq.dorglq(m, n, k, a, 0, lda, tau, 0, work, 0, lwork);

            double[] qqt = matMul(a, m, n, transpose(a, m, n), n, m);

            double[] identity = new double[m * m];
            for (int i = 0; i < m; i++) {
                identity[i * m + i] = 1;
            }

            assertMatrixClose(identity, qqt, m, m, TOL * max(1, m),
                String.format("Q*Q^T != I for m=%d, n=%d", m, n));
        }
    }

    private static double[] randomMatrix(int m, int n, Random rnd) {
        double[] a = new double[m * n];
        for (int i = 0; i < a.length; i++) {
            a[i] = rnd.nextDouble();
        }
        return a;
    }

    private static double[] extractL(double[] a, int m, int n) {
        int k = min(m, n);
        double[] l = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (j < k && j <= i) {
                    l[i * n + j] = a[i * n + j];
                } else {
                    l[i * n + j] = 0;
                }
            }
        }
        return l;
    }

    private static double[] generateQ(double[] a, int m, int n, int k, double[] tau) {
        double[] q = new double[n * n];
        for (int i = 0; i < n; i++) {
            q[i * n + i] = 1;
        }

        for (int i = k - 1; i >= 0; i--) {
            double[] v = new double[n];
            for (int j = i; j < n; j++) {
                v[j] = a[i * n + j];
            }
            v[i] = 1;

            for (int row = 0; row < n; row++) {
                double dot = 0;
                for (int col = i; col < n; col++) {
                    dot += q[row * n + col] * v[col];
                }
                for (int col = i; col < n; col++) {
                    q[row * n + col] -= tau[i] * dot * v[col];
                }
            }
        }

        return q;
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

    private static void assertMatrixClose(double[] expected, double[] actual, int rows, int cols, double tol, String msg) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double diff = abs(expected[i * cols + j] - actual[i * cols + j]);
                if (diff > tol) {
                    fail(String.format("%s: mismatch at (%d,%d): expected %g, got %g, diff %g",
                        msg, i, j, expected[i * cols + j], actual[i * cols + j], diff));
                }
            }
        }
    }

    private static void assertArrayClose(double[] expected, double[] actual, double tol, String msg) {
        for (int i = 0; i < expected.length; i++) {
            double diff = abs(expected[i] - actual[i]);
            if (diff > tol) {
                fail(String.format("%s: mismatch at index %d: expected %g, got %g, diff %g",
                    msg, i, expected[i], actual[i], diff));
            }
        }
    }
}
