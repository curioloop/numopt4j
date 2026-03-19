/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Random;

import static java.lang.Math.*;
import static org.junit.jupiter.api.Assertions.*;

class DormrqTest {

    private static final double TOL = 1e-10;
    private static final Random rnd = new Random(42);

    @Test
    @DisplayName("Dormrq: Left NoTrans vs Dormr2")
    void testDormrqLeftNoTrans() {
        int m = 100, n = 150;
        int k = min(m, n);
        int lda = m, ldc = n;

        double[] A = generateRandomMatrix(k, m, rnd);
        double[] tau = new double[k];
        double[] C = generateRandomMatrix(m, n, rnd);
        double[] C2 = C.clone();

        int nb = Ilaenv.ilaenv(1, "DORMRQ", "LN", m, n, k, -1);
        int lwork = max(1, n) * nb + 64 * 64;
        double[] work = new double[lwork];
        double[] work2 = new double[max(m, n)];

        Dgerq.dormrq(BLAS.Side.Left, BLAS.Trans.NoTrans, m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work, 0, lwork);
        Dgerq.dormr2(BLAS.Side.Left, BLAS.Trans.NoTrans, m, n, k, A, 0, lda, tau, 0, C2, 0, ldc, work2, 0);

        assertArrayEquals(C2, C, TOL, "Dormrq Left NoTrans should match Dormr2");
    }

    @Test
    @DisplayName("Dormrq: Left Trans vs Dormr2")
    void testDormrqLeftTrans() {
        int m = 100, n = 150;
        int k = min(m, n);
        int lda = m, ldc = n;

        double[] A = generateRandomMatrix(k, m, rnd);
        double[] tau = new double[k];
        double[] C = generateRandomMatrix(m, n, rnd);
        double[] C2 = C.clone();

        int nb = Ilaenv.ilaenv(1, "DORMRQ", "LT", m, n, k, -1);
        int lwork = max(1, n) * nb + 64 * 64;
        double[] work = new double[lwork];
        double[] work2 = new double[max(m, n)];

        Dgerq.dormrq(BLAS.Side.Left, BLAS.Trans.Trans, m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work, 0, lwork);
        Dgerq.dormr2(BLAS.Side.Left, BLAS.Trans.Trans, m, n, k, A, 0, lda, tau, 0, C2, 0, ldc, work2, 0);

        assertArrayEquals(C2, C, TOL, "Dormrq Left Trans should match Dormr2");
    }

    @Test
    @DisplayName("Dormrq: Right NoTrans vs Dormr2")
    void testDormrqRightNoTrans() {
        int m = 150, n = 100;
        int k = min(m, n);
        int lda = n, ldc = n;

        double[] A = generateRandomMatrix(k, n, rnd);
        double[] tau = new double[k];
        double[] C = generateRandomMatrix(m, n, rnd);
        double[] C2 = C.clone();

        int nb = Ilaenv.ilaenv(1, "DORMRQ", "RN", m, n, k, -1);
        int lwork = max(1, m) * nb + 64 * 64;
        double[] work = new double[lwork];
        double[] work2 = new double[max(m, n)];

        Dgerq.dormrq(BLAS.Side.Right, BLAS.Trans.NoTrans, m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work, 0, lwork);
        Dgerq.dormr2(BLAS.Side.Right, BLAS.Trans.NoTrans, m, n, k, A, 0, lda, tau, 0, C2, 0, ldc, work2, 0);

        assertArrayEquals(C2, C, TOL, "Dormrq Right NoTrans should match Dormr2");
    }

    @Test
    @DisplayName("Dormrq: Right Trans vs Dormr2")
    void testDormrqRightTrans() {
        int m = 150, n = 100;
        int k = min(m, n);
        int lda = n, ldc = n;

        double[] A = generateRandomMatrix(k, n, rnd);
        double[] tau = new double[k];
        double[] C = generateRandomMatrix(m, n, rnd);
        double[] C2 = C.clone();

        int nb = Ilaenv.ilaenv(1, "DORMRQ", "RT", m, n, k, -1);
        int lwork = max(1, m) * nb + 64 * 64;
        double[] work = new double[lwork];
        double[] work2 = new double[max(m, n)];

        Dgerq.dormrq(BLAS.Side.Right, BLAS.Trans.Trans, m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work, 0, lwork);
        Dgerq.dormr2(BLAS.Side.Right, BLAS.Trans.Trans, m, n, k, A, 0, lda, tau, 0, C2, 0, ldc, work2, 0);

        assertArrayEquals(C2, C, TOL, "Dormrq Right Trans should match Dormr2");
    }

    @Test
    @DisplayName("Dormrq: Various sizes Left")
    void testDormrqVariousSizesLeft() {
        int[][] sizes = {{50, 50}, {100, 150}, {150, 100}, {200, 200}};

        for (int[] size : sizes) {
            int m = size[0], n = size[1];
            int k = min(m, n);
            int lda = m, ldc = n;

            double[] A = generateRandomMatrix(k, m, rnd);
            double[] tau = new double[k];
            double[] C = generateRandomMatrix(m, n, rnd);
            double[] C2 = C.clone();

            int nb = Ilaenv.ilaenv(1, "DORMRQ", "LN", m, n, k, -1);
            int lwork = max(1, n) * nb + 64 * 64;
            double[] work = new double[lwork];
            double[] work2 = new double[max(m, n)];

            Dgerq.dormrq(BLAS.Side.Left, BLAS.Trans.NoTrans, m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work, 0, lwork);
            Dgerq.dormr2(BLAS.Side.Left, BLAS.Trans.NoTrans, m, n, k, A, 0, lda, tau, 0, C2, 0, ldc, work2, 0);

            assertArrayEquals(C2, C, TOL, "Dormrq Left should match Dormr2 for size " + m + "x" + n);
        }
    }

    @Test
    @DisplayName("Dormrq: Various sizes Right")
    void testDormrqVariousSizesRight() {
        int[][] sizes = {{50, 50}, {100, 150}, {150, 100}, {200, 200}};

        for (int[] size : sizes) {
            int m = size[0], n = size[1];
            int k = min(m, n);
            int lda = n, ldc = n;

            double[] A = generateRandomMatrix(k, n, rnd);
            double[] tau = new double[k];
            double[] C = generateRandomMatrix(m, n, rnd);
            double[] C2 = C.clone();

            int nb = Ilaenv.ilaenv(1, "DORMRQ", "RN", m, n, k, -1);
            int lwork = max(1, m) * nb + 64 * 64;
            double[] work = new double[lwork];
            double[] work2 = new double[max(m, n)];

            Dgerq.dormrq(BLAS.Side.Right, BLAS.Trans.NoTrans, m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work, 0, lwork);
            Dgerq.dormr2(BLAS.Side.Right, BLAS.Trans.NoTrans, m, n, k, A, 0, lda, tau, 0, C2, 0, ldc, work2, 0);

            assertArrayEquals(C2, C, TOL, "Dormrq Right should match Dormr2 for size " + m + "x" + n);
        }
    }

    @Test
    @DisplayName("Dormrq: Workspace query")
    void testDormrqWorkspaceQuery() {
        int m = 100, n = 150;
        int k = min(m, n);
        int lda = m, ldc = n;

        double[] A = generateRandomMatrix(k, m, rnd);
        double[] tau = new double[k];
        double[] C = generateRandomMatrix(m, n, rnd);
        double[] work = new double[1];

        int result = Dgerq.dormrq(BLAS.Side.Left, BLAS.Trans.NoTrans, m, n, k, A, 0, lda, tau, 0, C, 0, ldc, work, 0, -1);

        assertEquals(0, result);
        assertTrue(work[0] > 0, "Workspace query should return positive size");
    }

    private static double[] generateRandomMatrix(int m, int n, Random rnd) {
        double[] a = new double[m * n];
        for (int i = 0; i < a.length; i++) {
            a[i] = rnd.nextDouble();
        }
        return a;
    }
}
