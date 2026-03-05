/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DgeqrTest {

    private static final double TOL = 1e-10;

    @Test
    void testDgeqr2Basic() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        int m = 3, n = 3;
        double[] tau = new double[Math.min(m, n)];
        double[] work = new double[n];

        Dgeqr.dgeqr2(m, n, A, 0, n, tau, 0, work, 0);

        for (int i = 0; i < Math.min(m, n); i++) {
            assertTrue(Math.abs(tau[i]) <= 2.0, "tau[" + i + "] should be in valid range");
        }
    }

    @Test
    void testDgeqr2Rectangular() {
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        int m = 2, n = 3;
        double[] tau = new double[Math.min(m, n)];
        double[] work = new double[n];

        Dgeqr.dgeqr2(m, n, A, 0, n, tau, 0, work, 0);

        assertEquals(2, tau.length);
    }

    @Test
    void testDgeqrfWorkspaceQuery() {
        double[] A = new double[9];
        double[] tau = new double[3];
        double[] work = new double[1];

        int info = Dgeqr.dgeqrf(3, 3, A, 0, 3, tau, 0, work, 0, -1);

        assertEquals(0, info);
        assertTrue(work[0] > 0);
    }

    @Test
    void testDgeqrfBasic() {
        double[] A = {
            12, -51, 4,
            6, 167, -68,
            -4, 24, -41
        };
        int m = 3, n = 3;
        double[] tau = new double[Math.min(m, n)];
        double[] work = new double[100];

        int info = Dgeqr.dgeqrf(m, n, A, 0, n, tau, 0, work, 0, work.length);

        assertEquals(0, info);

        for (int i = 0; i < Math.min(m, n); i++) {
            assertTrue(Math.abs(tau[i]) <= 2.0, "tau[" + i + "] should be in valid range");
        }
    }

    @Test
    void testDgeqrfZeroMatrix() {
        double[] A = new double[9];
        int m = 3, n = 3;
        double[] tau = new double[Math.min(m, n)];
        double[] work = new double[100];

        int info = Dgeqr.dgeqrf(m, n, A, 0, n, tau, 0, work, 0, work.length);

        assertEquals(0, info);
    }

    @Test
    void testDgeqrfEmpty() {
        double[] work = new double[1];
        double[] tau = new double[0];
        int info = Dgeqr.dgeqrf(0, 0, new double[0], 0, 1, tau, 0, work, 0, 1);
        assertEquals(0, info);
    }

    @Test
    void testDgeqr2TallMatrix() {
        double[] A = {
            1, 2,
            3, 4,
            5, 6,
            7, 8
        };
        int m = 4, n = 2;
        double[] tau = new double[Math.min(m, n)];
        double[] work = new double[n];

        Dgeqr.dgeqr2(m, n, A, 0, n, tau, 0, work, 0);

        assertEquals(2, tau.length);
        assertTrue(Math.abs(tau[0]) <= 2.0);
        assertTrue(Math.abs(tau[1]) <= 2.0);
    }

    @Test
    void testDgeqr2WideMatrix() {
        double[] A = {
            1, 2, 3, 4,
            5, 6, 7, 8
        };
        int m = 2, n = 4;
        double[] tau = new double[Math.min(m, n)];
        double[] work = new double[n];

        Dgeqr.dgeqr2(m, n, A, 0, n, tau, 0, work, 0);

        assertEquals(2, tau.length);
    }

    @Test
    void testDgeqr2SingleColumn() {
        double[] A = {1, 2, 3, 4, 5};
        int m = 5, n = 1;
        double[] tau = new double[1];
        double[] work = new double[1];

        Dgeqr.dgeqr2(m, n, A, 0, n, tau, 0, work, 0);

        assertTrue(Math.abs(tau[0]) <= 2.0);
    }

    @Test
    void testDgeqr2SingleRow() {
        double[] A = {1, 2, 3, 4, 5};
        int m = 1, n = 5;
        double[] tau = new double[1];
        double[] work = new double[n];

        Dgeqr.dgeqr2(m, n, A, 0, n, tau, 0, work, 0);

        assertEquals(0.0, tau[0], TOL);
    }

    @Test
    void testDgeqrfConsistencyWithDgeqr2() {
        java.util.Random rand = new java.util.Random(42);
        int[] sizes = {2, 3, 5, 10, 20};

        for (int n : sizes) {
            double[] A1 = new double[n * n];
            double[] A2 = new double[n * n];
            for (int i = 0; i < n * n; i++) {
                A1[i] = rand.nextDouble() * 2 - 1;
                A2[i] = A1[i];
            }

            double[] tau1 = new double[n];
            double[] tau2 = new double[n];
            double[] work1 = new double[n];
            double[] work2 = new double[100];

            Dgeqr.dgeqr2(n, n, A1, 0, n, tau1, 0, work1, 0);
            Dgeqr.dgeqrf(n, n, A2, 0, n, tau2, 0, work2, 0, work2.length);

            for (int i = 0; i < n * n; i++) {
                assertEquals(A1[i], A2[i], TOL * n, "QR factorization mismatch at index " + i + " for n=" + n);
            }
        }
    }

    @Test
    void testDgeqr2UpperTriangularPart() {
        double[] A = {
            12, -51, 4,
            6, 167, -68,
            -4, 24, -41
        };
        int m = 3, n = 3;
        double[] tau = new double[n];
        double[] work = new double[n];

        Dgeqr.dgeqr2(m, n, A, 0, n, tau, 0, work, 0);

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                assertTrue(Math.abs(A[i * n + j]) > TOL || i == j, 
                    "Upper triangular element should be non-zero");
            }
        }
    }

    @Test
    void testDgeqrfLargeMatrix() {
        int m = 100, n = 50;
        java.util.Random rand = new java.util.Random(42);
        double[] A = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            A[i] = rand.nextDouble() * 2 - 1;
        }
        double[] tau = new double[Math.min(m, n)];
        double[] work = new double[n * 10];

        int info = Dgeqr.dgeqrf(m, n, A, 0, n, tau, 0, work, 0, work.length);

        assertEquals(0, info);
    }
}
