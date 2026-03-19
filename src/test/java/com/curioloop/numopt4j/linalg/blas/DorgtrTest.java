/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DorgtrTest {

    private static final double TOL = 1e-10;

    @Test
    void testUpper() {
        double[] A = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 6
        };
        int n = 3;
        double[] tau = {0.5, 0.3};
        double[] work = new double[100];

        Dorgtr.dorgtr(BLAS.Uplo.Upper, n, A, n, tau, 0, work, 0, work.length);

        for (int i = 0; i < n * n; i++) {
            assertTrue(Double.isFinite(A[i]), "A[" + i + "] should be finite");
        }
    }

    @Test
    void testLower() {
        double[] A = {
            1, 0, 0,
            2, 4, 0,
            3, 5, 6
        };
        int n = 3;
        double[] tau = {0.5, 0.3};
        double[] work = new double[100];

        Dorgtr.dorgtr(BLAS.Uplo.Lower, n, A, n, tau, 0, work, 0, work.length);

        for (int i = 0; i < n * n; i++) {
            assertTrue(Double.isFinite(A[i]), "A[" + i + "] should be finite");
        }
    }

    @Test
    void testEmpty() {
        Dorgtr.dorgtr(BLAS.Uplo.Upper, 0, new double[0], 0, new double[0], 0, new double[1], 0, 0);
    }

    @Test
    void testSingleElement() {
        double[] A = {1};
        double[] tau = new double[0];
        double[] work = new double[1];

        Dorgtr.dorgtr(BLAS.Uplo.Upper, 1, A, 1, tau, 0, work, 0, work.length);

        assertEquals(1.0, A[0], TOL);
    }

    @Test
    void testTwoByTwoUpper() {
        double[] A = {
            1, 2,
            0, 3
        };
        int n = 2;
        double[] tau = {0.5};
        double[] work = new double[10];

        Dorgtr.dorgtr(BLAS.Uplo.Upper, n, A, n, tau, 0, work, 0, work.length);

        for (int i = 0; i < n * n; i++) {
            assertTrue(Double.isFinite(A[i]));
        }
    }

    @Test
    void testTwoByTwoLower() {
        double[] A = {
            1, 0,
            2, 3
        };
        int n = 2;
        double[] tau = {0.5};
        double[] work = new double[10];

        Dorgtr.dorgtr(BLAS.Uplo.Lower, n, A, n, tau, 0, work, 0, work.length);

        for (int i = 0; i < n * n; i++) {
            assertTrue(Double.isFinite(A[i]));
        }
    }

    @Test
    void testWorkspaceQuery() {
        double[] A = new double[9];
        double[] tau = new double[2];
        double[] work = new double[1];

        Dorgtr.dorgtr(BLAS.Uplo.Upper, 3, A, 3, tau, 0, work, 0, -1);

        assertTrue(work[0] > 0);
    }

    @Test
    void testRandomUpper() {
        int n = 5;
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                A[i * n + j] = Math.random();
            }
        }
        double[] tau = new double[n - 1];
        for (int i = 0; i < n - 1; i++) {
            tau[i] = Math.random();
        }
        double[] work = new double[100];

        Dorgtr.dorgtr(BLAS.Uplo.Upper, n, A, n, tau, 0, work, 0, work.length);

        for (int i = 0; i < n * n; i++) {
            assertTrue(Double.isFinite(A[i]));
        }
    }

    @Test
    void testRandomLower() {
        int n = 5;
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                A[i * n + j] = Math.random();
            }
        }
        double[] tau = new double[n - 1];
        for (int i = 0; i < n - 1; i++) {
            tau[i] = Math.random();
        }
        double[] work = new double[100];

        Dorgtr.dorgtr(BLAS.Uplo.Lower, n, A, n, tau, 0, work, 0, work.length);

        for (int i = 0; i < n * n; i++) {
            assertTrue(Double.isFinite(A[i]));
        }
    }

    @Test
    void testLargeMatrix() {
        int n = 20;
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                A[i * n + j] = Math.random();
            }
        }
        double[] tau = new double[n - 1];
        for (int i = 0; i < n - 1; i++) {
            tau[i] = Math.random();
        }
        double[] work = new double[500];

        Dorgtr.dorgtr(BLAS.Uplo.Upper, n, A, n, tau, 0, work, 0, work.length);

        for (int i = 0; i < n * n; i++) {
            assertTrue(Double.isFinite(A[i]));
        }
    }

    @Test
    void testIdentityUpper() {
        int n = 4;
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            A[i * n + i] = 1.0;
        }
        double[] tau = new double[n - 1];
        double[] work = new double[100];

        Dorgtr.dorgtr(BLAS.Uplo.Upper, n, A, n, tau, 0, work, 0, work.length);

        for (int i = 0; i < n; i++) {
            assertEquals(1.0, A[i * n + i], TOL);
        }
    }

    @Test
    void testIdentityLower() {
        int n = 4;
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            A[i * n + i] = 1.0;
        }
        double[] tau = new double[n - 1];
        double[] work = new double[100];

        Dorgtr.dorgtr(BLAS.Uplo.Lower, n, A, n, tau, 0, work, 0, work.length);

        for (int i = 0; i < n; i++) {
            assertEquals(1.0, A[i * n + i], TOL);
        }
    }
}
