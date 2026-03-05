/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DhseqrTest {

    private static final double TOL = 1e-10;

    @Test
    void testEigenvaluesOnly() {
        double[] H = {
            3, 1, 0,
            0, 2, 1,
            0, 0, 1
        };
        int n = 3;
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];

        int info = Dhseqr.dhseqr('E', 'N', n, 0, n - 1, H, n, wr, wi, null, n, work, 0, work.length);

        assertEquals(0, info);
    }

    @Test
    void testEmpty() {
        double[] work = new double[1];
        int info = Dhseqr.dhseqr('E', 'N', 0, 0, -1, new double[0], 0,
                new double[0], new double[0], null, 0, work, 0, 0);
        assertEquals(0, info);
    }

    @Test
    void testSingleElement() {
        double[] H = {5};
        double[] wr = new double[1];
        double[] wi = new double[1];
        double[] work = new double[10];

        int info = Dhseqr.dhseqr('E', 'N', 1, 0, 0, H, 1, wr, wi, null, 1, work, 0, work.length);

        assertEquals(0, info);
        assertEquals(5, wr[0], TOL);
        assertEquals(0, wi[0], TOL);
    }

    @Test
    void testTwoByTwo() {
        double[] H = {
            2, 1,
            0, 3
        };
        int n = 2;
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[20];

        int info = Dhseqr.dhseqr('E', 'N', n, 0, n - 1, H, n, wr, wi, null, n, work, 0, work.length);

        assertEquals(0, info);
        assertTrue(Double.isFinite(wr[0]));
        assertTrue(Double.isFinite(wr[1]));
    }

    @Test
    void testDiagonalMatrix() {
        int n = 4;
        double[] H = new double[n * n];
        double[] diag = {4, 3, 2, 1};
        for (int i = 0; i < n; i++) {
            H[i * n + i] = diag[i];
        }
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];

        int info = Dhseqr.dhseqr('E', 'N', n, 0, n - 1, H, n, wr, wi, null, n, work, 0, work.length);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) {
            assertEquals(diag[i], wr[i], TOL);
            assertEquals(0, wi[i], TOL);
        }
    }

    @Test
    void testIdentityMatrix() {
        int n = 3;
        double[] H = new double[n * n];
        for (int i = 0; i < n; i++) {
            H[i * n + i] = 1.0;
        }
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];

        int info = Dhseqr.dhseqr('E', 'N', n, 0, n - 1, H, n, wr, wi, null, n, work, 0, work.length);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) {
            assertEquals(1.0, wr[i], TOL);
            assertEquals(0, wi[i], TOL);
        }
    }

    @Test
    void testRandomMatrix() {
        int n = 5;
        double[] H = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                H[i * n + j] = Math.random() * 10 - 5;
            }
        }
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[200];

        int info = Dhseqr.dhseqr('E', 'N', n, 0, n - 1, H, n, wr, wi, null, n, work, 0, work.length);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) {
            assertTrue(Double.isFinite(wr[i]));
        }
    }

    @Test
    void testWorkspaceQuery() {
        double[] H = new double[9];
        double[] wr = new double[3];
        double[] wi = new double[3];
        double[] work = new double[1];

        int info = Dhseqr.dhseqr('E', 'N', 3, 0, 2, H, 3, wr, wi, null, 3, work, 0, -1);

        assertEquals(0, info);
        assertTrue(work[0] > 0);
    }

    @Test
    void testSchurForm() {
        double[] H = {
            3, 1, 0,
            0, 2, 1,
            0, 0, 1
        };
        int n = 3;
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] Z = new double[n * n];
        for (int i = 0; i < n; i++) {
            Z[i * n + i] = 1.0;
        }
        double[] work = new double[100];

        int info = Dhseqr.dhseqr('S', 'I', n, 0, n - 1, H, n, wr, wi, Z, n, work, 0, work.length);

        assertEquals(0, info);
    }

    @Test
    void testLargeMatrix() {
        int n = 20;
        double[] H = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                H[i * n + j] = Math.random() * 10 - 5;
            }
        }
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[500];

        int info = Dhseqr.dhseqr('E', 'N', n, 0, n - 1, H, n, wr, wi, null, n, work, 0, work.length);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) {
            assertTrue(Double.isFinite(wr[i]));
        }
    }

    @Test
    void testUpperTriangular() {
        int n = 4;
        double[] H = new double[n * n];
        double[] diag = {4, 3, 2, 1};
        for (int i = 0; i < n; i++) {
            H[i * n + i] = diag[i];
            for (int j = i + 1; j < n; j++) {
                H[i * n + j] = Math.random();
            }
        }
        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];

        int info = Dhseqr.dhseqr('E', 'N', n, 0, n - 1, H, n, wr, wi, null, n, work, 0, work.length);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) {
            assertEquals(diag[i], wr[i], TOL);
        }
    }

    @Test
    void testEigenvalueConservation() {
        int n = 4;
        double[] H = new double[n * n];
        for (int i = 0; i < n; i++) {
            H[i * n + i] = i + 1;
            if (i > 0) {
                H[i * n + i - 1] = Math.random();
            }
        }

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[200];

        int info = Dhseqr.dhseqr('E', 'N', n, 0, n - 1, H, n, wr, wi, null, n, work, 0, work.length);

        assertEquals(0, info);

        double traceOriginal = 0;
        for (int i = 0; i < n; i++) {
            traceOriginal += i + 1;
        }

        double traceAfter = 0;
        for (int i = 0; i < n; i++) {
            traceAfter += wr[i];
        }

        assertEquals(traceOriginal, traceAfter, TOL, "Trace should be preserved");
    }
}
