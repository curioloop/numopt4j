/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DgeesTest {

    private static final double TOL = 1e-10;

    @Test
    void testSimpleMatrix() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        int n = 3;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);
        
        double[] expectedReal = {16.116843969807043, -1.1168439698070427, 0};
        double[] expectedImag = {0, 0, 0};

        for (int i = 0; i < n; i++) {
            assertEquals(expectedReal[i], wr[i], 1e-8, "Real eigenvalue mismatch at " + i);
            assertEquals(expectedImag[i], wi[i], 1e-8, "Imag eigenvalue mismatch at " + i);
        }
    }

    @Test
    void testDiagonalMatrix() {
        double[] A = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        int n = 3;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);

        double[] expected = {1, 2, 3};
        for (int i = 0; i < n; i++) {
            assertEquals(expected[i], wr[i], TOL);
            assertEquals(0, wi[i], TOL);
        }
    }

    @Test
    void testComplexEigenvalues() {
        double[] A = {
            0, -1,
            1, 0
        };
        int n = 2;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);

        assertEquals(0, wr[0], TOL);
        assertEquals(1, wi[0], TOL);
        assertEquals(0, wr[1], TOL);
        assertEquals(-1, wi[1], TOL);
    }

    @Test
    void testNoVectors() {
        double[] A = {
            1, 2,
            3, 4
        };
        int n = 2;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('N', 'N', null,  n, A, n, wr, wi, null, n, work, 0, work.length, bwork);

        assertEquals(0, info);

        double trace = 5;
        double det = -2;
        double discriminant = trace * trace - 4 * det;
        double sqrtDisc = Math.sqrt(discriminant);

        double e1 = (trace + sqrtDisc) / 2;
        double e2 = (trace - sqrtDisc) / 2;

        boolean[] used = new boolean[n];
        for (int i = 0; i < n; i++) {
            boolean found = false;
            for (int j = 0; j < 2; j++) {
                if (!used[j]) {
                    double expected = (j == 0) ? e1 : e2;
                    if (Math.abs(wr[i] - expected) < TOL && Math.abs(wi[i]) < TOL) {
                        used[j] = true;
                        found = true;
                        break;
                    }
                }
            }
            assertTrue(found, "Unexpected eigenvalue " + wr[i]);
        }
    }

    @Test
    void testWithSort() {
        double[] A = {
            3, 1, 0,
            0, 2, 0,
            0, 0, 1
        };
        int n = 3;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        Select select = (r, i) -> r > 1.5;

        int info = Dgees.dgees('V', 'S', select, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);
        int sdim = (int) work[0];
        assertEquals(2, sdim);
    }

    @Test
    void testRandomMatrix() {
        Random rnd = new Random(42);
        int n = 10;

        double[] A = new double[n * n];
        for (int i = 0; i < n * n; i++) {
            A[i] = rnd.nextGaussian();
        }

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[n * 10];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            if (wi[i] != 0) {
                assertTrue(i < n - 1, "Complex eigenvalue at last position");
                assertEquals(wr[i], wr[i + 1], TOL, "Conjugate pair real parts differ");
                assertEquals(wi[i], -wi[i + 1], TOL, "Conjugate pair imag parts not opposite");
                assertTrue(wi[i] > 0, "First of conjugate pair should have positive imag");
                i++;
            }
        }
    }

    @Test
    void testWorkspaceQuery() {
        double[] A = new double[4];
        double[] wr = new double[2];
        double[] wi = new double[2];
        double[] work = new double[1];

        int info = Dgees.dgees('V', 'N', null, 2, A, 2, wr, wi, null, 2, work, 0, -1, null);

        assertEquals(0, info);
        assertTrue(work[0] > 0);
    }

    @Test
    void testZeroMatrix() {
        double[] A = new double[4];
        int n = 2;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);
        assertEquals(0, wr[0], TOL);
        assertEquals(0, wr[1], TOL);
    }

    @Test
    void testIdentityMatrix() {
        double[] A = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        int n = 3;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);
        for (int i = 0; i < n; i++) {
            assertEquals(1, wr[i], TOL);
            assertEquals(0, wi[i], TOL);
        }
    }

    @Test
    void testSchurForm() {
        double[] A = {
            4, 1, 0,
            0, 3, 1,
            0, 0, 2
        };
        int n = 3;

        double[] wr = new double[n];
        double[] wi = new double[n];
        double[] vs = new double[n * n];
        double[] work = new double[100];
        boolean[] bwork = new boolean[n];

        int info = Dgees.dgees('V', 'N', null, n, A, n, wr, wi, vs, n, work, 0, work.length, bwork);

        assertEquals(0, info);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                assertEquals(0, A[i * n + j], TOL, "Schur form violation at (" + i + "," + j + ")");
            }
        }
    }
}
