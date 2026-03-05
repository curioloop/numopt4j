/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DgetrTest {

    private static final double TOL = 1e-10;

    @Test
    void testBasic() {
        double[] A = {
            2, 1, 1,
            4, 3, 3,
            8, 7, 9
        };
        int n = 3;
        int[] ipiv = new int[n];

        int info = Dgetr.dgetrf(n, n, A, 0, n, ipiv, 0);

        assertEquals(0, info);
    }

    @Test
    void testEmpty() {
        int info = Dgetr.dgetrf(0, 0, new double[0], 0, 1, new int[0], 0);
        assertEquals(0, info);
    }

    @Test
    void testSingular() {
        double[] A = {
            1, 2, 3,
            1, 2, 3,
            1, 2, 3
        };
        int n = 3;
        int[] ipiv = new int[n];

        int info = Dgetr.dgetrf(n, n, A, 0, n, ipiv, 0);

        assertNotEquals(0, info);
    }

    @Test
    void testSingleElement() {
        double[] A = {5};
        int[] ipiv = new int[1];

        int info = Dgetr.dgetrf(1, 1, A, 0, 1, ipiv, 0);

        assertEquals(0, info);
        assertEquals(0, ipiv[0]);
    }

    @Test
    void testRectangularTall() {
        double[] A = {
            1, 2,
            3, 4,
            5, 6
        };
        int m = 3, n = 2;
        int[] ipiv = new int[Math.min(m, n)];

        int info = Dgetr.dgetrf(m, n, A, 0, n, ipiv, 0);

        assertEquals(0, info);
    }

    @Test
    void testRectangularWide() {
        double[] A = {
            1, 2, 3,
            4, 5, 6
        };
        int m = 2, n = 3;
        int[] ipiv = new int[Math.min(m, n)];

        int info = Dgetr.dgetrf(m, n, A, 0, n, ipiv, 0);

        assertEquals(0, info);
    }

    @Test
    void testPLUDecomposition() {
        double[] A = {
            4, 3,
            6, 3
        };
        double[] Aorig = A.clone();
        int n = 2;
        int[] ipiv = new int[n];

        int info = Dgetr.dgetrf(n, n, A, 0, n, ipiv, 0);
        assertEquals(0, info);

        double[] L = new double[n * n];
        double[] U = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i > j) {
                    L[i * n + j] = A[i * n + j];
                } else {
                    U[i * n + j] = A[i * n + j];
                }
                if (i == j) {
                    L[i * n + j] = 1.0;
                }
            }
        }

        double[] LU = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    LU[i * n + j] += L[i * n + k] * U[k * n + j];
                }
            }
        }

        double[] P = new double[n * n];
        for (int i = 0; i < n; i++) {
            P[i * n + i] = 1.0;
        }
        for (int i = n - 1; i >= 0; i--) {
            if (ipiv[i] != i) {
                for (int j = 0; j < n; j++) {
                    double tmp = P[i * n + j];
                    P[i * n + j] = P[ipiv[i] * n + j];
                    P[ipiv[i] * n + j] = tmp;
                }
            }
        }

        double[] PA = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    PA[i * n + j] += P[i * n + k] * Aorig[k * n + j];
                }
            }
        }

        for (int i = 0; i < n * n; i++) {
            assertEquals(PA[i], LU[i], TOL);
        }
    }

    @Test
    void testDgetriBasic() {
        double[] A = {
            4, 3,
            6, 3
        };
        double[] Aorig = A.clone();
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n * n];

        int info = Dgetr.dgetrf(n, n, A, 0, n, ipiv, 0);
        assertEquals(0, info);

        boolean ok = Dgetr.dgetri(n, A, 0, n, ipiv, 0, work, 0, work.length);
        assertTrue(ok);

        double[] I = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    I[i * n + j] += Aorig[i * n + k] * A[k * n + j];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, I[i * n + j], TOL);
            }
        }
    }

    @Test
    void testDgetrsBasic() {
        double[] A = {
            3, 2,
            1, 2
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] b = {5, 5};

        int info = Dgetr.dgetrf(n, n, A, 0, n, ipiv, 0);
        assertEquals(0, info);

        Dgetr.dgetrs(BLAS.Transpose.NoTrans, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        assertEquals(0.0, b[0], TOL);
        assertEquals(2.5, b[1], TOL);
    }

    @Test
    void testDgetrsTransposed() {
        double[] A = {
            3, 2,
            1, 2
        };
        double[] Aorig = A.clone();
        int n = 2;
        int[] ipiv = new int[n];
        double[] b = {5, 5};
        double[] bOrig = b.clone();

        int info = Dgetr.dgetrf(n, n, A, 0, n, ipiv, 0);
        assertEquals(0, info);

        Dgetr.dgetrs(BLAS.Transpose.Trans, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        double[] Atb = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Atb[i] += Aorig[j * n + i] * b[j];
            }
        }

        for (int i = 0; i < n; i++) {
            assertEquals(bOrig[i], Atb[i], TOL);
        }
    }

    @Test
    void testLargeMatrix() {
        int n = 100;
        java.util.Random rand = new java.util.Random(42);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = rand.nextDouble();
            }
            A[i * n + i] += n;
        }
        int[] ipiv = new int[n];

        int info = Dgetr.dgetrf(n, n, A, 0, n, ipiv, 0);
        assertEquals(0, info);
    }
}
