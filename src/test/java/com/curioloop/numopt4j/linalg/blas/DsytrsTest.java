/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DsytrsTest {

    private static final double TOL = 1e-10;

    @Test
    void testEmpty() {
        Dsytrs.dsytrs(BLAS.Uplo.Lower, 0, 1, new double[0], 0, 1, new int[0], 0, new double[0], 0, 1);
    }

    @Test
    void testSingleElementLower() {
        double[] A = {4.0};
        int[] ipiv = {1};
        double[] b = {8.0};

        Dsytrs.dsytrs(BLAS.Uplo.Lower, 1, 1, A, 0, 1, ipiv, 0, b, 0, 1);

        assertEquals(2.0, b[0], TOL);
    }

    @Test
    void testSingleElementUpper() {
        double[] A = {4.0};
        int[] ipiv = {1};
        double[] b = {8.0};

        Dsytrs.dsytrs(BLAS.Uplo.Upper, 1, 1, A, 0, 1, ipiv, 0, b, 0, 1);

        assertEquals(2.0, b[0], TOL);
    }

    @Test
    void testPositiveDefiniteLower() {
        double[] A = {
            4, 0, 0,
            2, 5, 0,
            2, 3, 6
        };
        int n = 3;
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];

        Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = {8, 7, 11};
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        double[] Ax = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Ax[i] += getOriginalA(i, j, 'L') * b[j];
            }
        }

        for (int i = 0; i < n; i++) {
            assertEquals(bOrig[i], Ax[i], TOL);
        }
    }

    @Test
    void testPositiveDefiniteUpper() {
        double[] A = {
            4, 2, 2,
            0, 5, 3,
            0, 0, 6
        };
        int n = 3;
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];

        Dsytrf.dsytrf(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = {8, 7, 11};
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Upper, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        double[] Ax = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Ax[i] += getOriginalA(i, j, 'U') * b[j];
            }
        }

        for (int i = 0; i < n; i++) {
            assertEquals(bOrig[i], Ax[i], TOL);
        }
    }

    @Test
    void testIndefiniteLower() {
        double[] A = {
            1, 0,
            2, 1
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];

        Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = {3, 4};
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        double[] Ax = new double[n];
        Ax[0] = 1 * b[0] + 2 * b[1];
        Ax[1] = 2 * b[0] + 1 * b[1];

        for (int i = 0; i < n; i++) {
            assertEquals(bOrig[i], Ax[i], TOL);
        }
    }

    @Test
    void testIndefiniteUpper() {
        double[] A = {
            1, 2,
            0, 1
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];

        Dsytrf.dsytrf(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = {3, 4};
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Upper, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        double[] Ax = new double[n];
        Ax[0] = 1 * b[0] + 2 * b[1];
        Ax[1] = 2 * b[0] + 1 * b[1];

        for (int i = 0; i < n; i++) {
            assertEquals(bOrig[i], Ax[i], TOL);
        }
    }

    @Test
    void test2x2BlockPivotLower() {
        double[] A = {
            1, 0,
            2, 1
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];

        Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = {5, 4};
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        double[] Ax = new double[n];
        Ax[0] = 1 * b[0] + 2 * b[1];
        Ax[1] = 2 * b[0] + 1 * b[1];

        for (int i = 0; i < n; i++) {
            assertEquals(bOrig[i], Ax[i], TOL);
        }
    }

    @Test
    void testMultipleRHS() {
        double[] A = {
            4, 0, 0,
            2, 5, 0,
            2, 3, 6
        };
        int n = 3;
        int nrhs = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];

        Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] B = {
            8, 16,
            7, 14,
            11, 22
        };
        double[] BOrig = B.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, nrhs, A, 0, n, ipiv, 0, B, 0, nrhs);

        for (int rhs = 0; rhs < nrhs; rhs++) {
            double[] Ax = new double[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    Ax[i] += getOriginalA(i, j, 'L') * B[j * nrhs + rhs];
                }
            }
            for (int i = 0; i < n; i++) {
                assertEquals(BOrig[i * nrhs + rhs], Ax[i], TOL);
            }
        }
    }

    @Test
    void testLargeMatrixLower() {
        int n = 50;
        java.util.Random rand = new java.util.Random(42);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                A[i * n + j] = rand.nextDouble();
                A[j * n + i] = A[i * n + j];
            }
        }
        for (int i = 0; i < n; i++) {
            A[i * n + i] += n;
        }
        double[] AOrig = A.clone();

        int[] ipiv = new int[n];
        double[] work = new double[n * 64];
        Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = new double[n];
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = rand.nextDouble();
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i] += AOrig[i * n + j] * x[j];
            }
        }
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        for (int i = 0; i < n; i++) {
            assertEquals(x[i], b[i], 1e-8);
        }
    }

    @Test
    void testLargeMatrixUpper() {
        int n = 50;
        java.util.Random rand = new java.util.Random(42);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                A[i * n + j] = rand.nextDouble();
                A[j * n + i] = A[i * n + j];
            }
        }
        for (int i = 0; i < n; i++) {
            A[i * n + i] += n;
        }
        double[] AOrig = A.clone();

        int[] ipiv = new int[n];
        double[] work = new double[n * 64];
        Dsytrf.dsytrf(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = new double[n];
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = rand.nextDouble();
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i] += AOrig[i * n + j] * x[j];
            }
        }
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Upper, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        for (int i = 0; i < n; i++) {
            assertEquals(x[i], b[i], 1e-8);
        }
    }

    @Test
    void testWithDsytrf() {
        int n = 30;
        java.util.Random rand = new java.util.Random(789);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                A[i * n + j] = rand.nextDouble();
                A[j * n + i] = A[i * n + j];
            }
        }
        for (int i = 0; i < n; i++) {
            A[i * n + i] += n;
        }
        double[] AOrig = A.clone();

        int[] ipiv = new int[n];
        double[] work = new double[n * 64];
        Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = new double[n];
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = rand.nextDouble();
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i] += AOrig[i * n + j] * x[j];
            }
        }

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        for (int i = 0; i < n; i++) {
            assertEquals(x[i], b[i], 1e-8);
        }
    }

    @Test
    void testIdentityMatrix() {
        double[] A = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        int n = 3;
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];

        Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = {1, 2, 3};
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        for (int i = 0; i < n; i++) {
            assertEquals(bOrig[i], b[i], TOL);
        }
    }

    @Test
    void testIndefinite3x3Lower() {
        int n = 3;
        double[][] sym = {
            { 1,  2,  3},
            { 2,  1,  2},
            { 3,  2, -1}
        };
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                A[i * n + j] = sym[i][j];
            }
        }
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];
        double[] AOrig = A.clone();

        Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = {6, 8, 4};
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        double[] Ax = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Ax[i] += sym[i][j] * b[j];
            }
        }

        for (int i = 0; i < n; i++) {
            assertEquals(bOrig[i], Ax[i], TOL);
        }
    }

    @Test
    void testIndefinite3x3Upper() {
        int n = 3;
        double[][] sym = {
            { 1,  2,  3},
            { 2,  1,  2},
            { 3,  2, -1}
        };
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                A[i * n + j] = sym[i][j];
            }
        }
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];
        double[] AOrig = A.clone();

        Dsytrf.dsytrf(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = {6, 8, 4};
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Upper, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        double[] Ax = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Ax[i] += sym[i][j] * b[j];
            }
        }

        for (int i = 0; i < n; i++) {
            assertEquals(bOrig[i], Ax[i], TOL);
        }
    }

    @Test
    void testIndefiniteMultiple2x2Blocks() {
        int n = 4;
        double[][] sym = {
            { 1,  2,  0,  0},
            { 2,  1,  1,  0},
            { 0,  1,  1,  2},
            { 0,  0,  2,  1}
        };
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                A[i * n + j] = sym[i][j];
            }
        }
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];
        double[] AOrig = A.clone();

        Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = {5, 10, 8, 6};
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        double[] Ax = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Ax[i] += sym[i][j] * b[j];
            }
        }

        for (int i = 0; i < n; i++) {
            assertEquals(bOrig[i], Ax[i], TOL);
        }
    }

    @Test
    void testIndefiniteMixedPivots() {
        int n = 5;
        double[][] sym = {
            { 4,  1,  2,  0,  0},
            { 1,  1,  2,  0,  0},
            { 2,  2,  1,  1,  0},
            { 0,  0,  1,  5,  2},
            { 0,  0,  0,  2,  3}
        };
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                A[i * n + j] = sym[i][j];
            }
        }
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];
        double[] AOrig = A.clone();

        Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = {12, 8, 15, 25, 18};
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        double[] Ax = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Ax[i] += sym[i][j] * b[j];
            }
        }

        for (int i = 0; i < n; i++) {
            assertEquals(bOrig[i], Ax[i], TOL);
        }
    }

    @Test
    void testIndefiniteLargeMatrix() {
        int n = 30;
        java.util.Random rand = new java.util.Random(999);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                A[i * n + j] = rand.nextDouble() * 10 - 5;
                A[j * n + i] = A[i * n + j];
            }
        }
        double[] AOrig = A.clone();

        int[] ipiv = new int[n];
        double[] work = new double[n * 64];
        Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] b = new double[n];
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = rand.nextDouble() * 10 - 5;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i] += AOrig[i * n + j] * x[j];
            }
        }
        double[] bOrig = b.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        for (int i = 0; i < n; i++) {
            assertEquals(x[i], b[i], 1e-8);
        }
    }

    @Test
    void testIndefiniteMultipleRHS() {
        int n = 2;
        double[][] sym = {
            { 1,  2},
            { 2,  1}
        };
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                A[i * n + j] = sym[i][j];
            }
        }
        int nrhs = 3;
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];
        double[] AOrig = A.clone();

        Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);

        double[] B = {
            3, 6, 9,
            4, 8, 12
        };
        double[] BOrig = B.clone();

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, nrhs, A, 0, n, ipiv, 0, B, 0, nrhs);

        for (int rhs = 0; rhs < nrhs; rhs++) {
            double[] Ax = new double[n];
            Ax[0] = sym[0][0] * B[rhs] + sym[0][1] * B[nrhs + rhs];
            Ax[1] = sym[1][0] * B[rhs] + sym[1][1] * B[nrhs + rhs];
            assertEquals(BOrig[rhs], Ax[0], TOL);
            assertEquals(BOrig[nrhs + rhs], Ax[1], TOL);
        }
    }

    @Test
    void testIndefiniteWithDsytrf() {
        int n = 50;
        java.util.Random rand = new java.util.Random(12345);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                A[i * n + j] = rand.nextDouble() * 20 - 10;
                A[j * n + i] = A[i * n + j];
            }
        }
        double[] AOrig = A.clone();

        int[] ipiv = new int[n];
        double[] work = new double[n * 64];
        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);
        assertEquals(0, info);

        double[] b = new double[n];
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = rand.nextDouble() * 10 - 5;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i] += AOrig[i * n + j] * x[j];
            }
        }

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        for (int i = 0; i < n; i++) {
            assertEquals(x[i], b[i], 1e-7);
        }
    }

    @Test
    void testIndefiniteUpperLarge() {
        int n = 40;
        java.util.Random rand = new java.util.Random(54321);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                A[i * n + j] = rand.nextDouble() * 20 - 10;
                A[j * n + i] = A[i * n + j];
            }
        }
        double[] AOrig = A.clone();

        int[] ipiv = new int[n];
        double[] work = new double[n * 64];
        int info = Dsytrf.dsytrf(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work, n * 64);
        assertEquals(0, info);

        double[] b = new double[n];
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = rand.nextDouble() * 10 - 5;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i] += AOrig[i * n + j] * x[j];
            }
        }

        Dsytrs.dsytrs(BLAS.Uplo.Upper, n, 1, A, 0, n, ipiv, 0, b, 0, 1);

        for (int i = 0; i < n; i++) {
            assertEquals(x[i], b[i], 1e-7);
        }
    }

    private double getOriginalA(int i, int j, char uplo) {
        double[][] origLower = {
            {4, 2, 2},
            {2, 5, 3},
            {2, 3, 6}
        };
        double[][] origUpper = {
            {4, 2, 2},
            {2, 5, 3},
            {2, 3, 6}
        };
        return uplo == 'L' ? origLower[i][j] : origUpper[i][j];
    }
}
