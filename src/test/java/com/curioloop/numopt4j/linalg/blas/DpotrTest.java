/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DpotrTest {

    private static final double TOL = 1e-10;

    @Test
    void testUpper() {
        double[] A = {
            4, 2, 2,
            0, 5, 3,
            0, 0, 6
        };
        int n = 3;

        int info = Dpotr.dpotrf(BLAS.Uplo.Upper, n, A, 0, n);

        assertEquals(0, info);
    }

    @Test
    void testLower() {
        double[] A = {
            4, 0, 0,
            2, 5, 0,
            2, 3, 6
        };
        int n = 3;

        int info = Dpotr.dpotrf(BLAS.Uplo.Lower, n, A, 0, n);

        assertEquals(0, info);
    }

    @Test
    void testEmpty() {
        int info = Dpotr.dpotrf(BLAS.Uplo.Upper, 0, new double[0], 0, 1);
        assertEquals(0, info);
    }

    @Test
    void testSingleElement() {
        double[] A = {4.0};

        int info = Dpotr.dpotrf(BLAS.Uplo.Lower, 1, A, 0, 1);

        assertEquals(0, info);
        assertEquals(2.0, A[0], TOL);
    }

    @Test
    void testNotPositiveDefinite() {
        double[] A = {
            1, 2, 3,
            2, 4, 5,
            3, 5, 6
        };
        int n = 3;

        int info = Dpotr.dpotrf(BLAS.Uplo.Lower, n, A, 0, n);

        assertNotEquals(0, info);
    }

    @Test
    void testDecompositionCorrectnessLower() {
        double[] A = {
            4, 0, 0,
            2, 5, 0,
            2, 3, 6
        };
        double[] Aorig = A.clone();
        int n = 3;

        int info = Dpotr.dpotrf(BLAS.Uplo.Lower, n, A, 0, n);
        assertEquals(0, info);

        double[] LLT = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    LLT[i * n + j] += A[i * n + k] * A[j * n + k];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                assertEquals(Aorig[i * n + j], LLT[i * n + j], TOL);
            }
        }
    }

    @Test
    void testDecompositionCorrectnessUpper() {
        double[] A = {
            4, 2, 2,
            0, 5, 3,
            0, 0, 6
        };
        double[] Aorig = A.clone();
        int n = 3;

        int info = Dpotr.dpotrf(BLAS.Uplo.Upper, n, A, 0, n);
        assertEquals(0, info);

        double[] UTU = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    UTU[i * n + j] += A[k * n + i] * A[k * n + j];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                assertEquals(Aorig[i * n + j], UTU[i * n + j], TOL);
            }
        }
    }

    @Test
    void testDpotriBasic() {
        double[] A = {
            4, 2,
            2, 5
        };
        double[] Aorig = {4, 2, 2, 5};
        int n = 2;

        int info = Dpotr.dpotrf(BLAS.Uplo.Upper, n, A, 0, n);
        assertEquals(0, info);

        boolean ok = Dpotr.dpotri(BLAS.Uplo.Upper, n, A, 0, n);
        assertTrue(ok);

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                A[j * n + i] = A[i * n + j];
            }
        }

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
    void testDpotrsBasic() {
        double[] A = {
            4, 2,
            2, 5
        };
        int n = 2;
        double[] b = {6, 7};

        int info = Dpotr.dpotrf(BLAS.Uplo.Upper, n, A, 0, n);
        assertEquals(0, info);

        Dpotr.dpotrs(BLAS.Uplo.Upper, n, 1, A, 0, n, b, 0, 1);

        assertEquals(1.0, b[0], TOL);
        assertEquals(1.0, b[1], TOL);
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
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double avg = (A[i * n + j] + A[j * n + i]) / 2;
                A[i * n + j] = avg;
                A[j * n + i] = avg;
            }
        }
        for (int i = 0; i < n; i++) {
            A[i * n + i] += n;
        }

        int info = Dpotr.dpotrf(BLAS.Uplo.Lower, n, A, 0, n);
        assertEquals(0, info);
    }
}
