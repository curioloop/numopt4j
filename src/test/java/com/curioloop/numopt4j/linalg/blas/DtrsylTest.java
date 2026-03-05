/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Dtrsyl routine.
 *
 * <p>Dtrsyl solves the real Sylvester matrix equation:
 * op(A)*X + X*op(B) = scale*C</p>
 */
class DtrsylTest {

    private static final double EPSILON = 1e-10;

    @Test
    @DisplayName("dtrsyl: 1x1 identity matrices")
    void test1x1Identity() {
        double[] a = {2.0};
        double[] b = {3.0};
        double[] c = {10.0};

        boolean[] okOut = new boolean[1];
        double scale = Dtrsyl.dtrsyl(false, false, 1, 1, 1,
                a, 0, 1, b, 0, 1, c, 0, 1, okOut);

        assertTrue(okOut[0]);
        assertEquals(1.0, scale, EPSILON);
        assertEquals(2.0, c[0], EPSILON);
    }

    @Test
    @DisplayName("dtrsyl: 2x2 identity matrices")
    void test2x2Identity() {
        double[] a = {1, 0, 0, 1};
        double[] b = {1, 0, 0, 1};
        double[] c = {1, 2, 3, 4};

        boolean[] okOut = new boolean[1];
        double scale = Dtrsyl.dtrsyl(false, false, 1, 2, 2,
                a, 0, 2, b, 0, 2, c, 0, 2, okOut);

        assertTrue(okOut[0]);
        assertEquals(1.0, scale, EPSILON);
        assertEquals(0.5, c[0], EPSILON);
        assertEquals(1.0, c[1], EPSILON);
        assertEquals(1.5, c[2], EPSILON);
        assertEquals(2.0, c[3], EPSILON);
    }

    @Test
    @DisplayName("dtrsyl: A*X + X*B = C with diagonal matrices")
    void testDiagonalMatrices() {
        double[] a = {2, 0, 0, 3};
        double[] b = {4, 0, 0, 5};
        double[] c = {24, 0, 0, 32};

        boolean[] okOut = new boolean[1];
        double scale = Dtrsyl.dtrsyl(false, false, 1, 2, 2,
                a, 0, 2, b, 0, 2, c, 0, 2, okOut);

        assertTrue(okOut[0]);
        assertEquals(1.0, scale, EPSILON);
        assertEquals(4.0, c[0], EPSILON);
        assertEquals(0.0, c[1], EPSILON);
        assertEquals(0.0, c[2], EPSILON);
        assertEquals(4.0, c[3], EPSILON);
    }

    @Test
    @DisplayName("dtrsyl: A*X - X*B = C (isgn=-1)")
    void testNegativeSign() {
        double[] a = {5, 0, 0, 6};
        double[] b = {2, 0, 0, 3};
        double[] c = {6, 0, 0, 12};

        boolean[] okOut = new boolean[1];
        double scale = Dtrsyl.dtrsyl(false, false, -1, 2, 2,
                a, 0, 2, b, 0, 2, c, 0, 2, okOut);

        assertTrue(okOut[0]);
        assertEquals(1.0, scale, EPSILON);
        assertEquals(2.0, c[0], EPSILON);
        assertEquals(0.0, c[1], EPSILON);
        assertEquals(0.0, c[2], EPSILON);
        assertEquals(4.0, c[3], EPSILON);
    }

    @Test
    @DisplayName("dtrsyl: upper triangular matrices")
    void testUpperTriangular() {
        double[] a = {2, 1, 0, 3};
        double[] b = {4, 1, 0, 5};
        double[] c = {24, 13, 0, 32};

        double[] cOrig = c.clone();

        boolean[] okOut = new boolean[1];
        double scale = Dtrsyl.dtrsyl(false, false, 1, 2, 2,
                a, 0, 2, b, 0, 2, c, 0, 2, okOut);

        assertTrue(okOut[0]);
        verifyResidual(a, b, c, cOrig, scale, 1, 2);
    }

    @Test
    @DisplayName("dtrsyl: 3x3 matrices")
    void test3x3Matrices() {
        double[] a = {
            2, 0, 0,
            0, 3, 0,
            0, 0, 4
        };
        double[] b = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        double[] c = {
            6, 0, 0,
            0, 10, 0,
            0, 0, 14
        };

        boolean[] okOut = new boolean[1];
        double scale = Dtrsyl.dtrsyl(false, false, 1, 3, 3,
                a, 0, 3, b, 0, 3, c, 0, 3, okOut);

        assertTrue(okOut[0]);
        assertEquals(1.0, scale, EPSILON);
        assertEquals(2.0, c[0], EPSILON);
        assertEquals(2.0, c[4], EPSILON);
        assertEquals(2.0, c[8], EPSILON);
    }

    @Test
    @DisplayName("dtrsyl: transpose A")
    void testTransposeA() {
        double[] a = {2, 1, 0, 3};
        double[] b = {4, 0, 0, 5};
        double[] c = {24, 0, 13, 32};

        boolean[] okOut = new boolean[1];
        Dtrsyl.dtrsyl(true, false, 1, 2, 2,
                a, 0, 2, b, 0, 2, c, 0, 2, okOut);

        assertTrue(okOut[0]);
    }

    @Test
    @DisplayName("dtrsyl: transpose B")
    void testTransposeB() {
        double[] a = {2, 0, 0, 3};
        double[] b = {4, 1, 0, 5};
        double[] c = {24, 13, 0, 32};

        boolean[] okOut = new boolean[1];
        Dtrsyl.dtrsyl(false, true, 1, 2, 2,
                a, 0, 2, b, 0, 2, c, 0, 2, okOut);

        assertTrue(okOut[0]);
    }

    @Test
    @DisplayName("dtrsyl: transpose both A and B")
    void testTransposeBoth() {
        double[] a = {2, 1, 0, 3};
        double[] b = {4, 1, 0, 5};
        double[] c = {24, 0, 13, 32};

        boolean[] okOut = new boolean[1];
        Dtrsyl.dtrsyl(true, true, 1, 2, 2,
                a, 0, 2, b, 0, 2, c, 0, 2, okOut);

        assertTrue(okOut[0]);
    }

    @Test
    @DisplayName("dtrsyl: 2x2 block in A (quasi-triangular)")
    void testQuasiTriangularA() {
        double[] a = {
            1, 2, 0, 0,
            -3, 4, 0, 0,
            0, 0, 5, 0,
            0, 0, 0, 6
        };
        double[] b = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        double[] c = new double[12];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                c[i * 3 + j] = (i + 1) * (j + 1);
            }
        }

        boolean[] okOut = new boolean[1];
        Dtrsyl.dtrsyl(false, false, 1, 4, 3,
                a, 0, 4, b, 0, 3, c, 0, 3, okOut);

        assertTrue(okOut[0]);
    }

    @Test
    @DisplayName("dtrsyl: verify residual A*X + X*B = scale*C")
    void testVerifyResidual() {
        double[] a = {3, 1, 0, 4};
        double[] b = {2, 0.5, 0, 3};
        double[] c = {10, 5, 8, 12};

        double[] cOrig = c.clone();

        boolean[] okOut = new boolean[1];
        double scale = Dtrsyl.dtrsyl(false, false, 1, 2, 2,
                a, 0, 2, b, 0, 2, c, 0, 2, okOut);

        assertTrue(okOut[0]);

        double[][] ax = new double[2][2];
        double[][] xb = new double[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    ax[i][j] += a[i * 2 + k] * c[k * 2 + j];
                    xb[i][j] += c[i * 2 + k] * b[k * 2 + j];
                }
            }
        }

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double expected = scale * cOrig[i * 2 + j];
                double actual = ax[i][j] + xb[i][j];
                assertEquals(expected, actual, Math.abs(expected) * EPSILON + EPSILON);
            }
        }
    }

    @Test
    @DisplayName("dtrsyl: m=0 returns scale=1")
    void testMZero() {
        double[] a = {};
        double[] b = {1, 0, 0, 1};
        double[] c = {};

        boolean[] okOut = new boolean[1];
        double scale = Dtrsyl.dtrsyl(false, false, 1, 0, 2,
                a, 0, 1, b, 0, 2, c, 0, 1, okOut);

        assertEquals(1.0, scale, EPSILON);
        assertTrue(okOut[0]);
    }

    @Test
    @DisplayName("dtrsyl: n=0 returns scale=1")
    void testNZero() {
        double[] a = {1, 0, 0, 1};
        double[] b = {};
        double[] c = {};

        boolean[] okOut = new boolean[1];
        double scale = Dtrsyl.dtrsyl(false, false, 1, 2, 0,
                a, 0, 2, b, 0, 1, c, 0, 2, okOut);

        assertEquals(1.0, scale, EPSILON);
        assertTrue(okOut[0]);
    }

    @Test
    @DisplayName("dtrsyl: okOut=null skips ok output")
    void testOkOutNull() {
        double[] a = {2.0};
        double[] b = {3.0};
        double[] c = {10.0};

        double scale = Dtrsyl.dtrsyl(false, false, 1, 1, 1,
                a, 0, 1, b, 0, 1, c, 0, 1, null);

        assertEquals(1.0, scale, EPSILON);
        assertEquals(2.0, c[0], EPSILON);
    }

    private void verifyResidual(double[] a, double[] b, double[] x, double[] cOrig, double scale, int isgn, int n) {
        double[][] ax = new double[n][n];
        double[][] xb = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    ax[i][j] += a[i * n + k] * x[k * n + j];
                    xb[i][j] += x[i * n + k] * b[k * n + j];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double lhs = ax[i][j] + isgn * xb[i][j];
                double rhs = scale * cOrig[i * n + j];
                assertEquals(rhs, lhs, Math.abs(rhs) * EPSILON + EPSILON,
                    "Residual check failed at (" + i + "," + j + ")");
            }
        }
    }
}
