/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class DlasrTest {

    private static final double TOL = 1e-10;

    @Test
    void testLeftForwardVariable() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] c = {0.6, 0.8};
        double[] s = {0.8, 0.6};

        Dlasr.dlasr(BLAS.Side.Left, 'V', 'F', 3, 3, c, 0, s, 0, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
        assertTrue(Double.isFinite(A[1]));
    }

    @Test
    void testRightForwardVariable() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] c = {0.6, 0.8};
        double[] s = {0.8, 0.6};

        Dlasr.dlasr(BLAS.Side.Right, 'V', 'F', 3, 3, c, 0, s, 0, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
        assertTrue(Double.isFinite(A[1]));
    }

    @Test
    void testIdentityRotation() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] AOrig = A.clone();
        double[] c = {1, 1};
        double[] s = {0, 0};

        Dlasr.dlasr(BLAS.Side.Left, 'V', 'F', 3, 3, c, 0, s, 0, A, 0, 3);

        for (int i = 0; i < A.length; i++) {
            assertEquals(AOrig[i], A[i], TOL);
        }
    }

    @Test
    void testEmpty() {
        Dlasr.dlasr(BLAS.Side.Left, 'V', 'F', 0, 0, new double[0], 0, new double[0], 0, new double[0], 0, 0);
    }

    @Test
    void testSingleRow() {
        double[] A = {1, 2, 3};
        double[] c = {1};
        double[] s = {0};

        Dlasr.dlasr(BLAS.Side.Left, 'V', 'F', 1, 3, c, 0, s, 0, A, 0, 3);

        assertEquals(1, A[0], TOL);
        assertEquals(2, A[1], TOL);
        assertEquals(3, A[2], TOL);
    }

    @Test
    void testSingleColumn() {
        double[] A = {1, 2, 3};
        double[] c = {1};
        double[] s = {0};

        Dlasr.dlasr(BLAS.Side.Right, 'V', 'F', 3, 1, c, 0, s, 0, A, 0, 1);

        assertEquals(1, A[0], TOL);
        assertEquals(2, A[1], TOL);
        assertEquals(3, A[2], TOL);
    }

    @Test
    void testLeftBackwardVariable() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] c = {0.6, 0.8};
        double[] s = {0.8, 0.6};

        Dlasr.dlasr(BLAS.Side.Left, 'V', 'B', 3, 3, c, 0, s, 0, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
        assertTrue(Double.isFinite(A[1]));
    }

    @Test
    void testRightBackwardVariable() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] c = {0.6, 0.8};
        double[] s = {0.8, 0.6};

        Dlasr.dlasr(BLAS.Side.Right, 'V', 'B', 3, 3, c, 0, s, 0, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
        assertTrue(Double.isFinite(A[1]));
    }

    @Test
    void testLeftForwardTop() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] c = {0.6, 0.8};
        double[] s = {0.8, 0.6};

        Dlasr.dlasr(BLAS.Side.Left, 'T', 'F', 3, 3, c, 0, s, 0, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
        assertTrue(Double.isFinite(A[1]));
    }

    @Test
    void testLeftBackwardTop() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] c = {0.6, 0.8};
        double[] s = {0.8, 0.6};

        Dlasr.dlasr(BLAS.Side.Left, 'T', 'B', 3, 3, c, 0, s, 0, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
        assertTrue(Double.isFinite(A[1]));
    }

    @Test
    void testRightForwardTop() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] c = {0.6, 0.8};
        double[] s = {0.8, 0.6};

        Dlasr.dlasr(BLAS.Side.Right, 'T', 'F', 3, 3, c, 0, s, 0, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
        assertTrue(Double.isFinite(A[1]));
    }

    @Test
    void testRightBackwardTop() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] c = {0.6, 0.8};
        double[] s = {0.8, 0.6};

        Dlasr.dlasr(BLAS.Side.Right, 'T', 'B', 3, 3, c, 0, s, 0, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
        assertTrue(Double.isFinite(A[1]));
    }

    @Test
    void testLeftForwardBottom() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] c = {0.6, 0.8};
        double[] s = {0.8, 0.6};

        Dlasr.dlasr(BLAS.Side.Left, 'B', 'F', 3, 3, c, 0, s, 0, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
        assertTrue(Double.isFinite(A[1]));
    }

    @Test
    void testLeftBackwardBottom() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] c = {0.6, 0.8};
        double[] s = {0.8, 0.6};

        Dlasr.dlasr(BLAS.Side.Left, 'B', 'B', 3, 3, c, 0, s, 0, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
        assertTrue(Double.isFinite(A[1]));
    }

    @Test
    void testRightForwardBottom() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] c = {0.6, 0.8};
        double[] s = {0.8, 0.6};

        Dlasr.dlasr(BLAS.Side.Right, 'B', 'F', 3, 3, c, 0, s, 0, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
        assertTrue(Double.isFinite(A[1]));
    }

    @Test
    void testRightBackwardBottom() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        double[] c = {0.6, 0.8};
        double[] s = {0.8, 0.6};

        Dlasr.dlasr(BLAS.Side.Right, 'B', 'B', 3, 3, c, 0, s, 0, A, 0, 3);

        assertTrue(Double.isFinite(A[0]));
        assertTrue(Double.isFinite(A[1]));
    }

    @Test
    void testLeftVariableForwardRectangular() {
        int m = 5, n = 10;
        double[] A = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            A[i] = Math.random();
        }
        double[] c = new double[m - 1];
        double[] s = new double[m - 1];
        for (int i = 0; i < m - 1; i++) {
            double theta = Math.random() * 2 * Math.PI;
            c[i] = Math.cos(theta);
            s[i] = Math.sin(theta);
        }

        Dlasr.dlasr(BLAS.Side.Left, 'V', 'F', m, n, c, 0, s, 0, A, 0, n);

        for (int i = 0; i < m * n; i++) {
            assertTrue(Double.isFinite(A[i]));
        }
    }

    @Test
    void testRightVariableForwardRectangular() {
        int m = 10, n = 5;
        double[] A = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            A[i] = Math.random();
        }
        double[] c = new double[n - 1];
        double[] s = new double[n - 1];
        for (int i = 0; i < n - 1; i++) {
            double theta = Math.random() * 2 * Math.PI;
            c[i] = Math.cos(theta);
            s[i] = Math.sin(theta);
        }

        Dlasr.dlasr(BLAS.Side.Right, 'V', 'F', m, n, c, 0, s, 0, A, 0, n);

        for (int i = 0; i < m * n; i++) {
            assertTrue(Double.isFinite(A[i]));
        }
    }

    @Test
    void testOrthogonalityLeft() {
        double[] A = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        double theta = Math.PI / 4;
        double[] c = {Math.cos(theta), Math.cos(theta)};
        double[] s = {Math.sin(theta), Math.sin(theta)};

        Dlasr.dlasr(BLAS.Side.Left, 'V', 'F', 3, 3, c, 0, s, 0, A, 0, 3);

        double row0Norm = Math.sqrt(A[0]*A[0] + A[1]*A[1] + A[2]*A[2]);
        double row1Norm = Math.sqrt(A[3]*A[3] + A[4]*A[4] + A[5]*A[5]);
        assertEquals(1.0, row0Norm, TOL);
        assertEquals(1.0, row1Norm, TOL);
    }

    @Test
    void testOrthogonalityRight() {
        double[] A = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };
        double theta = Math.PI / 4;
        double[] c = {Math.cos(theta), Math.cos(theta)};
        double[] s = {Math.sin(theta), Math.sin(theta)};

        Dlasr.dlasr(BLAS.Side.Right, 'V', 'F', 3, 3, c, 0, s, 0, A, 0, 3);

        double col0Norm = Math.sqrt(A[0]*A[0] + A[3]*A[3] + A[6]*A[6]);
        double col1Norm = Math.sqrt(A[1]*A[1] + A[4]*A[4] + A[7]*A[7]);
        assertEquals(1.0, col0Norm, TOL);
        assertEquals(1.0, col1Norm, TOL);
    }

    @Test
    void testRandomParityAllModes() {
        BLAS.Side[] sides = {BLAS.Side.Left, BLAS.Side.Right};
        char[] pivots = {'V', 'T', 'B'};
        char[] directs = {'F', 'B'};
        int[][] shapes = {
            {4, 7},
            {9, 5},
            {16, 13}
        };

        for (BLAS.Side side : sides) {
            for (char pivot : pivots) {
                for (char direct : directs) {
                    for (int[] shape : shapes) {
                        verifyParity(side, pivot, direct, shape[0], shape[1], shape[1] + 3,
                            5, 2, 3,
                            20260423L + side.ordinal() * 97L + pivot * 31L + direct * 17L + shape[0] * 13L + shape[1]);
                    }
                }
            }
        }
    }

    private static void verifyParity(BLAS.Side side, char pivot, char direct, int m, int n, int lda,
                                     int aOff, int cOff, int sOff, long seed) {
        int rotations = side == BLAS.Side.Left ? m - 1 : n - 1;
        double[] c = new double[cOff + rotations + 2];
        double[] s = new double[sOff + rotations + 2];
        double[] expected = new double[aOff + m * lda + 7];
        double[] actual = new double[expected.length];
        Random random = new Random(seed);

        fillRotations(random, c, cOff, s, sOff, rotations);
        fillDense(random, expected);
        System.arraycopy(expected, 0, actual, 0, expected.length);

        BlasTestSupport.scalarDlasr(side, pivot, direct, m, n, c, cOff, s, sOff, expected, aOff, lda);
        Dlasr.dlasr(side, pivot, direct, m, n, c, cOff, s, sOff, actual, aOff, lda);

        assertMatrixClose(expected, actual, 1e-12);
    }

    private static void fillRotations(Random random, double[] c, int cOff, double[] s, int sOff, int rotations) {
        for (int i = 0; i < rotations; i++) {
            if ((i & 3) == 0) {
                c[cOff + i] = 1.0;
                s[sOff + i] = 0.0;
            } else {
                double theta = (random.nextDouble() - 0.5) * (2.0 * Math.PI);
                c[cOff + i] = Math.cos(theta);
                s[sOff + i] = Math.sin(theta);
            }
        }
    }

    private static void fillDense(Random random, double[] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = random.nextDouble() - 0.5;
        }
    }

    private static void assertMatrixClose(double[] expected, double[] actual, double tolerance) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            final int index = i;
            double diff = Math.abs(expected[i] - actual[i]);
            double scale = Math.max(1.0, Math.max(Math.abs(expected[i]), Math.abs(actual[i])));
            assertTrue(diff <= tolerance * scale,
                () -> "Mismatch at index " + index + ": expected=" + expected[index] + ", actual=" + actual[index]);
        }
    }
}
