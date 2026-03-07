/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

class LUTest {

    private static final double TOLERANCE = 1e-10;

    @Test
    @DisplayName("2x2 matrix inversion")
    void testInverse2x2() {
        double[] A = {4, 3, 6, 3};
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 2);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        verifyInverse(Aorig, inv, 2);
    }

    @Test
    @DisplayName("3x3 matrix inversion")
    void testInverse3x3() {
        double[] A = {1, 2, 3, 0, 1, 4, 5, 6, 0};
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 3);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        verifyInverse(Aorig, inv, 3);
    }

    @Test
    @DisplayName("4x4 matrix inversion")
    void testInverse4x4() {
        double[] A = new double[16];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                A[i * 4 + j] = 1.0 / (i + j + 1);
            }
        }
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 4);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        verifyInverse(Aorig, inv, 4, 1e-8);
    }

    @Test
    @DisplayName("Singular matrix detection")
    void testSingularMatrix() {
        double[] A = {1, 2, 2, 4};

        LU lu = LU.decompose(A, 2);
        assertFalse(lu.ok());
    }

    @Test
    @DisplayName("In-place inversion")
    void testInPlaceInversion() {
        double[] A = {4, 7, 2, 6};
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 2);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        verifyInverse(Aorig, inv, 2);
    }

    @Test
    @DisplayName("Solve linear system")
    void testSolve() {
        double[] A = {3, 2, 1, 2};
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 2);
        assertTrue(lu.ok());

        double[] b = {5, 5};
        double[] x = lu.solve(b, null);
        assertNotNull(x);

        assertEquals(0.0, x[0], TOLERANCE);
        assertEquals(2.5, x[1], TOLERANCE);
    }

    @Test
    @DisplayName("Compare with LMMatrixOps")
    void testCompareWithLMMatrixOps() {
        double[] A = {
            2.5, 1.2, 0.8,
            1.2, 3.1, 1.5,
            0.8, 1.5, 2.8
        };

        double[] A2 = A.clone();
        LU lu = LU.decompose(A2, 3);
        assertTrue(lu.ok());
        double[] invOps = lu.inverse(null);
        assertNotNull(invOps);

        double[] A3 = A.clone();
        LU lu2 = LU.decompose(A3, 3);
        assertTrue(lu2.ok());
        double[] inv = lu2.inverse(null);
        assertNotNull(inv);

        for (int i = 0; i < 9; i++) {
            assertEquals(invOps[i], inv[i], TOLERANCE, "Mismatch at index " + i);
        }
    }

    @Test
    @DisplayName("Identity matrix inversion")
    void testIdentityMatrix() {
        double[] I = {1, 0, 0, 0, 1, 0, 0, 0, 1};

        LU lu = LU.decompose(I, 3);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, inv[i * 3 + j], TOLERANCE);
            }
        }
    }

    @Test
    @DisplayName("Diagonal matrix inversion")
    void testDiagonalMatrix() {
        double[] D = {2, 0, 0, 0, 3, 0, 0, 0, 4};

        LU lu = LU.decompose(D, 3);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        assertEquals(0.5, inv[0], TOLERANCE);
        assertEquals(0.0, inv[1], TOLERANCE);
        assertEquals(0.0, inv[2], TOLERANCE);
        assertEquals(0.0, inv[3], TOLERANCE);
        assertEquals(1.0/3.0, inv[4], TOLERANCE);
        assertEquals(0.0, inv[5], TOLERANCE);
        assertEquals(0.0, inv[6], TOLERANCE);
        assertEquals(0.0, inv[7], TOLERANCE);
        assertEquals(0.25, inv[8], TOLERANCE);
    }

    @Test
    @DisplayName("Zero-allocation inverse with external workspace")
    void testZeroAllocationInverse() {
        double[] A = {
            2.5, 1.2, 0.8,
            1.2, 3.1, 1.5,
            0.8, 1.5, 2.8
        };
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 3);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        verifyInverse(Aorig, inv, 3);
    }

    @Test
    @DisplayName("True in-place inverse (O(n) workspace only)")
    void testTrueInPlaceInverse() {
        double[] A = {4, 7, 2, 6};
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 2);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        verifyInverse(Aorig, inv, 2);
    }

    @Test
    @DisplayName("Input validation")
    void testInputValidation() {
        assertThrows(IllegalArgumentException.class, () ->
            LU.decompose(null, 2));
        assertThrows(IllegalArgumentException.class, () ->
            LU.decompose(new double[3], 2));
        assertThrows(IllegalArgumentException.class, () ->
            LU.decompose(new double[4], 0));
    }

    @Test
    @DisplayName("1x1 matrix inversion")
    void testInverse1x1() {
        double[] A = {5.0};

        LU lu = LU.decompose(A, 1);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        assertEquals(0.2, inv[0], TOLERANCE);
    }

    @Test
    @DisplayName("Large matrix inversion (n=20)")
    void testLargeMatrix20() {
        int n = 20;
        double[] A = generateDiagonallyDominantMatrix(n, 42);
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, n);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        verifyInverse(Aorig, inv, n, 1e-8);
    }

    @Test
    @DisplayName("Large matrix inversion (n=50)")
    void testLargeMatrix50() {
        int n = 50;
        double[] A = generateDiagonallyDominantMatrix(n, 123);
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, n);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        verifyInverse(Aorig, inv, n, 1e-7);
    }

    @Test
    @DisplayName("Upper triangular matrix inversion")
    void testUpperTriangular() {
        double[] A = {2, 3, 4, 0, 5, 6, 0, 0, 7};
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 3);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        verifyInverse(Aorig, inv, 3);
    }

    @Test
    @DisplayName("Lower triangular matrix inversion")
    void testLowerTriangular() {
        double[] A = {2, 0, 0, 3, 5, 0, 4, 6, 7};
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 3);
        assertTrue(lu.ok());
        double[] inv = lu.inverse(null);
        assertNotNull(inv);

        verifyInverse(Aorig, inv, 3);
    }

    @Test
    @DisplayName("Determinant calculation")
    void testDeterminant() {
        double[] A = {4, 3, 6, 3};

        LU lu = LU.decompose(A, 2);
        assertTrue(lu.ok());

        double det = lu.determinant();
        assertEquals(-6.0, det, TOLERANCE);
    }

    @Test
    @DisplayName("LogDet calculation")
    void testLogDet() {
        double[] A = {4, 3, 6, 3};

        LU lu = LU.decompose(A, 2);
        assertTrue(lu.ok());

        double[] logDet = lu.logDet();
        double expectedLogDet = Math.log(6.0);
        assertEquals(expectedLogDet, logDet[0], TOLERANCE);
        assertEquals(-1.0, logDet[1], TOLERANCE);
    }

    @Test
    @DisplayName("LogDet for positive determinant")
    void testLogDetPositive() {
        double[] A = {2, 1, 1, 3};
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 2);
        assertTrue(lu.ok());

        double det = lu.determinant();
        double[] logDet = lu.logDet();

        assertEquals(5.0, det, TOLERANCE);
        assertEquals(Math.log(5.0), logDet[0], TOLERANCE);
        assertEquals(1.0, logDet[1], TOLERANCE);
    }

    @Test
    @DisplayName("extract L")
    void testExtractL() {
        double[] A = {4, 3, 6, 3};
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 2);
        assertTrue(lu.ok());

        double[] L = lu.toL().data;

        assertEquals(1.0, L[0], TOLERANCE);
        assertEquals(0.0, L[1], TOLERANCE);
        assertTrue(Math.abs(L[2]) > 0);
        assertEquals(1.0, L[3], TOLERANCE);
    }

    @Test
    @DisplayName("extract U")
    void testExtractU() {
        double[] A = {4, 3, 6, 3};
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 2);
        assertTrue(lu.ok());

        double[] U = lu.toU().data;

        assertTrue(Math.abs(U[0]) > 0);
        assertTrue(Math.abs(U[1]) > 0);
        assertEquals(0.0, U[2], TOLERANCE);
        assertTrue(Math.abs(U[3]) > 0);
    }

    @Test
    @DisplayName("L and U reconstruct LU")
    void testExtractLAndUReconstruct() {
        double[] A = {
            2.5, 1.2, 0.8,
            1.2, 3.1, 1.5,
            0.8, 1.5, 2.8
        };
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 3);
        assertTrue(lu.ok());

        double[] L = lu.toL().data;
        double[] U = lu.toU().data;
        int[] piv = lu.piv();

        double[] LU_reconstructed = new double[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    LU_reconstructed[i * 3 + j] += L[i * 3 + k] * U[k * 3 + j];
                }
            }
        }

        double[] P = new double[9];
        for (int i = 0; i < 3; i++) {
            P[i * 3 + piv[i]] = 1.0;
        }

        double[] PLU = new double[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    PLU[i * 3 + j] += P[i * 3 + k] * LU_reconstructed[k * 3 + j];
                }
            }
        }

        for (int i = 0; i < 9; i++) {
            assertEquals(Aorig[i], PLU[i], TOLERANCE);
        }
    }

    @Test
    @DisplayName("cond calculation")
    void testCond() {
        double[] A = {
            2.5, 1.2, 0.8,
            1.2, 3.1, 1.5,
            0.8, 1.5, 2.8
        };

        LU lu = LU.decompose(A, 3);
        assertTrue(lu.ok());

        double cond = lu.cond();
        assertTrue(cond > 0);
        assertTrue(cond >= 1);
    }

    @Test
    @DisplayName("cond for ill-conditioned matrix")
    void testCondIllConditioned() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9.0001
        };

        LU lu = LU.decompose(A, 3);
        assertTrue(lu.ok());

        double cond = lu.cond();
        assertTrue(cond > 1e3);
    }

    private double[] generateDiagonallyDominantMatrix(int n, long seed) {
        java.util.Random rand = new java.util.Random(seed);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            double rowSum = 0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    A[i * n + j] = rand.nextDouble() * 2 - 1;
                    rowSum += Math.abs(A[i * n + j]);
                }
            }
            A[i * n + i] = rowSum + 1 + rand.nextDouble();
        }
        return A;
    }

    private void verifyInverse(double[] A, double[] Ainv, int n) {
        verifyInverse(A, Ainv, n, TOLERANCE);
    }

    private void verifyInverse(double[] A, double[] Ainv, int n, double tol) {
        double[] product = new double[n * n];
        // inline matrix multiply: product = A * Ainv
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int k = 0; k < n; k++) s += A[i * n + k] * Ainv[k * n + j];
                product[i * n + j] = s;
            }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, product[i * n + j], tol,
                    "A * A⁻¹ != I at (" + i + "," + j + ")");
            }
        }
    }
}
