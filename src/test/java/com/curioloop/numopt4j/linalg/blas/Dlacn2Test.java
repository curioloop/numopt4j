/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Dlacn2 routine.
 *
 * <p>Dlacn2 estimates the 1-norm of a square matrix using reverse communication.
 * The caller must apply the matrix (or its transpose) between calls.</p>
 */
class Dlacn2Test {

    private static final double EPSILON = 1e-10;

    @Test
    @DisplayName("dlacn2: identity matrix")
    void testIdentityMatrix() {
        int n = 3;
        double[][] A = {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
        };

        double est = estimateNorm(n, A);
        assertEquals(1.0, est, EPSILON);
    }

    @Test
    @DisplayName("dlacn2: 2x2 matrix")
    void test2x2Matrix() {
        int n = 2;
        double[][] A = {
            {1, 2},
            {3, 4}
        };

        double est = estimateNorm(n, A);
        double expected = Math.abs(1) + Math.abs(3);
        if (Math.abs(2) + Math.abs(4) > expected) {
            expected = Math.abs(2) + Math.abs(4);
        }
        assertEquals(expected, est, expected * EPSILON);
    }

    @Test
    @DisplayName("dlacn2: 3x3 matrix")
    void test3x3Matrix() {
        int n = 3;
        double[][] A = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        double est = estimateNorm(n, A);
        double expected = Math.abs(3) + Math.abs(6) + Math.abs(9);
        assertEquals(expected, est, expected * EPSILON);
    }

    @Test
    @DisplayName("dlacn2: 4x4 matrix")
    void test4x4Matrix() {
        int n = 4;
        double[][] A = {
            {1, 0, 0, 0},
            {0, 2, 0, 0},
            {0, 0, 3, 0},
            {0, 0, 0, 4}
        };

        double est = estimateNorm(n, A);
        assertEquals(4.0, est, EPSILON);
    }

    @Test
    @DisplayName("dlacn2: single element")
    void testSingleElement() {
        int n = 1;
        double[][] A = {{5.0}};

        double est = estimateNorm(n, A);
        assertEquals(5.0, est, EPSILON);
    }

    @Test
    @DisplayName("dlacn2: negative values")
    void testNegativeValues() {
        int n = 2;
        double[][] A = {
            {-1, -2},
            {-3, -4}
        };

        double est = estimateNorm(n, A);
        double expected = Math.abs(-1) + Math.abs(-3);
        if (Math.abs(-2) + Math.abs(-4) > expected) {
            expected = Math.abs(-2) + Math.abs(-4);
        }
        assertEquals(expected, est, expected * EPSILON);
    }

    @Test
    @DisplayName("dlacn2: large matrix")
    void testLargeMatrix() {
        int n = 10;
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = i + j;
            }
        }

        double est = estimateNorm(n, A);
        double expected = 0;
        for (int i = 0; i < n; i++) {
            double colSum = 0;
            for (int j = 0; j < n; j++) {
                colSum += Math.abs(A[j][i]);
            }
            expected = Math.max(expected, colSum);
        }
        assertEquals(expected, est, expected * EPSILON);
    }

    @Test
    @DisplayName("dlacn2: upper triangular")
    void testUpperTriangular() {
        int n = 3;
        double[][] A = {
            {1, 2, 3},
            {0, 4, 5},
            {0, 0, 6}
        };

        double est = estimateNorm(n, A);
        double expected = Math.abs(3) + Math.abs(5) + Math.abs(6);
        assertEquals(expected, est, expected * EPSILON);
    }

    @Test
    @DisplayName("dlacn2: lower triangular")
    void testLowerTriangular() {
        int n = 3;
        double[][] A = {
            {6, 0, 0},
            {5, 4, 0},
            {3, 2, 1}
        };

        double est = estimateNorm(n, A);
        double expected = Math.abs(6) + Math.abs(5) + Math.abs(3);
        assertEquals(expected, est, expected * EPSILON);
    }

    private double estimateNorm(int n, double[][] A) {
        double[] v = new double[n];
        double[] x = new double[n];
        int[] isgn = new int[n];
        int[] isave = new int[3];

        double est = 0.0;
        int[] kase = {0};

        do {
            est = Dlacn2.dlacn2(n, v, 0, x, 0, isgn, 0, est, kase, 0, isave, 0);

            if (kase[0] == 0) break;

            if (kase[0] == 1) {
                double[] y = new double[n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        y[i] += A[i][j] * x[j];
                    }
                }
                System.arraycopy(y, 0, x, 0, n);
            } else if (kase[0] == 2) {
                double[] y = new double[n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        y[i] += A[j][i] * x[j];
                    }
                }
                System.arraycopy(y, 0, x, 0, n);
            }
        } while (true);

        return est;
    }
}
