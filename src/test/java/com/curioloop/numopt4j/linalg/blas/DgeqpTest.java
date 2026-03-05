/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static com.curioloop.numopt4j.linalg.blas.BLAS.*;
import static java.lang.Math.*;
import static org.junit.jupiter.api.Assertions.*;

public class DgeqpTest {

    private static final double TOL = 1e-10;

    @Test
    public void testDgeqp3Simple() {
        Random rand = new Random(42);
        testDgeqp3(rand, 3, 3, 3, 0);
    }

    @Test
    public void testDgeqp3Fixed() {
        Random rand = new Random(42);
        testDgeqp3(rand, 4, 3, 4, 2);
    }

    @Test
    public void testDgeqp3Free() {
        Random rand = new Random(42);
        testDgeqp3(rand, 4, 3, 4, 0);
    }

    @Test
    public void testDgeqp3Basic() {
        Random rand = new Random(42);
        for (int m : new int[]{1, 2, 3, 4, 5}) {
            for (int n : new int[]{1, 2, 3, 4, 5}) {
                for (int free = 0; free < 3; free++) {
                    testDgeqp3(rand, m, n, n, free);
                }
            }
        }
    }

    @Test
    public void testDgeqp3Larger() {
        Random rand = new Random(42);
        testDgeqp3(rand, 64, 64, 64, 0);
        testDgeqp3(rand, 128, 128, 128, 0);
    }

    private void testDgeqp3(Random rand, int m, int n, int lda, int freeMode) {
        int mn = min(m, n);
        if (lda < max(1, n)) lda = max(1, n);

        double[] A = randomMatrix(m, n, lda, rand);
        double[] AOrig = A.clone();

        int[] jpvt = new int[n];
        String freeName;
        switch (freeMode) {
            case 0:
                for (int j = 0; j < n; j++) jpvt[j] = -1;
                freeName = "all";
                break;
            case 1:
                for (int j = 0; j < n; j++) jpvt[j] = rand.nextBoolean() ? -1 : 0;
                freeName = "some";
                break;
            default:
                for (int j = 0; j < n; j++) jpvt[j] = 0;
                freeName = "none";
                break;
        }

        double[] tau = new double[mn];
        double[] work = new double[1];
        Dgeqp.dgeqp3(m, n, A, 0, lda, jpvt, tau, work, 0, -1);
        int lwork = max(3 * n + 1, (int) work[0]);
        work = new double[lwork];

        Dgeqp.dgeqp3(m, n, A, 0, lda, jpvt, tau, work, 0, lwork);

        if (mn == 0) return;

        double[][] Q = constructQ(m, mn, A, lda, tau);
        assertOrthogonal(Q, "Case m=" + m + ",n=" + n + ",lda=" + lda + ",free=" + freeName);

        double[][] R = extractR(m, n, mn, A, lda);

        double[][] AP = applyPermutation(AOrig, m, n, lda, jpvt);

        double[][] QR = matMul(Q, R);
        double[][] diff = matSub(QR, AP);
        double resid = maxColumnSum(diff);
        String msg = "Case m=" + m + ",n=" + n + ",lda=" + lda + ",free=" + freeName;
        assertTrue(resid <= TOL * max(m, n), msg + ": |Q*R - A*P|=" + resid + " too large");
    }

    @Test
    public void testDlapmt() {
        Random rand = new Random(42);
        for (int m : new int[]{1, 3, 5}) {
            for (int n : new int[]{1, 3, 5}) {
                for (boolean forward : new boolean[]{true, false}) {
                    testDlapmt(rand, m, n, forward);
                }
            }
        }
    }

    private void testDlapmt(Random rand, int m, int n, boolean forward) {
        int lda = n;
        double[] A = new double[m * lda];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i * lda + j] = i * n + j;
            }
        }

        int[] k = new int[n];
        for (int i = 0; i < n; i++) {
            k[i] = i;
        }
        shuffleArray(k, rand);

        Dgeqp.dlapmt(forward, m, n, A, 0, lda, k, 0);

        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                assertTrue(A[i * lda + j] >= 0 && A[i * lda + j] < m * n,
                        "Value out of range at (" + i + "," + j + ")");
            }
        }
    }

    private static double[] randomMatrix(int m, int n, int lda, Random rand) {
        double[] A = new double[m * lda];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i * lda + j] = rand.nextGaussian();
            }
        }
        return A;
    }

    private static double[][] constructQ(int m, int k, double[] A, int lda, double[] tau) {
        double[][] Q = new double[m][m];
        for (int i = 0; i < m; i++) Q[i][i] = 1;

        for (int i = k - 1; i >= 0; i--) {
            double[] v = new double[m];
            v[i] = 1;
            for (int j = i + 1; j < m; j++) {
                v[j] = A[j * lda + i];
            }

            double[][] H = new double[m][m];
            for (int j = 0; j < m; j++) H[j][j] = 1;
            for (int r = i; r < m; r++) {
                for (int c = i; c < m; c++) {
                    H[r][c] -= tau[i] * v[r] * v[c];
                }
            }

            double[][] newQ = new double[m][m];
            for (int r = 0; r < m; r++) {
                for (int c = 0; c < m; c++) {
                    for (int l = 0; l < m; l++) {
                        newQ[r][c] += H[r][l] * Q[l][c];
                    }
                }
            }
            Q = newQ;
        }
        return Q;
    }

    private static double[][] extractR(int m, int n, int k, double[] A, int lda) {
        double[][] R = new double[m][n];
        for (int i = 0; i < k; i++) {
            for (int j = i; j < n; j++) {
                R[i][j] = A[i * lda + j];
            }
        }
        return R;
    }

    private static double[][] applyPermutation(double[] AOrig, int m, int n, int lda, int[] jpvt) {
        double[][] AP = new double[m][n];
        for (int j = 0; j < n; j++) {
            int srcCol = jpvt[j];
            for (int i = 0; i < m; i++) {
                AP[i][j] = AOrig[i * lda + srcCol];
            }
        }
        return AP;
    }

    private static void assertOrthogonal(double[][] Q, String msg) {
        int m = Q.length;
        double[][] QtQ = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < m; k++) {
                    QtQ[i][j] += Q[k][i] * Q[k][j];
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double diff = abs(QtQ[i][j] - expected);
                assertTrue(diff <= TOL * m,
                        msg + ": Q not orthogonal at (" + i + "," + j + "), diff=" + diff);
            }
        }
    }

    private static double[][] matMul(double[][] A, double[][] B) {
        int m = A.length;
        int n = B[0].length;
        int k = B.length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int l = 0; l < k; l++) {
                    C[i][j] += A[i][l] * B[l][j];
                }
            }
        }
        return C;
    }

    private static double[][] matSub(double[][] A, double[][] B) {
        int m = A.length;
        int n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        return C;
    }

    private static double maxColumnSum(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        double maxSum = 0;
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int i = 0; i < m; i++) {
                sum += abs(A[i][j]);
            }
            maxSum = max(maxSum, sum);
        }
        return maxSum;
    }

    private static void shuffleArray(int[] arr, Random rand) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
}
