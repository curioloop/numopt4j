/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class Dgelq2Test {

    private static final double TOL = 1e-14;

    @Test
    @DisplayName("Dgelq2: simple 1x2 case")
    void testSimple1x2() {
        double[] a = {1.0, 2.0};
        double[] aCopy = a.clone();
        int m = 1, n = 2, lda = 2;
        
        double[] tau = new double[1];
        double[] work = new double[m];
        
        Dgelq.dgelq2(m, n, a, 0, lda, tau, 0, work, 0);
        
        System.out.println("A after dgelq2: [" + a[0] + ", " + a[1] + "]");
        System.out.println("tau: " + tau[0]);
        
        double[] q = constructQ(m, n, a, lda, tau);
        System.out.println("Q:");
        System.out.println("  [" + q[0] + ", " + q[1] + "]");
        System.out.println("  [" + q[2] + ", " + q[3] + "]");
        
        double[] l = new double[m * n];
        l[0] = a[0];
        System.out.println("L: [" + l[0] + ", " + l[1] + "]");
        
        double[] lq = matMul(l, m, n, q, n, n);
        System.out.println("L*Q: [" + lq[0] + ", " + lq[1] + "]");
        System.out.println("Original A: [" + aCopy[0] + ", " + aCopy[1] + "]");
        
        double resid = maxColSumDiff(lq, aCopy, m, n, n, lda);
        System.out.println("Residual: " + resid);
        
        assertTrue(resid <= TOL, "|L*Q - A| too large: " + resid);
    }

    @Test
    @DisplayName("Dgelq2: all sizes")
    void testAllSizes() {
        Random rnd = new Random(12345);
        for (int m : new int[]{0, 1, 2, 3, 5, 10}) {
            for (int n : new int[]{0, 1, 2, 3, 5, 10}) {
                for (int lda : new int[]{Math.max(1, n), n + 4}) {
                    testDgelq2(m, n, lda, rnd);
                }
            }
        }
    }

    private void testDgelq2(int m, int n, int lda, Random rnd) {
        double[] a = new double[m * lda];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a[i * lda + j] = rnd.nextGaussian();
            }
        }

        double[] aCopy = a.clone();

        int k = Math.min(m, n);
        double[] tau = new double[Math.max(1, k)];
        double[] work = new double[Math.max(1, m)];

        Dgelq.dgelq2(m, n, a, 0, lda, tau, 0, work, 0);

        if (m == 0 || n == 0) {
            return;
        }

        double[] q = constructQ(m, n, a, lda, tau);

        assertTrue(isOrthogonal(q, n, TOL), "Q should be orthogonal for m=" + m + ", n=" + n);

        double[] l = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= Math.min(i, n - 1); j++) {
                l[i * n + j] = a[i * lda + j];
            }
        }

        double[] lq = matMul(l, m, n, q, n, n);
        double resid = maxColSumDiff(lq, aCopy, m, n, n, lda);

        assertTrue(resid <= TOL * Math.max(m, n), "|L*Q - A| too large: " + resid + " for m=" + m + ", n=" + n);
    }

    private double[] constructQ(int m, int n, double[] a, int lda, double[] tau) {
        int k = Math.min(m, n);
        double[] q = new double[n * n];

        for (int i = 0; i < n; i++) {
            q[i * n + i] = 1;
        }

        for (int i = 0; i < k; i++) {
            double[] v = new double[n];
            v[i] = 1;
            for (int j = i + 1; j < n; j++) {
                v[j] = a[i * lda + j];
            }

            double[] h = new double[n * n];
            for (int j = 0; j < n; j++) {
                h[j * n + j] = 1;
            }
            for (int row = 0; row < n; row++) {
                for (int col = 0; col < n; col++) {
                    h[row * n + col] -= tau[i] * v[row] * v[col];
                }
            }

            double[] qCopy = q.clone();
            for (int row = 0; row < n; row++) {
                for (int col = 0; col < n; col++) {
                    q[row * n + col] = 0;
                    for (int l = 0; l < n; l++) {
                        q[row * n + col] += h[row * n + l] * qCopy[l * n + col];
                    }
                }
            }
        }

        return q;
    }

    private boolean isOrthogonal(double[] q, int n, double tol) {
        double[] qtq = new double[n * n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    qtq[i * n + j] += q[k * n + i] * q[k * n + j];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                if (Math.abs(qtq[i * n + j] - expected) > tol) {
                    return false;
                }
            }
        }

        return true;
    }

    private double[] matMul(double[] a, int m, int k, double[] b, int k2, int n) {
        double[] c = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int l = 0; l < k; l++) {
                    c[i * n + j] += a[i * k + l] * b[l * n + j];
                }
            }
        }
        return c;
    }

    private double maxColSumDiff(double[] a, double[] b, int m, int n, int lda, int ldb) {
        double max = 0;
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int i = 0; i < m; i++) {
                sum += Math.abs(a[i * n + j] - b[i * ldb + j]);
            }
            max = Math.max(max, sum);
        }
        return max;
    }
}
