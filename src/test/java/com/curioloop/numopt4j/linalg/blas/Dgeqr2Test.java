/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class Dgeqr2Test {

    private static final double TOL = 1e-14;

    @Test
    @DisplayName("Dgeqr2: all sizes")
    void testAllSizes() {
        Random rnd = new Random(12345);
        for (int m : new int[]{0, 1, 2, 3, 5, 10}) {
            for (int n : new int[]{0, 1, 2, 3, 5, 10}) {
                for (int lda : new int[]{Math.max(1, n), n + 4}) {
                    testDgeqr2(m, n, lda, rnd);
                }
            }
        }
    }

    private void testDgeqr2(int m, int n, int lda, Random rnd) {
        double[] a = new double[m * lda];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a[i * lda + j] = rnd.nextGaussian();
            }
        }

        double[] aCopy = a.clone();

        int k = Math.min(m, n);
        double[] tau = new double[k];
        double[] work = new double[n];

        Dgeqr.dgeqr2(m, n, a, 0, lda, tau, 0, work, 0);

        if (m == 0 || n == 0) {
            return;
        }

        double[] q = constructQ(m, n, a, lda, tau);

        assertTrue(isOrthogonal(q, m, TOL), "Q should be orthogonal for m=" + m + ", n=" + n);

        double[] r = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = i; j < n; j++) {
                r[i * n + j] = a[i * lda + j];
            }
        }

        double[] qr = matMul(q, m, m, r, m, n);
        double resid = maxColSumDiff(qr, aCopy, m, n, n, lda);

        assertTrue(resid <= TOL * m, "|Q*R - A| too large: " + resid + " for m=" + m + ", n=" + n);
    }

    private double[] constructQ(int m, int n, double[] a, int lda, double[] tau) {
        int k = Math.min(m, n);
        double[] q = new double[m * m];

        for (int i = 0; i < m; i++) {
            q[i * m + i] = 1;
        }

        for (int i = 0; i < k; i++) {
            double[] v = new double[m];
            v[i] = 1;
            for (int j = i + 1; j < m; j++) {
                v[j] = a[j * lda + i];
            }

            double[] h = new double[m * m];
            for (int j = 0; j < m; j++) {
                h[j * m + j] = 1;
            }
            for (int row = 0; row < m; row++) {
                for (int col = 0; col < m; col++) {
                    h[row * m + col] -= tau[i] * v[row] * v[col];
                }
            }

            double[] qCopy = q.clone();
            for (int row = 0; row < m; row++) {
                for (int col = 0; col < m; col++) {
                    q[row * m + col] = 0;
                    for (int l = 0; l < m; l++) {
                        q[row * m + col] += qCopy[row * m + l] * h[l * m + col];
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
