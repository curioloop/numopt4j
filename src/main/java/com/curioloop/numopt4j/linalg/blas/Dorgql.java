/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DORGQL: Generates the m×n matrix Q with orthonormal columns defined as the
 * last n columns of a product of k elementary reflectors of order m.
 * 
 * <p>Q = H_{k-1} * ... * H_1 * H_0</p>
 * 
 * <p>Based on gonum/lapack implementation.</p>
 */
interface Dorgql {

    static void dorgql(int m, int n, int k, double[] a, int lda, int aOff,
                       double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        if (n == 0) {
            return;
        }

        if (k == 0) {
            return;
        }

        int nb = Ilaenv.ilaenv(1, "DORGQL", " ", m, n, k, -1);

        if (lwork == -1) {
            work[workOff] = n * nb;
            return;
        }

        int iws = n;
        int ldwork = nb;

        int kk = 0;
        if (nb > 1 && nb < k) {
            kk = Math.min(k, ((k - nb + nb - 1) / nb) * nb);

            for (int row = m - kk; row < m; row++) {
                for (int col = 0; col < n - kk; col++) {
                    a[aOff + row * lda + col] = 0.0;
                }
            }
        }

        int m1 = m - kk;
        int n1 = n - kk;
        int k1 = k - kk;
        if (m1 > 0 && n1 > 0 && k1 > 0) {
            dorg2l(m1, n1, k1, a, lda, aOff, tau, tauOff, work, workOff);
        }

        if (kk > 0) {
            for (int i = k - kk; i < k; i += nb) {
                int ib = Math.min(nb, k - i);

                if (n - k + i > 0) {
                    int vCol = n - k + i;
                    int vOff = aOff + vCol;
                    int mV = m - k + i + ib;
                    Dlarft.dlarftBackward(a, vOff, lda, tau, tauOff + i, work, workOff, ldwork, mV, ib);

                    int m2 = m - k + i + ib;
                    int n2 = n - k + i;
                    Dlarfb.dlarfbLeftBackward(a, vOff, lda,
                                     work, workOff, ldwork,
                                     a, aOff, lda,
                                     work, workOff + ib * ldwork, ldwork, m2, n2, ib);
                }

                int m3 = m - k + i + ib;
                int blockCol = n - k + i;
                int aBlockOff = aOff + blockCol;
                dorg2l(m3, ib, ib, a, lda, aBlockOff, tau, tauOff + i, work, workOff);

                for (int j = n - k + i; j < n - k + i + ib; j++) {
                    for (int l = m - k + i + ib; l < m; l++) {
                        a[aOff + l * lda + j] = 0.0;
                    }
                }
            }
        }

        work[workOff] = iws;
    }

    static void dorg2l(int m, int n, int k, double[] a, int lda, int aOff,
                       double[] tau, int tauOff, double[] work, int wOff) {
        if (n == 0) {
            return;
        }

        for (int j = 0; j < n - k; j++) {
            for (int l = 0; l < m; l++) {
                a[aOff + l * lda + j] = 0.0;
            }
            a[aOff + (m - n + j) * lda + j] = 1.0;
        }

        for (int i = 0; i < k; i++) {
            int ii = n - k + i;

            a[aOff + (m - n + ii) * lda + ii] = 1.0;
            
            int vStart = aOff + ii;
            if (ii > 0) {
                Dlarf.dlarf(BLAS.Side.Left, m - n + ii + 1, ii,
                           a, vStart, lda, tau[tauOff + i],
                           a, aOff, lda, work, wOff);
            }
            
            for (int l = 0; l < m - n + ii; l++) {
                a[aOff + l * lda + ii] *= -tau[tauOff + i];
            }
            
            a[aOff + (m - n + ii) * lda + ii] = 1.0 - tau[tauOff + i];

            for (int l = m - n + ii + 1; l < m; l++) {
                a[aOff + l * lda + ii] = 0.0;
            }
        }
    }
}
