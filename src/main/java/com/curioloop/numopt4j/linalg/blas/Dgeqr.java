/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * Computes the QR factorization of a general m×n matrix A.
 *
 * <p>Computes A = Q * R where Q is orthogonal and R is upper trapezoidal.
 * Also provides blocked (dgeqrf) and unblocked (dgeqr2) variants, as well as
 * routines for generating (dorgqr/dorg2r) the orthogonal matrix Q.
 *
 *
 * @see #dgeqrf(int, int, double[], int, int, double[], int, double[], int, int)
 * @see #dgeqr2(int, int, double[], int, int, double[], int, double[], int)
 * @see #dorgqr(int, int, int, double[], int, int, double[], int, double[], int, int)
 */
interface Dgeqr {

    static void dgeqr2(int m, int n, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        if (m < 0 || n < 0 || lda < max(1, n)) {
            return;
        }

        int k = min(m, n);
        if (k == 0) {
            return;
        }

        for (int i = 0; i < k; i++) {
            int aii = aOff + i * lda + i;
            double alpha = A[aii];
            int xOff = aOff + min(i + 1, m - 1) * lda + i;
            A[aii] = BLAS.dlarfg(m - i, alpha, A, xOff, lda, tau, tauOff + i);
            
            if (i < n - 1) {
                double savedAii = A[aii];
                A[aii] = 1;
                BLAS.dlarf(BLAS.Side.Left, m - i, n - i - 1,
                           A, aii, lda,
                           tau[tauOff + i],
                           A, aii + 1, lda,
                           work, workOff);
                A[aii] = savedAii;
            }
        }
    }

    static int dgeqrf(int m, int n, double[] A, int aOff, int lda,
                      double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        if (m < 0 || n < 0 || lda < max(1, n)) {
            return -1;
        }

        int k = min(m, n);
        if (k == 0) {
            work[workOff] = 1;
            return 0;
        }

        int nb = Ilaenv.ilaenv(1, "DGEQRF", " ", m, n, -1, -1);
        if (lwork == -1) {
            work[workOff] = n * nb;
            return 0;
        }

        int nbmin = 2;
        int nx = 0;
        int iws = n;

        if (1 < nb && nb < k) {
            nx = max(0, Ilaenv.ilaenv(3, "DGEQRF", " ", m, n, -1, -1));
            if (k > nx) {
                iws = n * nb;
                if (lwork < iws) {
                    nb = lwork / n;
                    nbmin = max(2, Ilaenv.ilaenv(2, "DGEQRF", " ", m, n, -1, -1));
                }
            }
        }

        int i = 0;
        if (nbmin <= nb && nb < k && nx < k) {
            int ldwork = nb;
            for (i = 0; i < k - nx; i += nb) {
                int ib = min(k - i, nb);
                dgeqr2(m - i, ib, A, aOff + i * lda + i, lda, tau, tauOff + i, work, workOff);
                if (i + ib < n) {
                    Dlarft.dlarftForwardColWise(A, aOff + i * lda + i, lda, tau, tauOff + i, work, workOff, ldwork, m - i, ib);
                    Dlarfb.dlarfbLeftForwardColWise(A, aOff + i * lda + i, lda, work, workOff, ldwork,
                                A, aOff + i * lda + i + ib, lda, work, workOff + ib * ldwork, ldwork, m - i, n - i - ib, ib);
                }
            }
        }

        if (i < k) {
            dgeqr2(m - i, n - i, A, aOff + i * lda + i, lda, tau, tauOff + i, work, workOff);
        }

        work[workOff] = iws;
        return 0;
    }

    static void dorg2r(int m, int n, int k, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        if (m < 0 || n < 0 || k < 0 || k > min(m, n) || lda < max(1, n)) {
            return;
        }

        if (n == 0) {
            return;
        }

        for (int l = 0; l < m; l++) {
            for (int j = k; j < n; j++) {
                A[aOff + l * lda + j] = 0;
            }
        }
        for (int j = k; j < n; j++) {
            A[aOff + j * lda + j] = 1;
        }

        for (int i = k - 1; i >= 0; i--) {
            for (int w = workOff; w < workOff + n; w++) {
                work[w] = 0;
            }
            if (i < n - 1) {
                A[aOff + i * lda + i] = 1;
                BLAS.dlarf(BLAS.Side.Left, m - i, n - i - 1,
                           A, aOff + i * lda + i, lda,
                           tau[tauOff + i],
                           A, aOff + i * lda + i + 1, lda,
                           work, workOff);
            }
            if (i < m - 1) {
                BLAS.dscal(m - i - 1, -tau[tauOff + i], A, aOff + (i + 1) * lda + i, lda);
            }
            A[aOff + i * lda + i] = 1 - tau[tauOff + i];
            for (int l = 0; l < i; l++) {
                A[aOff + l * lda + i] = 0;
            }
        }
    }

    static int dorgqr(int m, int n, int k, double[] A, int aOff, int lda,
                      double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        if (m < 0 || n < 0 || k < 0 || k > min(m, n) || lda < max(1, n)) {
            return -1;
        }

        if (n == 0) {
            work[workOff] = 1;
            return 0;
        }

        int nb = Ilaenv.ilaenv(1, "DORGQR", " ", m, n, k, -1);
        if (lwork == -1) {
            work[workOff] = n * nb;
            return 0;
        }

        int nbmin = 2;
        int nx = 0;
        int iws = n;
        int ldwork = nb;

        if (1 < nb && nb < k) {
            nx = max(0, Ilaenv.ilaenv(3, "DORGQR", " ", m, n, k, -1));
            if (nx < k) {
                iws = n * ldwork;
            }
        }

        int ki = 0;
        int kk = 0;
        if (nbmin <= nb && nb < k && nx < k) {
            ki = ((k - nx - 1) / nb) * nb;
            kk = min(k, ki + nb);
            for (int j = kk; j < n; j++) {
                for (int l = 0; l < kk; l++) {
                    A[aOff + l * lda + j] = 0;
                }
            }
        }

        if (kk < n) {
            dorg2r(m - kk, n - kk, k - kk, A, aOff + kk * lda + kk, lda, tau, tauOff + kk, work, workOff);
        }

        if (kk > 0) {
            for (int i = ki; i >= 0; i -= nb) {
                int ib = min(nb, k - i);
                if (i + ib < n) {
                    Dlarft.dlarftForwardColWise(A, aOff + i * lda + i, lda, tau, tauOff + i, work, workOff, ldwork, m - i, ib);
                    Dlarfb.dlarfbLeftForwardColWise(A, aOff + i * lda + i, lda, work, workOff, ldwork,
                                A, aOff + i * lda + i + ib, lda, work, workOff + ib * ldwork, ldwork, m - i, n - i - ib, ib);
                }
                dorg2r(m - i, ib, ib, A, aOff + i * lda + i, lda, tau, tauOff + i, work, workOff);
                for (int j = i; j < i + ib; j++) {
                    for (int l = 0; l < i; l++) {
                        A[aOff + l * lda + j] = 0;
                    }
                }
            }
        }

        work[workOff] = iws;
        return 0;
    }
}
