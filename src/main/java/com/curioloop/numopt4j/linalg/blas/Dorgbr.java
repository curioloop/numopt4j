/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * DORGBR generates one of the real orthogonal matrices Q or P**T
 * determined by DGEBRD when reducing a real matrix A to bidiagonal form:
 * A = Q * B * P**T.
 *
 * <p>If {@code vect = 'Q'}, generates Q (or the leading columns of Q if m < k).
 * If {@code vect = 'P'}, generates P**T (or the leading rows of P**T if n < k).
 *
 * <p>Reference: LAPACK Working Note, DORGBR.
 *
 * @see Dgebrd
 */
interface Dorgbr {

    static int dorgbr(char vect, int m, int n, int k,
                      double[] A, int aOff, int lda,
                      double[] tau, int tauOff,
                      double[] work, int workOff, int lwork) {
        boolean wantq = (vect == 'Q');

        if (m == 0 || n == 0) {
            work[workOff] = 1;
            return 0;
        }

        int mn = wantq ? m : n;
        int nb = Ilaenv.ilaenv(1, wantq ? "DORGQR" : "DORGLQ", " ", m, n, k, -1);
        int lwkopt = max(1, mn) * nb;
        if (lwork == -1) {
            work[workOff] = lwkopt;
            return 0;
        }

        if (wantq) {
            if (m >= k) {
                Dgeqr.dorgqr(m, n, k, A, aOff, lda, tau, tauOff, work, workOff, lwork);
            } else {
                for (int j = m - 1; j >= 1; j--) {
                    A[aOff + j] = 0;
                    for (int i = j + 1; i < m; i++) {
                        A[aOff + i * lda + j] = A[aOff + i * lda + j - 1];
                    }
                }
                A[aOff] = 1;
                for (int i = 1; i < m; i++) {
                    A[aOff + i * lda] = 0;
                }
                if (m > 1) {
                    Dgeqr.dorgqr(m - 1, m - 1, m - 1, A, aOff + lda + 1, lda, tau, tauOff, work, workOff, lwork);
                }
            }
        } else {
            if (k < n) {
                Dgelq.dorglq(m, n, k, A, aOff, lda, tau, tauOff, work, workOff, lwork);
            } else {
                A[aOff] = 1;
                for (int i = 1; i < n; i++) {
                    A[aOff + i * lda] = 0;
                }
                for (int j = 1; j < n; j++) {
                    for (int i = j - 1; i >= 1; i--) {
                        A[aOff + i * lda + j] = A[aOff + (i - 1) * lda + j];
                    }
                    A[aOff + j] = 0;
                }
                if (n > 1) {
                    Dgelq.dorglq(n - 1, n - 1, n - 1, A, aOff + lda + 1, lda, tau, tauOff, work, workOff, lwork);
                }
            }
        }

        work[workOff] = lwkopt;
        return 0;
    }
}
