/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

interface Dgebrd {

    /**
     * Reduces a general m×n matrix A to upper or lower bidiagonal form using a blocked algorithm.
     *
     * <p>Computes Qᵀ * A * P = B using a blocked Householder reduction. If m >= n, B is upper
     * bidiagonal; if m < n, B is lower bidiagonal. Uses Dlabrd for blocked reduction and
     * Dgebd2 for the unblocked remainder.
     *
     * @param m      number of rows of A
     * @param n      number of columns of A
     * @param A      m×n matrix (row-major, overwritten on exit)
     * @param aOff   offset into A
     * @param lda    leading dimension of A
     * @param d      diagonal elements of B (length min(m,n))
     * @param dOff   offset into d
     * @param e      off-diagonal elements of B (length min(m,n)-1)
     * @param eOff   offset into e
     * @param tauQ   scalar factors for Q reflectors (length min(m,n))
     * @param tauQOff offset into tauQ
     * @param tauP   scalar factors for P reflectors (length min(m,n))
     * @param tauPOff offset into tauP
     * @param work   workspace (length >= max(m,n), or (m+n)*nb for blocked)
     * @param workOff offset into work
     * @param lwork  workspace size; use -1 for workspace query
     * @return 0 on success
     */
    static int dgebrd(int m, int n, double[] A, int aOff, int lda,
                      double[] d, int dOff, double[] e, int eOff,
                      double[] tauQ, int tauQOff, double[] tauP, int tauPOff,
                      double[] work, int workOff, int lwork) {
        int minmn = min(m, n);
        if (minmn == 0) {
            work[workOff] = 1;
            return 0;
        }

        int nb = Ilaenv.ilaenv(1, "DGEBRD", " ", m, n, -1, -1);
        int lwkopt = (m + n) * nb;
        if (lwork == -1) {
            work[workOff] = lwkopt;
            return 0;
        }

        int nx = minmn;
        int ws = max(m, n);
        if (1 < nb && nb < minmn) {
            nx = max(nb, Ilaenv.ilaenv(3, "DGEBRD", " ", m, n, -1, -1));
            if (nx < minmn) {
                ws = (m + n) * nb;
                if (lwork < ws) {
                    int nbmin = Ilaenv.ilaenv(2, "DGEBRD", " ", m, n, -1, -1);
                    if (lwork >= (m + n) * nbmin) {
                        nb = lwork / (m + n);
                    } else {
                        nb = minmn;
                        nx = minmn;
                    }
                }
            }
        }

        int ldworkx = nb;
        int ldworky = nb;
        int i = 0;
        for (i = 0; i < minmn - nx; i += nb) {
            int ib = min(minmn - i, nb);
            Dlabrd.dlabrd(m - i, n - i, ib, A, aOff + i * lda + i, lda,
                          d, dOff + i, e, eOff + i, tauQ, tauQOff + i, tauP, tauPOff + i,
                          work, workOff, ldworkx, work, workOff + m * ldworkx, ldworky);

            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m - i - nb, n - i - nb, nb,
                       -1, A, aOff + (i + nb) * lda + i, lda,
                       work, workOff + nb * ldworky, ldworky,
                       1, A, aOff + (i + nb) * lda + i + nb, lda);

            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m - i - nb, n - i - nb, nb,
                       -1, work, workOff + nb * ldworkx, ldworkx,
                       A, aOff + i * lda + i + nb, lda,
                       1, A, aOff + (i + nb) * lda + i + nb, lda);

            if (m >= n) {
                for (int j = i; j < i + nb; j++) {
                    A[aOff + j * lda + j] = d[dOff + j];
                    A[aOff + j * lda + j + 1] = e[eOff + j];
                }
            } else {
                for (int j = i; j < i + nb; j++) {
                    A[aOff + j * lda + j] = d[dOff + j];
                    A[aOff + (j + 1) * lda + j] = e[eOff + j];
                }
            }
        }

        Dgebd2.dgebd2(m - i, n - i, A, aOff + i * lda + i, lda,
                      d, dOff + i, e, eOff + i, tauQ, tauQOff + i, tauP, tauPOff + i,
                      work, workOff);

        work[workOff] = ws;
        return 0;
    }
}
