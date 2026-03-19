/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.max;
import static java.lang.Math.min;

interface Dgehrd {

    int NBMAX = 64;

    static int dgehrd(int n, int ilo, int ihi, double[] A, int aOff, int lda,
                      double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        if (n <= 0) {
            return 0;
        }

        int nh = ihi - ilo + 1;
        if (nh <= 1) {
            work[workOff] = 1;
            return 0;
        }

        int nb = min(NBMAX, Ilaenv.ilaenv(1, "DGEHRD", " ", n, ilo, ihi, -1));
        int ldt = NBMAX + 1;
        int tsize = ldt * NBMAX;
        int lwkopt = n * nb + tsize;

        if (lwork == -1) {
            work[workOff] = lwkopt;
            return 0;
        }

        for (int i = 0; i < ilo; i++) {
            tau[tauOff + i] = 0;
        }
        for (int i = ihi; i < n - 1; i++) {
            tau[tauOff + i] = 0;
        }

        int nbmin = 2;
        int nx = nh;

        if (nb > 1 && nb < nh) {
            nx = max(nb, Ilaenv.ilaenv(3, "DGEHRD", " ", n, ilo, ihi, -1));
            if (nx < nh) {
                if (lwork < n * nb + tsize) {
                    nbmin = max(2, Ilaenv.ilaenv(2, "DGEHRD", " ", n, ilo, ihi, -1));
                    if (lwork >= n * nbmin + tsize) {
                        nb = (lwork - tsize) / n;
                    } else {
                        nb = 1;
                    }
                }
            }
        }

        int ldwork = nb;

        int i = ilo;
        if (nb >= nbmin && nb < nh && nx < nh) {
            int iwt = workOff + n * nb;

            for (i = ilo; i < ihi - nx; i += nb) {
                int ib = min(nb, ihi - i);

                Dlahr2.dlahr2(ihi - i + 1, 1, ib, A, aOff + i * lda + i, lda, tau, tauOff + i, work, iwt, ldt, work, workOff, ldwork);

                double ei = A[aOff + (i + ib) * lda + i + ib - 1];
                A[aOff + (i + ib) * lda + i + ib - 1] = 1.0;

                Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, ihi + 1, ihi - i - ib + 1, ib,
                        -1.0, work, workOff, ldwork,
                        A, aOff + (i + ib) * lda + i, lda,
                        1.0, A, aOff + i + ib, lda);
                A[aOff + (i + ib) * lda + i + ib - 1] = ei;

                if (ib > 1) {
                    Dtrmm.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, i + 1, ib - 1,
                            1.0, A, aOff + (i + 1) * lda + i, lda, work, workOff, ldwork);
                    for (int j = 0; j <= ib - 2; j++) {
                        Daxpy.daxpy(i + 1, -1.0, work, workOff + j, ldwork, A, aOff + i + j + 1, lda);
                    }
                }

                Dlarfb.dlarfbLeftTransForwardColWise(A, aOff + (i + 1) * lda + i, lda, work, iwt, ldt,
                        A, aOff + (i + 1) * lda + i + ib, lda, work, workOff, ldwork, ihi - i, n - i - ib, ib);
            }
        }

        Dgehd2.dgehd2(n, i, ihi, A, aOff, lda, tau, tauOff, work, workOff);

        work[workOff] = lwkopt;
        return 0;
    }

    /**
     * Dormhr multiplies an m×n general matrix C by the orthogonal matrix Q produced by {@link #dgehrd}:
     *
     * <pre>
     *   Q * C    if side == Left  and trans == NoTrans
     *   Qᵀ * C   if side == Left  and trans == Trans
     *   C * Q    if side == Right and trans == NoTrans
     *   C * Qᵀ   if side == Right and trans == Trans
     * </pre>
     *
     * Q is defined as the product of (ihi - ilo) elementary reflectors returned by dgehrd:
     *   Q = H_{ilo} * H_{ilo+1} * ... * H_{ihi-1}
     *
     * <p>Corresponds to LAPACK DORMHR.
     *
     * @param side    apply Q from the Left or Right
     * @param trans   apply Q (NoTrans) or Qᵀ (Trans)
     * @param m       number of rows of C
     * @param n       number of columns of C
     * @param ilo     lower bound of the Hessenberg reduction range (0-based)
     * @param ihi     upper bound of the Hessenberg reduction range (0-based, inclusive)
     * @param a       matrix returned by dgehrd containing the elementary reflectors
     * @param aOff    offset into a
     * @param lda     leading dimension of a
     * @param tau     scalar factors of the elementary reflectors
     * @param tauOff  offset into tau
     * @param c       m×n matrix C; overwritten on return
     * @param cOff    offset into c
     * @param ldc     leading dimension of c
     * @param work    workspace array; on return work[workOff] = optimal lwork
     * @param workOff offset into work
     * @param lwork   length of work; use -1 for a workspace query
     */
    static void dormhr(BLAS.Side side, BLAS.Trans trans, int m, int n, int ilo, int ihi,
                       double[] a, int aOff, int lda, double[] tau, int tauOff,
                       double[] c, int cOff, int ldc, double[] work, int workOff, int lwork) {

        if (m == 0 || n == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return;
        }

        int nh = ihi - ilo;
        if (nh == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return;
        }

        // nq = order of Q (m when left, n when right); nw = minimum workspace.
        @SuppressWarnings("unused")
        int nq = (side == BLAS.Side.Left) ? m : n;
        int nw = (side == BLAS.Side.Left) ? n : m;

        String opts;
        if (side == BLAS.Side.Left) {
            opts = (trans == BLAS.Trans.NoTrans) ? "LN" : "LT";
        } else {
            opts = (trans == BLAS.Trans.NoTrans) ? "RN" : "RT";
        }
        int nb = min(64, Ilaenv.ilaenv(1, "DORMQR", opts, nh, nw, nh, -1));
        int lwkopt = max(1, nw) * nb;

        if (lwork == -1) {
            work[workOff] = lwkopt;
            return;
        }

        // Delegate to Dormqr on the nh×nh active submatrix of Q.
        if (side == BLAS.Side.Left) {
            Dormqr.dormqr(side, trans, nh, n, nh,
                          a, aOff + (ilo + 1) * lda + ilo, lda,
                          tau, tauOff + ilo,
                          c, cOff + (ilo + 1) * ldc, ldc,
                          work, workOff, lwork);
        } else {
            Dormqr.dormqr(side, trans, m, nh, nh,
                          a, aOff + (ilo + 1) * lda + ilo, lda,
                          tau, tauOff + ilo,
                          c, cOff + ilo + 1, ldc,
                          work, workOff, lwork);
        }
        work[workOff] = lwkopt;
    }

}
