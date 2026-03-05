/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * DORMBR overwrites the general real M-by-N matrix C with one of:
 * <pre>
 *   SIDE = 'L'  SIDE = 'R'
 *   TRANS = 'N':  Q * C    C * Q
 *   TRANS = 'T':  Q**T * C  C * Q**T
 * </pre>
 * where Q is an orthogonal matrix determined by DGEBRD.
 * If {@code vect = 'Q'}, Q is the left orthogonal factor;
 * if {@code vect = 'P'}, Q is the right orthogonal factor (P**T).
 *
 * <p>Reference: LAPACK Working Note, DORMBR.
 *
 * @see Dgebrd
 * @see Dormqr
 */
interface Dormbr {

    static int dormbr(char vect, BLAS.Side side, BLAS.Transpose trans, int m, int n, int k,
                      double[] A, int aOff, int lda,
                      double[] tau, int tauOff,
                      double[] C, int cOff, int ldc,
                      double[] work, int workOff, int lwork) {
        boolean applyQ = vect == 'Q' || vect == 'q';
        boolean left = side == BLAS.Side.Left;

        int nq = left ? m : n;
        int nw = left ? n : m;

        if (!applyQ && vect != 'P' && vect != 'p') {
            return -1;
        }
        if (m < 0) {
            return -4;
        }
        if (n < 0) {
            return -5;
        }
        if (k < 0) {
            return -6;
        }
        if (applyQ && lda < max(1, min(nq, k))) {
            return -8;
        }
        if (!applyQ && lda < max(1, nq)) {
            return -8;
        }
        if (ldc < max(1, n)) {
            return -11;
        }
        if (lwork < max(1, nw) && lwork != -1) {
            return -14;
        }

        if (m == 0 || n == 0) {
            if (lwork >= 1) {
                work[workOff] = 1;
            }
            return 0;
        }

        String opts = (left ? "L" : "R") + (trans == BLAS.Transpose.Trans ? "T" : "N");
        int nb;
        if (applyQ) {
            if (left) {
                nb = Ilaenv.ilaenv(1, "DORMQR", opts, m - 1, n, m - 1, -1);
            } else {
                nb = Ilaenv.ilaenv(1, "DORMQR", opts, m, n - 1, n - 1, -1);
            }
        } else {
            if (left) {
                nb = Ilaenv.ilaenv(1, "DORMLQ", opts, m - 1, n, m - 1, -1);
            } else {
                nb = Ilaenv.ilaenv(1, "DORMLQ", opts, m, n - 1, n - 1, -1);
            }
        }
        int lworkopt = max(1, nw) * nb;

        if (lwork == -1) {
            work[workOff] = lworkopt;
            return 0;
        }

        int minnqk = min(nq, k);

        if (applyQ) {
            if (nq >= k) {
                return dormqrBlocked(side, trans, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff, lwork);
            } else if (nq > 1) {
                int mi = m;
                int ni = n - 1;
                int i1 = 0;
                int i2 = 1;
                if (left) {
                    mi = m - 1;
                    ni = n;
                    i1 = 1;
                    i2 = 0;
                }
                int info = dormqrBlocked(side, trans, mi, ni, nq - 1, A, aOff + lda, lda, tau, tauOff, C, cOff + i1 * ldc + i2, ldc, work, workOff, lwork);
                if (info != 0) {
                    return info;
                }
            }
            work[workOff] = lworkopt;
            return 0;
        }

        BLAS.Transpose transt = (trans == BLAS.Transpose.NoTrans) ? BLAS.Transpose.Trans : BLAS.Transpose.NoTrans;

        if (nq > k) {
            return Dgelq.dormlq(side, transt, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff, lwork);
        } else if (nq > 1) {
            int mi = m;
            int ni = n - 1;
            int i1 = 0;
            int i2 = 1;
            if (left) {
                mi = m - 1;
                ni = n;
                i1 = 1;
                i2 = 0;
            }
            int info = Dgelq.dormlq(side, transt, mi, ni, nq - 1, A, aOff + 1, lda, tau, tauOff, C, cOff + i1 * ldc + i2, ldc, work, workOff, lwork);
            if (info != 0) {
                return info;
            }
        }

        work[workOff] = lworkopt;
        return 0;
    }

    static int dormqrBlocked(BLAS.Side side, BLAS.Transpose trans, int m, int n, int k,
                             double[] A, int aOff, int lda,
                             double[] tau, int tauOff,
                             double[] C, int cOff, int ldc,
                             double[] work, int workOff, int lwork) {
        boolean left = side == BLAS.Side.Left;
        boolean notrans = trans == BLAS.Transpose.NoTrans;

        int nw = left ? n : m;

        if (m < 0 || n < 0 || k < 0) {
            return -1;
        }
        if (left && k > m) {
            return -4;
        }
        if (!left && k > n) {
            return -4;
        }
        if (lda < max(1, k)) {
            return -6;
        }
        if (ldc < max(1, n)) {
            return -10;
        }
        if (lwork < max(1, nw) && lwork != -1) {
            return -13;
        }

        if (m == 0 || n == 0 || k == 0) {
            if (lwork >= 1) {
                work[workOff] = 1;
            }
            return 0;
        }

        int nbmax = 64;
        int ldt = nbmax;
        int tsize = nbmax * ldt;

        String opts = String.valueOf(side.code) + trans.code;
        int nb = min(nbmax, Ilaenv.ilaenv(1, "DORMQR", opts, m, n, k, -1));
        int lworkopt = max(1, nw) * nb + tsize;

        if (lwork == -1) {
            work[workOff] = lworkopt;
            return 0;
        }

        int nbmin = 2;
        if (1 < nb && nb < k) {
            int iws = nw * nb + tsize;
            if (lwork < iws) {
                nb = (lwork - tsize) / nw;
                nbmin = max(2, Ilaenv.ilaenv(2, "DORMQR", opts, m, n, k, -1));
            }
        }

        if (nb < nbmin || k <= nb) {
            dormqrUnblocked(side, trans, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff);
            work[workOff] = lworkopt;
            return 0;
        }

        int ldwork = nb;
        int tOff = workOff;
        int wrkOff = workOff + tsize;

        if (left && notrans) {
            for (int i = 0; i < k; i += nb) {
                int ib = min(nb, k - i);
                Dlarft.dlarft(A, aOff + i * lda + i, lda, tau, tauOff + i, work, tOff, ldt, m - i, ib);
                Dlarfb.dlarfbLeft(A, aOff + i * lda + i, lda, work, tOff, ldt, C, cOff + i * ldc, ldc, work, wrkOff, ldwork, m - i, n, ib);
            }
        } else if (left) {
            for (int i = ((k - 1) / nb) * nb; i >= 0; i -= nb) {
                int ib = min(nb, k - i);
                Dlarft.dlarft(A, aOff + i * lda + i, lda, tau, tauOff + i, work, tOff, ldt, m - i, ib);
                Dlarfb.dlarfbLeftTrans(A, aOff + i * lda + i, lda, work, tOff, ldt, C, cOff + i * ldc, ldc, work, wrkOff, ldwork, m - i, n, ib);
            }
        } else if (notrans) {
            for (int i = 0; i < k; i += nb) {
                int ib = min(nb, k - i);
                Dlarft.dlarft(A, aOff + i * lda + i, lda, tau, tauOff + i, work, tOff, ldt, m, ib);
                dlarfbRight(A, aOff + i * lda + i, lda, work, tOff, ldt, C, cOff + i, ldc, work, wrkOff, ldwork, m, n - i, ib);
            }
        } else {
            for (int i = ((k - 1) / nb) * nb; i >= 0; i -= nb) {
                int ib = min(nb, k - i);
                Dlarft.dlarft(A, aOff + i * lda + i, lda, tau, tauOff + i, work, tOff, ldt, m, ib);
                dlarfbRightTrans(A, aOff + i * lda + i, lda, work, tOff, ldt, C, cOff + i, ldc, work, wrkOff, ldwork, m, n - i, ib);
            }
        }

        work[workOff] = lworkopt;
        return 0;
    }

    static void dormqrUnblocked(BLAS.Side side, BLAS.Transpose trans, int m, int n, int k,
                                double[] A, int aOff, int lda,
                                double[] tau, int tauOff,
                                double[] C, int cOff, int ldc,
                                double[] work, int workOff) {
        boolean left = side == BLAS.Side.Left;
        boolean notrans = trans == BLAS.Transpose.NoTrans;

        if (m == 0 || n == 0 || k == 0) {
            return;
        }

        if (left && notrans) {
            for (int i = 0; i < k; i++) {
                int aiiOff = aOff + i * lda + i;
                double aii = A[aiiOff];
                A[aiiOff] = 1;
                Dlarf.dlarf(side, m - i, n, A, aiiOff, lda, tau[tauOff + i], C, cOff + i * ldc, ldc, work, workOff);
                A[aiiOff] = aii;
            }
        } else if (left) {
            for (int i = k - 1; i >= 0; i--) {
                int aiiOff = aOff + i * lda + i;
                double aii = A[aiiOff];
                A[aiiOff] = 1;
                Dlarf.dlarf(side, m - i, n, A, aiiOff, lda, tau[tauOff + i], C, cOff + i * ldc, ldc, work, workOff);
                A[aiiOff] = aii;
            }
        } else if (notrans) {
            for (int i = 0; i < k; i++) {
                int aiiOff = aOff + i * lda + i;
                double aii = A[aiiOff];
                A[aiiOff] = 1;
                Dlarf.dlarf(side, m, n - i, A, aiiOff, lda, tau[tauOff + i], C, cOff + i, ldc, work, workOff);
                A[aiiOff] = aii;
            }
        } else {
            for (int i = k - 1; i >= 0; i--) {
                int aiiOff = aOff + i * lda + i;
                double aii = A[aiiOff];
                A[aiiOff] = 1;
                Dlarf.dlarf(side, m, n - i, A, aiiOff, lda, tau[tauOff + i], C, cOff + i, ldc, work, workOff);
                A[aiiOff] = aii;
            }
        }
    }

    static void dlarfbRight(double[] V, int vOff, int ldv,
                            double[] T, int tOff, int ldt,
                            double[] C, int cOff, int ldc,
                            double[] work, int wOff, int ldwork, int m, int nC, int k) {
        if (m <= 0 || nC <= 0 || k <= 0) return;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                work[wOff + i * ldwork + j] = C[cOff + i * ldc + j];
            }
        }

        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        if (nC > k) {
            BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m, k, nC - k,
                    1.0, C, cOff + k, ldc,
                    V, vOff + k * ldv, ldv,
                    1.0, work, wOff, ldwork);
        }

        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit, m, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        if (nC > k) {
            BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.Trans, m, nC - k, k,
                    -1.0, work, wOff, ldwork,
                    V, vOff + k * ldv, ldv,
                    1.0, C, cOff + k, ldc);
        }

        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Transpose.Trans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + i * ldc + j] -= work[wOff + i * ldwork + j];
            }
        }
    }

    static void dlarfbRightTrans(double[] V, int vOff, int ldv,
                                 double[] T, int tOff, int ldt,
                                 double[] C, int cOff, int ldc,
                                 double[] work, int wOff, int ldwork, int m, int nC, int k) {
        if (m <= 0 || nC <= 0 || k <= 0) return;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                work[wOff + i * ldwork + j] = C[cOff + i * ldc + j];
            }
        }

        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        if (nC > k) {
            BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m, k, nC - k,
                    1.0, C, cOff + k, ldc,
                    V, vOff + k * ldv, ldv,
                    1.0, work, wOff, ldwork);
        }

        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Transpose.Trans, BLAS.Diag.NonUnit, m, k, 1.0, T, tOff, ldt, work, wOff, ldwork);

        if (nC > k) {
            BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.Trans, m, nC - k, k,
                    -1.0, work, wOff, ldwork,
                    V, vOff + k * ldv, ldv,
                    1.0, C, cOff + k, ldc);
        }

        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Transpose.Trans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, ldwork);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + i * ldc + j] -= work[wOff + i * ldwork + j];
            }
        }
    }
}
