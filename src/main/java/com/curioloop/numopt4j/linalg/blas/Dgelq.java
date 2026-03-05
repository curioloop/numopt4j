/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * Computes the LQ factorization of a general m×n matrix A.
 *
 * <p>Computes A = L * Q where L is lower trapezoidal and Q is orthogonal.
 * Also provides blocked (dgelqf) and unblocked (dgelq2) variants, as well as
 * routines for generating (dorglq/dorgl2) and applying (dormlq/dorml2) the
 * orthogonal matrix Q.
 *
 * @see #dgelqf(int, int, double[], int, int, double[], int, double[], int, int)
 * @see #dgelq2(int, int, double[], int, int, double[], int, double[], int)
 * @see #dorglq(int, int, int, double[], int, int, double[], int, double[], int, int)
 * @see #dormlq(BLAS.Side, BLAS.Transpose, int, int, int, double[], int, int, double[], int, double[], int, int, double[], int, int)
 */
interface Dgelq {

    static void dgelq2(int m, int n, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        if (m < 0 || n < 0 || lda < max(1, n)) {
            return;
        }

        int k = min(m, n);
        if (k == 0) {
            return;
        }

        for (int i = 0; i < k; i++) {
            int rowOff = aOff + i * lda;
            double aii = BLAS.dlarfg(n - i, A[rowOff + i], A, rowOff + min(i + 1, n - 1), 1, tau, tauOff + i);
            A[rowOff + i] = aii;
            
            if (i < m - 1) {
                double savedAii = A[rowOff + i];
                A[rowOff + i] = 1;
                BLAS.dlarf(BLAS.Side.Right, m - i - 1, n - i,
                           A, rowOff + i, 1,
                           tau[tauOff + i],
                           A, aOff + (i + 1) * lda + i, lda,
                           work, workOff);
                A[rowOff + i] = savedAii;
            }
        }
    }

    static int dgelqf(int m, int n, double[] A, int aOff, int lda,
                      double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        if (m < 0 || n < 0) {
            return -1;
        }

        int k = min(m, n);
        if (k == 0) {
            work[workOff] = 1;
            return 0;
        }

        if (lda < max(1, n)) {
            return -1;
        }

        int nb = Ilaenv.ilaenv(1, "DGELQF", " ", m, n, -1, -1);
        if (lwork == -1) {
            work[workOff] = m * nb;
            return 0;
        }

        int nbmin = 2;
        int nx = 0;
        int iws = m;
        
        if (1 < nb && nb < k) {
            nx = max(0, Ilaenv.ilaenv(3, "DGELQF", " ", m, n, -1, -1));
            if (nx < k) {
                iws = m * nb;
                if (lwork < iws) {
                    nb = lwork / m;
                    nbmin = max(2, Ilaenv.ilaenv(2, "DGELQF", " ", m, n, -1, -1));
                }
            }
        }

        int ldwork = nb;
        int i = 0;
        
        if (nbmin <= nb && nb < k && nx < k) {
            for (i = 0; i < k - nx; i += nb) {
                int ib = min(k - i, nb);
                dgelq2(ib, n - i, A, aOff + i * lda + i, lda, tau, tauOff + i, work, workOff);
                
                if (i + ib < m) {
                    Dlarft.dlarftForwardRowWise(A, aOff + i * lda + i, lda, tau, tauOff + i, work, workOff, ldwork, n - i, ib);
                    Dlarfb.dlarfbRightForwardRowWise(A, aOff + i * lda + i, lda,
                            work, workOff, ldwork,
                            A, aOff + (i + ib) * lda + i, lda,
                            work, workOff + ib * ldwork, ldwork, m - i - ib, n - i, ib);
                }
            }
        }

        if (i < k) {
            dgelq2(m - i, n - i, A, aOff + i * lda + i, lda, tau, tauOff + i, work, workOff);
        }

        work[workOff] = iws;
        return 0;
    }

    static void dorgl2(int m, int n, int k, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        if (m < 0 || n < m || k < 0 || k > m || lda < max(1, n)) {
            return;
        }

        if (m == 0) {
            return;
        }

        if (k < m) {
            for (int i = k; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    A[aOff + i * lda + j] = 0;
                }
            }
            for (int j = k; j < m; j++) {
                A[aOff + j * lda + j] = 1;
            }
        }

        for (int i = k - 1; i >= 0; i--) {
            int rowOff = aOff + i * lda;
            if (i < n - 1) {
                if (i < m - 1) {
                    A[rowOff + i] = 1;
                    Dlarf.dlarf(BLAS.Side.Right, m - i - 1, n - i,
                               A, rowOff + i, 1,
                               tau[tauOff + i],
                               A, aOff + (i + 1) * lda + i, lda,
                               work, workOff);
                }
                Dscal.dscal(n - i - 1, -tau[tauOff + i], A, rowOff + i + 1, 1);
            }
            A[rowOff + i] = 1 - tau[tauOff + i];
            for (int l = 0; l < i; l++) {
                A[rowOff + l] = 0;
            }
        }
    }

    static int dorglq(int m, int n, int k, double[] A, int aOff, int lda,
                      double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        if (m < 0 || n < m || k < 0 || k > m || lda < max(1, n)) {
            return -1;
        }

        if (m == 0) {
            work[workOff] = 1;
            return 0;
        }

        int nb = Ilaenv.ilaenv(1, "DORGLQ", " ", m, n, k, -1);
        if (lwork == -1) {
            work[workOff] = m * nb;
            return 0;
        }

        int nbmin = 2;
        int nx = 0;
        int iws = m;
        int ldwork = 0;
        
        if (1 < nb && nb < k) {
            nx = max(0, Ilaenv.ilaenv(3, "DORGLQ", " ", m, n, k, -1));
            if (nx < k) {
                ldwork = nb;
                iws = m * ldwork;
                if (lwork < iws) {
                    nb = lwork / m;
                    ldwork = nb;
                    nbmin = max(2, Ilaenv.ilaenv(2, "DORGLQ", " ", m, n, k, -1));
                }
            }
        }

        int ki = 0;
        int kk = 0;
        
        if (nbmin <= nb && nb < k && nx < k) {
            ki = ((k - nx - 1) / nb) * nb;
            kk = min(k, ki + nb);
            for (int i = kk; i < m; i++) {
                for (int j = 0; j < kk; j++) {
                    A[aOff + i * lda + j] = 0;
                }
            }
        }

        if (kk < m) {
            dorgl2(m - kk, n - kk, k - kk, A, aOff + kk * lda + kk, lda, tau, tauOff + kk, work, workOff);
        }

        if (kk > 0) {
            for (int i = ki; i >= 0; i -= nb) {
                int ib = min(nb, k - i);
                if (i + ib < m) {
                    Dlarft.dlarftForwardRowWise(A, aOff + i * lda + i, lda, tau, tauOff + i, work, workOff, ldwork, n - i, ib);
                    Dlarfb.dlarfbRightTransForwardRowWise(A, aOff + i * lda + i, lda,
                            work, workOff, ldwork,
                            A, aOff + (i + ib) * lda + i, lda,
                            work, workOff + ib * ldwork, ldwork, m - i - ib, n - i, ib);
                }
                dorgl2(ib, n - i, ib, A, aOff + i * lda + i, lda, tau, tauOff + i, work, workOff);
                for (int l = i; l < i + ib; l++) {
                    for (int j = 0; j < i; j++) {
                        A[aOff + l * lda + j] = 0;
                    }
                }
            }
        }

        work[workOff] = iws;
        return 0;
    }

    static int dormlq(BLAS.Side side, BLAS.Transpose trans, int m, int n, int k,
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
        if (lda < max(1, left ? m : n)) {
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
        int nb = min(nbmax, Ilaenv.ilaenv(1, "DORMLQ", opts, m, n, k, -1));
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
                nbmin = max(2, Ilaenv.ilaenv(2, "DORMLQ", opts, m, n, k, -1));
            }
        }

        if (nb < nbmin || k <= nb) {
            dorml2(side, trans, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff);
            work[workOff] = lworkopt;
            return 0;
        }

        int ldwork = nb;
        int tOff = workOff;
        int wrkOff = workOff + tsize;

        BLAS.Transpose transt = notrans ? BLAS.Transpose.Trans : BLAS.Transpose.NoTrans;

        if (left && notrans) {
            for (int i = 0; i < k; i += nb) {
                int ib = min(nb, k - i);
                dlarftRowWise(m - i, ib, A, aOff + i * lda + i, lda, tau, tauOff + i, work, tOff, ldt);
                dlarfbRowWiseLeft(side, transt, m - i, n, ib,
                        A, aOff + i * lda + i, lda,
                        work, tOff, ldt,
                        C, cOff + i * ldc, ldc,
                        work, wrkOff, ldwork);
            }
        } else if (left) {
            for (int i = ((k - 1) / nb) * nb; i >= 0; i -= nb) {
                int ib = min(nb, k - i);
                dlarftRowWise(m - i, ib, A, aOff + i * lda + i, lda, tau, tauOff + i, work, tOff, ldt);
                dlarfbRowWiseLeft(side, transt, m - i, n, ib,
                        A, aOff + i * lda + i, lda,
                        work, tOff, ldt,
                        C, cOff + i * ldc, ldc,
                        work, wrkOff, ldwork);
            }
        } else if (notrans) {
            for (int i = ((k - 1) / nb) * nb; i >= 0; i -= nb) {
                int ib = min(nb, k - i);
                dlarftRowWise(n - i, ib, A, aOff + i * lda + i, lda, tau, tauOff + i, work, tOff, ldt);
                dlarfbRowWiseRight(side, transt, m, n - i, ib,
                        A, aOff + i * lda + i, lda,
                        work, tOff, ldt,
                        C, cOff + i, ldc,
                        work, wrkOff, ldwork);
            }
        } else {
            for (int i = 0; i < k; i += nb) {
                int ib = min(nb, k - i);
                dlarftRowWise(n - i, ib, A, aOff + i * lda + i, lda, tau, tauOff + i, work, tOff, ldt);
                dlarfbRowWiseRight(side, transt, m, n - i, ib,
                        A, aOff + i * lda + i, lda,
                        work, tOff, ldt,
                        C, cOff + i, ldc,
                        work, wrkOff, ldwork);
            }
        }

        work[workOff] = lworkopt;
        return 0;
    }

    static void dorml2(BLAS.Side side, BLAS.Transpose trans, int m, int n, int k,
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
                Dlarf.dlarf(side, m - i, n, A, aiiOff, 1, tau[tauOff + i], C, cOff + i * ldc, ldc, work, workOff);
                A[aiiOff] = aii;
            }
        } else if (left) {
            for (int i = k - 1; i >= 0; i--) {
                int aiiOff = aOff + i * lda + i;
                double aii = A[aiiOff];
                A[aiiOff] = 1;
                Dlarf.dlarf(side, m - i, n, A, aiiOff, 1, tau[tauOff + i], C, cOff + i * ldc, ldc, work, workOff);
                A[aiiOff] = aii;
            }
        } else if (notrans) {
            for (int i = k - 1; i >= 0; i--) {
                int aiiOff = aOff + i * lda + i;
                double aii = A[aiiOff];
                A[aiiOff] = 1;
                Dlarf.dlarf(side, m, n - i, A, aiiOff, 1, tau[tauOff + i], C, cOff + i, ldc, work, workOff);
                A[aiiOff] = aii;
            }
        } else {
            for (int i = 0; i < k; i++) {
                int aiiOff = aOff + i * lda + i;
                double aii = A[aiiOff];
                A[aiiOff] = 1;
                Dlarf.dlarf(side, m, n - i, A, aiiOff, 1, tau[tauOff + i], C, cOff + i, ldc, work, workOff);
                A[aiiOff] = aii;
            }
        }
    }

    static void dlarftRowWise(int n, int k, double[] V, int vOff, int ldv,
                              double[] tau, int tauOff, double[] T, int tOff, int ldt) {
        if (k <= 0) return;

        int prevlastv = n - 1;
        for (int i = 0; i < k; i++) {
            prevlastv = max(i, prevlastv);
            if (tau[tauOff + i] == 0.0) {
                for (int j = 0; j <= i; j++) {
                    T[tOff + j * ldt + i] = 0.0;
                }
                continue;
            }

            int lastv;
            for (lastv = n - 1; lastv >= i + 1; lastv--) {
                if (V[vOff + i * ldv + lastv] != 0.0) {
                    break;
                }
            }

            for (int j = 0; j < i; j++) {
                T[tOff + j * ldt + i] = -tau[tauOff + i] * V[vOff + j * ldv + i];
            }

            int j = min(lastv, prevlastv);
            if (j > i) {
                for (int col = 0; col < i; col++) {
                    double sum = T[tOff + col * ldt + i];
                    for (int row = i + 1; row <= j; row++) {
                        sum += -tau[tauOff + i] * V[vOff + col * ldv + row] * V[vOff + i * ldv + row];
                    }
                    T[tOff + col * ldt + i] = sum;
                }
            }

            if (i > 0) {
                for (int col = 0; col < i; col++) {
                    double sum = 0;
                    for (int p = col; p < i; p++) {
                        sum += T[tOff + col * ldt + p] * T[tOff + p * ldt + i];
                    }
                    T[tOff + col * ldt + i] = sum;
                }
            }

            T[tOff + i * ldt + i] = tau[tauOff + i];
            if (i > 1) {
                prevlastv = max(prevlastv, lastv);
            } else {
                prevlastv = lastv;
            }
        }
    }

    static void dlarfbRowWiseLeft(BLAS.Side side, BLAS.Transpose trans, int m, int n, int k,
                                  double[] V, int vOff, int ldv,
                                  double[] T, int tOff, int ldt,
                                  double[] C, int cOff, int ldc,
                                  double[] work, int wOff, int ldwork) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        BLAS.Transpose transEnum = (trans == BLAS.Transpose.NoTrans) ? BLAS.Transpose.Trans : BLAS.Transpose.NoTrans;

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < n; i++) {
                work[wOff + j * n + i] = C[cOff + j * ldc + i];
            }
        }

        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Transpose.Trans, BLAS.Diag.Unit, n, k, 1.0, V, vOff, ldv, work, wOff, n);

        if (m > k) {
            BLAS.dgemm(BLAS.Transpose.Trans, BLAS.Transpose.Trans, n, k, m - k,
                    1.0, C, cOff + k * ldc, ldc,
                    V, vOff + k, ldv,
                    1.0, work, wOff, n);
        }

        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, transEnum, BLAS.Diag.NonUnit, n, k, 1.0, T, tOff, ldt, work, wOff, n);

        if (m > k) {
            BLAS.dgemm(BLAS.Transpose.Trans, BLAS.Transpose.Trans, m - k, n, k,
                    -1.0, V, vOff + k, ldv,
                    work, wOff, n,
                    1.0, C, cOff + k * ldc, ldc);
        }

        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.Unit, n, k, 1.0, V, vOff, ldv, work, wOff, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + j * ldc + i] -= work[wOff + j * n + i];
            }
        }
    }

    static void dlarfbRowWiseRight(BLAS.Side side, BLAS.Transpose trans, int m, int n, int k,
                                   double[] V, int vOff, int ldv,
                                   double[] T, int tOff, int ldt,
                                   double[] C, int cOff, int ldc,
                                   double[] work, int wOff, int ldwork) {
        if (m <= 0 || n <= 0 || k <= 0) return;

        BLAS.Transpose transEnum = (trans == BLAS.Transpose.NoTrans) ? BLAS.Transpose.Trans : BLAS.Transpose.NoTrans;

        for (int j = 0; j < k; j++) {
            for (int i = 0; i < m; i++) {
                work[wOff + j * m + i] = C[cOff + i * ldc + j];
            }
        }

        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Transpose.Trans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, m);

        if (n > k) {
            BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.Trans, m, k, n - k,
                    1.0, C, cOff + k, ldc,
                    V, vOff + k, ldv,
                    1.0, work, wOff, m);
        }

        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, transEnum, BLAS.Diag.NonUnit, m, k, 1.0, T, tOff, ldt, work, wOff, m);

        if (n > k) {
            BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m, n - k, k,
                    -1.0, work, wOff, m,
                    V, vOff + k, ldv,
                    1.0, C, cOff + k, ldc);
        }

        BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.Unit, m, k, 1.0, V, vOff, ldv, work, wOff, m);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                C[cOff + i * ldc + j] -= work[wOff + j * m + i];
            }
        }
    }
}
