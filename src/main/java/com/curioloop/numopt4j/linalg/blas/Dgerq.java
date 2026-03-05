/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * Computes the RQ factorization of a general m×n matrix A.
 *
 * <p>Computes A = R * Q where R is upper trapezoidal and Q is orthogonal.
 * Also provides blocked (dgerqf) and unblocked (dgerq2) variants, as well as
 * routines for generating (dorgrq/dorgr2) and applying (dormrq/dormr2) the
 * orthogonal matrix Q.
 *
 * @see #dgerqf(int, int, double[], int, int, double[], int, double[], int, int)
 * @see #dgerq2(int, int, double[], int, int, double[], int, double[], int)
 * @see #dorgrq(int, int, int, double[], int, int, double[], int, double[], int, int)
 * @see #dormrq(BLAS.Side, BLAS.Transpose, int, int, int, double[], int, int, double[], int, double[], int, int, double[], int, int)
 */
interface Dgerq {

    static void dgerq2(int m, int n, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        if (m < 0 || n < 0 || lda < max(1, n)) return;

        int k = min(m, n);
        if (k == 0) return;

        for (int i = k - 1; i >= 0; i--) {
            int mki = m - k + i;
            int nki = n - k + i;
            int aiiOff = aOff + mki * lda + nki;

            double aii = Dlarfg.dlarfg(nki + 1, A[aiiOff], A, aOff + mki * lda, 1, tau, tauOff + i);

            A[aiiOff] = 1;
            Dlarf.dlarf(BLAS.Side.Right, mki, nki + 1, A, aOff + mki * lda, 1, tau[tauOff + i], A, aOff, lda, work, workOff);
            A[aiiOff] = aii;
        }
    }

    static void dormr2(BLAS.Side side, BLAS.Transpose trans, int m, int n, int k,
                       double[] A, int aOff, int lda, double[] tau, int tauOff,
                       double[] C, int cOff, int ldc, double[] work, int workOff) {
        if (m < 0 || n < 0 || k < 0 || k > min(m, n)) return;

        boolean left = (side == BLAS.Side.Left);
        boolean notran = (trans == BLAS.Transpose.NoTrans);

        if (left) {
            if (notran) {
                for (int i = k - 1; i >= 0; i--) {
                    int aiiOff = aOff + i * lda + (m - k + i);
                    double aii = A[aiiOff];
                    A[aiiOff] = 1;
                    Dlarf.dlarf(BLAS.Side.Left, m - k + i + 1, n, A, aOff + i * lda, 1, tau[tauOff + i], C, cOff, ldc, work, workOff);
                    A[aiiOff] = aii;
                }
            } else {
                for (int i = 0; i < k; i++) {
                    int aiiOff = aOff + i * lda + (m - k + i);
                    double aii = A[aiiOff];
                    A[aiiOff] = 1;
                    Dlarf.dlarf(BLAS.Side.Left, m - k + i + 1, n, A, aOff + i * lda, 1, tau[tauOff + i], C, cOff, ldc, work, workOff);
                    A[aiiOff] = aii;
                }
            }
        } else {
            if (notran) {
                for (int i = 0; i < k; i++) {
                    int aiiOff = aOff + i * lda + (n - k + i);
                    double aii = A[aiiOff];
                    A[aiiOff] = 1;
                    Dlarf.dlarf(BLAS.Side.Right, m, n - k + i + 1, A, aOff + i * lda, 1, tau[tauOff + i], C, cOff, ldc, work, workOff);
                    A[aiiOff] = aii;
                }
            } else {
                for (int i = k - 1; i >= 0; i--) {
                    int aiiOff = aOff + i * lda + (n - k + i);
                    double aii = A[aiiOff];
                    A[aiiOff] = 1;
                    Dlarf.dlarf(BLAS.Side.Right, m, n - k + i + 1, A, aOff + i * lda, 1, tau[tauOff + i], C, cOff, ldc, work, workOff);
                    A[aiiOff] = aii;
                }
            }
        }
    }

    static void dorgr2(int m, int n, int k, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        if (m < 0 || n < 0 || k < 0 || k > m || n < m || lda < max(1, n)) return;

        if (m == 0) return;

        for (int l = 0; l < m - k; l++) {
            for (int j = 0; j < n; j++) {
                A[aOff + l * lda + j] = 0;
            }
            A[aOff + l * lda + (n - m + l)] = 1;
        }

        for (int i = 0; i < k; i++) {
            int ii = m - k + i;
            int nmi = n - m + ii;

            A[aOff + ii * lda + nmi] = 1;
            Dlarf.dlarf(BLAS.Side.Right, ii, nmi + 1, A, aOff + ii * lda, 1, tau[tauOff + i], A, aOff, lda, work, workOff);
            Dscal.dscal(nmi, -tau[tauOff + i], A, aOff + ii * lda, 1);
            A[aOff + ii * lda + nmi] = 1 - tau[tauOff + i];

            for (int l = nmi + 1; l < n; l++) {
                A[aOff + ii * lda + l] = 0;
            }
        }
    }

    static int dgerqf(int m, int n, double[] A, int aOff, int lda,
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

        int nb = Ilaenv.ilaenv(1, "DGERQF", " ", m, n, -1, -1);
        if (lwork == -1) {
            work[workOff] = m * nb;
            return 0;
        }

        int nbmin = 2;
        int nx = 1;
        int iws = m;
        int ldwork = 0;

        if (1 < nb && nb < k) {
            nx = max(0, Ilaenv.ilaenv(3, "DGERQF", " ", m, n, -1, -1));
            if (nx < k) {
                iws = m * nb;
                if (lwork < iws) {
                    nb = lwork / m;
                    nbmin = max(2, Ilaenv.ilaenv(2, "DGERQF", " ", m, n, -1, -1));
                }
                ldwork = nb;
            }
        }

        int mu = m;
        int nu = n;

        if (nbmin <= nb && nb < k && nx < k) {
            int ki = ((k - nx - 1) / nb) * nb;
            int kk = min(k, ki + nb);

            int i;
            for (i = k - kk + ki; i >= k - kk; i -= nb) {
                int ib = min(k - i, nb);

                dgerq2(ib, n - k + i + ib, A, aOff + (m - k + i) * lda, lda, tau, tauOff + i, work, workOff);

                if (m - k + i > 0) {
                    Dlarft.dlarftBackwardRowWise(A, aOff + (m - k + i) * lda, lda, tau, tauOff + i, work, workOff, ldwork, n - k + i + ib, ib);
                    Dlarfb.dlarfbRightBackwardRowWise(A, aOff + (m - k + i) * lda, lda,
                            work, workOff, ldwork,
                            A, aOff, lda,
                            work, workOff + ib * ldwork, ldwork, m - k + i, n - k + i + ib, ib);
                }
            }
            mu = m - k + i + nb;
            nu = n - k + i + nb;
        }

        if (mu > 0 && nu > 0) {
            dgerq2(mu, nu, A, aOff, lda, tau, tauOff, work, workOff);
        }

        work[workOff] = iws;
        return 0;
    }

    static int dorgrq(int m, int n, int k, double[] A, int aOff, int lda,
                      double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        if (m < 0 || n < m || k < 0 || k > m || lda < max(1, n)) {
            return -1;
        }

        if (m == 0) {
            work[workOff] = 1;
            return 0;
        }

        int nb = Ilaenv.ilaenv(1, "DORGRQ", " ", m, n, k, -1);
        if (lwork == -1) {
            work[workOff] = m * nb;
            return 0;
        }

        int nbmin = 2;
        int nx = 0;
        int iws = m;
        int ldwork = 0;

        if (1 < nb && nb < k) {
            nx = max(0, Ilaenv.ilaenv(3, "DORGRQ", " ", m, n, k, -1));
            if (nx < k) {
                ldwork = nb;
                iws = m * ldwork;
                if (lwork < iws) {
                    nb = lwork / m;
                    ldwork = nb;
                    nbmin = max(2, Ilaenv.ilaenv(2, "DORGRQ", " ", m, n, k, -1));
                }
            }
        }

        int ki = 0;
        int kk = 0;

        if (nbmin <= nb && nb < k && nx < k) {
            ki = ((k - nx - 1) / nb) * nb;
            kk = min(k, ki + nb);

            for (int l = 0; l < m - kk; l++) {
                for (int j = 0; j < n - kk; j++) {
                    A[aOff + l * lda + j] = 0;
                }
                A[aOff + l * lda + (n - kk + l)] = 1;
            }
        }

        if (kk < m) {
            dorgr2WholeMatrix(m, n, k - kk, kk, A, aOff, lda, tau, tauOff + kk, work, workOff);
        }

        if (kk > 0) {
            for (int i = ki; i >= 0; i -= nb) {
                int ib = min(nb, k - i);
                int nki = n - k + i;

                if (i + ib < m) {
                    Dlarft.dlarftBackwardRowWise(A, aOff + i * lda, lda, tau, tauOff + i, work, workOff, ldwork, nki + ib, ib);
                    Dlarfb.dlarfbRightTransBackwardRowWise(A, aOff + i * lda, lda,
                            work, workOff, ldwork,
                            A, aOff + (i + ib) * lda, lda,
                            work, workOff + ib * ldwork, ldwork, m - i - ib, nki + ib, ib);
                }

                dorgr2WholeMatrix(m, n, ib, i, A, aOff, lda, tau, tauOff + i, work, workOff);

                for (int l = i; l < i + ib; l++) {
                    for (int j = 0; j < nki; j++) {
                        A[aOff + l * lda + j] = 0;
                    }
                }
            }
        }

        work[workOff] = iws;
        return 0;
    }

    static void dorgr2WholeMatrix(int m, int n, int k, int rowOffset,
                                  double[] A, int aOff, int lda,
                                  double[] tau, int tauOff,
                                  double[] work, int workOff) {
        if (k == 0) return;

        for (int i = 0; i < k; i++) {
            int ii = m - k + i;
            int nmi = n - m + ii;

            A[aOff + (rowOffset + i) * lda + nmi] = 1;
            if (rowOffset + i > 0) {
                Dlarf.dlarf(BLAS.Side.Right, rowOffset + i, nmi + 1, A, aOff + (rowOffset + i) * lda, 1, tau[tauOff + i], A, aOff, lda, work, workOff);
            }
            Dscal.dscal(nmi, -tau[tauOff + i], A, aOff + (rowOffset + i) * lda, 1);
            A[aOff + (rowOffset + i) * lda + nmi] = 1 - tau[tauOff + i];

            for (int l = nmi + 1; l < n; l++) {
                A[aOff + (rowOffset + i) * lda + l] = 0;
            }
        }
    }

    static int dormrq(BLAS.Side side, BLAS.Transpose trans, int m, int n, int k,
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
        int nb = min(nbmax, Ilaenv.ilaenv(1, "DORMRQ", opts, m, n, k, -1));
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
                nbmin = max(2, Ilaenv.ilaenv(2, "DORMRQ", opts, m, n, k, -1));
            }
        }

        if (nb < nbmin || k <= nb) {
            dormr2(side, trans, m, n, k, A, aOff, lda, tau, tauOff, C, cOff, ldc, work, workOff);
            work[workOff] = lworkopt;
            return 0;
        }

        int ldwork = nb;
        int tOff = workOff;
        int wrkOff = workOff + tsize;

        if (left && notrans) {
            for (int i = ((k - 1) / nb) * nb; i >= 0; i -= nb) {
                int ib = min(nb, k - i);
                int mi = m - k + i + ib;
                int ic = m - k + i;
                Dlarft.dlarftBackwardRowWise(A, aOff + i * lda, lda, tau, tauOff + i, work, tOff, ldt, mi, ib);
                Dlarfb.dlarfbLeftTransBackwardRowWise(A, aOff + i * lda, lda,
                        work, tOff, ldt,
                        C, cOff, ldc,
                        work, wrkOff, ldwork, mi, n, ib);
            }
        } else if (left) {
            for (int i = 0; i < k; i += nb) {
                int ib = min(nb, k - i);
                int mi = m - k + i + ib;
                Dlarft.dlarftBackwardRowWise(A, aOff + i * lda, lda, tau, tauOff + i, work, tOff, ldt, mi, ib);
                Dlarfb.dlarfbLeftBackwardRowWise(A, aOff + i * lda, lda,
                        work, tOff, ldt,
                        C, cOff, ldc,
                        work, wrkOff, ldwork, mi, n, ib);
            }
        } else if (notrans) {
            for (int i = 0; i < k; i += nb) {
                int ib = min(nb, k - i);
                int ni = n - k + i + ib;
                int ic = n - k + i;
                Dlarft.dlarftBackwardRowWise(A, aOff + i * lda, lda, tau, tauOff + i, work, tOff, ldt, ni, ib);
                Dlarfb.dlarfbRightTransBackwardRowWise(A, aOff + i * lda, lda,
                        work, tOff, ldt,
                        C, cOff, ldc,
                        work, wrkOff, ldwork, m, ni, ib);
            }
        } else {
            for (int i = ((k - 1) / nb) * nb; i >= 0; i -= nb) {
                int ib = min(nb, k - i);
                int ni = n - k + i + ib;
                Dlarft.dlarftBackwardRowWise(A, aOff + i * lda, lda, tau, tauOff + i, work, tOff, ldt, ni, ib);
                Dlarfb.dlarfbRightBackwardRowWise(A, aOff + i * lda, lda,
                        work, tOff, ldt,
                        C, cOff, ldc,
                        work, wrkOff, ldwork, m, ni, ib);
            }
        }

        work[workOff] = lworkopt;
        return 0;
    }
}
