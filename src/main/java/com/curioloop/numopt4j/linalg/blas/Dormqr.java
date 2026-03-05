/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

interface Dormqr {

    /**
     * DORMQR overwrites the general real M-by-N matrix C with one of:
     * <pre>
     *   SIDE = 'L'  SIDE = 'R'
     *   TRANS = 'N':  Q * C    C * Q
     *   TRANS = 'T':  Q**T * C  C * Q**T
     * </pre>
     * where Q is a real orthogonal matrix defined as the product of k
     * elementary reflectors as returned by DGEQRF.
     *
     * <p>Uses a blocked algorithm (DLARFT + DLARFB) when the block size
     * is large enough, falling back to the unblocked algorithm (DORM2R)
     * for small problems.
     *
     * <p>Reference: LAPACK Working Note, DORMQR.
     *
     * @see Dgeqr
     * @see Dlarft
     * @see Dlarfb
     */
    static void dorm2r(BLAS.Side side, BLAS.Transpose trans, int m, int n, int k,
                       double[] A, int aOff, int lda, double[] tau, int tauOff,
                       double[] C, int cOff, int ldc, double[] work, int workOff) {
        if (m < 0 || n < 0 || k < 0) return;

        boolean left = (side == BLAS.Side.Left);
        if (left && k > m) return;
        if (!left && k > n) return;

        boolean notran = (trans == BLAS.Transpose.NoTrans);

        if (left) {
            if (notran) {
                for (int i = k - 1; i >= 0; i--) {
                    int aii = aOff + i * lda + i;
                    double t = tau[tauOff + i];
                    if (t != 0) {
                        double savedAii = A[aii];
                        A[aii] = 1;
                        BLAS.dlarf(BLAS.Side.Left, m - i, n, A, aii, lda, t, C, cOff + i * ldc, ldc, work, workOff);
                        A[aii] = savedAii;
                    }
                }
            } else {
                for (int i = 0; i < k; i++) {
                    int aii = aOff + i * lda + i;
                    double t = tau[tauOff + i];
                    if (t != 0) {
                        double savedAii = A[aii];
                        A[aii] = 1;
                        BLAS.dlarf(BLAS.Side.Left, m - i, n, A, aii, lda, t, C, cOff + i * ldc, ldc, work, workOff);
                        A[aii] = savedAii;
                    }
                }
            }
        } else {
            if (notran) {
                for (int i = 0; i < k; i++) {
                    int aii = aOff + i * lda + i;
                    double t = tau[tauOff + i];
                    if (t != 0) {
                        double savedAii = A[aii];
                        A[aii] = 1;
                        BLAS.dlarf(BLAS.Side.Right, m, n - i, A, aii, lda, t, C, cOff + i, ldc, work, workOff);
                        A[aii] = savedAii;
                    }
                }
            } else {
                for (int i = k - 1; i >= 0; i--) {
                    int aii = aOff + i * lda + i;
                    double t = tau[tauOff + i];
                    if (t != 0) {
                        double savedAii = A[aii];
                        A[aii] = 1;
                        BLAS.dlarf(BLAS.Side.Right, m, n - i, A, aii, lda, t, C, cOff + i, ldc, work, workOff);
                        A[aii] = savedAii;
                    }
                }
            }
        }
    }

    static int dormqr(BLAS.Side side, BLAS.Transpose trans, int m, int n, int k,
                      double[] V, int vOff, int ldv,
                      double[] tau, int tauOff,
                      double[] C, int cOff, int ldc,
                      double[] work, int workOff) {
        return dormqr(side, trans, m, n, k, V, vOff, ldv, tau, tauOff, C, cOff, ldc, work, workOff, -1);
    }

    static int dormqr(BLAS.Side side, BLAS.Transpose trans, int m, int n, int k,
                      double[] V, int vOff, int ldv,
                      double[] tau, int tauOff,
                      double[] C, int cOff, int ldc,
                      double[] work, int workOff, int lwork) {
        boolean left = (side == BLAS.Side.Left);
        int nq = left ? m : n;
        int nw = left ? n : m;

        if (m < 0 || n < 0 || k < 0) return -1;
        if (left && k > m) return -1;
        if (!left && k > n) return -1;

        if (m == 0 || n == 0 || k == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return 0;
        }

        int nbmax = 64;
        int ldt = nbmax;
        int tsize = nbmax * ldt;

        String opts = "" + side.code + trans.code;
        int nb = min(nbmax, Ilaenv.ilaenv(1, "DORMQR", opts, nq, n, k, -1));
        int lworkopt = max(1, nw) * nb + tsize;

        if (lwork == -1) {
            work[workOff] = lworkopt;
            return 0;
        }

        int nbmin = 2;
        if (1 < nb && nb < k) {
            if (lwork < nw * nb + tsize) {
                nb = (lwork - tsize) / nw;
                nbmin = max(2, Ilaenv.ilaenv(2, "DORMQR", opts, nq, n, k, -1));
            }
        }

        if (nb < nbmin || k <= nb) {
            dorm2r(side, trans, m, n, k, V, vOff, ldv, tau, tauOff, C, cOff, ldc, work, workOff);
            return 0;
        }

        int ldwork = nb;
        boolean notran = (trans == BLAS.Transpose.NoTrans);

        if (left) {
            if (notran) {
                for (int i = ((k - 1) / nb) * nb; i >= 0; i -= nb) {
                    int ib = min(nb, k - i);
                    Dlarft.dlarftForward(V, vOff + i * ldv + i, ldv, tau, tauOff + i,
                                         work, workOff, ldt, m - i, ib);
                    Dlarfb.dlarfbLeft(V, vOff + i * ldv + i, ldv,
                                      work, workOff, ldt,
                                      C, cOff + i * ldc, ldc,
                                      work, workOff + tsize, ldwork, m - i, n, ib);
                }
            } else {
                for (int i = 0; i < k; i += nb) {
                    int ib = min(nb, k - i);
                    Dlarft.dlarftForward(V, vOff + i * ldv + i, ldv, tau, tauOff + i,
                                         work, workOff, ldt, m - i, ib);
                    Dlarfb.dlarfbLeftTrans(V, vOff + i * ldv + i, ldv,
                                           work, workOff, ldt,
                                           C, cOff + i * ldc, ldc,
                                           work, workOff + tsize, ldwork, m - i, n, ib);
                }
            }
        } else {
            if (notran) {
                for (int i = 0; i < k; i += nb) {
                    int ib = min(nb, k - i);
                    Dlarft.dlarftForward(V, vOff + i * ldv + i, ldv, tau, tauOff + i,
                                         work, workOff, ldt, n - i, ib);
                    Dlarfb.dlarfbRight(V, vOff + i * ldv + i, ldv,
                                       work, workOff, ldt,
                                       C, cOff + i, ldc,
                                       work, workOff + tsize, ldwork, m, n - i, ib);
                }
            } else {
                for (int i = ((k - 1) / nb) * nb; i >= 0; i -= nb) {
                    int ib = min(nb, k - i);
                    Dlarft.dlarftForward(V, vOff + i * ldv + i, ldv, tau, tauOff + i,
                                         work, workOff, ldt, n - i, ib);
                    Dlarfb.dlarfbRightTrans(V, vOff + i * ldv + i, ldv,
                                            work, workOff, ldt,
                                            C, cOff + i, ldc,
                                            work, workOff + tsize, ldwork, m, n - i, ib);
                }
            }
        }

        return 0;
    }
}
