/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Forms the triangular factor T in a block reflector H = I - V*T*V^T.
 * LAPACK DLARFT algorithm.
 *
 * <p>For Forward direction: H = H_0 * H_1 * ... * H_{k-1}, T is upper triangular.</p>
 * <p>For Backward direction: H = H_{k-1} * ... * H_1 * H_0, T is lower triangular.</p>
 *
 * <p>Storage conventions:</p>
 * <ul>
 *   <li>ColumnWise ('C'): V columns contain Householder vectors</li>
 *   <li>RowWise ('R'): V rows contain Householder vectors</li>
 * </ul>
 */
public interface Dlarft {

    static void dlarft(char direct, char storev, int m, int k,
                       double[] V, int vOff, int ldv,
                       double[] tau, int tauOff,
                       double[] T, int tOff, int ldt) {
        boolean forward = direct == 'F' || direct == 'f';
        boolean colwise = storev == 'C' || storev == 'c';

        if (forward) {
            if (colwise) {
                dlarftForwardColWise(V, vOff, ldv, tau, tauOff, T, tOff, ldt, m, k);
            } else {
                dlarftForwardRowWise(V, vOff, ldv, tau, tauOff, T, tOff, ldt, m, k);
            }
        } else {
            if (colwise) {
                dlarftBackwardColWise(V, vOff, ldv, tau, tauOff, T, tOff, ldt, m, k);
            } else {
                dlarftBackwardRowWise(V, vOff, ldv, tau, tauOff, T, tOff, ldt, m, k);
            }
        }
    }

    static void dlarftForward(double[] V, int vOff, int ldv, double[] tau, int tauOff,
                              double[] T, int tOff, int ldt, int m, int k) {
        dlarftForwardColWise(V, vOff, ldv, tau, tauOff, T, tOff, ldt, m, k);
    }

    static void dlarftForwardColWise(double[] V, int vOff, int ldv, double[] tau, int tauOff,
                                     double[] T, int tOff, int ldt, int m, int k) {
        if (k <= 0) return;

        int prevlastv = m - 1;
        for (int i = 0; i < k; i++) {
            prevlastv = Math.max(i, prevlastv);
            if (tau[tauOff + i] == 0.0) {
                for (int j = 0; j <= i; j++) {
                    T[tOff + j * ldt + i] = 0.0;
                }
                continue;
            }

            int lastv;
            for (lastv = m - 1; lastv >= i + 1; lastv--) {
                if (V[vOff + lastv * ldv + i] != 0.0) {
                    break;
                }
            }

            for (int j = 0; j < i; j++) {
                T[tOff + j * ldt + i] = -tau[tauOff + i] * V[vOff + i * ldv + j];
            }

            int j = Math.min(lastv, prevlastv);
            if (j > i) {
                BLAS.dgemv(BLAS.Trans.Trans, j - i, i, -tau[tauOff + i],
                           V, vOff + (i + 1) * ldv, ldv,
                           V, vOff + (i + 1) * ldv + i, ldv,
                           1.0, T, tOff + i, ldt);
            }

            if (i > 0) {
                BLAS.dtrmv(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, i, T, tOff, ldt, T, tOff + i, ldt);
            }

            T[tOff + i * ldt + i] = tau[tauOff + i];
            if (i > 1) {
                prevlastv = Math.max(prevlastv, lastv);
            } else {
                prevlastv = lastv;
            }
        }
    }

    static void dlarftForwardRowWise(double[] V, int vOff, int ldv, double[] tau, int tauOff,
                                     double[] T, int tOff, int ldt, int m, int k) {
        if (k <= 0) return;

        int prevlastv = m - 1;
        for (int i = 0; i < k; i++) {
            prevlastv = Math.max(i, prevlastv);
            if (tau[tauOff + i] == 0.0) {
                for (int j = 0; j <= i; j++) {
                    T[tOff + j * ldt + i] = 0.0;
                }
                continue;
            }

            int lastv;
            for (lastv = m - 1; lastv >= i + 1; lastv--) {
                if (V[vOff + i * ldv + lastv] != 0.0) {
                    break;
                }
            }

            for (int j = 0; j < i; j++) {
                T[tOff + j * ldt + i] = -tau[tauOff + i] * V[vOff + j * ldv + i];
            }

            int j = Math.min(lastv, prevlastv);
            if (j > i) {
                BLAS.dgemv(BLAS.Trans.NoTrans, i, j - i, -tau[tauOff + i],
                           V, vOff + i + 1, ldv,
                           V, vOff + i * ldv + i + 1, 1,
                           1.0, T, tOff + i, ldt);
            }

            if (i > 0) {
                BLAS.dtrmv(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, i, T, tOff, ldt, T, tOff + i, ldt);
            }

            T[tOff + i * ldt + i] = tau[tauOff + i];
            if (i > 1) {
                prevlastv = Math.max(prevlastv, lastv);
            } else {
                prevlastv = lastv;
            }
        }
    }

    static void dlarftBackward(double[] V, int vOff, int ldv, double[] tau, int tauOff,
                               double[] T, int tOff, int ldt, int m, int k) {
        dlarftBackwardColWise(V, vOff, ldv, tau, tauOff, T, tOff, ldt, m, k);
    }

    static void dlarftBackwardColWise(double[] V, int vOff, int ldv, double[] tau, int tauOff,
                                      double[] T, int tOff, int ldt, int m, int k) {
        if (k <= 0) return;

        int prevlastv = 0;
        for (int i = k - 1; i >= 0; i--) {
            if (tau[tauOff + i] == 0.0) {
                for (int j = i; j < k; j++) {
                    T[tOff + j * ldt + i] = 0.0;
                }
                continue;
            }

            int lastv = 0;
            if (i < k - 1) {
                for (lastv = 0; lastv < i; lastv++) {
                    if (V[vOff + lastv * ldv + i] != 0.0) {
                        break;
                    }
                }
                for (int j = i + 1; j < k; j++) {
                    T[tOff + j * ldt + i] = -tau[tauOff + i] * V[vOff + (m - k + i) * ldv + j];
                }
                int j = Math.max(lastv, prevlastv);
                BLAS.dgemv(BLAS.Trans.Trans, m - k + i - j, k - i - 1, -tau[tauOff + i],
                           V, vOff + j * ldv + i + 1, ldv,
                           V, vOff + j * ldv + i, ldv,
                           1.0, T, tOff + (i + 1) * ldt + i, ldt);
                
                BLAS.dtrmv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, k - i - 1,
                           T, tOff + (i + 1) * ldt + i + 1, ldt,
                           T, tOff + (i + 1) * ldt + i, ldt);
                
                if (i > 0) {
                    prevlastv = Math.min(prevlastv, lastv);
                } else {
                    prevlastv = lastv;
                }
            }

            T[tOff + i * ldt + i] = tau[tauOff + i];
        }
    }

    static void dlarftBackwardRowWise(double[] V, int vOff, int ldv, double[] tau, int tauOff,
                                      double[] T, int tOff, int ldt, int n, int k) {
        if (k <= 0) return;

        int prevlastv = 0;
        for (int i = k - 1; i >= 0; i--) {
            if (tau[tauOff + i] == 0.0) {
                for (int j = i; j < k; j++) {
                    T[tOff + j * ldt + i] = 0.0;
                }
                continue;
            }

            int lastv = 0;
            if (i < k - 1) {
                for (lastv = 0; lastv < i; lastv++) {
                    if (V[vOff + i * ldv + lastv] != 0.0) {
                        break;
                    }
                }
                for (int j = i + 1; j < k; j++) {
                    T[tOff + j * ldt + i] = -tau[tauOff + i] * V[vOff + j * ldv + (n - k + i)];
                }
                int j = Math.max(lastv, prevlastv);
                BLAS.dgemv(BLAS.Trans.NoTrans, k - i - 1, n - k + i - j, -tau[tauOff + i],
                           V, vOff + (i + 1) * ldv + j, ldv,
                           V, vOff + i * ldv + j, 1,
                           1.0, T, tOff + (i + 1) * ldt + i, ldt);
                
                BLAS.dtrmv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, k - i - 1,
                           T, tOff + (i + 1) * ldt + i + 1, ldt,
                           T, tOff + (i + 1) * ldt + i, ldt);
                
                if (i > 0) {
                    prevlastv = Math.min(prevlastv, lastv);
                } else {
                    prevlastv = lastv;
                }
            }

            T[tOff + i * ldt + i] = tau[tauOff + i];
        }
    }
}
