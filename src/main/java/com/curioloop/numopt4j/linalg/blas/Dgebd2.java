/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

interface Dgebd2 {

    /**
     * Reduces a general m×n matrix A to upper or lower bidiagonal form B by an orthogonal transformation.
     *
     * <p>Computes Qᵀ * A * P = B, where B is bidiagonal. If m >= n, B is upper bidiagonal;
     * if m < n, B is lower bidiagonal. The matrices Q and P are represented as products
     * of elementary reflectors stored in A along with tauQ and tauP.
     *
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
     * @param work   workspace (length max(m,n))
     * @param workOff offset into work
     */
    static void dgebd2(int m, int n, double[] A, int aOff, int lda,
                       double[] d, int dOff, double[] e, int eOff,
                       double[] tauQ, int tauQOff, double[] tauP, int tauPOff,
                       double[] work, int workOff) {
        if (m < 0 || n < 0 || lda < max(1, n)) {
            return;
        }

        int minmn = min(m, n);
        if (minmn == 0) {
            return;
        }

        if (m >= n) {
            for (int i = 0; i < n; i++) {
                int rowOff = aOff + i * lda;
                double aii = Dlarfg.dlarfg(m - i, A[rowOff + i], A, aOff + min(i + 1, m - 1) * lda + i, lda, tauQ, tauQOff + i);
                d[dOff + i] = aii;
                A[rowOff + i] = 1;

                if (i < n - 1) {
                    Dlarf.dlarf(BLAS.Side.Left, m - i, n - i - 1, A, rowOff + i, lda, tauQ[tauQOff + i],
                                A, rowOff + i + 1, lda, work, workOff);
                }
                A[rowOff + i] = d[dOff + i];

                if (i < n - 1) {
                    double aii1 = Dlarfg.dlarfg(n - i - 1, A[rowOff + i + 1], A, rowOff + min(i + 2, n - 1), 1, tauP, tauPOff + i);
                    e[eOff + i] = aii1;
                    A[rowOff + i + 1] = 1;
                    Dlarf.dlarf(BLAS.Side.Right, m - i - 1, n - i - 1, A, rowOff + i + 1, 1, tauP[tauPOff + i],
                                A, aOff + (i + 1) * lda + i + 1, lda, work, workOff);
                    A[rowOff + i + 1] = e[eOff + i];
                } else {
                    tauP[tauPOff + i] = 0;
                }
            }
        } else {
            for (int i = 0; i < m; i++) {
                int rowOff = aOff + i * lda;
                double aii = Dlarfg.dlarfg(n - i, A[rowOff + i], A, rowOff + min(i + 1, n - 1), 1, tauP, tauPOff + i);
                d[dOff + i] = aii;
                A[rowOff + i] = 1;

                if (i < m - 1) {
                    Dlarf.dlarf(BLAS.Side.Right, m - i - 1, n - i, A, rowOff + i, 1, tauP[tauPOff + i],
                                A, aOff + (i + 1) * lda + i, lda, work, workOff);
                }
                A[rowOff + i] = d[dOff + i];

                if (i < m - 1) {
                    double aii1 = Dlarfg.dlarfg(m - i - 1, A[aOff + (i + 1) * lda + i], A, aOff + min(i + 2, m - 1) * lda + i, lda, tauQ, tauQOff + i);
                    e[eOff + i] = aii1;
                    A[aOff + (i + 1) * lda + i] = 1;
                    Dlarf.dlarf(BLAS.Side.Left, m - i - 1, n - i - 1, A, aOff + (i + 1) * lda + i, lda, tauQ[tauQOff + i],
                                A, aOff + (i + 1) * lda + i + 1, lda, work, workOff);
                    A[aOff + (i + 1) * lda + i] = e[eOff + i];
                } else {
                    tauQ[tauQOff + i] = 0;
                }
            }
        }
    }
}
