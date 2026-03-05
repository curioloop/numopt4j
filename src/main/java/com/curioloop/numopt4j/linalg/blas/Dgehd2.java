/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.min;

interface Dgehd2 {

    /**
     * Reduces a block of a general n×n matrix A to upper Hessenberg form using an unblocked algorithm.
     *
     * <p>Computes Qᵀ * A * Q = H by applying a sequence of elementary reflectors H_i.
     * Only the submatrix A[ilo:ihi+1, ilo:ihi+1] is reduced; the rest of A is unchanged.
     * The reflectors are stored below the first subdiagonal of A, and their scalar factors in tau.
     *
     *
     * @param n      order of the matrix A
     * @param ilo    lower bound of the active submatrix (0-based)
     * @param ihi    upper bound of the active submatrix (0-based, inclusive)
     * @param A      n×n matrix (row-major, overwritten on exit)
     * @param aOff   offset into A
     * @param lda    leading dimension of A
     * @param tau    scalar factors of elementary reflectors (length n-1)
     * @param tauOff offset into tau
     * @param work   workspace (length n)
     * @param workOff offset into work
     */
    static void dgehd2(int n, int ilo, int ihi, double[] A, int aOff, int lda,
                       double[] tau, int tauOff, double[] work, int workOff) {
        if (n <= 0) {
            return;
        }

        for (int i = ilo; i < ihi; i++) {
            double alpha = A[aOff + (i + 1) * lda + i];

            double aii = Dlarfg.dlarfg(ihi - i, alpha, A, aOff + min(i + 2, n - 1) * lda + i, lda, tau, tauOff + i);

            A[aOff + (i + 1) * lda + i] = 1.0;

            Dlarf.dlarf(BLAS.Side.Right, ihi + 1, ihi - i,
                    A, aOff + (i + 1) * lda + i, lda, tau[tauOff + i],
                    A, aOff + i + 1, lda, work, workOff);

            Dlarf.dlarf(BLAS.Side.Left, ihi - i, n - i - 1,
                    A, aOff + (i + 1) * lda + i, lda, tau[tauOff + i],
                    A, aOff + (i + 1) * lda + i + 1, lda, work, workOff);

            A[aOff + (i + 1) * lda + i] = aii;
        }
    }
}
