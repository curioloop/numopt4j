/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * DLABRD: Reduces the first NB rows and columns of a real general m×n matrix A
 * to upper or lower bidiagonal form by an orthogonal transformation Q**T * A * P.
 * Also returns the matrices X and Y needed to apply the transformation to the
 * unreduced part of A: A := A - V*Y**T - X*U**T.
 *
 */
interface Dlabrd {

    static void dlabrd(int m, int n, int nb, double[] A, int aOff, int lda,
                       double[] d, int dOff, double[] e, int eOff,
                       double[] tauQ, int tauQOff, double[] tauP, int tauPOff,
                       double[] x, int xOff, int ldx, double[] y, int yOff, int ldy) {
        if (m == 0 || n == 0 || nb == 0) {
            return;
        }

        if (m >= n) {
            for (int i = 0; i < nb; i++) {
                BLAS.dgemv(BLAS.Transpose.NoTrans, m - i, i, -1, A, aOff + i * lda, lda, y, yOff + i * ldy, 1, 1, A, aOff + i * lda + i, lda);
                BLAS.dgemv(BLAS.Transpose.NoTrans, m - i, i, -1, x, xOff + i * ldx, ldx, A, aOff + i, lda, 1, A, aOff + i * lda + i, lda);

                double aii = Dlarfg.dlarfg(m - i, A[aOff + i * lda + i], A, aOff + min(i + 1, m - 1) * lda + i, lda, tauQ, tauQOff + i);
                d[dOff + i] = aii;

                if (i < n - 1) {
                    A[aOff + i * lda + i] = 1;
                    BLAS.dgemv(BLAS.Transpose.Trans, m - i, n - i - 1, 1, A, aOff + i * lda + i + 1, lda, A, aOff + i * lda + i, lda, 0, y, yOff + (i + 1) * ldy + i, ldy);
                    BLAS.dgemv(BLAS.Transpose.Trans, m - i, i, 1, A, aOff + i * lda, lda, A, aOff + i * lda + i, lda, 0, y, yOff + i, ldy);
                    BLAS.dgemv(BLAS.Transpose.NoTrans, n - i - 1, i, -1, y, yOff + (i + 1) * ldy, ldy, y, yOff + i, ldy, 1, y, yOff + (i + 1) * ldy + i, ldy);
                    BLAS.dgemv(BLAS.Transpose.Trans, m - i, i, 1, x, xOff + i * ldx, ldx, A, aOff + i * lda + i, lda, 0, y, yOff + i, ldy);
                    BLAS.dgemv(BLAS.Transpose.Trans, i, n - i - 1, -1, A, aOff + i + 1, lda, y, yOff + i, ldy, 1, y, yOff + (i + 1) * ldy + i, ldy);
                    BLAS.dscal(n - i - 1, tauQ[tauQOff + i], y, yOff + (i + 1) * ldy + i, ldy);

                    BLAS.dgemv(BLAS.Transpose.NoTrans, n - i - 1, i + 1, -1, y, yOff + (i + 1) * ldy, ldy, A, aOff + i * lda, 1, 1, A, aOff + i * lda + i + 1, 1);
                    BLAS.dgemv(BLAS.Transpose.Trans, i, n - i - 1, -1, A, aOff + i + 1, lda, x, xOff + i * ldx, 1, 1, A, aOff + i * lda + i + 1, 1);

                    double aii1 = Dlarfg.dlarfg(n - i - 1, A[aOff + i * lda + i + 1], A, aOff + i * lda + min(i + 2, n - 1), 1, tauP, tauPOff + i);
                    e[eOff + i] = aii1;
                    A[aOff + i * lda + i + 1] = 1;

                    BLAS.dgemv(BLAS.Transpose.NoTrans, m - i - 1, n - i - 1, 1, A, aOff + (i + 1) * lda + i + 1, lda, A, aOff + i * lda + i + 1, 1, 0, x, xOff + (i + 1) * ldx + i, ldx);
                    BLAS.dgemv(BLAS.Transpose.Trans, n - i - 1, i + 1, 1, y, yOff + (i + 1) * ldy, ldy, A, aOff + i * lda + i + 1, 1, 0, x, xOff + i, ldx);
                    BLAS.dgemv(BLAS.Transpose.NoTrans, m - i - 1, i + 1, -1, A, aOff + (i + 1) * lda, lda, x, xOff + i, ldx, 1, x, xOff + (i + 1) * ldx + i, ldx);
                    BLAS.dgemv(BLAS.Transpose.NoTrans, i, n - i - 1, 1, A, aOff + i + 1, lda, A, aOff + i * lda + i + 1, 1, 0, x, xOff + i, ldx);
                    BLAS.dgemv(BLAS.Transpose.NoTrans, m - i - 1, i, -1, x, xOff + (i + 1) * ldx, ldx, x, xOff + i, ldx, 1, x, xOff + (i + 1) * ldx + i, ldx);
                    BLAS.dscal(m - i - 1, tauP[tauPOff + i], x, xOff + (i + 1) * ldx + i, ldx);
                }
            }
        } else {
            for (int i = 0; i < nb; i++) {
                BLAS.dgemv(BLAS.Transpose.NoTrans, n - i, i, -1, y, yOff + i * ldy, ldy, A, aOff + i * lda, 1, 1, A, aOff + i * lda + i, 1);
                BLAS.dgemv(BLAS.Transpose.Trans, i, n - i, -1, A, aOff + i, lda, x, xOff + i * ldx, 1, 1, A, aOff + i * lda + i, 1);

                double aii = Dlarfg.dlarfg(n - i, A[aOff + i * lda + i], A, aOff + i * lda + min(i + 1, n - 1), 1, tauP, tauPOff + i);
                d[dOff + i] = aii;

                if (i < m - 1) {
                    A[aOff + i * lda + i] = 1;
                    BLAS.dgemv(BLAS.Transpose.NoTrans, m - i - 1, n - i, 1, A, aOff + (i + 1) * lda + i, lda, A, aOff + i * lda + i, 1, 0, x, xOff + (i + 1) * ldx + i, ldx);
                    BLAS.dgemv(BLAS.Transpose.Trans, n - i, i, 1, y, yOff + i * ldy, ldy, A, aOff + i * lda + i, 1, 0, x, xOff + i, ldx);
                    BLAS.dgemv(BLAS.Transpose.NoTrans, m - i - 1, i, -1, A, aOff + (i + 1) * lda, lda, x, xOff + i, ldx, 1, x, xOff + (i + 1) * ldx + i, ldx);
                    BLAS.dgemv(BLAS.Transpose.NoTrans, i, n - i, 1, A, aOff + i, lda, A, aOff + i * lda + i, 1, 0, x, xOff + i, ldx);
                    BLAS.dgemv(BLAS.Transpose.NoTrans, m - i - 1, i, -1, x, xOff + (i + 1) * ldx, ldx, x, xOff + i, ldx, 1, x, xOff + (i + 1) * ldx + i, ldx);
                    BLAS.dscal(m - i - 1, tauP[tauPOff + i], x, xOff + (i + 1) * ldx + i, ldx);

                    BLAS.dgemv(BLAS.Transpose.NoTrans, m - i - 1, i, -1, A, aOff + (i + 1) * lda, lda, y, yOff + i * ldy, 1, 1, A, aOff + (i + 1) * lda + i, lda);
                    BLAS.dgemv(BLAS.Transpose.NoTrans, m - i - 1, i + 1, -1, x, xOff + (i + 1) * ldx, ldx, A, aOff + i, lda, 1, A, aOff + (i + 1) * lda + i, lda);

                    double aii1 = Dlarfg.dlarfg(m - i - 1, A[aOff + (i + 1) * lda + i], A, aOff + min(i + 2, m - 1) * lda + i, lda, tauQ, tauQOff + i);
                    e[eOff + i] = aii1;
                    A[aOff + (i + 1) * lda + i] = 1;

                    BLAS.dgemv(BLAS.Transpose.Trans, m - i - 1, n - i - 1, 1, A, aOff + (i + 1) * lda + i + 1, lda, A, aOff + (i + 1) * lda + i, lda, 0, y, yOff + (i + 1) * ldy + i, ldy);
                    BLAS.dgemv(BLAS.Transpose.Trans, m - i - 1, i, 1, A, aOff + (i + 1) * lda, lda, A, aOff + (i + 1) * lda + i, lda, 0, y, yOff + i, ldy);
                    BLAS.dgemv(BLAS.Transpose.NoTrans, n - i - 1, i, -1, y, yOff + (i + 1) * ldy, ldy, y, yOff + i, ldy, 1, y, yOff + (i + 1) * ldy + i, ldy);
                    BLAS.dgemv(BLAS.Transpose.Trans, m - i - 1, i + 1, 1, x, xOff + (i + 1) * ldx, ldx, A, aOff + (i + 1) * lda + i, lda, 0, y, yOff + i, ldy);
                    BLAS.dgemv(BLAS.Transpose.Trans, i + 1, n - i - 1, -1, A, aOff + i + 1, lda, y, yOff + i, ldy, 1, y, yOff + (i + 1) * ldy + i, ldy);
                    BLAS.dscal(n - i - 1, tauQ[tauQOff + i], y, yOff + (i + 1) * ldy + i, ldy);
                }
            }
        }
    }
}
