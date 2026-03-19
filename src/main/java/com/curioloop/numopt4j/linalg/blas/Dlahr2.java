/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.min;

/**
 * DLAHR2: Reduces the first NB columns of a real general n×(n-k+1) matrix A
 * so that elements below the k-th subdiagonal are zero. Returns matrices V, T
 * (block reflector I - V*T*V**T) and Y = A*V*T for updating the unreduced part.
 *
 */
interface Dlahr2 {

    static void dlahr2(int n, int k, int nb,
                       double[] a, int aOff, int lda,
                       double[] tau, int tauOff,
                       double[] t, int tOff, int ldt,
                       double[] y, int yOff, int ldy) {

        if (n <= 0 || nb <= 0) {
            return;
        }

        double ei = 0;
        for (int i = 0; i < nb; i++) {
            if (i > 0) {
                Dgemv.dgemv(BLAS.Trans.NoTrans, n - k, i, -1.0, y, yOff + k * ldy, ldy, a, aOff + (k + i - 1) * lda, 1, 1.0, a, aOff + k * lda + i, lda);

                Dlamv.dcopy(i, a, aOff + k * lda + i, lda, t, tOff + nb - 1, ldt);
                Dtrmv.dtrmv(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, i, a, aOff + k * lda, lda, t, tOff + nb - 1, ldt);

                if (n - k - i > 0) {
                    Dgemv.dgemv(BLAS.Trans.Trans, n - k - i, i, 1.0, a, aOff + (k + i) * lda, lda, a, aOff + (k + i) * lda + i, lda, 1.0, t, tOff + nb - 1, ldt);
                }

                Dtrmv.dtrmv(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, i, t, tOff, ldt, t, tOff + nb - 1, ldt);

                if (n - k - i > 0) {
                    Dgemv.dgemv(BLAS.Trans.NoTrans, n - k - i, i, -1.0, a, aOff + (k + i) * lda, lda, t, tOff + nb - 1, ldt, 1.0, a, aOff + (k + i) * lda + i, lda);
                }

                Dtrmv.dtrmv(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, i, a, aOff + k * lda, lda, t, tOff + nb - 1, ldt);
                Daxpy.daxpy(i, -1.0, t, tOff + nb - 1, ldt, a, aOff + k * lda + i, lda);

                a[aOff + (k + i - 1) * lda + i - 1] = ei;
            }

            ei = Dlarfg.dlarfg(n - k - i, a[aOff + (k + i) * lda + i], a, aOff + min(k + i + 1, n - 1) * lda + i, lda, tau, tauOff + i);
            a[aOff + (k + i) * lda + i] = 1.0;

            if (n - k - i - 1 > 0) {
                Dgemv.dgemv(BLAS.Trans.NoTrans, n - k, n - k - i - 1, 1.0, a, aOff + k * lda + i + 1, lda, a, aOff + (k + i) * lda + i, lda, 0.0, y, yOff + k * ldy + i, ldy);
            }

            Dgemv.dgemv(BLAS.Trans.Trans, n - k - i, i, 1.0, a, aOff + (k + i) * lda, lda, a, aOff + (k + i) * lda + i, lda, 0.0, t, tOff + i, ldt);
            Dgemv.dgemv(BLAS.Trans.NoTrans, n - k, i, -1.0, y, yOff + k * ldy, ldy, t, tOff + i, ldt, 1.0, y, yOff + k * ldy + i, ldy);

            Dscal.dscal(n - k, tau[tauOff + i], y, yOff + k * ldy + i, ldy);

            Dscal.dscal(i, -tau[tauOff + i], t, tOff + i, ldt);
            Dtrmv.dtrmv(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, i, t, tOff, ldt, t, tOff + i, ldt);

            t[tOff + i * ldt + i] = tau[tauOff + i];
        }
        a[aOff + (k + nb - 1) * lda + nb - 1] = ei;

        if (k > 0) {
            Dlamv.dlacpy('A', k, nb, a, aOff + 1, lda, y, yOff, ldy);
            Dtrmm.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, k, nb, 1.0, a, aOff + k * lda, lda, y, yOff, ldy);
            if (n > k + nb) {
                Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, k, nb, n - k - nb, 1.0, a, aOff + 1 + nb, lda, a, aOff + (k + nb) * lda, lda, 1.0, y, yOff, ldy);
            }
            Dtrmm.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, k, nb, 1.0, t, tOff, ldt, y, yOff, ldy);
        }
    }
}
