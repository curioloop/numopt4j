/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * LAPACK DLAUUM: Computes the product U * U^T (upper) or L^T * L (lower),
 * where U or L is a triangular factor stored in the upper or lower triangle of A.
 *
 * <p>The result overwrites the corresponding triangle of A. Uses a blocked
 * algorithm (BLOCK_SIZE) that calls the unblocked {@code dlauu2} for small blocks.</p>
 *
 * <p>Corresponds to LAPACK routine DLAUUM.
 */
interface Dlauum {

    int BLOCK_SIZE = 64;

    static void dlauum(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        if (n == 0) return;

        boolean lower = uplo == BLAS.Uplo.Lower;

        if (BLOCK_SIZE <= 1 || n <= BLOCK_SIZE) {
            dlauu2(uplo, n, A, aOff, lda);
            return;
        }

        if (lower) {
            for (int i = 0; i < n; i += BLOCK_SIZE) {
                int ib = min(BLOCK_SIZE, n - i);

                BLAS.dtrmm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, ib, i, 1.0,
                        A, aOff + i * lda + i, lda,
                        A, aOff + i * lda, lda);
                dlauu2(uplo, ib, A, aOff + i * lda + i, lda);

                if (n - i - ib > 0) {
                    BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, ib, i, n - i - ib, 1.0,
                            A, aOff + (i + ib) * lda + i, lda,
                            A, aOff + (i + ib) * lda, lda,
                            1.0, A, aOff + i * lda, lda);
                    BLAS.dsyrk(BLAS.Uplo.Lower, BLAS.Trans.Trans, ib, n - i - ib, 1.0,
                            A, aOff + (i + ib) * lda + i, lda,
                            1.0, A, aOff + i * lda + i, lda);
                }
            }
        } else {
            for (int i = 0; i < n; i += BLOCK_SIZE) {
                int ib = min(BLOCK_SIZE, n - i);

                BLAS.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, i, ib, 1.0,
                        A, aOff + i * lda + i, lda,
                        A, aOff + i, lda);
                dlauu2(uplo, ib, A, aOff + i * lda + i, lda);

                if (n - i - ib > 0) {
                    BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, i, ib, n - i - ib, 1.0,
                            A, aOff + i + ib, lda,
                            A, aOff + i * lda + i + ib, lda,
                            1.0, A, aOff + i, lda);
                    BLAS.dsyrk(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, ib, n - i - ib, 1.0,
                            A, aOff + i * lda + i + ib, lda,
                            1.0, A, aOff + i * lda + i, lda);
                }
            }
        }
    }

    static void dlauu2(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        boolean lower = uplo == BLAS.Uplo.Lower;

        if (lower) {
            for (int i = 0; i < n; i++) {
                int rowI = aOff + i * lda;
                double aii = A[rowI + i];
                if (i < n - 1) {
                    A[rowI + i] = BLAS.ddot(n - i, A, rowI + i, lda, A, rowI + i, lda);
                    BLAS.dgemv(BLAS.Trans.Trans, n - i - 1, i, 1.0,
                            A, aOff + (i + 1) * lda, lda,
                            A, aOff + (i + 1) * lda + i, lda,
                            aii, A, rowI, 1);
                } else {
                    BLAS.dscal(i + 1, aii, A, rowI, 1);
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                int rowI = aOff + i * lda;
                double aii = A[rowI + i];
                if (i < n - 1) {
                    A[rowI + i] = BLAS.ddot(n - i, A, rowI + i, 1, A, rowI + i, 1);
                    BLAS.dgemv(BLAS.Trans.NoTrans, i, n - i - 1, 1.0,
                            A, aOff + i + 1, lda,
                            A, rowI + i + 1, 1,
                            aii, A, aOff + i, lda);
                } else {
                    BLAS.dscal(i + 1, aii, A, aOff + i, lda);
                }
            }
        }
    }
}
