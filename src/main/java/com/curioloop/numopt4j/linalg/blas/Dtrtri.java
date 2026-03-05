/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * LAPACK DTRTRI: Computes the inverse of a triangular matrix A in-place.
 *
 * <p>Uses a blocked Level-3 BLAS algorithm (BLOCK_SIZE) that calls the unblocked
 * {@code dtrti2} for diagonal blocks. Returns 0 on success, or a positive integer
 * indicating the position of the first zero diagonal element if A is singular.</p>
 *
 * <p>Also provides {@code dtrti2} (unblocked inversion) and {@code dtrcon}
 * (triangular matrix condition number estimate).</p>
 *
 * <p>Corresponds to LAPACK routine DTRTRI.
 * Go reference: gonum/lapack/gonum/dtrtri.go</p>
 */
interface Dtrtri {

    int BLOCK_SIZE = 64;

    static int dtrtri(BLAS.Uplo uplo, BLAS.Diag diag, int n, double[] A, int aOff, int lda) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean unit = diag == BLAS.Diag.Unit;

        if (n == 0) return 0;

        if (!unit) {
            for (int i = 0; i < n; i++) {
                if (A[aOff + i * lda + i] == 0) return i + 1;
            }
        }

        if (BLOCK_SIZE <= 1 || n <= BLOCK_SIZE) {
            dtrti2(uplo, diag, n, A, aOff, lda);
            return 0;
        }

        if (upper) {
            for (int j = 0; j < n; j += BLOCK_SIZE) {
                int jb = min(BLOCK_SIZE, n - j);

                BLAS.dtrmm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, unit ? BLAS.Diag.Unit : BLAS.Diag.NonUnit, j, jb, 1.0, A, aOff, lda, A, aOff + j, lda);
                BLAS.dtrsm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, unit ? BLAS.Diag.Unit : BLAS.Diag.NonUnit, j, jb, -1.0, A, aOff + j * lda + j, lda, A, aOff + j, lda);
                dtrti2(uplo, diag, jb, A, aOff + j * lda + j, lda);
            }
        } else {
            int nn = ((n - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            for (int j = nn; j >= 0; j -= BLOCK_SIZE) {
                int jb = min(BLOCK_SIZE, n - j);

                if (j + jb <= n - 1) {
                    BLAS.dtrmm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, unit ? BLAS.Diag.Unit : BLAS.Diag.NonUnit, n - j - jb, jb, 1.0,
                            A, aOff + (j + jb) * lda + j + jb, lda,
                            A, aOff + (j + jb) * lda + j, lda);
                    BLAS.dtrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, unit ? BLAS.Diag.Unit : BLAS.Diag.NonUnit, n - j - jb, jb, -1.0,
                            A, aOff + j * lda + j, lda,
                            A, aOff + (j + jb) * lda + j, lda);
                }
                dtrti2(uplo, diag, jb, A, aOff + j * lda + j, lda);
            }
        }

        return 0;
    }

    static void dtrti2(BLAS.Uplo uplo, BLAS.Diag diag, int n, double[] A, int aOff, int lda) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean unit = diag == BLAS.Diag.Unit;

        if (upper) {
            for (int j = 0; j < n; j++) {
                int rowJ = aOff + j * lda;
                if (!unit) {
                    A[rowJ + j] = 1.0 / A[rowJ + j];
                }
                double ajj = unit ? -1.0 : -A[rowJ + j];

                if (j > 0) {
                    BLAS.dtrmv(BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, diag, j, A, aOff, lda, A, aOff + j, lda);
                    BLAS.dscal(j, ajj, A, aOff + j, lda);
                }
            }
        } else {
            for (int j = n - 1; j >= 0; j--) {
                int rowJ = aOff + j * lda;
                if (!unit) {
                    A[rowJ + j] = 1.0 / A[rowJ + j];
                }
                double ajj = unit ? -1.0 : -A[rowJ + j];

                if (j < n - 1) {
                    BLAS.dtrmv(BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, diag, n - j - 1, A, aOff + (j + 1) * lda + j + 1, lda, A, aOff + (j + 1) * lda + j, lda);
                    BLAS.dscal(n - j - 1, ajj, A, aOff + (j + 1) * lda + j, lda);
                }
            }
        }
    }

    static double dtrcon(char norm, BLAS.Uplo uplo, BLAS.Diag diag, int n,
                         double[] A, int lda, double[] work, int[] iwork) {
        if (n == 0) return 1.0;

        double smlnum = Dlamch.dlamch('S') * n;
        double rcond = 0.0;

        double anorm = Dlantr.dlantr(norm, uplo, diag, n, n, A, lda, work);
        if (anorm <= 0) return rcond;

        double ainvnm = 0.0;
        boolean normin = false;
        int kase1 = (norm == '1' || norm == 'O' || norm == 'o') ? 1 : 2;

        int[] isave = new int[3];

        while (true) {
            ainvnm = Dlacn2.dlacn2(n, work, n, work, 0, iwork, 0, ainvnm, isave, 2, isave, 0);

            if (isave[2] == 0) {
                if (ainvnm != 0) {
                    rcond = (1.0 / anorm) / ainvnm;
                }
                return rcond;
            }

            double scale;
            if (isave[2] == kase1) {
                scale = Dlatrs.dlatrs(uplo, BLAS.Transpose.Trans, diag, normin, n, A, lda, work, 0, work, 2 * n);
            } else {
                scale = Dlatrs.dlatrs(uplo, BLAS.Transpose.NoTrans, diag, normin, n, A, lda, work, 0, work, 2 * n);
            }

            normin = true;

            if (scale != 1) {
                int ix = 0;
                double xnorm = abs(work[0]);
                for (int i = 1; i < n; i++) {
                    if (abs(work[i]) > xnorm) {
                        xnorm = abs(work[i]);
                        ix = i;
                    }
                }
                if (scale == 0 || scale < xnorm * smlnum) {
                    return rcond;
                }
                Drscl.drscl(n, scale, work, 0, 1);
            }
        }
    }
}
