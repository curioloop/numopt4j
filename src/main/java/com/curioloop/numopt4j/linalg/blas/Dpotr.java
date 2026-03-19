/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * LAPACK DPOTR: Cholesky factorization and related routines for symmetric positive definite matrices.
 *
 * <p>Provides the following LAPACK routines as static methods:</p>
 * <ul>
 *   <li>{@code dpotf2} — unblocked Cholesky factorization (DPOTF2)</li>
 *   <li>{@code dpotrf} — blocked Cholesky factorization (DPOTRF)</li>
 *   <li>{@code dpotri} — matrix inversion using Cholesky factorization (DPOTRI)</li>
 *   <li>{@code dpotrs} — solve A*X=B using Cholesky factorization (DPOTRS)</li>
 *   <li>{@code dpocon} — estimate reciprocal condition number (DPOCON)</li>
 * </ul>
 *
 * <p>For upper triangular: computes A = U^T * U; for lower triangular: A = L * L^T.
 * Returns false (or positive info) if the matrix is not positive definite.</p>
 *
 */
interface Dpotr {

    static boolean dpotf2(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        if (n < 0 || lda < Math.max(1, n)) {
            return false;
        }
        if (n == 0) {
            return true;
        }

        boolean lower = uplo == BLAS.Uplo.Lower;
        return lower ? dpotf2Lower(n, A, aOff, lda) : dpotf2Upper(n, A, aOff, lda);
    }

    static boolean dpotf2Lower(int n, double[] A, int aOff, int lda) {
        for (int j = 0; j < n; j++) {
            int rowJ = aOff + j * lda;
            double ajj = A[rowJ + j];
            if (j > 0) {
                ajj -= BLAS.ddot(j, A, rowJ, 1, A, rowJ, 1);
            }
            if (ajj <= 0 || Double.isNaN(ajj)) {
                A[rowJ + j] = ajj;
                return false;
            }
            ajj = sqrt(ajj);
            A[rowJ + j] = ajj;
            if (j < n - 1) {
                BLAS.dgemv(BLAS.Trans.NoTrans, n - j - 1, j, -1.0,
                        A, aOff + (j + 1) * lda, lda,
                        A, rowJ, 1,
                        1.0, A, aOff + (j + 1) * lda + j, lda);
                BLAS.dscal(n - j - 1, 1.0 / ajj, A, aOff + (j + 1) * lda + j, lda);
            }
        }
        return true;
    }

    static boolean dpotf2Upper(int n, double[] A, int aOff, int lda) {
        for (int j = 0; j < n; j++) {
            int rowJ = aOff + j * lda;
            double ajj = A[rowJ + j];
            if (j > 0) {
                ajj -= BLAS.ddot(j, A, aOff + j, lda, A, aOff + j, lda);
            }
            if (ajj <= 0 || Double.isNaN(ajj)) {
                A[rowJ + j] = ajj;
                return false;
            }
            ajj = sqrt(ajj);
            A[rowJ + j] = ajj;
            if (j < n - 1) {
                BLAS.dgemv(BLAS.Trans.Trans, j, n - j - 1, -1.0,
                        A, aOff + j + 1, lda,
                        A, aOff + j, lda,
                        1.0, A, rowJ + j + 1, 1);
                BLAS.dscal(n - j - 1, 1.0 / ajj, A, rowJ + j + 1, 1);
            }
        }
        return true;
    }

    static int dpotrf(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        if (n < 0 || lda < max(1, n)) {
            return -1;
        }
        if (n == 0) {
            return 0;
        }

        boolean lower = uplo == BLAS.Uplo.Lower;
        int nb = Ilaenv.ilaenv(1, "DPOTRF", String.valueOf(uplo.code), n, -1, -1, -1);

        if (nb <= 1 || n <= nb) {
            return dpotf2(uplo, n, A, aOff, lda) ? 0 : 1;
        }

        return lower ? dpotrfLower(n, A, aOff, lda, nb) : dpotrfUpper(n, A, aOff, lda, nb);
    }

    static int dpotrfLower(int n, double[] A, int aOff, int lda, int nb) {
        for (int j = 0; j < n; j += nb) {
            int jb = min(nb, n - j);

            BLAS.dsyrk(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, jb, j, -1.0,
                    A, aOff + j * lda, lda,
                    1.0, A, aOff + j * lda + j, lda);

            if (!dpotf2Lower(jb, A, aOff + j * lda + j, lda)) return j + 1;

            if (j + jb < n) {
                BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, n - j - jb, jb, j, -1.0,
                        A, aOff + (j + jb) * lda, lda,
                        A, aOff + j * lda, lda,
                        1.0, A, aOff + (j + jb) * lda + j, lda);

                BLAS.dtrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, n - j - jb, jb, 1.0,
                        A, aOff + j * lda + j, lda,
                        A, aOff + (j + jb) * lda + j, lda);
            }
        }
        return 0;
    }

    static int dpotrfUpper(int n, double[] A, int aOff, int lda, int nb) {
        for (int j = 0; j < n; j += nb) {
            int jb = min(nb, n - j);

            BLAS.dsyrk(BLAS.Uplo.Upper, BLAS.Trans.Trans, jb, j, -1.0,
                    A, aOff + j, lda,
                    1.0, A, aOff + j * lda + j, lda);

            if (!dpotf2Upper(jb, A, aOff + j * lda + j, lda)) return j + 1;

            if (j + jb < n) {
                BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, jb, n - j - jb, j, -1.0,
                        A, aOff + j, lda,
                        A, aOff + j + jb, lda,
                        1.0, A, aOff + j * lda + j + jb, lda);

                BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, jb, n - j - jb, 1.0,
                        A, aOff + j * lda + j, lda,
                        A, aOff + j * lda + j + jb, lda);
            }
        }
        return 0;
    }

    static boolean dpotri(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda) {
        if (n == 0) return true;

        if (Dtrtri.dtrtri(uplo, BLAS.Diag.NonUnit, n, A, aOff, lda) != 0) {
            return false;
        }

        Dlauum.dlauum(uplo, n, A, aOff, lda);
        return true;
    }

    static void dpotrs(BLAS.Uplo uplo, int n, int nrhs, double[] A, int aOff, int lda,
                       double[] B, int bOff, int ldb) {
        if (n == 0 || nrhs == 0) return;

        boolean lower = uplo == BLAS.Uplo.Lower;
        if (lower) {
            BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, n, nrhs, 1.0, A, aOff, lda, B, bOff, ldb);
            BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, n, nrhs, 1.0, A, aOff, lda, B, bOff, ldb);
        } else {
            BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, n, nrhs, 1.0, A, aOff, lda, B, bOff, ldb);
            BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, n, nrhs, 1.0, A, aOff, lda, B, bOff, ldb);
        }
    }

    static double dpocon(BLAS.Uplo uplo, int n, double[] A, int lda, double anorm,
                         double[] work, int[] iwork) {
        if (n == 0) return 1.0;
        if (anorm == 0) return 0.0;

        boolean upper = uplo == BLAS.Uplo.Upper;
        double smlnum = Dlamch.dlamch('S');
        double rcond = 0.0;
        double ainvnm = 0.0;
        boolean normin = false;

        int[] isave = new int[3];

        while (true) {
            ainvnm = Dlacn2.dlacn2(n, work, n, work, 0, iwork, 0, ainvnm, isave, 2, isave, 0);

            if (isave[2] == 0) {
                if (ainvnm != 0) {
                    rcond = (1.0 / ainvnm) / anorm;
                }
                return rcond;
            }

            double sl, su;
            if (upper) {
                sl = Dlatrs.dlatrs(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, normin, n, A, lda, work, 0, work, 2 * n);
                normin = true;
                su = Dlatrs.dlatrs(BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, normin, n, A, lda, work, 0, work, 2 * n);
            } else {
                sl = Dlatrs.dlatrs(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, normin, n, A, lda, work, 0, work, 2 * n);
                normin = true;
                su = Dlatrs.dlatrs(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, normin, n, A, lda, work, 0, work, 2 * n);
            }

            double scale = sl * su;

            if (scale != 1) {
                int ix = 0;
                double xmax = abs(work[0]);
                for (int i = 1; i < n; i++) {
                    if (abs(work[i]) > xmax) {
                        xmax = abs(work[i]);
                        ix = i;
                    }
                }
                if (scale == 0 || scale < xmax * smlnum) {
                    return rcond;
                }
                Drscl.drscl(n, scale, work, 0, 1);
            }
        }
    }
}
