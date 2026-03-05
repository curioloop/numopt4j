/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * LAPACK DGETR: LU factorization and related routines for general matrices.
 *
 * <p>Provides the following LAPACK routines as static methods:</p>
 * <ul>
 *   <li>{@code dgetf2} — unblocked LU factorization with partial pivoting (DGETF2)</li>
 *   <li>{@code dgetrf} — blocked LU factorization with partial pivoting (DGETRF)</li>
 *   <li>{@code dgetri} — matrix inversion using LU factorization (DGETRI)</li>
 *   <li>{@code dgetrs} — solve A*X=B or A^T*X=B using LU factorization (DGETRS)</li>
 *   <li>{@code dgecon} — estimate reciprocal condition number (DGECON)</li>
 * </ul>
 *
 * <p>The LU factorization computes A = P * L * U where P is a permutation matrix,
 * L is unit lower triangular, and U is upper triangular. Row interchanges are
 * recorded in {@code ipiv} (0-indexed).</p>
 *
 * <p>Go reference: gonum/lapack/gonum/dgetf2.go, dgetrf.go, dgetri.go, dgetrs.go, dgecon.go</p>
 */
interface Dgetr {

    double SFMIN = Double.MIN_NORMAL;

    static int dgetf2(int m, int n, double[] A, int aOff, int lda, int[] ipiv, int ipivOff) {
        if (m < 0 || n < 0 || lda < max(1, n)) {
            return -1;
        }

        int mn = min(m, n);

        if (mn == 0) return 0;

        int info = 0;

        for (int j = 0; j < mn; j++) {
            int jpLocal = BLAS.idamax(m - j, A, aOff + j * lda + j, lda);
            int jp = j + jpLocal;
            ipiv[ipivOff + j] = jp;

            if (A[aOff + jp * lda + j] == 0) {
                if (info == 0) info = j + 1;
            } else {
                if (jp != j) {
                    BLAS.dswap(n, A, aOff + j * lda, 1, A, aOff + jp * lda, 1);
                }
                if (j < m - 1) {
                    double aj = A[aOff + j * lda + j];
                    if (abs(aj) >= SFMIN) {
                        BLAS.dscal(m - j - 1, 1.0 / aj, A, aOff + (j + 1) * lda + j, lda);
                    } else {
                        for (int i = 0; i < m - j - 1; i++) {
                            A[aOff + (j + 1 + i) * lda + j] = A[aOff + (j + 1 + i) * lda + j] / aj;
                        }
                    }
                }
            }

            if (j < mn - 1) {
                BLAS.dger(m - j - 1, n - j - 1, -1.0,
                        A, aOff + (j + 1) * lda + j, lda,
                        A, aOff + j * lda + j + 1, 1,
                        A, aOff + (j + 1) * lda + j + 1, lda);
            }
        }

        return info;
    }

    static int dgetrf(int m, int n, double[] A, int aOff, int lda, int[] ipiv, int ipivOff) {
        if (m < 0 || n < 0 || lda < max(1, n)) {
            return -1;
        }

        int mn = min(m, n);

        if (mn == 0) return 0;

        int nb = Ilaenv.ilaenv(1, "DGETRF", " ", m, n, -1, -1);
        if (nb <= 1 || mn <= nb) {
            return dgetf2(m, n, A, aOff, lda, ipiv, ipivOff);
        }

        int info = 0;

        for (int j = 0; j < mn; j += nb) {
            int jb = min(mn - j, nb);

            int blockInfo = dgetf2(m - j, jb, A, aOff + j * lda + j, lda, ipiv, ipivOff + j);
            if (blockInfo != 0 && info == 0) {
                info = blockInfo + j;
            }

            for (int i = j; i <= min(m - 1, j + jb - 1); i++) {
                ipiv[ipivOff + i] = j + ipiv[ipivOff + i];
            }

            BLAS.dlaswp(j, A, aOff, lda, j, j + jb - 1, ipiv, ipivOff, 1);

            if (j + jb < n) {
                BLAS.dlaswp(n - j - jb, A, aOff + j + jb, lda, j, j + jb - 1, ipiv, ipivOff, 1);

                BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, BLAS.Diag.Unit, jb, n - j - jb, 1.0,
                        A, aOff + j * lda + j, lda, A, aOff + j * lda + j + jb, lda);

                if (j + jb < m) {
                    BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m - j - jb, n - j - jb, jb, -1.0,
                            A, aOff + (j + jb) * lda + j, lda,
                            A, aOff + j * lda + j + jb, lda,
                            1.0, A, aOff + (j + jb) * lda + j + jb, lda);
                }
            }
        }

        return info;
    }

    static boolean dgetri(int n, double[] A, int aOff, int lda, int[] ipiv, int ipivOff,
                          double[] work, int workOff, int lwork) {
        if (n == 0) return true;

        if (Dtrtri.dtrtri(BLAS.Uplo.Upper, BLAS.Diag.NonUnit, n, A, aOff, lda) != 0) {
            return false;
        }

        int nb = Ilaenv.ilaenv(1, "DGETRI", " ", n, -1, -1, -1);
        int nbmin = 2;
        int ldwork = nb;

        if (nb >= nbmin && n > nb) {
            int iws = max(n * nb, 1);
            if (lwork < iws) {
                nb = lwork / n;
                nbmin = 2;
            }
            ldwork = nb;
        }

        if (nb < nbmin || n <= nb) {
            for (int j = n - 1; j >= 0; j--) {
                for (int i = j + 1; i < n; i++) {
                    work[workOff + i] = A[aOff + i * lda + j];
                    A[aOff + i * lda + j] = 0;
                }
                if (j < n - 1) {
                    for (int i = 0; i < n; i++) {
                        double sum = 0.0;
                        for (int k = j + 1; k < n; k++) {
                            sum += A[aOff + i * lda + k] * work[workOff + k];
                        }
                        A[aOff + i * lda + j] -= sum;
                    }
                }
            }
        } else {
            int nn = ((n - 1) / nb) * nb;
            for (int j = nn; j >= 0; j -= nb) {
                int jb = min(nb, n - j);

                for (int jj = j; jj < j + jb; jj++) {
                    for (int i = jj + 1; i < n; i++) {
                        work[workOff + i * ldwork + (jj - j)] = A[aOff + i * lda + jj];
                        A[aOff + i * lda + jj] = 0;
                    }
                }

                if (j + jb < n) {
                    BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, n, jb, n - j - jb, -1.0,
                            A, aOff + j + jb, lda,
                            work, workOff + (j + jb) * ldwork, ldwork,
                            1.0, A, aOff + j, lda);
                }
                BLAS.dtrsm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, BLAS.Diag.Unit, n, jb, 1.0,
                        work, workOff + j * ldwork, ldwork,
                        A, aOff + j, lda);
            }
        }

        for (int j = n - 2; j >= 0; j--) {
            int jp = ipiv[ipivOff + j];
            if (jp != j) {
                BLAS.dswap(n, A, aOff + j, lda, A, aOff + jp, lda);
            }
        }

        return true;
    }

    static void dgetrs(BLAS.Transpose trans, int n, int nrhs, double[] A, int aOff, int lda,
                       int[] ipiv, int ipivOff, double[] B, int bOff, int ldb) {
        if (n == 0 || nrhs == 0) return;

        if (trans == BLAS.Transpose.NoTrans) {
            BLAS.dlaswp(nrhs, B, bOff, ldb, 0, n - 1, ipiv, ipivOff, 1);
            BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, BLAS.Diag.Unit, n, nrhs, 1.0, A, aOff, lda, B, bOff, ldb);
            BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit, n, nrhs, 1.0, A, aOff, lda, B, bOff, ldb);
        } else {
            BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Transpose.Trans, BLAS.Diag.NonUnit, n, nrhs, 1.0, A, aOff, lda, B, bOff, ldb);
            BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Transpose.Trans, BLAS.Diag.Unit, n, nrhs, 1.0, A, aOff, lda, B, bOff, ldb);
            BLAS.dlaswp(nrhs, B, bOff, ldb, 0, n - 1, ipiv, ipivOff, -1);
        }
    }

    static double dgecon(char norm, int n, double[] A, int lda, double anorm,
                         double[] work, int[] iwork) {
        if (n == 0) return 1.0;
        if (anorm == 0) return 0.0;
        if (Double.isNaN(anorm)) return anorm;
        if (Double.isInfinite(anorm)) return 0.0;

        double smlnum = Dlamch.dlamch('S');
        double rcond = 0.0;
        double ainvnm = 0.0;
        boolean normin = false;
        boolean onenrm = norm == '1' || norm == 'O' || norm == 'o';
        int kase1 = onenrm ? 1 : 2;

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
            if (isave[2] == kase1) {
                sl = Dlatrs.dlatrs(BLAS.Uplo.Upper, BLAS.Transpose.Trans, BLAS.Diag.NonUnit, normin, n, A, lda, work, 0, work, 2 * n);
                su = Dlatrs.dlatrs(BLAS.Uplo.Lower, BLAS.Transpose.Trans, BLAS.Diag.Unit, normin, n, A, lda, work, 0, work, 3 * n);
            } else {
                su = Dlatrs.dlatrs(BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, BLAS.Diag.Unit, normin, n, A, lda, work, 0, work, 3 * n);
                sl = Dlatrs.dlatrs(BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit, normin, n, A, lda, work, 0, work, 2 * n);
            }

            double scale = sl * su;
            normin = true;

            if (scale != 1) {
                int ix = 0;
                double xmax = abs(work[0]);
                for (int i = 1; i < n; i++) {
                    if (abs(work[i]) > xmax) {
                        xmax = abs(work[i]);
                        ix = i;
                    }
                }
                if (scale == 0 || scale < abs(work[ix]) * smlnum) {
                    return rcond;
                }
                Drscl.drscl(n, scale, work, 0, 1);
            }
        }
    }
}
