/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Reduces a symmetric matrix to tridiagonal form.
 * LAPACK DSYTRD algorithm.
 *
 * <p>Reduces a real symmetric matrix A to symmetric tridiagonal form T
 * by orthogonal similarity transformations:
 * A = Q * T * Q^T</p>
 *
 * <p>The matrix A is stored in row-major format. On output, the diagonal
 * elements are stored in d, the off-diagonal in e, and the Householder
 * scalars in tau.</p>
 */
interface Dsytrd {

    /**
     * Reduces a symmetric matrix to tridiagonal form.
     *
     * @param uplo  'L' for lower triangular, 'U' for upper triangular
     * @param n     the order of the matrix
     * @param A     the symmetric matrix (n x n, row-major), overwritten
     * @param lda   leading dimension of A
     * @param d     the diagonal elements (output, length n)
     * @param e     the off-diagonal elements (output, length n-1)
     * @param tau   the Householder scalars (output, length n-1)
     * @param work  workspace
     * @param lwork size of work
     */
    static void dsytrd(BLAS.Uplo uplo, int n, double[] A, int lda,
                       double[] d, int dOff, double[] e, int eOff, double[] tau, int tauOff,
                       double[] work, int lwork) {
        if (n <= 0) {
            return;
        }

        if (n == 1) {
            d[dOff] = A[0];
            return;
        }

        boolean upper = uplo == BLAS.Uplo.Upper;

        // Get optimal block size using ILAENV
        int nb = Ilaenv.ilaenv(1, "DSYTRD", String.valueOf(upper ? 'U' : 'L'), n, 0, 0, -1);
        
        // Calculate optimal workspace size
        int lworkopt = n * nb;
        
        // If lwork == -1, return optimal workspace size
        if (lwork == -1) {
            work[0] = lworkopt;
            return;
        }

        // Determine when to cross over from blocked to unblocked code
        int nx = n;
        int iws = 1;
        int ldwork = nb;
        
        if (1 < nb && nb < n) {
            nx = Math.max(nb, Ilaenv.ilaenv(3, "DSYTRD", String.valueOf(upper ? 'U' : 'L'), n, 0, 0, -1));
            if (nx < n) {
                ldwork = nb;
                iws = n * ldwork;
                if (lwork < iws) {
                    nb = Math.max(lwork / n, 1);
                    int nbmin = Ilaenv.ilaenv(2, "DSYTRD", String.valueOf(upper ? 'U' : 'L'), n, 0, 0, -1);
                    if (nb < nbmin) {
                        nx = n;
                    }
                }
            } else {
                nx = n;
            }
        } else {
            nb = 1;
        }
        ldwork = nb;

        if (upper) {
            int kk = n - ((n - nx + nb - 1) / nb) * nb;
            for (int i = n - nb; i >= kk; i -= nb) {
                BLAS.dlatrd(uplo, i + nb, nb, A, i * lda + i, lda, e, eOff, tau, tauOff, work, 0, ldwork);

                BLAS.dsyr2k(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, i, nb, -1.0,
                            A, i * lda + i, lda,
                            work, 0, ldwork,
                            1.0, A, 0, lda);

                for (int j = i; j < i + nb; j++) {
                    A[(j - 1) * lda + j] = e[eOff + j - 1];
                    d[dOff + j] = A[j * lda + j];
                }
            }
            Dsytd2.dsytd2(uplo, kk, A, 0, lda, d, dOff, e, eOff, tau, tauOff, work, 0);
        } else {
            int kk = n - ((n - nx + nb - 1) / nb) * nb;
            int i;
            for (i = 0; i < n - nx; i += nb) {
                BLAS.dlatrd(uplo, n - i, nb, A, i * lda + i, lda, e, eOff + i, tau, tauOff + i, work, 0, ldwork);

                int remainingRows = n - i - nb;
                BLAS.dsyr2k(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, remainingRows, nb, -1.0,
                            A, (i + nb) * lda + i, lda,
                            work, nb * ldwork, ldwork,
                            1.0, A, (i + nb) * lda + i + nb, lda);

                for (int j = i; j < i + nb; j++) {
                    A[(j + 1) * lda + j] = e[eOff + j];
                    d[dOff + j] = A[j * lda + j];
                }
            }
            Dsytd2.dsytd2(uplo, n - i, A, i * lda + i, lda, d, dOff + i, e, eOff + i, tau, tauOff + i, work, 0);
        }
        
        work[0] = iws;
    }

    static void dsytrd(BLAS.Uplo uplo, int n, double[] A, int lda,
                       double[] d, int dOff, double[] e, int eOff, double[] tau, int tauOff,
                       double[] work, int workOff, int lwork) {
        if (n <= 0) {
            return;
        }

        if (n == 1) {
            d[dOff] = A[0];
            return;
        }

        boolean upper = uplo == BLAS.Uplo.Upper;

        int nb = Ilaenv.ilaenv(1, "DSYTRD", String.valueOf(upper ? 'U' : 'L'), n, 0, 0, -1);
        int lworkopt = n * nb;
        
        if (lwork == -1) {
            work[workOff] = lworkopt;
            return;
        }

        int nx = n;
        int iws = 1;
        int ldwork = nb;
        
        if (1 < nb && nb < n) {
            nx = Math.max(nb, Ilaenv.ilaenv(3, "DSYTRD", String.valueOf(upper ? 'U' : 'L'), n, 0, 0, -1));
            if (nx < n) {
                ldwork = nb;
                iws = n * ldwork;
                if (lwork < iws) {
                    nb = Math.max(lwork / n, 1);
                    int nbmin = Ilaenv.ilaenv(2, "DSYTRD", String.valueOf(upper ? 'U' : 'L'), n, 0, 0, -1);
                    if (nb < nbmin) {
                        nx = n;
                    }
                }
            } else {
                nx = n;
            }
        } else {
            nb = 1;
        }
        ldwork = nb;

        if (upper) {
            int kk = n - ((n - nx + nb - 1) / nb) * nb;
            for (int i = n - nb; i >= kk; i -= nb) {
                BLAS.dlatrd(uplo, i + nb, nb, A, i * lda + i, lda, e, eOff, tau, tauOff, work, workOff, ldwork);

                BLAS.dsyr2k(BLAS.Uplo.Upper, BLAS.Trans.NoTrans, i, nb, -1.0,
                            A, i * lda + i, lda,
                            work, workOff, ldwork,
                            1.0, A, 0, lda);

                for (int j = i; j < i + nb; j++) {
                    A[(j - 1) * lda + j] = e[eOff + j - 1];
                    d[dOff + j] = A[j * lda + j];
                }
            }
            Dsytd2.dsytd2(uplo, kk, A, 0, lda, d, dOff, e, eOff, tau, tauOff, work, workOff);
        } else {
            int kk = n - ((n - nx + nb - 1) / nb) * nb;
            int i;
            for (i = 0; i < n - nx; i += nb) {
                BLAS.dlatrd(uplo, n - i, nb, A, i * lda + i, lda, e, eOff + i, tau, tauOff + i, work, workOff, ldwork);

                int remainingRows = n - i - nb;
                BLAS.dsyr2k(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, remainingRows, nb, -1.0,
                            A, (i + nb) * lda + i, lda,
                            work, workOff + nb * ldwork, ldwork,
                            1.0, A, (i + nb) * lda + i + nb, lda);

                for (int j = i; j < i + nb; j++) {
                    A[(j + 1) * lda + j] = e[eOff + j];
                    d[dOff + j] = A[j * lda + j];
                }
            }
            Dsytd2.dsytd2(uplo, n - i, A, i * lda + i, lda, d, dOff + i, e, eOff + i, tau, tauOff + i, work, workOff);
        }
        
        work[workOff] = iws;
    }
}
