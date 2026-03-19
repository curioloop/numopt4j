/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * LAPACK-style triangular solve.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DTRTRS - solves a triangular system of equations A * X = B or A^T * X = B.
 *
 * <p>A is a triangular matrix of order N, and B is an N-by-NRHS matrix.</p>
 */
interface Dtrtrs {

    /**
     * DTRTRS - solves a triangular system of the form A * X = B or A^T * X = B.
     *
     * @param uplo 'U' if A is upper triangular, 'L' if lower
     * @param trans 'N' for A*X=B, 'T' for A^T*X=B
     * @param diag 'U' if A is unit triangular, 'N' otherwise
     * @param n order of matrix A
     * @param nrhs number of right-hand sides (columns in B)
     * @param A triangular matrix A (n×n)
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param B right-hand side matrix (n×nrhs), overwritten with solution
     * @param bOff offset into B
     * @param ldb leading dimension of B
     * @return true if successful, false if singular
     */
    static boolean dtrtrs(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag, int n, int nrhs,
                          double[] A, int aOff, int lda,
                          double[] B, int bOff, int ldb) {
        boolean upper = (uplo == BLAS.Uplo.Upper);
        boolean transpose = (trans == BLAS.Trans.Trans || trans == BLAS.Trans.Conj);
        boolean unitDiag = (diag == BLAS.Diag.Unit);

        // Quick return for empty problems
        if (n == 0 || nrhs == 0) return true;

        // Check for zero diagonal elements
        if (!unitDiag) {
            for (int i = 0; i < n; i++) {
                if (A[aOff + i * lda + i] == 0.0) {
                    return false; // Singular
                }
            }
        }

        // Use a blocked Level-3 algorithm: solve diagonal blocks with DTRSM
        // and update the trailing matrix with DGEMM. This reduces the number
        // of vector updates and leverages the high-performance GEMM kernel.

        final int NB = Math.max(1, Math.min(Dgemm.BLOCK_M, n));
        BLAS.Diag diagEnum = unitDiag ? BLAS.Diag.Unit : BLAS.Diag.NonUnit;

        if (!transpose) {
            // Solve A * X = B
            if (upper) {
                // Back substitution: process blocks from bottom to top
                for (int ii = n; ii > 0; ii -= NB) {
                    int ib = Math.min(NB, ii);
                    int i0 = ii - ib;

                    // Solve diagonal block: A[i0:i0+ib-1, i0:i0+ib-1] * X[i0] = B[i0]
                    Dtrsm.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, diagEnum, ib, nrhs,
                                 1.0, A, aOff + i0 * lda + i0, lda,
                                 B, bOff + i0 * ldb, ldb);

                    // Update above blocks: B[0:i0-1,:] -= A[0:i0-1, i0:i0+ib-1] * X[i0]
                    if (i0 > 0) {
                        Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, i0, nrhs, ib,
                                     -1.0,
                                     A, aOff + i0, lda,
                                     B, bOff + i0 * ldb, ldb,
                                     1.0,
                                     B, bOff, ldb);
                    }
                }
            } else {
                // Lower: forward substitution
                for (int i0 = 0; i0 < n; i0 += NB) {
                    int ib = Math.min(NB, n - i0);

                    // Solve diagonal block
                    Dtrsm.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, diagEnum, ib, nrhs,
                                 1.0, A, aOff + i0 * lda + i0, lda,
                                 B, bOff + i0 * ldb, ldb);

                    // Update below blocks: B[i0+ib:n-1,:] -= A[i0+ib:n-1, i0:i0+ib-1] * X[i0]
                    int r0 = i0 + ib;
                    if (r0 < n) {
                        Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, n - r0, nrhs, ib,
                                     -1.0,
                                     A, aOff + r0 * lda + i0, lda,
                                     B, bOff + i0 * ldb, ldb,
                                     1.0,
                                     B, bOff + r0 * ldb, ldb);
                    }
                }
            }
        } else {
            // Solve A^T * X = B
            if (upper) {
                // A^T is lower: forward substitution (process top->bottom)
                for (int i0 = 0; i0 < n; i0 += NB) {
                    int ib = Math.min(NB, n - i0);

                    // Solve diagonal block: op(A_block) = A_block^T
                    Dtrsm.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Trans, diagEnum, ib, nrhs,
                                 1.0, A, aOff + i0 * lda + i0, lda,
                                 B, bOff + i0 * ldb, ldb);

                    // Update trailing blocks: B[i0+ib:n-1,:] -= A[i0, i0+ib:n-1]^T * X[i0]
                    int r0 = i0 + ib;
                    if (r0 < n) {
                        Dgemm.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, n - r0, nrhs, ib,
                                     -1.0,
                                     A, aOff + i0 * lda + r0, lda,
                                     B, bOff + i0 * ldb, ldb,
                                     1.0,
                                     B, bOff + r0 * ldb, ldb);
                    }
                }
            } else {
                // A^T is upper: back substitution (process bottom->top)
                for (int ii = n; ii > 0; ii -= NB) {
                    int ib = Math.min(NB, ii);
                    int i0 = ii - ib;

                    // Solve diagonal block: op(A_block) = A_block^T
                    Dtrsm.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans, diagEnum, ib, nrhs,
                                 1.0, A, aOff + i0 * lda + i0, lda,
                                 B, bOff + i0 * ldb, ldb);

                    // Update above blocks: B[0:i0-1,:] -= A[i0, 0:i0-1]^T * X[i0]
                    if (i0 > 0) {
                        Dgemm.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, i0, nrhs, ib,
                                     -1.0,
                                     A, aOff + i0 * lda, lda,
                                     B, bOff + i0 * ldb, ldb,
                                     1.0,
                                     B, bOff, ldb);
                    }
                }
            }
        }

        return true;
    }
}
