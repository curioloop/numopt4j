/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DTRSM solves triangular matrix equations (in-place).
 *
 * <p>BLAS Level-3 operation: solves one of the matrix equations</p>
 * <ul>
 *   <li>op(A) * X = alpha * B</li>
 *   <li>X * op(A) = alpha * B</li>
 * </ul>
 * <p>where A is a triangular matrix, and X overwrites B.</p>
 *
 * <p>Reference: LAPACK/BLAS DTRSM</p>
 *
 * <h2>Storage</h2>
 * <p>Row-major storage is used throughout. A and B are row-major matrices.</p>
 */
public interface Dtrsm {

    /**
     * Solves a triangular matrix equation (in-place).
     *
     * <p>B is overwritten with the solution X. Uses only O(1) additional memory.</p>
     *
     * @param side 'L' for op(A) * X = alpha * B, 'R' for X * op(A) = alpha * B
     * @param uplo 'U' for upper triangular, 'L' for lower
     * @param trans 'N' for no transpose, 'T' for transpose
     * @param diag 'U' for unit diagonal, 'N' for non-unit
     * @param m number of rows in B
     * @param n number of columns in B
     * @param alpha scalar multiplier
     * @param A triangular matrix A (row-major, not modified)
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param B matrix B (row-major, overwritten with solution X)
     * @param bOff offset into B
     * @param ldb leading dimension of B
     */
    static void dtrsm(BLAS.Side side, BLAS.Uplo uplo, BLAS.Transpose trans, BLAS.Diag diag,
                      int m, int n, double alpha,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb) {

        // Normalize parameters
        boolean leftSide = side == BLAS.Side.Left;
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean transA = trans == BLAS.Transpose.Trans || trans == BLAS.Transpose.ConjTrans;
        boolean unitDiag = diag == BLAS.Diag.Unit;

        // Quick returns
        if (m <= 0 || n <= 0) {
            return;
        }

        if (alpha == 0.0) {
            // B := 0 — use Dlamv.dset which is optimized for fills
            for (int i = 0; i < m; i++) {
                int bRow = bOff + i * ldb;
                Dlamv.dset(n, 0.0, B, bRow, 1);
            }
            return;
        }

        // Scale B by alpha first — prefer Dscal which has an optimized path for xInc == 1
        if (alpha != 1.0) {
            for (int i = 0; i < m; i++) {
                int bRow = bOff + i * ldb;
                Dscal.dscal(n, alpha, B, bRow, 1);
            }
        }

        // Dispatch to specialized implementation
        if (leftSide) {
            // Solve A * X = B (X overwrites B, A is m x m)
            if (upper && !transA) {
                dtrsmLeftUpperNN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else if (!upper && !transA) {
                dtrsmLeftLowerNN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else if (upper && transA) {
                dtrsmLeftUpperTN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else {
                dtrsmLeftLowerTN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
            }
        } else {
            // Solve X * A = B (X overwrites B, A is n x n)
            if (upper && !transA) {
                dtrsmRightUpperNN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else if (!upper && !transA) {
                dtrsmRightLowerNN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else if (upper && transA) {
                dtrsmRightUpperTN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else {
                dtrsmRightLowerTN(m, n, A, aOff, lda, B, bOff, ldb, unitDiag);
            }
        }
    }

    // ==================== Left Side: Solve A * X = B ====================

    /**
     * Left Upper NoTranspose: Solve A * X = B where A is upper triangular.
     * Process rows from bottom to top (back substitution).
     */
    static void dtrsmLeftUpperNN(int m, int n,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        // Back substitution: process rows from bottom to top
        for (int i = m - 1; i >= 0; i--) {
            int aRow = aOff + i * lda;
            int bRow = bOff + i * ldb;

            // Subtract known contributions from rows below
            for (int k = i + 1; k < m; k++) {
                double aik = A[aRow + k];
                int bRowK = bOff + k * ldb;
                // B[bRow,:] -= aik * B[bRowK,:]
                Daxpy.daxpy(n, -aik, B, bRowK, 1, B, bRow, 1);
            }

            // Divide by diagonal
            if (!unitDiag) {
                Dscal.dscal(n, 1.0 / A[aRow + i], B, bRow, 1);
            }
        }
    }

    /**
     * Left Lower NoTranspose: Solve A * X = B where A is lower triangular.
     * Process rows from top to bottom (forward substitution).
     */
    static void dtrsmLeftLowerNN(int m, int n,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        // Forward substitution: process rows from top to bottom
        for (int i = 0; i < m; i++) {
            int aRow = aOff + i * lda;
            int bRow = bOff + i * ldb;

            // Subtract known contributions from rows above
            for (int k = 0; k < i; k++) {
                double aik = A[aRow + k];
                int bRowK = bOff + k * ldb;
                // B[bRow,:] -= aik * B[bRowK,:]
                Daxpy.daxpy(n, -aik, B, bRowK, 1, B, bRow, 1);
            }

            // Divide by diagonal
            if (!unitDiag) {
                Dscal.dscal(n, 1.0 / A[aRow + i], B, bRow, 1);
            }
        }
    }

    /**
     * Left Upper Transpose: Solve A^T * X = B where A is upper triangular.
     * Process rows from top to bottom.
     */
    static void dtrsmLeftUpperTN(int m, int n,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        // For A^T * X = B with A upper:
        // Row 0: A[0,0]*X[0] + A[0,1]*X[1] + ... = B[0]
        // => X[0] = (B[0] - A[0,1]*X[1] - ...) / A[0,0]
        // Process from top to bottom
        for (int i = 0; i < m; i++) {
            int aCol = aOff + i;  // Column i of A (stride lda)
            int bRow = bOff + i * ldb;

            // Subtract known contributions
            for (int k = 0; k < i; k++) {
                // A[k,i] is at A[k*lda + i]
                double aki = A[aOff + k * lda + i];
                int bRowK = bOff + k * ldb;
                // B[bRow,:] -= aki * B[bRowK,:]
                Daxpy.daxpy(n, -aki, B, bRowK, 1, B, bRow, 1);
            }

            // Divide by diagonal
            if (!unitDiag) {
                Dscal.dscal(n, 1.0 / A[aOff + i * lda + i], B, bRow, 1);
            }
        }
    }

    /**
     * Left Lower Transpose: Solve A^T * X = B where A is lower triangular.
     * Process rows from bottom to top.
     */
    static void dtrsmLeftLowerTN(int m, int n,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        // For A^T * X = B with A lower:
        // Process from bottom to top
        for (int i = m - 1; i >= 0; i--) {
            int bRow = bOff + i * ldb;

            // Subtract known contributions
            for (int k = i + 1; k < m; k++) {
                // A[k,i] is at A[k*lda + i]
                double aki = A[aOff + k * lda + i];
                int bRowK = bOff + k * ldb;
                // B[bRow,:] -= aki * B[bRowK,:]
                Daxpy.daxpy(n, -aki, B, bRowK, 1, B, bRow, 1);
            }

            // Divide by diagonal
            if (!unitDiag) {
                Dscal.dscal(n, 1.0 / A[aOff + i * lda + i], B, bRow, 1);
            }
        }
    }

    // ==================== Right Side: Solve X * A = B ====================

    /**
     * Right Upper NoTranspose: Solve X * A = B where A is upper triangular.
     * Process columns from left to right (forward substitution).
     * 
     * X * A = B where A is n x n upper triangular
     * X[i,j] = sum_{k=0}^{j} X[i,k] * A[k,j]
     * 
     * Algorithm: for k = 0 to n-1:
     *   X[i,k] = B[i,k] / A[k,k]
     *   for j = k+1 to n-1: X[i,j] -= X[i,k] * A[k,j]
     */
    static void dtrsmRightUpperNN(int m, int n,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb;
            for (int k = 0; k < n; k++) {
                double bk = B[bRow + k];
                if (bk == 0) continue;
                if (!unitDiag) {
                    B[bRow + k] = bk / A[aOff + k * lda + k];
                    bk = B[bRow + k];
                }
                for (int j = k + 1; j < n; j++) {
                    B[bRow + j] -= bk * A[aOff + k * lda + j];
                }
            }
        }
    }

    /**
     * Right Lower NoTranspose: Solve X * A = B where A is lower triangular.
     * Process columns from right to left.
     * 
     * X * A = B where A is n x n lower triangular
     * X[i,k] = (B[i,k] - sum_{j=0}^{k-1} X[i,j] * A[j,k]) / A[k,k]
     * 
     * Algorithm: for k = n-1 to 0:
     *   X[i,k] = B[i,k] / A[k,k]
     *   for j = 0 to k-1: X[i,j] -= X[i,k] * A[k,j]
     */
    static void dtrsmRightLowerNN(int m, int n,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb;
            for (int k = n - 1; k >= 0; k--) {
                double bk = B[bRow + k];
                if (bk == 0) continue;
                if (!unitDiag) {
                    B[bRow + k] = bk / A[aOff + k * lda + k];
                    bk = B[bRow + k];
                }
                for (int j = 0; j < k; j++) {
                    B[bRow + j] -= bk * A[aOff + k * lda + j];
                }
            }
        }
    }

    /**
     * Right Upper Transpose: Solve X * A^T = B where A is upper triangular.
     * Process columns from right to left.
     */
    static void dtrsmRightUpperTN(int m, int n,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb;
            for (int j = n - 1; j >= 0; j--) {
                double sum = B[bRow + j];
                for (int k = j + 1; k < n; k++) {
                    sum -= A[aOff + j * lda + k] * B[bRow + k];
                }
                if (!unitDiag) {
                    sum /= A[aOff + j * lda + j];
                }
                B[bRow + j] = sum;
            }
        }
    }

    /**
     * Right Lower Transpose: Solve X * A^T = B where A is lower triangular.
     * Process columns from left to right.
     */
    static void dtrsmRightLowerTN(int m, int n,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb;
            for (int j = 0; j < n; j++) {
                double sum = B[bRow + j];
                for (int k = 0; k < j; k++) {
                    sum -= A[aOff + j * lda + k] * B[bRow + k];
                }
                if (!unitDiag) {
                    sum /= A[aOff + j * lda + j];
                }
                B[bRow + j] = sum;
            }
        }
    }
}
