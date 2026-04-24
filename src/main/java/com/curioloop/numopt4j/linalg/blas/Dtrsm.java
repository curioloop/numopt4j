/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

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

    int ROW_UPDATE_SIMD_MIN_N = 32;

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
    static void dtrsm(BLAS.Side side, BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
                      int m, int n, double alpha,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb) {

        // Normalize parameters
        boolean leftSide = side == BLAS.Side.Left;
        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean transA = trans == BLAS.Trans.Trans || trans == BLAS.Trans.Conj;
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
            int k = i + 1;
            for (; k + 1 < m; k += 2) {
                double aik0 = -A[aRow + k];
                double aik1 = -A[aRow + k + 1];
                if (aik0 != 0.0 || aik1 != 0.0) {
                    dtrsmRowUpdate2(n,
                        aik0, B, bOff + k * ldb,
                        aik1, B, bOff + (k + 1) * ldb,
                        B, bRow);
                }
            }
            for (; k < m; k++) {
                double aik = A[aRow + k];
                if (aik != 0.0) {
                    dtrsmRowUpdate1(n, -aik, B, bOff + k * ldb, B, bRow);
                }
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
            int k = 0;
            int k2 = (i / 2) * 2;
            for (; k < k2; k += 2) {
                double aik0 = -A[aRow + k];
                double aik1 = -A[aRow + k + 1];
                if (aik0 != 0.0 || aik1 != 0.0) {
                    dtrsmRowUpdate2(n,
                        aik0, B, bOff + k * ldb,
                        aik1, B, bOff + (k + 1) * ldb,
                        B, bRow);
                }
            }
            for (; k < i; k++) {
                double aik = A[aRow + k];
                if (aik != 0.0) {
                    dtrsmRowUpdate1(n, -aik, B, bOff + k * ldb, B, bRow);
                }
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
            int bRow = bOff + i * ldb;

            // Subtract known contributions
            int k = 0;
            int k2 = (i / 2) * 2;
            for (; k < k2; k += 2) {
                double aki0 = -A[aOff + k * lda + i];
                double aki1 = -A[aOff + (k + 1) * lda + i];
                if (aki0 != 0.0 || aki1 != 0.0) {
                    dtrsmRowUpdate2(n,
                        aki0, B, bOff + k * ldb,
                        aki1, B, bOff + (k + 1) * ldb,
                        B, bRow);
                }
            }
            for (; k < i; k++) {
                double aki = A[aOff + k * lda + i];
                if (aki != 0.0) {
                    dtrsmRowUpdate1(n, -aki, B, bOff + k * ldb, B, bRow);
                }
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
            int k = i + 1;
            for (; k + 1 < m; k += 2) {
                double aki0 = -A[aOff + k * lda + i];
                double aki1 = -A[aOff + (k + 1) * lda + i];
                if (aki0 != 0.0 || aki1 != 0.0) {
                    dtrsmRowUpdate2(n,
                        aki0, B, bOff + k * ldb,
                        aki1, B, bOff + (k + 1) * ldb,
                        B, bRow);
                }
            }
            for (; k < m; k++) {
                double aki = A[aOff + k * lda + i];
                if (aki != 0.0) {
                    dtrsmRowUpdate1(n, -aki, B, bOff + k * ldb, B, bRow);
                }
            }

            // Divide by diagonal
            if (!unitDiag) {
                Dscal.dscal(n, 1.0 / A[aOff + i * lda + i], B, bRow, 1);
            }
        }
    }

    static void dtrsmRowUpdate1(int n,
                                double alpha, double[] x, int xOff,
                                double[] y, int yOff) {
        if (n <= 0 || alpha == 0.0) {
            return;
        }

        if (isSIMDRowUpdateCandidate(n) && DtrsmRowUpdateSIMD.dtrsmRowUpdate1(n, alpha, x, xOff, y, yOff)) {
            return;
        }

        int j = 0;
        for (; j + 3 < n; j += 4) {
            y[yOff + j] = Math.fma(alpha, x[xOff + j], y[yOff + j]);
            y[yOff + j + 1] = Math.fma(alpha, x[xOff + j + 1], y[yOff + j + 1]);
            y[yOff + j + 2] = Math.fma(alpha, x[xOff + j + 2], y[yOff + j + 2]);
            y[yOff + j + 3] = Math.fma(alpha, x[xOff + j + 3], y[yOff + j + 3]);
        }
        for (; j < n; j++) {
            y[yOff + j] = Math.fma(alpha, x[xOff + j], y[yOff + j]);
        }
    }

    static void dtrsmRowUpdate2(int n,
                                double alpha0, double[] x0, int x0Off,
                                double alpha1, double[] x1, int x1Off,
                                double[] y, int yOff) {
        if (n <= 0 || (alpha0 == 0.0 && alpha1 == 0.0)) {
            return;
        }
        if (alpha0 == 0.0) {
            dtrsmRowUpdate1(n, alpha1, x1, x1Off, y, yOff);
            return;
        }
        if (alpha1 == 0.0) {
            dtrsmRowUpdate1(n, alpha0, x0, x0Off, y, yOff);
            return;
        }
        // Two-row fused updates stayed ahead on the accepted skinny-RHS replay shapes, so keep the pair kernel inline.
        if (isSIMDRowUpdateCandidate(n) && DtrsmRowUpdateSIMD.dtrsmRowUpdate2(n, alpha0, x0, x0Off, alpha1, x1, x1Off, y, yOff)) {
            return;
        }

        int j = 0;
        for (; j + 3 < n; j += 4) {
            y[yOff + j] = Math.fma(alpha0, x0[x0Off + j], Math.fma(alpha1, x1[x1Off + j], y[yOff + j]));
            y[yOff + j + 1] = Math.fma(alpha0, x0[x0Off + j + 1], Math.fma(alpha1, x1[x1Off + j + 1], y[yOff + j + 1]));
            y[yOff + j + 2] = Math.fma(alpha0, x0[x0Off + j + 2], Math.fma(alpha1, x1[x1Off + j + 2], y[yOff + j + 2]));
            y[yOff + j + 3] = Math.fma(alpha0, x0[x0Off + j + 3], Math.fma(alpha1, x1[x1Off + j + 3], y[yOff + j + 3]));
        }
        for (; j < n; j++) {
            y[yOff + j] = Math.fma(alpha0, x0[x0Off + j], Math.fma(alpha1, x1[x1Off + j], y[yOff + j]));
        }
    }

    private static boolean isSIMDRowUpdateCandidate(int n) {
        return n >= ROW_UPDATE_SIMD_MIN_N && SIMD.supportDaxpy();
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
                int tail = n - k - 1;
                if (tail > 0) {
                    if (tail >= ROW_UPDATE_SIMD_MIN_N) {
                        dtrsmRowUpdate1(tail, -bk, A, aOff + k * lda + k + 1, B, bRow + k + 1);
                    } else {
                        for (int j = k + 1; j < n; j++) {
                            B[bRow + j] = Math.fma(-bk, A[aOff + k * lda + j], B[bRow + j]);
                        }
                    }
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
                if (k > 0) {
                    if (k >= ROW_UPDATE_SIMD_MIN_N) {
                        dtrsmRowUpdate1(k, -bk, A, aOff + k * lda, B, bRow);
                    } else {
                        for (int j = 0; j < k; j++) {
                            B[bRow + j] = Math.fma(-bk, A[aOff + k * lda + j], B[bRow + j]);
                        }
                    }
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
                int aRow = aOff + j * lda;
                double sum0 = B[bRow + j];
                double sum1 = 0.0;
                double sum2 = 0.0;
                double sum3 = 0.0;
                int k = j + 1;
                for (; k + 3 < n; k += 4) {
                    sum0 = Math.fma(-A[aRow + k], B[bRow + k], sum0);
                    sum1 = Math.fma(-A[aRow + k + 1], B[bRow + k + 1], sum1);
                    sum2 = Math.fma(-A[aRow + k + 2], B[bRow + k + 2], sum2);
                    sum3 = Math.fma(-A[aRow + k + 3], B[bRow + k + 3], sum3);
                }
                double sum = (sum0 + sum1) + (sum2 + sum3);
                for (; k < n; k++) {
                    sum = Math.fma(-A[aRow + k], B[bRow + k], sum);
                }
                if (!unitDiag) {
                    sum /= A[aRow + j];
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
                    sum = Math.fma(-A[aOff + j * lda + k], B[bRow + k], sum);
                }
                if (!unitDiag) {
                    sum /= A[aOff + j * lda + j];
                }
                B[bRow + j] = sum;
            }
        }
    }
}

final class DtrsmRowUpdateSIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;

    private DtrsmRowUpdateSIMD() {
    }

    static boolean dtrsmRowUpdate1(int n,
                                   double alpha, double[] x, int xOff,
                                   double[] y, int yOff) {
        if (LANES <= 1) {
            return false;
        }

        DoubleVector alphaVec = DoubleVector.broadcast(SPECIES, alpha);
        int limit = SPECIES.loopBound(n);
        int j = 0;
        for (; j < limit; j += LANES) {
            DoubleVector xVec = DoubleVector.fromArray(SPECIES, x, xOff + j);
            DoubleVector yVec = DoubleVector.fromArray(SPECIES, y, yOff + j);
            alphaVec.fma(xVec, yVec).intoArray(y, yOff + j);
        }
        for (; j < n; j++) {
            y[yOff + j] = Math.fma(alpha, x[xOff + j], y[yOff + j]);
        }
        return true;
    }

    static boolean dtrsmRowUpdate2(int n,
                                   double alpha0, double[] x0, int x0Off,
                                   double alpha1, double[] x1, int x1Off,
                                   double[] y, int yOff) {
        if (LANES <= 1) {
            return false;
        }

        DoubleVector alpha0Vec = DoubleVector.broadcast(SPECIES, alpha0);
        DoubleVector alpha1Vec = DoubleVector.broadcast(SPECIES, alpha1);
        int limit = SPECIES.loopBound(n);
        int j = 0;
        for (; j < limit; j += LANES) {
            DoubleVector x0Vec = DoubleVector.fromArray(SPECIES, x0, x0Off + j);
            DoubleVector x1Vec = DoubleVector.fromArray(SPECIES, x1, x1Off + j);
            DoubleVector yVec = DoubleVector.fromArray(SPECIES, y, yOff + j);
            alpha0Vec.fma(x0Vec, alpha1Vec.fma(x1Vec, yVec)).intoArray(y, yOff + j);
        }
        for (; j < n; j++) {
            y[yOff + j] = Math.fma(alpha0, x0[x0Off + j], Math.fma(alpha1, x1[x1Off + j], y[yOff + j]));
        }
        return true;
    }
}
