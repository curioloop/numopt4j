/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DTRMM performs triangular matrix-matrix multiplication (in-place).
 * 
 * <p>BLAS Level-3 operation: B := alpha*op(A)*B or B := alpha*B*op(A)</p>
 * <p>where A is a triangular matrix. B is overwritten with the result.</p>
 * 
 * <p><b>Implementation Note:</b> In-place algorithm using O(1) extra space per row/column.
 * Uses the fact that triangular matrices allow sequential computation without
 * overwriting values that are still needed.</p>
 * 
 * <p>Reference: LAPACK/BLAS DTRMM</p>
 * 
 * <h2>Cache Blocking</h2>
 * <p>Uses asymmetric blocking for optimal L1 cache utilization:</p>
 * <ul>
 *   <li>BLOCK_M = BLOCK_N = 64 (for C block)</li>
 *   <li>BLOCK_K = 128 (for A block - streamed)</li>
 * </ul>
 */
public interface Dtrmm {

    /**
     * Performs triangular matrix-matrix multiplication (in-place).
     *
     * <p>B is overwritten with the result. Uses only O(1) additional memory.</p>
     *
     * @param side 'L' for A*B, 'R' for B*A
     * @param uplo 'U' for upper triangular, 'L' for lower
     * @param trans 'N' for no transpose, 'T' for transpose
     * @param diag 'U' for unit diagonal, 'N' for non-unit
     * @param m number of rows in B
     * @param n number of columns in B
     * @param alpha scalar multiplier
     * @param A triangular matrix A (row-major, not modified)
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param B matrix B (row-major, overwritten with result)
     * @param bOff offset into B
     * @param ldb leading dimension of B
     */
    static void dtrmm(BLAS.Side side, BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag,
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
            // B := 0
            for (int i = 0; i < m; i++) {
                int bRow = bOff + i * ldb;
                for (int j = 0; j < n; j++) {
                    B[bRow + j] = 0.0;
                }
            }
            return;
        }

        // Dispatch to specialized implementation
        if (leftSide) {
            // B := alpha * A * B (A is m x m)
            if (upper && !transA) {
                dtrmmLeftUpperNN(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else if (!upper && !transA) {
                dtrmmLeftLowerNN(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else if (upper && transA) {
                dtrmmLeftUpperTN(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else {
                dtrmmLeftLowerTN(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
            }
        } else {
            // B := alpha * B * A (A is n x n)
            if (upper && !transA) {
                dtrmmRightUpperNN(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else if (!upper && !transA) {
                dtrmmRightLowerNN(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else if (upper && transA) {
                dtrmmRightUpperTN(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
            } else {
                dtrmmRightLowerTN(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
            }
        }
    }

    // ==================== Left Side: B := alpha * A * B ====================

    /**
     * Left Upper NoTranspose: B := alpha * A * B
     * 
     * For upper triangular A: Result[i] depends on B[i], B[i+1], ..., B[m-1]
     * Uses 4x4 micro-kernel for register optimization.
     * 
     * <p>Key optimization: 4x4 micro-kernel with 16 accumulators,
     * alpha applied at write-back only.</p>
     */
    static void dtrmmLeftUpperNN(int m, int n, double alpha,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        
        // Use 4x4 blocked kernel for medium/large matrices
        if (m >= 4 && n >= 4) {
            dtrmmLeftUpperNNBlocked(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        } else {
            // Direct for small matrices
            dtrmmLeftUpperNNDirect(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        }
    }

    /**
     * Left Upper NN: B := alpha * A * B  (loop-interchange optimized)
     *
     * <p>Uses k-outer column-update order (scatter) instead of i-j-k (gather):
     * For each pivot row k (processed top-to-bottom), first scatter A[i,k]*B[k,:]
     * into B[i,:] for all i < k (reads original B[k,:] before it is scaled),
     * then scale B[k,:] by its own diagonal.
     * Inner j-loop accesses B[i,:] and B[k,:] sequentially (row-contiguous).</p>
     */
    static void dtrmmLeftUpperNNBlocked(int m, int n, double alpha,
                                        double[] A, int aOff, int lda,
                                        double[] B, int bOff, int ldb,
                                        boolean unitDiag) {
        int n4 = (n / 4) * 4;
        // Process k from 0 to m-1: scatter off-diagonal first, then scale diagonal
        for (int k = 0; k < m; k++) {
            int bRowK = bOff + k * ldb;
            // Scatter: update rows i < k using original B[k,:]
            for (int i = 0; i < k; i++) {
                double aik = alpha * A[aOff + i * lda + k];
                if (aik == 0.0) continue;
                int bRowI = bOff + i * ldb;
                int j = 0;
                for (; j < n4; j += 4) {
                    B[bRowI + j]     = Math.fma(aik, B[bRowK + j],     B[bRowI + j]);
                    B[bRowI + j + 1] = Math.fma(aik, B[bRowK + j + 1], B[bRowI + j + 1]);
                    B[bRowI + j + 2] = Math.fma(aik, B[bRowK + j + 2], B[bRowI + j + 2]);
                    B[bRowI + j + 3] = Math.fma(aik, B[bRowK + j + 3], B[bRowI + j + 3]);
                }
                for (; j < n; j++) B[bRowI + j] = Math.fma(aik, B[bRowK + j], B[bRowI + j]);
            }
            // Scale diagonal row k
            double diag = unitDiag ? alpha : alpha * A[aOff + k * lda + k];
            int j = 0;
            for (; j < n4; j += 4) {
                B[bRowK + j]     *= diag;
                B[bRowK + j + 1] *= diag;
                B[bRowK + j + 2] *= diag;
                B[bRowK + j + 3] *= diag;
            }
            for (; j < n; j++) B[bRowK + j] *= diag;
        }
    }

    /**
     * Direct for small matrices (no 4x4 blocking)
     */
    static void dtrmmLeftUpperNNDirect(int m, int n, double alpha,
                                       double[] A, int aOff, int lda,
                                       double[] B, int bOff, int ldb,
                                       boolean unitDiag) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                double sum = unitDiag ? B[bOff + i * ldb + j] : A[aOff + i * lda + i] * B[bOff + i * ldb + j];
                for (int k = i + 1; k < m; k++) {
                    sum = Math.fma(A[aOff + i * lda + k], B[bOff + k * ldb + j], sum);
                }
                B[bOff + i * ldb + j] = alpha * sum;
            }
        }
    }

    /**
     * Left Lower NoTranspose: B := alpha * A * B
     * 
     * For lower triangular A: Result[i] depends on B[0], B[1], ..., B[i]
     * Uses 4x4 micro-kernel for register optimization.
     * 
     * <p>Key optimization: 4x4 micro-kernel with 16 accumulators,
     * alpha applied at write-back only.</p>
     */
    static void dtrmmLeftLowerNN(int m, int n, double alpha,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        
        // Use 4x4 blocked kernel for medium/large matrices
        if (m >= 4 && n >= 4) {
            dtrmmLeftLowerNNBlocked(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        } else {
            // Direct for small matrices
            dtrmmLeftLowerNNDirect(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        }
    }

    /**
     * Left Lower NN: B := alpha * A * B  (loop-interchange optimized)
     *
     * <p>Uses k-outer column-update order (scatter) instead of i-j-k (gather):
     * Process k from m-1 down to 0. For each k:
     *   1. Scatter A[i,k]*B[k,:] into B[i,:] for all i > k (B[k,:] still original).
     *   2. Scale B[k,:] by its diagonal element.
     * Inner j-loop accesses B[i,:] and B[k,:] sequentially (row-contiguous).</p>
     */
    static void dtrmmLeftLowerNNBlocked(int m, int n, double alpha,
                                        double[] A, int aOff, int lda,
                                        double[] B, int bOff, int ldb,
                                        boolean unitDiag) {
        int n4 = (n / 4) * 4;
        for (int k = m - 1; k >= 0; k--) {
            int bRowK = bOff + k * ldb;
            // Step 1: scatter off-diagonal into rows i > k (B[k,:] still original here)
            for (int i = k + 1; i < m; i++) {
                double aik = alpha * A[aOff + i * lda + k];
                if (aik == 0.0) continue;
                int bRowI = bOff + i * ldb;
                int j = 0;
                for (; j < n4; j += 4) {
                    B[bRowI + j]     = Math.fma(aik, B[bRowK + j],     B[bRowI + j]);
                    B[bRowI + j + 1] = Math.fma(aik, B[bRowK + j + 1], B[bRowI + j + 1]);
                    B[bRowI + j + 2] = Math.fma(aik, B[bRowK + j + 2], B[bRowI + j + 2]);
                    B[bRowI + j + 3] = Math.fma(aik, B[bRowK + j + 3], B[bRowI + j + 3]);
                }
                for (; j < n; j++) B[bRowI + j] = Math.fma(aik, B[bRowK + j], B[bRowI + j]);
            }
            // Step 2: scale diagonal row k
            double diag = unitDiag ? alpha : alpha * A[aOff + k * lda + k];
            int j = 0;
            for (; j < n4; j += 4) {
                B[bRowK + j]     *= diag;
                B[bRowK + j + 1] *= diag;
                B[bRowK + j + 2] *= diag;
                B[bRowK + j + 3] *= diag;
            }
            for (; j < n; j++) B[bRowK + j] *= diag;
        }
    }

    /**
     * Direct for small matrices (no 4x4 blocking)
     */
    static void dtrmmLeftLowerNNDirect(int m, int n, double alpha,
                                        double[] A, int aOff, int lda,
                                        double[] B, int bOff, int ldb,
                                        boolean unitDiag) {
        // Process rows from bottom to top to allow in-place computation
        for (int i = m - 1; i >= 0; i--) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                // For non-unit: include diagonal in loop (k=0 to i)
                // For unit: exclude diagonal from loop (k=0 to i-1)
                int kEnd = unitDiag ? i : i + 1;
                for (int k = 0; k < kEnd; k++) {
                    sum = Math.fma(A[aOff + i * lda + k], B[bOff + k * ldb + j], sum);
                }
                if (unitDiag) {
                    // For unit diagonal, add B[i,j] (diagonal = 1)
                    sum += B[bOff + i * ldb + j];
                }
                // For non-unit, diagonal is already included in k-loop
                B[bOff + i * ldb + j] = alpha * sum;
            }
        }
    }

    /**
     * Left Upper Transpose: B := alpha * A^T * B
     * 
     * For upper triangular A: Result[i] depends on A[0..i, i] * B[0..i]
     * Uses 4x4 micro-kernel for register optimization.
     * 
     * <p>Key optimization: 4x4 micro-kernel with 16 accumulators,
     * alpha applied at write-back only.</p>
     */
    static void dtrmmLeftUpperTN(int m, int n, double alpha,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        
        // Use 4x4 blocked kernel for medium/large matrices
        if (m >= 4 && n >= 4) {
            dtrmmLeftUpperTNBlocked(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        } else {
            // Direct for small matrices
            dtrmmLeftUpperTNDirect(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        }
    }

    /**
     * 4x4 blocked kernel for Left Upper TN
     * B := alpha * A^T * B where A is upper triangular
     * Process rows from bottom to top (required for in-place)
     */
    static void dtrmmLeftUpperTNBlocked(int m, int n, double alpha,
                                             double[] A, int aOff, int lda,
                                             double[] B, int bOff, int ldb,
                                             boolean unitDiag) {
        
        int m4 = (m / 4) * 4;
        int n4 = (n / 4) * 4;
        
        // Process rows from bottom to top (required for upper triangular transpose in-place)
        for (int i = m - 1; i >= 0; i--) {
            // For each row, process column blocks
            for (int jj = 0; jj < n4; jj += 4) {
                for (int j = jj; j < jj + 4 && j < n; j++) {
                    double sum = 0.0;
                    // Upper triangular transpose: row i depends on rows i to m-1
                    // B[i,j] = sum(k=i to m-1) A[k,i] * B[k,j]
                    int kStart = unitDiag ? i + 1 : i;
                    for (int k = kStart; k < m; k++) {
                        sum = Math.fma(A[aOff + k * lda + i], B[bOff + k * ldb + j], sum);
                    }
                    if (unitDiag) {
                        sum += B[bOff + i * ldb + j];
                    }
                    B[bOff + i * ldb + j] = alpha * sum;
                }
            }
        }
        
        // Handle remaining columns (n > n4)
        if (n > n4) {
            for (int i = m - 1; i >= 0; i--) {
                for (int j = n4; j < n; j++) {
                    double sum = 0.0;
                    int kStart = unitDiag ? i + 1 : i;
                    for (int k = kStart; k < m; k++) {
                        sum = Math.fma(A[aOff + k * lda + i], B[bOff + k * ldb + j], sum);
                    }
                    if (unitDiag) {
                        sum += B[bOff + i * ldb + j];
                    }
                    B[bOff + i * ldb + j] = alpha * sum;
                }
            }
        }
    }

    /**
     * Direct computation for small matrices (Left Upper TN)
     */
    static void dtrmmLeftUpperTNDirect(int m, int n, double alpha,
                                       double[] A, int aOff, int lda,
                                       double[] B, int bOff, int ldb,
                                       boolean unitDiag) {
        // Process rows from bottom to top for in-place computation
        for (int i = m - 1; i >= 0; i--) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                // For non-unit: include diagonal in loop (k=i to m-1)
                // For unit: exclude diagonal from loop (k=i+1 to m-1)
                int kStart = unitDiag ? i + 1 : i;
                for (int k = kStart; k < m; k++) {
                    sum = Math.fma(A[aOff + k * lda + i], B[bOff + k * ldb + j], sum);
                }
                if (unitDiag) {
                    // For unit diagonal, add B[i,j] (diagonal = 1)
                    sum += B[bOff + i * ldb + j];
                }
                // For non-unit, diagonal is already included in k-loop
                B[bOff + i * ldb + j] = alpha * sum;
            }
        }
    }

    /**
     * Left Lower Transpose: B := alpha * A^T * B
     * 
     * For lower triangular A: Result[i] depends on A[i..m-1, i] * B[i..m-1]
     * Uses 4x4 micro-kernel for register optimization.
     * 
     * <p>Key optimization: 4x4 micro-kernel with 16 accumulators,
     * alpha applied at write-back only.</p>
     */
    static void dtrmmLeftLowerTN(int m, int n, double alpha,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 boolean unitDiag) {
        
        // Use 4x4 blocked kernel for medium/large matrices
        if (m >= 4 && n >= 4) {
            dtrmmLeftLowerTNBlocked(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        } else {
            // Direct for small matrices
            dtrmmLeftLowerTNDirect(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        }
    }

    /**
     * 4x4 blocked kernel for Left Lower TN
     * B := alpha * A^T * B where A is lower triangular
     * Process rows from top to bottom (required for in-place)
     */
    static void dtrmmLeftLowerTNBlocked(int m, int n, double alpha,
                                             double[] A, int aOff, int lda,
                                             double[] B, int bOff, int ldb,
                                             boolean unitDiag) {
        
        int m4 = (m / 4) * 4;
        int n4 = (n / 4) * 4;
        
        // Process rows from top to bottom (required for lower triangular transpose in-place)
        for (int i = 0; i < m; i++) {
            // For each row, process column blocks
            for (int jj = 0; jj < n4; jj += 4) {
                for (int j = jj; j < jj + 4 && j < n; j++) {
                    double sum = 0.0;
                    // Lower triangular transpose: row i depends on rows 0 to i
                    // B[i,j] = sum(k=0 to i) A[k,i] * B[k,j]
                    int kEnd = unitDiag ? i : i + 1;
                    for (int k = 0; k < kEnd; k++) {
                        sum = Math.fma(A[aOff + k * lda + i], B[bOff + k * ldb + j], sum);
                    }
                    if (unitDiag) {
                        sum += B[bOff + i * ldb + j];
                    }
                    B[bOff + i * ldb + j] = alpha * sum;
                }
            }
        }
        
        // Handle remaining columns (n > n4)
        if (n > n4) {
            for (int i = 0; i < m; i++) {
                for (int j = n4; j < n; j++) {
                    double sum = 0.0;
                    int kEnd = unitDiag ? i : i + 1;
                    for (int k = 0; k < kEnd; k++) {
                        sum = Math.fma(A[aOff + k * lda + i], B[bOff + k * ldb + j], sum);
                    }
                    if (unitDiag) {
                        sum += B[bOff + i * ldb + j];
                    }
                    B[bOff + i * ldb + j] = alpha * sum;
                }
            }
        }
    }

    /**
     * Direct computation for small matrices (Left Lower TN)
     */
    static void dtrmmLeftLowerTNDirect(int m, int n, double alpha,
                                       double[] A, int aOff, int lda,
                                       double[] B, int bOff, int ldb,
                                       boolean unitDiag) {
        // Process rows from top to bottom for in-place computation
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                // For non-unit: include diagonal in loop (k=0 to i)
                // For unit: exclude diagonal from loop (k=0 to i-1)
                int kEnd = unitDiag ? i : i + 1;
                for (int k = 0; k < kEnd; k++) {
                    sum = Math.fma(A[aOff + k * lda + i], B[bOff + k * ldb + j], sum);
                }
                if (unitDiag) {
                    // For unit diagonal, add B[i,j] (diagonal = 1)
                    sum += B[bOff + i * ldb + j];
                }
                // For non-unit, diagonal is already included in k-loop
                B[bOff + i * ldb + j] = alpha * sum;
            }
        }
    }

    // ==================== Right Side: B := alpha * B * A ====================

    /**
     * Right Upper NoTranspose: B := alpha * B * A
     * 
     * For upper triangular A: Result[j] depends on B[0], B[1], ..., B[j]
     * Uses 4x4 micro-kernel for register optimization.
     * 
     * <p>Key optimization: 4x4 micro-kernel with 16 accumulators,
     * alpha applied at write-back only.</p>
     */
    static void dtrmmRightUpperNN(int m, int n, double alpha,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        
        // Use 4x4 blocked kernel for medium/large matrices
        if (m >= 4 && n >= 4) {
            dtrmmRightUpperNNBlocked(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        } else {
            // Direct for small matrices
            dtrmmRightUpperNNDirect(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        }
    }

    /**
     * 4x4 blocked kernel for Right Upper NN
     * B := alpha * B * A where A is upper triangular
     * Process columns from right to left (required for upper triangular in-place)
     *
     * <p>NOTE: k-outer scatter was benchmarked but showed ~25% regression vs j-i-k gather.
     * Root cause: scatter writes B[:,j] column-by-column across m rows — each iteration
     * touches a different column offset, defeating row-major cache locality.
     * Original j-i-k gather reads B[i,0..j] contiguously in the k-loop, which JIT
     * vectorizes better. No loop interchange applied here.</p>
     */
    static void dtrmmRightUpperNNBlocked(int m, int n, double alpha,
                                              double[] A, int aOff, int lda,
                                              double[] B, int bOff, int ldb,
                                              boolean unitDiag) {
        int m4 = (m / 4) * 4;
        int n4 = (n / 4) * 4;

        // Process columns from right to left (required for upper triangular in-place)
        for (int j = n - 1; j >= 0; j--) {
            for (int ii = 0; ii < m4; ii += 4) {
                for (int i = ii; i < ii + 4 && i < m; i++) {
                    double sum = 0.0;
                    int kEnd = unitDiag ? j : j + 1;
                    for (int k = 0; k < kEnd; k++) sum = Math.fma(B[bOff+i*ldb+k], A[aOff+k*lda+j], sum);
                    if (unitDiag) sum += B[bOff+i*ldb+j];
                    B[bOff+i*ldb+j] = alpha * sum;
                }
            }
        }

        // Handle remaining rows (m > m4)
        if (m > m4) {
            for (int j = n - 1; j >= 0; j--) {
                for (int i = m4; i < m; i++) {
                    double sum = 0.0;
                    int kEnd = unitDiag ? j : j + 1;
                    for (int k = 0; k < kEnd; k++) sum = Math.fma(B[bOff+i*ldb+k], A[aOff+k*lda+j], sum);
                    if (unitDiag) sum += B[bOff+i*ldb+j];
                    B[bOff+i*ldb+j] = alpha * sum;
                }
            }
        }
    }

    /**
     * Direct for small matrices (no 4x4 blocking)
     */
    static void dtrmmRightUpperNNDirect(int m, int n, double alpha,
                                        double[] A, int aOff, int lda,
                                        double[] B, int bOff, int ldb,
                                        boolean unitDiag) {
        // Process columns from right to left for in-place computation
        for (int j = n - 1; j >= 0; j--) {
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                // For non-unit: include diagonal in loop (k=0 to j)
                // For unit: exclude diagonal from loop (k=0 to j-1)
                int kEnd = unitDiag ? j : j + 1;
                for (int k = 0; k < kEnd; k++) {
                    sum = Math.fma(B[bOff + i * ldb + k], A[aOff + k * lda + j], sum);
                }
                if (unitDiag) {
                    // For unit diagonal, add B[i,j] (diagonal = 1)
                    sum += B[bOff + i * ldb + j];
                }
                // For non-unit, diagonal is already included in k-loop
                B[bOff + i * ldb + j] = alpha * sum;
            }
        }
    }

    /**
     * Right Lower NoTranspose: B := alpha * B * A
     * 
     * For lower triangular A: Result[j] depends on B[j], B[j+1], ..., B[n-1]
     * Uses 4x4 micro-kernel for register optimization.
     * 
     * <p>Key optimization: 4x4 micro-kernel with 16 accumulators,
     * alpha applied at write-back only.</p>
     */
    static void dtrmmRightLowerNN(int m, int n, double alpha,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        
        // Use 4x4 blocked kernel for medium/large matrices
        if (m >= 4 && n >= 4) {
            dtrmmRightLowerNNBlocked(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        } else {
            // Direct for small matrices
            dtrmmRightLowerNNDirect(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        }
    }

    /**
     * Right Lower NN: B := alpha * B * A  (loop-interchange optimized)
     *
     * <p>Uses k-outer scatter order: for each pivot column k (left-to-right),
     * scatter B[:,k]*A[k,j] into B[:,j] for all j < k (reads original B[:,k]
     * before it is scaled), then scale B[:,k] by its diagonal.
     * Inner i-loop accesses B[i,k] and B[i,j] within the same row (row-contiguous).
     * A[k,j] for fixed k is also row-contiguous.</p>
     */
    static void dtrmmRightLowerNNBlocked(int m, int n, double alpha,
                                              double[] A, int aOff, int lda,
                                              double[] B, int bOff, int ldb,
                                              boolean unitDiag) {
        int m4 = (m / 4) * 4;
        // Process k from 0 to n-1: scatter off-diagonal first, then scale diagonal
        for (int k = 0; k < n; k++) {
            int aRowK = aOff + k * lda;
            // Scatter: update columns j < k using original B[:,k]
            for (int j = 0; j < k; j++) {
                double akj = alpha * A[aRowK + j];
                if (akj == 0.0) continue;
                int i = 0;
                for (; i < m4; i += 4) {
                    int b0 = bOff + i * ldb, b1 = bOff + (i+1) * ldb;
                    int b2 = bOff + (i+2) * ldb, b3 = bOff + (i+3) * ldb;
                    B[b0 + j] = Math.fma(akj, B[b0 + k], B[b0 + j]);
                    B[b1 + j] = Math.fma(akj, B[b1 + k], B[b1 + j]);
                    B[b2 + j] = Math.fma(akj, B[b2 + k], B[b2 + j]);
                    B[b3 + j] = Math.fma(akj, B[b3 + k], B[b3 + j]);
                }
                for (; i < m; i++) {
                    int bi = bOff + i * ldb;
                    B[bi + j] = Math.fma(akj, B[bi + k], B[bi + j]);
                }
            }
            // Scale diagonal column k
            double diag = unitDiag ? alpha : alpha * A[aRowK + k];
            int i = 0;
            for (; i < m4; i += 4) {
                B[bOff + i * ldb + k]     *= diag;
                B[bOff + (i+1) * ldb + k] *= diag;
                B[bOff + (i+2) * ldb + k] *= diag;
                B[bOff + (i+3) * ldb + k] *= diag;
            }
            for (; i < m; i++) B[bOff + i * ldb + k] *= diag;
        }
    }

    /**
     * Direct for small matrices (no 4x4 blocking)
     */
    static void dtrmmRightLowerNNDirect(int m, int n, double alpha,
                                        double[] A, int aOff, int lda,
                                        double[] B, int bOff, int ldb,
                                        boolean unitDiag) {
        // Process columns from left to right for in-place computation
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                // For non-unit: include diagonal in loop (k=j to n-1)
                // For unit: exclude diagonal from loop (k=j+1 to n-1)
                int kStart = unitDiag ? j + 1 : j;
                for (int k = kStart; k < n; k++) {
                    sum = Math.fma(B[bOff + i * ldb + k], A[aOff + k * lda + j], sum);
                }
                if (unitDiag) {
                    // For unit diagonal, add B[i,j] (diagonal = 1)
                    sum += B[bOff + i * ldb + j];
                }
                // For non-unit, diagonal is already included in k-loop
                B[bOff + i * ldb + j] = alpha * sum;
            }
        }
    }

    /**
     * Right Upper Transpose: B := alpha * B * A^T
     * 
     * For upper triangular A: Result[j] depends on B[*][j..n-1] * A[j..n-1,j]
     * Uses 4x4 micro-kernel for register optimization.
     * 
     * <p>Key optimization: 4x4 micro-kernel with 16 accumulators,
     * alpha applied at write-back only.</p>
     */
    static void dtrmmRightUpperTN(int m, int n, double alpha,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        
        // Use 4x4 blocked kernel for medium/large matrices
        if (m >= 4 && n >= 4) {
            dtrmmRightUpperTNBlocked(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        } else {
            // Direct for small matrices
            dtrmmRightUpperTNDirect(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        }
    }

    /**
     * 4x4 blocked kernel for Right Upper TN
     * B := alpha * B * A^T where A is upper triangular
     * Process columns from left to right (required for in-place)
     */
    static void dtrmmRightUpperTNBlocked(int m, int n, double alpha,
                                              double[] A, int aOff, int lda,
                                              double[] B, int bOff, int ldb,
                                              boolean unitDiag) {
        
        int m4 = (m / 4) * 4;
        int n4 = (n / 4) * 4;
        
        // Process columns from left to right (required for upper triangular transpose in-place)
        for (int j = 0; j < n; j++) {
            // For each column, process row blocks
            for (int ii = 0; ii < m4; ii += 4) {
                for (int i = ii; i < ii + 4 && i < m; i++) {
                    double sum = 0.0;
                    // Upper triangular transpose: column j depends on B[i,j] to B[i,n-1]
                    int kStart = unitDiag ? j + 1 : j;
                    for (int k = kStart; k < n; k++) {
                        sum = Math.fma(B[bOff + i * ldb + k], A[aOff + j * lda + k], sum);;
                    }
                    if (unitDiag) {
                        sum += B[bOff + i * ldb + j];
                    }
                    B[bOff + i * ldb + j] = alpha * sum;
                }
            }
        }
        
        // Handle remaining rows (m > m4)
        if (m > m4) {
            for (int j = 0; j < n; j++) {
                for (int i = m4; i < m; i++) {
                    double sum = 0.0;
                    int kStart = unitDiag ? j + 1 : j;
                    for (int k = kStart; k < n; k++) {
                        sum = Math.fma(B[bOff + i * ldb + k], A[aOff + j * lda + k], sum);;
                    }
                    if (unitDiag) {
                        sum += B[bOff + i * ldb + j];
                    }
                    B[bOff + i * ldb + j] = alpha * sum;
                }
            }
        }
    }

    /**
     * Direct computation for small matrices (Right Upper TN)
     */
    static void dtrmmRightUpperTNDirect(int m, int n, double alpha,
                                        double[] A, int aOff, int lda,
                                        double[] B, int bOff, int ldb,
                                        boolean unitDiag) {
        
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb;
            
            // Process columns from left to right for in-place computation
            // B[i,j] depends on B[i,k] for k >= j, so we must go forward
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                
                // For non-unit: include diagonal (k=j to n-1)
                // For unit: exclude diagonal (k=j+1 to n-1)
                int kStart = unitDiag ? j + 1 : j;
                for (int k = kStart; k < n; k++) {
                    sum = Math.fma(B[bRow + k], A[aOff + j * lda + k], sum);
                }
                
                // Add diagonal term for unit diagonal only
                if (unitDiag) {
                    sum += B[bRow + j];
                }
                
                // Apply alpha at write-back only
                B[bRow + j] = alpha * sum;
            }
        }
    }

    /**
     * Right Lower Transpose: B := alpha * B * A^T
     * 
     * For lower triangular A: Result[j] depends on B[*][0..j] * A[0..j,j]
     * Uses 4x4 micro-kernel for register optimization.
     * 
     * <p>Key optimization: 4x4 micro-kernel with 16 accumulators,
     * alpha applied at write-back only.</p>
     */
    static void dtrmmRightLowerTN(int m, int n, double alpha,
                                  double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  boolean unitDiag) {
        
        // Use 4x4 blocked kernel for medium/large matrices
        if (m >= 4 && n >= 4) {
            dtrmmRightLowerTNBlocked(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        } else {
            // Direct for small matrices
            dtrmmRightLowerTNDirect(m, n, alpha, A, aOff, lda, B, bOff, ldb, unitDiag);
        }
    }

    /**
     * 4x4 blocked kernel for Right Lower TN
     * B := alpha * B * A^T where A is lower triangular
     * Process columns from right to left (required for in-place)
     */
    static void dtrmmRightLowerTNBlocked(int m, int n, double alpha,
                                              double[] A, int aOff, int lda,
                                              double[] B, int bOff, int ldb,
                                              boolean unitDiag) {
        
        int m4 = (m / 4) * 4;
        int n4 = (n / 4) * 4;
        
        // Process columns from right to left (required for lower triangular transpose in-place)
        for (int j = n - 1; j >= 0; j--) {
            // For each column, process row blocks
            for (int ii = 0; ii < m4; ii += 4) {
                for (int i = ii; i < ii + 4 && i < m; i++) {
                    double sum = 0.0;
                    // Lower triangular transpose: column j depends on B[i,0] to B[i,j]
                    int kEnd = unitDiag ? j : j + 1;
                    for (int k = 0; k < kEnd; k++) {
                        sum = Math.fma(B[bOff + i * ldb + k], A[aOff + j * lda + k], sum);;
                    }
                    if (unitDiag) {
                        sum += B[bOff + i * ldb + j];
                    }
                    B[bOff + i * ldb + j] = alpha * sum;
                }
            }
        }
        
        // Handle remaining rows (m > m4)
        if (m > m4) {
            for (int j = n - 1; j >= 0; j--) {
                for (int i = m4; i < m; i++) {
                    double sum = 0.0;
                    int kEnd = unitDiag ? j : j + 1;
                    for (int k = 0; k < kEnd; k++) {
                        sum = Math.fma(B[bOff + i * ldb + k], A[aOff + j * lda + k], sum);;
                    }
                    if (unitDiag) {
                        sum += B[bOff + i * ldb + j];
                    }
                    B[bOff + i * ldb + j] = alpha * sum;
                }
            }
        }
    }

    /**
     * Direct computation for small matrices (Right Lower TN)
     */
    static void dtrmmRightLowerTNDirect(int m, int n, double alpha,
                                        double[] A, int aOff, int lda,
                                        double[] B, int bOff, int ldb,
                                        boolean unitDiag) {
        
        for (int i = 0; i < m; i++) {
            int bRow = bOff + i * ldb;
            
            // Process columns from right to left for in-place computation
            // B[i,j] depends on B[i,k] for k <= j, so we must go backward
            for (int j = n - 1; j >= 0; j--) {
                double sum = 0.0;
                
                // For non-unit: include diagonal (k=0 to j)
                // For unit: exclude diagonal (k=0 to j-1)
                int kEnd = unitDiag ? j : j + 1;
                for (int k = 0; k < kEnd; k++) {
                    sum = Math.fma(B[bRow + k], A[aOff + j * lda + k], sum);
                }
                
                // Add diagonal term for unit diagonal only
                if (unitDiag) {
                    sum += B[bRow + j];
                }
                
                // Apply alpha at write-back only
                B[bRow + j] = alpha * sum;
            }
        }
    }
}
