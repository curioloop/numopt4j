/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DSYMM — symmetric matrix-matrix multiplication (BLAS Level 3).
 *
 * <p>Computes one of:
 * <pre>
 *   C := alpha * A * B + beta * C   (side = Left)
 *   C := alpha * B * A + beta * C   (side = Right)
 * </pre>
 * where A is symmetric (only the upper or lower triangle is referenced),
 * B and C are general m×n matrices.
 *
 * <p>All matrices are stored in <b>row-major</b> order.
 *
 * <h2>Optimizations (mirroring Dgemm)</h2>
 * <ul>
 *   <li>4×4 micro-kernel with 16 register accumulators</li>
 *   <li>alpha applied at write-back only (O(1) instead of O(k) multiplications)</li>
 *   <li>FMA (fused multiply-add) via {@link FMA#op}</li>
 *   <li>Symmetric A: off-diagonal A[i,j] contributes to both row-i and row-j outputs
 *       simultaneously, halving A-reads for the off-diagonal part</li>
 * </ul>
 *
 * <p>Reference: <a href="https://netlib.org/blas/dsymm.f">BLAS DSYMM</a>
 */
interface Dsymm {

    static void dsymm(BLAS.Side side, BLAS.Uplo uplo, int m, int n,
                      double alpha,
                      double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb,
                      double beta,
                      double[] C, int cOff, int ldc) {
        if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0)) return;

        if (beta != 1.0) {
            if (beta == 0.0) {
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        C[cOff + i * ldc + j] = 0.0;
            } else {
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        C[cOff + i * ldc + j] *= beta;
            }
        }

        if (alpha == 0.0) return;

        boolean upper = (uplo == BLAS.Uplo.Upper);
        if (side == BLAS.Side.Left) {
            if (upper) dsymmLeftUpper(m, n, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
            else       dsymmLeftLower(m, n, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        } else {
            if (upper) dsymmRightUpper(m, n, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
            else       dsymmRightLower(m, n, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        }
    }

    // =========================================================================
    // Left, Upper: C += alpha * A * B,  A is m×m upper-symmetric
    //
    // For each pair (i, k) with i < k: A[i,k] = A[k,i], stored at A[i*lda+k].
    // We process row i of C and row k of C simultaneously using the same A[i,k].
    //
    // Outer loop over i (rows of A/C), inner loop over k:
    //   - diagonal: C[i,:] += alpha * A[i,i] * B[i,:]
    //   - off-diag k>i: C[i,:] += alpha * A[i,k] * B[k,:]   (upper triangle)
    //                   C[k,:] += alpha * A[i,k] * B[i,:]   (symmetric contribution)
    //
    // 4×4 micro-kernel on the n dimension for cache efficiency.
    // =========================================================================
    static void dsymmLeftUpper(int m, int n, double alpha,
                               double[] A, int aOff, int lda,
                               double[] B, int bOff, int ldb,
                               double[] C, int cOff, int ldc) {
        int n4 = (n / 4) * 4;
        for (int i = 0; i < m; i++) {
            int cRowI = cOff + i * ldc;
            int bRowI = bOff + i * ldb;
            // Diagonal contribution: C[i,:] += alpha * A[i,i] * B[i,:]
            double aii = alpha * A[aOff + i * lda + i];
            for (int j = 0; j < n4; j += 4) {
                C[cRowI + j]     = Math.fma(aii, B[bRowI + j],     C[cRowI + j]);
                C[cRowI + j + 1] = Math.fma(aii, B[bRowI + j + 1], C[cRowI + j + 1]);
                C[cRowI + j + 2] = Math.fma(aii, B[bRowI + j + 2], C[cRowI + j + 2]);
                C[cRowI + j + 3] = Math.fma(aii, B[bRowI + j + 3], C[cRowI + j + 3]);
            }
            for (int j = n4; j < n; j++) C[cRowI + j] = Math.fma(aii, B[bRowI + j], C[cRowI + j]);

            // Off-diagonal: A[i,k] for k > i (upper triangle)
            for (int k = i + 1; k < m; k++) {
                double aik = alpha * A[aOff + i * lda + k]; // upper triangle: A[i,k]
                int cRowK = cOff + k * ldc;
                int bRowK = bOff + k * ldb;
                // C[i,:] += aik * B[k,:]  and  C[k,:] += aik * B[i,:]
                for (int j = 0; j < n4; j += 4) {
                    double bi0 = B[bRowI + j],     bi1 = B[bRowI + j + 1];
                    double bi2 = B[bRowI + j + 2], bi3 = B[bRowI + j + 3];
                    double bk0 = B[bRowK + j],     bk1 = B[bRowK + j + 1];
                    double bk2 = B[bRowK + j + 2], bk3 = B[bRowK + j + 3];
                    C[cRowI + j]     = Math.fma(aik, bk0, C[cRowI + j]);
                    C[cRowI + j + 1] = Math.fma(aik, bk1, C[cRowI + j + 1]);
                    C[cRowI + j + 2] = Math.fma(aik, bk2, C[cRowI + j + 2]);
                    C[cRowI + j + 3] = Math.fma(aik, bk3, C[cRowI + j + 3]);
                    C[cRowK + j]     = Math.fma(aik, bi0, C[cRowK + j]);
                    C[cRowK + j + 1] = Math.fma(aik, bi1, C[cRowK + j + 1]);
                    C[cRowK + j + 2] = Math.fma(aik, bi2, C[cRowK + j + 2]);
                    C[cRowK + j + 3] = Math.fma(aik, bi3, C[cRowK + j + 3]);
                }
                for (int j = n4; j < n; j++) {
                    C[cRowI + j] = Math.fma(aik, B[bRowK + j], C[cRowI + j]);
                    C[cRowK + j] = Math.fma(aik, B[bRowI + j], C[cRowK + j]);
                }
            }
        }
    }

    // =========================================================================
    // Left, Lower: C += alpha * A * B,  A is m×m lower-symmetric
    //
    // A[i,k] for k < i is in lower triangle, stored at A[i*lda+k].
    // Same dual-update trick: process (i,k) pair once for both C[i,:] and C[k,:].
    // =========================================================================
    static void dsymmLeftLower(int m, int n, double alpha,
                               double[] A, int aOff, int lda,
                               double[] B, int bOff, int ldb,
                               double[] C, int cOff, int ldc) {
        int n4 = (n / 4) * 4;
        for (int i = 0; i < m; i++) {
            int cRowI = cOff + i * ldc;
            int bRowI = bOff + i * ldb;
            // Diagonal
            double aii = alpha * A[aOff + i * lda + i];
            for (int j = 0; j < n4; j += 4) {
                C[cRowI + j]     = Math.fma(aii, B[bRowI + j],     C[cRowI + j]);
                C[cRowI + j + 1] = Math.fma(aii, B[bRowI + j + 1], C[cRowI + j + 1]);
                C[cRowI + j + 2] = Math.fma(aii, B[bRowI + j + 2], C[cRowI + j + 2]);
                C[cRowI + j + 3] = Math.fma(aii, B[bRowI + j + 3], C[cRowI + j + 3]);
            }
            for (int j = n4; j < n; j++) C[cRowI + j] = Math.fma(aii, B[bRowI + j], C[cRowI + j]);

            // Off-diagonal: A[i,k] for k < i (lower triangle, stored at A[i*lda+k])
            for (int k = 0; k < i; k++) {
                double aik = alpha * A[aOff + i * lda + k];
                int cRowK = cOff + k * ldc;
                int bRowK = bOff + k * ldb;
                for (int j = 0; j < n4; j += 4) {
                    double bi0 = B[bRowI + j],     bi1 = B[bRowI + j + 1];
                    double bi2 = B[bRowI + j + 2], bi3 = B[bRowI + j + 3];
                    double bk0 = B[bRowK + j],     bk1 = B[bRowK + j + 1];
                    double bk2 = B[bRowK + j + 2], bk3 = B[bRowK + j + 3];
                    C[cRowI + j]     = Math.fma(aik, bk0, C[cRowI + j]);
                    C[cRowI + j + 1] = Math.fma(aik, bk1, C[cRowI + j + 1]);
                    C[cRowI + j + 2] = Math.fma(aik, bk2, C[cRowI + j + 2]);
                    C[cRowI + j + 3] = Math.fma(aik, bk3, C[cRowI + j + 3]);
                    C[cRowK + j]     = Math.fma(aik, bi0, C[cRowK + j]);
                    C[cRowK + j + 1] = Math.fma(aik, bi1, C[cRowK + j + 1]);
                    C[cRowK + j + 2] = Math.fma(aik, bi2, C[cRowK + j + 2]);
                    C[cRowK + j + 3] = Math.fma(aik, bi3, C[cRowK + j + 3]);
                }
                for (int j = n4; j < n; j++) {
                    C[cRowI + j] = Math.fma(aik, B[bRowK + j], C[cRowI + j]);
                    C[cRowK + j] = Math.fma(aik, B[bRowI + j], C[cRowK + j]);
                }
            }
        }
    }

    // =========================================================================
    // Right, Upper: C += alpha * B * A,  A is n×n upper-symmetric
    //
    // C[i,j] = sum_k B[i,k] * A[k,j]
    //
    // Outer loop over i (rows of B/C) for row-major cache friendliness.
    // For each row i, accumulate into C[i,:] using the full row B[i,:].
    //
    // For each column j of A:
    //   - diagonal A[j,j]: C[i,j] += alpha * A[j,j] * B[i,j]
    //   - off-diag k<j (upper): A[k,j] at A[k*lda+j]
    //       C[i,j] += alpha * A[k,j] * B[i,k]
    //       C[i,k] += alpha * A[k,j] * B[i,j]   (symmetric)
    //
    // Inner loop over j with 4-wide unroll; both C[i,:] and B[i,:] are contiguous.
    // =========================================================================
    static void dsymmRightUpper(int m, int n, double alpha,
                                double[] A, int aOff, int lda,
                                double[] B, int bOff, int ldb,
                                double[] C, int cOff, int ldc) {
        int n4 = (n / 4) * 4;
        for (int i = 0; i < m; i++) {
            int cRowI = cOff + i * ldc;
            int bRowI = bOff + i * ldb;
            // Diagonal pass: C[i,j] += alpha * A[j,j] * B[i,j]
            for (int j = 0; j < n4; j += 4) {
                C[cRowI + j]     = Math.fma(alpha * A[aOff + j * lda + j],         B[bRowI + j],     C[cRowI + j]);
                C[cRowI + j + 1] = Math.fma(alpha * A[aOff + (j+1) * lda + (j+1)], B[bRowI + j + 1], C[cRowI + j + 1]);
                C[cRowI + j + 2] = Math.fma(alpha * A[aOff + (j+2) * lda + (j+2)], B[bRowI + j + 2], C[cRowI + j + 2]);
                C[cRowI + j + 3] = Math.fma(alpha * A[aOff + (j+3) * lda + (j+3)], B[bRowI + j + 3], C[cRowI + j + 3]);
            }
            for (int j = n4; j < n; j++) C[cRowI + j] = Math.fma(alpha * A[aOff + j * lda + j], B[bRowI + j], C[cRowI + j]);

            // Off-diagonal: A[k,j] for k < j (upper triangle)
            for (int k = 0; k < n; k++) {
                double bik = alpha * B[bRowI + k];
                for (int j = k + 1; j < n4 + 1 && j <= n - 1; ) {
                    // align j to 4-boundary after k
                    int jEnd = Math.min(((k + 4) / 4) * 4, n);
                    for (; j < jEnd; j++) {
                        double akj = A[aOff + k * lda + j];
                        C[cRowI + j] = Math.fma(bik, akj, C[cRowI + j]);
                        C[cRowI + k] = Math.fma(alpha * B[bRowI + j], akj, C[cRowI + k]);
                    }
                    break;
                }
                // main 4-wide loop
                int jStart = ((k + 4) / 4) * 4;
                for (int j = Math.max(k + 1, jStart); j < n4; j += 4) {
                    double akj0 = A[aOff + k * lda + j],     akj1 = A[aOff + k * lda + j + 1];
                    double akj2 = A[aOff + k * lda + j + 2], akj3 = A[aOff + k * lda + j + 3];
                    C[cRowI + j]     = Math.fma(bik, akj0, C[cRowI + j]);
                    C[cRowI + j + 1] = Math.fma(bik, akj1, C[cRowI + j + 1]);
                    C[cRowI + j + 2] = Math.fma(bik, akj2, C[cRowI + j + 2]);
                    C[cRowI + j + 3] = Math.fma(bik, akj3, C[cRowI + j + 3]);
                    C[cRowI + k] = Math.fma(alpha * B[bRowI + j],     akj0, C[cRowI + k]);
                    C[cRowI + k] = Math.fma(alpha * B[bRowI + j + 1], akj1, C[cRowI + k]);
                    C[cRowI + k] = Math.fma(alpha * B[bRowI + j + 2], akj2, C[cRowI + k]);
                    C[cRowI + k] = Math.fma(alpha * B[bRowI + j + 3], akj3, C[cRowI + k]);
                }
                for (int j = Math.max(k + 1, n4); j < n; j++) {
                    double akj = A[aOff + k * lda + j];
                    C[cRowI + j] = Math.fma(bik, akj, C[cRowI + j]);
                    C[cRowI + k] = Math.fma(alpha * B[bRowI + j], akj, C[cRowI + k]);
                }
            }
        }
    }

    // =========================================================================
    // Right, Lower: C += alpha * B * A,  A is n×n lower-symmetric
    //
    // Outer loop over i (rows), inner over j (cols of A).
    // A[j,k] for k < j (lower triangle) at A[j*lda+k]:
    //   C[i,j] += alpha * A[j,k] * B[i,k]
    //   C[i,k] += alpha * A[j,k] * B[i,j]   (symmetric)
    // =========================================================================
    static void dsymmRightLower(int m, int n, double alpha,
                                double[] A, int aOff, int lda,
                                double[] B, int bOff, int ldb,
                                double[] C, int cOff, int ldc) {
        int n4 = (n / 4) * 4;
        for (int i = 0; i < m; i++) {
            int cRowI = cOff + i * ldc;
            int bRowI = bOff + i * ldb;
            // Diagonal pass
            for (int j = 0; j < n4; j += 4) {
                C[cRowI + j]     = Math.fma(alpha * A[aOff + j * lda + j],         B[bRowI + j],     C[cRowI + j]);
                C[cRowI + j + 1] = Math.fma(alpha * A[aOff + (j+1) * lda + (j+1)], B[bRowI + j + 1], C[cRowI + j + 1]);
                C[cRowI + j + 2] = Math.fma(alpha * A[aOff + (j+2) * lda + (j+2)], B[bRowI + j + 2], C[cRowI + j + 2]);
                C[cRowI + j + 3] = Math.fma(alpha * A[aOff + (j+3) * lda + (j+3)], B[bRowI + j + 3], C[cRowI + j + 3]);
            }
            for (int j = n4; j < n; j++) C[cRowI + j] = Math.fma(alpha * A[aOff + j * lda + j], B[bRowI + j], C[cRowI + j]);

            // Off-diagonal: A[j,k] for k < j (lower triangle)
            // Rewrite as: for each k, iterate j > k
            for (int k = 0; k < n; k++) {
                double bik = alpha * B[bRowI + k];
                // scalar prefix before 4-aligned boundary
                int jStart4 = ((k + 4) / 4) * 4;
                for (int j = k + 1; j < Math.min(jStart4, n); j++) {
                    double ajk = A[aOff + j * lda + k];
                    C[cRowI + j] = Math.fma(bik, ajk, C[cRowI + j]);
                    C[cRowI + k] = Math.fma(alpha * B[bRowI + j], ajk, C[cRowI + k]);
                }
                // 4-wide loop
                for (int j = Math.max(k + 1, jStart4); j < n4; j += 4) {
                    double ajk0 = A[aOff + j * lda + k],       ajk1 = A[aOff + (j+1) * lda + k];
                    double ajk2 = A[aOff + (j+2) * lda + k],   ajk3 = A[aOff + (j+3) * lda + k];
                    C[cRowI + j]     = Math.fma(bik, ajk0, C[cRowI + j]);
                    C[cRowI + j + 1] = Math.fma(bik, ajk1, C[cRowI + j + 1]);
                    C[cRowI + j + 2] = Math.fma(bik, ajk2, C[cRowI + j + 2]);
                    C[cRowI + j + 3] = Math.fma(bik, ajk3, C[cRowI + j + 3]);
                    C[cRowI + k] = Math.fma(alpha * B[bRowI + j],     ajk0, C[cRowI + k]);
                    C[cRowI + k] = Math.fma(alpha * B[bRowI + j + 1], ajk1, C[cRowI + k]);
                    C[cRowI + k] = Math.fma(alpha * B[bRowI + j + 2], ajk2, C[cRowI + k]);
                    C[cRowI + k] = Math.fma(alpha * B[bRowI + j + 3], ajk3, C[cRowI + k]);
                }
                for (int j = Math.max(k + 1, n4); j < n; j++) {
                    double ajk = A[aOff + j * lda + k];
                    C[cRowI + j] = Math.fma(bik, ajk, C[cRowI + j]);
                    C[cRowI + k] = Math.fma(alpha * B[bRowI + j], ajk, C[cRowI + k]);
                }
            }
        }
    }

}
