/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DGEMM performs matrix-matrix multiplication.
 * 
 * <p>BLAS Level-3 operation: C := alpha*op(A)*op(B) + beta*C</p>
 * 
 * <p>Reference: LAPACK/BLAS DGEMM</p>
 * 
 * <h2>Numerical Considerations</h2>
 * <ul>
 *   <li>Uses blocked algorithm for cache efficiency</li>
 *   <li>Separates code paths for each transpose combination to avoid branches in hot loops</li>
 *   <li>No special overflow/underflow protection</li>
 * </ul>
 * 
 * <h2>Micro-Kernel</h2>
 * <p>Uses 4×4 micro-kernel with 16 accumulators, optimal for:</p>
 * <ul>
 *   <li>x86_64 with AVX2 (matches OpenBLAS/BLIS pattern)</li>
 *   <li>ARM64 with NEON (matches ARM Compute Library)</li>
 * </ul>
 * 
 * <p>Reference kernel sizes from high-performance libraries:</p>
 * <pre>
 *   OpenBLAS/BLIS (AVX2):   6×8
 *   BLIS/MKL (AVX512):      8×16
 *   ARM Compute (NEON):     4×4
 *   Eigen (generic):        4×4
 * </pre>
 */
public interface Dgemm {

    /**
     * Asymmetric block sizes tuned for L1 cache (32KB typical).
     * 
     * <p>Cache analysis:</p>
     * <ul>
     *   <li>A block: BLOCK_M × BLOCK_K × 8B</li>
     *   <li>B block: BLOCK_K × BLOCK_N × 8B</li>
     *   <li>C block: BLOCK_M × BLOCK_N × 8B</li>
     * </ul>
     * 
     * <p>With BLOCK_M=64, BLOCK_N=64, BLOCK_K=128:</p>
     * <ul>
     *   <li>A: 64 × 128 × 8 = 64KB (streamed, not fully resident)</li>
     *   <li>B: 128 × 64 × 8 = 64KB (streamed, not fully resident)</li>
     *   <li>C: 64 × 64 × 8 = 32KB (fully resident in registers during micro-kernel)</li>
     * </ul>
     * 
     * <p>Note: K dimension can be larger because it's streamed through,
     * while M and N need to be smaller for C to stay resident.</p>
     */
    int BLOCK_M = 64;
    int BLOCK_N = 64;
    int BLOCK_K = 128;  // K can be larger - it's streamed, not fully cached

    /**
     * Performs matrix-matrix multiplication with general matrices.
     *
     * <p>Operation: C := alpha*op(A)*op(B) + beta*C</p>
     * 
     * <p>Branches on transpose parameters once, then dispatches to specialized
     * implementations to avoid branches in hot loops.</p>
     */
    static void dgemm(BLAS.Trans transA, BLAS.Trans transB, int m, int n, int k,
                      double alpha, double[] A, int aOff, int lda,
                      double[] B, int bOff, int ldb,
                      double beta, double[] C, int cOff, int ldc) {
        
        // Normalize transpose parameters
        boolean transAFlag = transA == BLAS.Trans.Trans || transA == BLAS.Trans.Conj;
        boolean transBFlag = transB == BLAS.Trans.Trans || transB == BLAS.Trans.Conj;
        
        // Quick returns
        if (m <= 0 || n <= 0 || ((alpha == 0.0 || k <= 0) && beta == 1.0)) {
            return;
        }

        // Scale C by beta
        scaleC(C, cOff, ldc, m, n, beta);

        if (alpha == 0.0 || k <= 0) {
            return;
        }

        // Dispatch to specialized implementation based on transpose flags
        // This separates the code paths to avoid branches in hot loops
        if (!transAFlag && !transBFlag) {
            // C := alpha * A * B (most common case)
            dgemmNN(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        } else if (transAFlag && !transBFlag) {
            // C := alpha * A^T * B
            dgemmTN(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        } else if (!transAFlag && transBFlag) {
            // C := alpha * A * B^T
            dgemmNT(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        } else {
            // C := alpha * A^T * B^T
            dgemmTT(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        }
    }

    /**
     * Scales matrix C by beta.
     */
    static void scaleC(double[] C, int cOff, int ldc, int m, int n, double beta) {
        if (beta == 0.0) {
            for (int i = 0; i < m; i++) {
                int cRow = cOff + i * ldc;
                for (int j = 0; j < n; j++) {
                    C[cRow + j] = 0.0;
                }
            }
        } else if (beta != 1.0) {
            for (int i = 0; i < m; i++) {
                int cRow = cOff + i * ldc;
                for (int j = 0; j < n; j++) {
                    C[cRow + j] *= beta;
                }
            }
        }
    }

    /**
     * C := alpha * A * B + C (no transpose)
     * Most common case, heavily optimized.
     * 
     * Uses j-i-k loop order with register blocking for better cache efficiency:
     * - C is written once per block (good spatial locality)
     * - A is read sequentially within rows
     * - B is read sequentially within rows
     */
    static void dgemmNN(int m, int n, int k, double alpha,
                        double[] A, int aOff, int lda,
                        double[] B, int bOff, int ldb,
                        double[] C, int cOff, int ldc) {
        
        // Choose algorithm based on sizes (use max block dimension as threshold)
        int maxBlock = Math.max(BLOCK_M, Math.max(BLOCK_N, BLOCK_K));
        if (m < maxBlock && n < maxBlock && k < maxBlock) {
            dgemmNNDirect(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        } else {
            dgemmNNBlocked(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        }
    }

    /**
     * C := alpha * A^T * B + C (A transpose)
     * Uses blocked algorithm with index transformation for A.
     */
    static void dgemmTN(int m, int n, int k, double alpha,
                        double[] A, int aOff, int lda,
                        double[] B, int bOff, int ldb,
                        double[] C, int cOff, int ldc) {
        
        int maxBlock = Math.max(BLOCK_M, Math.max(BLOCK_N, BLOCK_K));
        if (m < maxBlock && n < maxBlock && k < maxBlock) {
            dgemmTNDirect(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        } else {
            dgemmTNBlocked(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        }
    }

    /**
     * C := alpha * A * B^T + C (B transpose)
     * Uses blocked algorithm with index transformation for B.
     */
    static void dgemmNT(int m, int n, int k, double alpha,
                        double[] A, int aOff, int lda,
                        double[] B, int bOff, int ldb,
                        double[] C, int cOff, int ldc) {
        
        int maxBlock = Math.max(BLOCK_M, Math.max(BLOCK_N, BLOCK_K));
        if (m < maxBlock && n < maxBlock && k < maxBlock) {
            dgemmNTDirect(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        } else {
            dgemmNTBlocked(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        }
    }

    /**
     * C := alpha * A^T * B^T + C (both transpose)
     * Uses blocked algorithm with index transformation for both A and B.
     */
    static void dgemmTT(int m, int n, int k, double alpha,
                        double[] A, int aOff, int lda,
                        double[] B, int bOff, int ldb,
                        double[] C, int cOff, int ldc) {
        
        int maxBlock = Math.max(BLOCK_M, Math.max(BLOCK_N, BLOCK_K));
        if (m < maxBlock && n < maxBlock && k < maxBlock) {
            dgemmTTDirect(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        } else {
            dgemmTTBlocked(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
        }
    }

    /**
     * Direct NN multiplication for small matrices (no blocking).
     * Uses 4×4 micro-kernel for optimal register utilization.
     */
    static void dgemmNNDirect(int m, int n, int k, double alpha,
                              double[] A, int aOff, int lda,
                              double[] B, int bOff, int ldb,
                              double[] C, int cOff, int ldc) {
        dgemmNNDirect4x4(m, n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
    }

    /**
     * 4×4 micro-kernel: processes 4 rows × 4 columns per iteration.
     * Optimal for x86_64 (AVX2) and ARM64 (NEON) architectures.
     * 
     * <p>Reference: This matches the kernel size used by OpenBLAS/BLIS (4-6 rows)
     * and ARM Compute Library (4×4 for NEON).</p>
     * 
     * <p>Key optimization: alpha is applied at write-back, not in the k-loop.</p>
     */
    static void dgemmNNDirect4x4(int m, int n, int k, double alpha,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 double[] C, int cOff, int ldc) {
        
        // Process in 4x4 micro-blocks for SIMD-optimal register utilization
        int m4 = (m / 4) * 4;
        int n4 = (n / 4) * 4;
        
        // Main 4x4 micro-kernel
        for (int i = 0; i < m4; i += 4) {
            int aRow0 = aOff + i * lda;
            int aRow1 = aOff + (i + 1) * lda;
            int aRow2 = aOff + (i + 2) * lda;
            int aRow3 = aOff + (i + 3) * lda;
            int cRow0 = cOff + i * ldc;
            int cRow1 = cOff + (i + 1) * ldc;
            int cRow2 = cOff + (i + 2) * ldc;
            int cRow3 = cOff + (i + 3) * ldc;
            
            for (int j = 0; j < n4; j += 4) {
                // 16 accumulators for 4×4 micro-kernel
                double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
                double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
                double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
                double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;
                
                for (int l = 0; l < k; l++) {
                    // No alpha here - move out of k-loop for O(k) -> O(1) reduction
                    double a0l = A[aRow0 + l];
                    double a1l = A[aRow1 + l];
                    double a2l = A[aRow2 + l];
                    double a3l = A[aRow3 + l];
                    int bRow = bOff + l * ldb;
                    double bj0 = B[bRow + j];
                    double bj1 = B[bRow + j + 1];
                    double bj2 = B[bRow + j + 2];
                    double bj3 = B[bRow + j + 3];
                    
                    c00 = FMA.op(a0l, bj0, c00);     c01 = FMA.op(a0l, bj1, c01);
                    c02 = FMA.op(a0l, bj2, c02); c03 = FMA.op(a0l, bj3, c03);
                    
                    c10 = FMA.op(a1l, bj0, c10);     c11 = FMA.op(a1l, bj1, c11);
                    c12 = FMA.op(a1l, bj2, c12); c13 = FMA.op(a1l, bj3, c13);
                    
                    c20 = FMA.op(a2l, bj0, c20);     c21 = FMA.op(a2l, bj1, c21);
                    c22 = FMA.op(a2l, bj2, c22); c23 = FMA.op(a2l, bj3, c23);
                    
                    c30 = FMA.op(a3l, bj0, c30);     c31 = FMA.op(a3l, bj1, c31);
                    c32 = FMA.op(a3l, bj2, c32); c33 = FMA.op(a3l, bj3, c33);
                }
                
                // Apply alpha at write-back (once per accumulator, not k times)
                C[cRow0 + j] += alpha * c00;     C[cRow0 + j + 1] += alpha * c01;
                C[cRow0 + j + 2] += alpha * c02; C[cRow0 + j + 3] += alpha * c03;
                
                C[cRow1 + j] += alpha * c10;     C[cRow1 + j + 1] += alpha * c11;
                C[cRow1 + j + 2] += alpha * c12; C[cRow1 + j + 3] += alpha * c13;
                
                C[cRow2 + j] += alpha * c20;     C[cRow2 + j + 1] += alpha * c21;
                C[cRow2 + j + 2] += alpha * c22; C[cRow2 + j + 3] += alpha * c23;
                
                C[cRow3 + j] += alpha * c30;     C[cRow3 + j + 1] += alpha * c31;
                C[cRow3 + j + 2] += alpha * c32; C[cRow3 + j + 3] += alpha * c33;
            }
            
            // Handle remainder columns (n % 4) for 4 rows
            for (int j = n4; j < n; j++) {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                for (int l = 0; l < k; l++) {
                    int bRow = bOff + l * ldb;
                    s0 += A[aRow0 + l] * B[bRow + j];
                    s1 += A[aRow1 + l] * B[bRow + j];
                    s2 += A[aRow2 + l] * B[bRow + j];
                    s3 += A[aRow3 + l] * B[bRow + j];
                }
                C[cRow0 + j] += alpha * s0;
                C[cRow1 + j] += alpha * s1;
                C[cRow2 + j] += alpha * s2;
                C[cRow3 + j] += alpha * s3;
            }
        }
        
        // Handle remainder rows (m % 4) using 2×4 or 1×4 kernel
        if (m4 < m) {
            int mRem = m - m4;
            
            if (mRem >= 2) {
                // Use 2×4 for remaining rows
                int i = m4;
                int aRow0 = aOff + i * lda;
                int aRow1 = aOff + (i + 1) * lda;
                int cRow0 = cOff + i * ldc;
                int cRow1 = cOff + (i + 1) * ldc;
                
                for (int j = 0; j < n4; j += 4) {
                    double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
                    double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
                    
                    for (int l = 0; l < k; l++) {
                        // No alpha here
                        double a0l = A[aRow0 + l];
                        double a1l = A[aRow1 + l];
                        int bRow = bOff + l * ldb;
                        double bj0 = B[bRow + j];
                        double bj1 = B[bRow + j + 1];
                        double bj2 = B[bRow + j + 2];
                        double bj3 = B[bRow + j + 3];
                        
                        c00 = FMA.op(a0l, bj0, c00);     c01 = FMA.op(a0l, bj1, c01);
                        c02 = FMA.op(a0l, bj2, c02); c03 = FMA.op(a0l, bj3, c03);
                        c10 = FMA.op(a1l, bj0, c10);     c11 = FMA.op(a1l, bj1, c11);
                        c12 = FMA.op(a1l, bj2, c12); c13 = FMA.op(a1l, bj3, c13);
                    }
                    
                    // Apply alpha at write-back
                    C[cRow0 + j] += alpha * c00;     C[cRow0 + j + 1] += alpha * c01;
                    C[cRow0 + j + 2] += alpha * c02; C[cRow0 + j + 3] += alpha * c03;
                    C[cRow1 + j] += alpha * c10;     C[cRow1 + j + 1] += alpha * c11;
                    C[cRow1 + j + 2] += alpha * c12; C[cRow1 + j + 3] += alpha * c13;
                }
                
                for (int j = n4; j < n; j++) {
                    double s0 = 0.0, s1 = 0.0;
                    for (int l = 0; l < k; l++) {
                        int bRow = bOff + l * ldb;
                        s0 += A[aRow0 + l] * B[bRow + j];
                        s1 += A[aRow1 + l] * B[bRow + j];
                    }
                    C[cRow0 + j] += alpha * s0;
                    C[cRow1 + j] += alpha * s1;
                }
                
                m4 += 2;
            }
            
            // Handle final single row if any
            if (m4 < m) {
                int i = m4;
                int aRow = aOff + i * lda;
                int cRow = cOff + i * ldc;
                
                for (int j = 0; j < n4; j += 4) {
                    double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
                    
                    for (int l = 0; l < k; l++) {
                        // No alpha here
                        double ail = A[aRow + l];
                        int bRow = bOff + l * ldb;
                        c0 = FMA.op(ail, B[bRow + j], c0);
                        c1 = FMA.op(ail, B[bRow + j + 1], c1);
                        c2 = FMA.op(ail, B[bRow + j + 2], c2);
                        c3 = FMA.op(ail, B[bRow + j + 3], c3);
                    }
                    
                    // Apply alpha at write-back
                    C[cRow + j] += alpha * c0;
                    C[cRow + j + 1] += alpha * c1;
                    C[cRow + j + 2] += alpha * c2;
                    C[cRow + j + 3] += alpha * c3;
                }
                
                for (int j = n4; j < n; j++) {
                    double sum = 0.0;
                    for (int l = 0; l < k; l++) {
                        sum += A[aRow + l] * B[bOff + l * ldb + j];
                    }
                    C[cRow + j] += alpha * sum;
                }
            }
        }
    }

    /**
     * Blocked NN multiplication for large matrices.
     * Uses asymmetric blocking tuned for L1 cache.
     * 
     * <p>BLIS-style optimization: write-back is deferred until all k-blocks are processed.</p>
     */
    static void dgemmNNBlocked(int m, int n, int k, double alpha,
                               double[] A, int aOff, int lda,
                               double[] B, int bOff, int ldb,
                               double[] C, int cOff, int ldc) {
        
        // Outer blocking for L2 cache (2x inner block size)
        final int OUTER_M = BLOCK_M * 2;
        final int OUTER_N = BLOCK_N * 2;
        final int OUTER_K = BLOCK_K * 2;
        
        // BLIS-style: i-j-kk loop order to defer C write-back
        for (int ii = 0; ii < m; ii += OUTER_M) {
            int iMax = Math.min(ii + OUTER_M, m);
            
            for (int jj = 0; jj < n; jj += OUTER_N) {
                int jMax = Math.min(jj + OUTER_N, n);
                
                // Process this (ii, jj) block across ALL k-blocks with deferred write-back
                dgemmNNBlockBlis(ii, iMax, jj, jMax, k, alpha,
                                 A, aOff, lda, B, bOff, ldb, C, cOff, ldc, BLOCK_K);
            }
        }
    }

    /**
     * BLIS-style block processing: accumulates across all k-blocks, writes to C once.
     * 
     * <p>Key optimization: instead of writing to C for each k-block, we keep
     * accumulators in registers for the entire k dimension, then write once.</p>
     * 
     * <p>Uses inner blocking for m and n dimensions to fit in registers.</p>
     */
    static void dgemmNNBlockBlis(int iStart, int iEnd, int jStart, int jEnd, int kTotal,
                                  double alpha, double[] A, int aOff, int lda,
                                  double[] B, int bOff, int ldb,
                                  double[] C, int cOff, int ldc, int kBlockSize) {
        
        // Inner blocking for m and n dimensions (register-level)
        for (int ii = iStart; ii < iEnd; ii += BLOCK_M) {
            int iMax = Math.min(ii + BLOCK_M, iEnd);
            
            for (int jj = jStart; jj < jEnd; jj += BLOCK_N) {
                int jMax = Math.min(jj + BLOCK_N, jEnd);
                
                // Process this micro-block with deferred write-back
                dgemmNNMicroBlis4x4(ii, iMax, jj, jMax, kTotal, alpha,
                                     A, aOff, lda, B, bOff, ldb, C, cOff, ldc, kBlockSize);
            }
        }
    }

    /**
     * 4×4 block kernel for x86_64 (AVX2) and ARM64 (NEON).
     * Key optimization: alpha is applied at write-back, not in the k-loop.
     */
    static void dgemmNNBlockOptimized4x4(int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd,
                                          double alpha, double[] A, int aOff, int lda,
                                          double[] B, int bOff, int ldb,
                                          double[] C, int cOff, int ldc) {
        
        // Align to 4×4 micro-kernel boundaries
        int i4End = iStart + ((iEnd - iStart) / 4) * 4;
        int j4End = jStart + ((jEnd - jStart) / 4) * 4;
        
        // Main 4×4 micro-kernel
        for (int i = iStart; i < i4End; i += 4) {
            int aRow0 = aOff + i * lda;
            int aRow1 = aOff + (i + 1) * lda;
            int aRow2 = aOff + (i + 2) * lda;
            int aRow3 = aOff + (i + 3) * lda;
            int cRow0 = cOff + i * ldc;
            int cRow1 = cOff + (i + 1) * ldc;
            int cRow2 = cOff + (i + 2) * ldc;
            int cRow3 = cOff + (i + 3) * ldc;
            
            for (int j = jStart; j < j4End; j += 4) {
                // 16 accumulators for 4×4 block
                double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
                double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
                double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
                double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;
                
                for (int k = kStart; k < kEnd; k++) {
                    // No alpha here - move out of k-loop
                    double a0k = A[aRow0 + k];
                    double a1k = A[aRow1 + k];
                    double a2k = A[aRow2 + k];
                    double a3k = A[aRow3 + k];
                    int bRow = bOff + k * ldb;
                    double bj0 = B[bRow + j];
                    double bj1 = B[bRow + j + 1];
                    double bj2 = B[bRow + j + 2];
                    double bj3 = B[bRow + j + 3];
                    
                    c00 = FMA.op(a0k, bj0, c00);     c01 = FMA.op(a0k, bj1, c01);
                    c02 = FMA.op(a0k, bj2, c02); c03 = FMA.op(a0k, bj3, c03);
                    
                    c10 = FMA.op(a1k, bj0, c10);     c11 = FMA.op(a1k, bj1, c11);
                    c12 = FMA.op(a1k, bj2, c12); c13 = FMA.op(a1k, bj3, c13);
                    
                    c20 = FMA.op(a2k, bj0, c20);     c21 = FMA.op(a2k, bj1, c21);
                    c22 = FMA.op(a2k, bj2, c22); c23 = FMA.op(a2k, bj3, c23);
                    
                    c30 = FMA.op(a3k, bj0, c30);     c31 = FMA.op(a3k, bj1, c31);
                    c32 = FMA.op(a3k, bj2, c32); c33 = FMA.op(a3k, bj3, c33);
                }
                
                // Apply alpha at write-back
                C[cRow0 + j] += alpha * c00;     C[cRow0 + j + 1] += alpha * c01;
                C[cRow0 + j + 2] += alpha * c02; C[cRow0 + j + 3] += alpha * c03;
                
                C[cRow1 + j] += alpha * c10;     C[cRow1 + j + 1] += alpha * c11;
                C[cRow1 + j + 2] += alpha * c12; C[cRow1 + j + 3] += alpha * c13;
                
                C[cRow2 + j] += alpha * c20;     C[cRow2 + j + 1] += alpha * c21;
                C[cRow2 + j + 2] += alpha * c22; C[cRow2 + j + 3] += alpha * c23;
                
                C[cRow3 + j] += alpha * c30;     C[cRow3 + j + 1] += alpha * c31;
                C[cRow3 + j + 2] += alpha * c32; C[cRow3 + j + 3] += alpha * c33;
            }
            
            // Remainder columns for 4 rows
            for (int j = j4End; j < jEnd; j++) {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                for (int k = kStart; k < kEnd; k++) {
                    int bRow = bOff + k * ldb;
                    s0 += A[aRow0 + k] * B[bRow + j];
                    s1 += A[aRow1 + k] * B[bRow + j];
                    s2 += A[aRow2 + k] * B[bRow + j];
                    s3 += A[aRow3 + k] * B[bRow + j];
                }
                C[cRow0 + j] += alpha * s0;
                C[cRow1 + j] += alpha * s1;
                C[cRow2 + j] += alpha * s2;
                C[cRow3 + j] += alpha * s3;
            }
        }
        
        // Remainder rows (process 2 or 1 at a time)
        for (int i = i4End; i < iEnd; i++) {
            int aRow = aOff + i * lda;
            int cRow = cOff + i * ldc;
            
            for (int j = jStart; j < j4End; j += 4) {
                double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
                
                for (int k = kStart; k < kEnd; k++) {
                    // No alpha here
                    double aik = A[aRow + k];
                    int bRow = bOff + k * ldb;
                    c0 = FMA.op(aik, B[bRow + j], c0);
                    c1 = FMA.op(aik, B[bRow + j + 1], c1);
                    c2 = FMA.op(aik, B[bRow + j + 2], c2);
                    c3 = FMA.op(aik, B[bRow + j + 3], c3);
                }
                
                // Apply alpha at write-back
                C[cRow + j] += alpha * c0;
                C[cRow + j + 1] += alpha * c1;
                C[cRow + j + 2] += alpha * c2;
                C[cRow + j + 3] += alpha * c3;
            }
            
            for (int j = j4End; j < jEnd; j++) {
                double sum = 0.0;
                for (int k = kStart; k < kEnd; k++) {
                    sum += A[aRow + k] * B[bOff + k * ldb + j];
                }
                C[cRow + j] += alpha * sum;
            }
        }
    }

    /**
     * BLIS-style 4×4 micro-kernel: accumulates across ALL k-blocks, writes to C ONCE.
     * 
     * <p>This is the key BLIS optimization: instead of writing to C for each k-block,
     * we keep accumulators in registers for the entire k dimension.</p>
     * 
     * <p>Benefits:</p>
     * <ul>
     *   <li>Reduces C memory writes from O(k/kBlockSize) to O(1) per micro-block</li>
     *   <li>Better register utilization - accumulators stay hot in registers</li>
     *   <li>Improved cache efficiency - C is read once, written once</li>
     * </ul>
     */
    static void dgemmNNMicroBlis4x4(int iStart, int iEnd, int jStart, int jEnd, int kTotal,
                                     double alpha, double[] A, int aOff, int lda,
                                     double[] B, int bOff, int ldb,
                                     double[] C, int cOff, int ldc, int kBlockSize) {
        
        // Align to 4×4 micro-kernel boundaries
        int i4End = iStart + ((iEnd - iStart) / 4) * 4;
        int j4End = jStart + ((jEnd - jStart) / 4) * 4;
        
        // Main 4×4 micro-kernel with BLIS-style deferred write-back
        for (int i = iStart; i < i4End; i += 4) {
            int aRow0 = aOff + i * lda;
            int aRow1 = aOff + (i + 1) * lda;
            int aRow2 = aOff + (i + 2) * lda;
            int aRow3 = aOff + (i + 3) * lda;
            int cRow0 = cOff + i * ldc;
            int cRow1 = cOff + (i + 1) * ldc;
            int cRow2 = cOff + (i + 2) * ldc;
            int cRow3 = cOff + (i + 3) * ldc;
            
            for (int j = jStart; j < j4End; j += 4) {
                // Accumulators for A*B product only (C already has beta applied via scaleC)
                double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
                double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
                double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
                double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;
                
                // Accumulate across ALL k-blocks (no write to C!)
                for (int kk = 0; kk < kTotal; kk += kBlockSize) {
                    int kEnd = Math.min(kk + kBlockSize, kTotal);
                    
                    for (int k = kk; k < kEnd; k++) {
                        double a0k = A[aRow0 + k];
                        double a1k = A[aRow1 + k];
                        double a2k = A[aRow2 + k];
                        double a3k = A[aRow3 + k];
                        int bRow = bOff + k * ldb;
                        double bj0 = B[bRow + j];
                        double bj1 = B[bRow + j + 1];
                        double bj2 = B[bRow + j + 2];
                        double bj3 = B[bRow + j + 3];
                        
                        c00 = FMA.op(a0k, bj0, c00);     c01 = FMA.op(a0k, bj1, c01);
                        c02 = FMA.op(a0k, bj2, c02); c03 = FMA.op(a0k, bj3, c03);
                        
                        c10 = FMA.op(a1k, bj0, c10);     c11 = FMA.op(a1k, bj1, c11);
                        c12 = FMA.op(a1k, bj2, c12); c13 = FMA.op(a1k, bj3, c13);
                        
                        c20 = FMA.op(a2k, bj0, c20);     c21 = FMA.op(a2k, bj1, c21);
                        c22 = FMA.op(a2k, bj2, c22); c23 = FMA.op(a2k, bj3, c23);
                        
                        c30 = FMA.op(a3k, bj0, c30);     c31 = FMA.op(a3k, bj1, c31);
                        c32 = FMA.op(a3k, bj2, c32); c33 = FMA.op(a3k, bj3, c33);
                    }
                }
                
                // Write back to C ONCE: C = C + alpha * (A*B)
                // Note: C already has beta applied via scaleC() at the start
                C[cRow0 + j] += alpha * c00;     C[cRow0 + j + 1] += alpha * c01;
                C[cRow0 + j + 2] += alpha * c02; C[cRow0 + j + 3] += alpha * c03;
                
                C[cRow1 + j] += alpha * c10;     C[cRow1 + j + 1] += alpha * c11;
                C[cRow1 + j + 2] += alpha * c12; C[cRow1 + j + 3] += alpha * c13;
                
                C[cRow2 + j] += alpha * c20;     C[cRow2 + j + 1] += alpha * c21;
                C[cRow2 + j + 2] += alpha * c22; C[cRow2 + j + 3] += alpha * c23;
                
                C[cRow3 + j] += alpha * c30;     C[cRow3 + j + 1] += alpha * c31;
                C[cRow3 + j + 2] += alpha * c32; C[cRow3 + j + 3] += alpha * c33;
            }
            
            // Remainder columns for 4 rows
            for (int j = j4End; j < jEnd; j++) {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                
                for (int k = 0; k < kTotal; k++) {
                    int bRow = bOff + k * ldb;
                    s0 += A[aRow0 + k] * B[bRow + j];
                    s1 += A[aRow1 + k] * B[bRow + j];
                    s2 += A[aRow2 + k] * B[bRow + j];
                    s3 += A[aRow3 + k] * B[bRow + j];
                }
                C[cRow0 + j] += alpha * s0;
                C[cRow1 + j] += alpha * s1;
                C[cRow2 + j] += alpha * s2;
                C[cRow3 + j] += alpha * s3;
            }
        }
        
        // Remainder rows
        for (int i = i4End; i < iEnd; i++) {
            int aRow = aOff + i * lda;
            int cRow = cOff + i * ldc;
            
            for (int j = jStart; j < j4End; j += 4) {
                double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
                
                for (int k = 0; k < kTotal; k++) {
                    double aik = A[aRow + k];
                    int bRow = bOff + k * ldb;
                    c0 = FMA.op(aik, B[bRow + j], c0);
                    c1 = FMA.op(aik, B[bRow + j + 1], c1);
                    c2 = FMA.op(aik, B[bRow + j + 2], c2);
                    c3 = FMA.op(aik, B[bRow + j + 3], c3);
                }
                
                C[cRow + j] += alpha * c0;
                C[cRow + j + 1] += alpha * c1;
                C[cRow + j + 2] += alpha * c2;
                C[cRow + j + 3] += alpha * c3;
            }
            
            for (int j = j4End; j < jEnd; j++) {
                double sum = 0.0;
                for (int k = 0; k < kTotal; k++) {
                    sum += A[aRow + k] * B[bOff + k * ldb + j];
                }
                C[cRow + j] += alpha * sum;
            }
        }
    }

    /**
     * Legacy block method for compatibility.
     */
    static void dgemmNNBlock(int iStart, int iEnd, int jStart, int jEnd, int kStart, int kMax,
                             double alpha, double[] A, int aOff, int lda,
                             double[] B, int bOff, int ldb,
                             double[] C, int cOff, int ldc) {
        dgemmNNBlockOptimized4x4(iStart, iEnd, jStart, jEnd, kStart, kMax,
                                  alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
    }

    // ==================== TN (A^T * B) Implementation ====================

    /**
     * Direct TN kernel for small matrices: C := alpha * A^T * B + C.
     * A is accessed column-major: A[l*lda + i] instead of A[i*lda + l].
     */
    static void dgemmTNDirect(int m, int n, int k, double alpha,
                              double[] A, int aOff, int lda,
                              double[] B, int bOff, int ldb,
                              double[] C, int cOff, int ldc) {
        
        int m4 = (m / 4) * 4;
        int n4 = (n / 4) * 4;
        
        for (int i = 0; i < m4; i += 4) {
            int cRow0 = cOff + i * ldc;
            int cRow1 = cOff + (i + 1) * ldc;
            int cRow2 = cOff + (i + 2) * ldc;
            int cRow3 = cOff + (i + 3) * ldc;
            
            for (int j = 0; j < n4; j += 4) {
                double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
                double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
                double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
                double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;
                
                for (int l = 0; l < k; l++) {
                    // A^T: access column l, rows i to i+3
                    double a0l = A[aOff + l * lda + i];
                    double a1l = A[aOff + l * lda + i + 1];
                    double a2l = A[aOff + l * lda + i + 2];
                    double a3l = A[aOff + l * lda + i + 3];
                    int bRow = bOff + l * ldb;
                    double bj0 = B[bRow + j];
                    double bj1 = B[bRow + j + 1];
                    double bj2 = B[bRow + j + 2];
                    double bj3 = B[bRow + j + 3];
                    
                    c00 = FMA.op(a0l, bj0, c00);     c01 = FMA.op(a0l, bj1, c01);
                    c02 = FMA.op(a0l, bj2, c02); c03 = FMA.op(a0l, bj3, c03);
                    
                    c10 = FMA.op(a1l, bj0, c10);     c11 = FMA.op(a1l, bj1, c11);
                    c12 = FMA.op(a1l, bj2, c12); c13 = FMA.op(a1l, bj3, c13);
                    
                    c20 = FMA.op(a2l, bj0, c20);     c21 = FMA.op(a2l, bj1, c21);
                    c22 = FMA.op(a2l, bj2, c22); c23 = FMA.op(a2l, bj3, c23);
                    
                    c30 = FMA.op(a3l, bj0, c30);     c31 = FMA.op(a3l, bj1, c31);
                    c32 = FMA.op(a3l, bj2, c32); c33 = FMA.op(a3l, bj3, c33);
                }
                
                C[cRow0 + j] += alpha * c00;     C[cRow0 + j + 1] += alpha * c01;
                C[cRow0 + j + 2] += alpha * c02; C[cRow0 + j + 3] += alpha * c03;
                
                C[cRow1 + j] += alpha * c10;     C[cRow1 + j + 1] += alpha * c11;
                C[cRow1 + j + 2] += alpha * c12; C[cRow1 + j + 3] += alpha * c13;
                
                C[cRow2 + j] += alpha * c20;     C[cRow2 + j + 1] += alpha * c21;
                C[cRow2 + j + 2] += alpha * c22; C[cRow2 + j + 3] += alpha * c23;
                
                C[cRow3 + j] += alpha * c30;     C[cRow3 + j + 1] += alpha * c31;
                C[cRow3 + j + 2] += alpha * c32; C[cRow3 + j + 3] += alpha * c33;
            }
            
            // Remainder columns
            for (int j = n4; j < n; j++) {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                for (int l = 0; l < k; l++) {
                    int bRow = bOff + l * ldb;
                    s0 += A[aOff + l * lda + i] * B[bRow + j];
                    s1 += A[aOff + l * lda + i + 1] * B[bRow + j];
                    s2 += A[aOff + l * lda + i + 2] * B[bRow + j];
                    s3 += A[aOff + l * lda + i + 3] * B[bRow + j];
                }
                C[cRow0 + j] += alpha * s0;
                C[cRow1 + j] += alpha * s1;
                C[cRow2 + j] += alpha * s2;
                C[cRow3 + j] += alpha * s3;
            }
        }
        
        // Remainder rows
        for (int i = m4; i < m; i++) {
            int cRow = cOff + i * ldc;
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[aOff + l * lda + i] * B[bOff + l * ldb + j];
                }
                C[cRow + j] += alpha * sum;
            }
        }
    }

    /**
     * Blocked TN kernel: C := alpha * A^T * B + C.
     */
    static void dgemmTNBlocked(int m, int n, int k, double alpha,
                                double[] A, int aOff, int lda,
                                double[] B, int bOff, int ldb,
                                double[] C, int cOff, int ldc) {
        
        final int OUTER_M = BLOCK_M * 2;
        final int OUTER_N = BLOCK_N * 2;
        
        for (int ii = 0; ii < m; ii += OUTER_M) {
            int iMax = Math.min(ii + OUTER_M, m);
            
            for (int jj = 0; jj < n; jj += OUTER_N) {
                int jMax = Math.min(jj + OUTER_N, n);
                
                for (int kk = 0; kk < k; kk += BLOCK_K) {
                    int kEnd = Math.min(kk + BLOCK_K, k);
                    
                    for (int i = ii; i < iMax; i += BLOCK_M) {
                        int iEnd2 = Math.min(i + BLOCK_M, iMax);
                        
                        for (int j = jj; j < jMax; j += BLOCK_N) {
                            int jEnd2 = Math.min(j + BLOCK_N, jMax);
                            
                            dgemmTNBlock(i, iEnd2, j, jEnd2, kk, kEnd, alpha,
                                         A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
                        }
                    }
                }
            }
        }
    }

    static void dgemmTNBlock(int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd,
                              double alpha, double[] A, int aOff, int lda,
                              double[] B, int bOff, int ldb,
                              double[] C, int cOff, int ldc) {
        
        int i4End = iStart + ((iEnd - iStart) / 4) * 4;
        int j4End = jStart + ((jEnd - jStart) / 4) * 4;
        
        for (int i = iStart; i < i4End; i += 4) {
            int cRow0 = cOff + i * ldc;
            int cRow1 = cOff + (i + 1) * ldc;
            int cRow2 = cOff + (i + 2) * ldc;
            int cRow3 = cOff + (i + 3) * ldc;
            
            for (int j = jStart; j < j4End; j += 4) {
                double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
                double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
                double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
                double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;
                
                for (int l = kStart; l < kEnd; l++) {
                    double a0l = A[aOff + l * lda + i];
                    double a1l = A[aOff + l * lda + i + 1];
                    double a2l = A[aOff + l * lda + i + 2];
                    double a3l = A[aOff + l * lda + i + 3];
                    int bRow = bOff + l * ldb;
                    double bj0 = B[bRow + j];
                    double bj1 = B[bRow + j + 1];
                    double bj2 = B[bRow + j + 2];
                    double bj3 = B[bRow + j + 3];
                    
                    c00 = FMA.op(a0l, bj0, c00);     c01 = FMA.op(a0l, bj1, c01);
                    c02 = FMA.op(a0l, bj2, c02); c03 = FMA.op(a0l, bj3, c03);
                    c10 = FMA.op(a1l, bj0, c10);     c11 = FMA.op(a1l, bj1, c11);
                    c12 = FMA.op(a1l, bj2, c12); c13 = FMA.op(a1l, bj3, c13);
                    c20 = FMA.op(a2l, bj0, c20);     c21 = FMA.op(a2l, bj1, c21);
                    c22 = FMA.op(a2l, bj2, c22); c23 = FMA.op(a2l, bj3, c23);
                    c30 = FMA.op(a3l, bj0, c30);     c31 = FMA.op(a3l, bj1, c31);
                    c32 = FMA.op(a3l, bj2, c32); c33 = FMA.op(a3l, bj3, c33);
                }
                
                C[cRow0 + j] += alpha * c00;     C[cRow0 + j + 1] += alpha * c01;
                C[cRow0 + j + 2] += alpha * c02; C[cRow0 + j + 3] += alpha * c03;
                C[cRow1 + j] += alpha * c10;     C[cRow1 + j + 1] += alpha * c11;
                C[cRow1 + j + 2] += alpha * c12; C[cRow1 + j + 3] += alpha * c13;
                C[cRow2 + j] += alpha * c20;     C[cRow2 + j + 1] += alpha * c21;
                C[cRow2 + j + 2] += alpha * c22; C[cRow2 + j + 3] += alpha * c23;
                C[cRow3 + j] += alpha * c30;     C[cRow3 + j + 1] += alpha * c31;
                C[cRow3 + j + 2] += alpha * c32; C[cRow3 + j + 3] += alpha * c33;
            }
            
            for (int j = j4End; j < jEnd; j++) {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                for (int l = kStart; l < kEnd; l++) {
                    int bRow = bOff + l * ldb;
                    s0 += A[aOff + l * lda + i] * B[bRow + j];
                    s1 += A[aOff + l * lda + i + 1] * B[bRow + j];
                    s2 += A[aOff + l * lda + i + 2] * B[bRow + j];
                    s3 += A[aOff + l * lda + i + 3] * B[bRow + j];
                }
                C[cRow0 + j] += alpha * s0;
                C[cRow1 + j] += alpha * s1;
                C[cRow2 + j] += alpha * s2;
                C[cRow3 + j] += alpha * s3;
            }
        }
        
        for (int i = i4End; i < iEnd; i++) {
            int cRow = cOff + i * ldc;
            for (int j = jStart; j < jEnd; j++) {
                double sum = 0.0;
                for (int l = kStart; l < kEnd; l++) {
                    sum += A[aOff + l * lda + i] * B[bOff + l * ldb + j];
                }
                C[cRow + j] += alpha * sum;
            }
        }
    }

    // ==================== NT (A * B^T) Implementation ====================

    /**
     * Direct NT kernel: C := alpha * A * B^T + C.
     * B is accessed column-major: B[j*ldb + l] instead of B[l*ldb + j].
     */
    static void dgemmNTDirect(int m, int n, int k, double alpha,
                              double[] A, int aOff, int lda,
                              double[] B, int bOff, int ldb,
                              double[] C, int cOff, int ldc) {
        
        int m4 = (m / 4) * 4;
        int n4 = (n / 4) * 4;
        
        for (int i = 0; i < m4; i += 4) {
            int aRow0 = aOff + i * lda;
            int aRow1 = aOff + (i + 1) * lda;
            int aRow2 = aOff + (i + 2) * lda;
            int aRow3 = aOff + (i + 3) * lda;
            int cRow0 = cOff + i * ldc;
            int cRow1 = cOff + (i + 1) * ldc;
            int cRow2 = cOff + (i + 2) * ldc;
            int cRow3 = cOff + (i + 3) * ldc;
            
            for (int j = 0; j < n4; j += 4) {
                double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
                double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
                double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
                double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;
                
                for (int l = 0; l < k; l++) {
                    // B^T: access column j to j+3, row l
                    int bColJ = bOff + j * ldb + l;
                    int bColJ1 = bOff + (j + 1) * ldb + l;
                    int bColJ2 = bOff + (j + 2) * ldb + l;
                    int bColJ3 = bOff + (j + 3) * ldb + l;
                    
                    double a0l = A[aRow0 + l];
                    double a1l = A[aRow1 + l];
                    double a2l = A[aRow2 + l];
                    double a3l = A[aRow3 + l];
                    
                    c00 += a0l * B[bColJ];  c01 += a0l * B[bColJ1];
                    c02 += a0l * B[bColJ2]; c03 += a0l * B[bColJ3];
                    c10 += a1l * B[bColJ];  c11 += a1l * B[bColJ1];
                    c12 += a1l * B[bColJ2]; c13 += a1l * B[bColJ3];
                    c20 += a2l * B[bColJ];  c21 += a2l * B[bColJ1];
                    c22 += a2l * B[bColJ2]; c23 += a2l * B[bColJ3];
                    c30 += a3l * B[bColJ];  c31 += a3l * B[bColJ1];
                    c32 += a3l * B[bColJ2]; c33 += a3l * B[bColJ3];
                }
                
                C[cRow0 + j] += alpha * c00;     C[cRow0 + j + 1] += alpha * c01;
                C[cRow0 + j + 2] += alpha * c02; C[cRow0 + j + 3] += alpha * c03;
                C[cRow1 + j] += alpha * c10;     C[cRow1 + j + 1] += alpha * c11;
                C[cRow1 + j + 2] += alpha * c12; C[cRow1 + j + 3] += alpha * c13;
                C[cRow2 + j] += alpha * c20;     C[cRow2 + j + 1] += alpha * c21;
                C[cRow2 + j + 2] += alpha * c22; C[cRow2 + j + 3] += alpha * c23;
                C[cRow3 + j] += alpha * c30;     C[cRow3 + j + 1] += alpha * c31;
                C[cRow3 + j + 2] += alpha * c32; C[cRow3 + j + 3] += alpha * c33;
            }
            
            for (int j = n4; j < n; j++) {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                for (int l = 0; l < k; l++) {
                    double bj = B[bOff + j * ldb + l];
                    s0 += A[aRow0 + l] * bj;
                    s1 += A[aRow1 + l] * bj;
                    s2 += A[aRow2 + l] * bj;
                    s3 += A[aRow3 + l] * bj;
                }
                C[cRow0 + j] += alpha * s0;
                C[cRow1 + j] += alpha * s1;
                C[cRow2 + j] += alpha * s2;
                C[cRow3 + j] += alpha * s3;
            }
        }
        
        for (int i = m4; i < m; i++) {
            int aRow = aOff + i * lda;
            int cRow = cOff + i * ldc;
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[aRow + l] * B[bOff + j * ldb + l];
                }
                C[cRow + j] += alpha * sum;
            }
        }
    }

    /**
     * Blocked NT kernel: C := alpha * A * B^T + C.
     */
    static void dgemmNTBlocked(int m, int n, int k, double alpha,
                                double[] A, int aOff, int lda,
                                double[] B, int bOff, int ldb,
                                double[] C, int cOff, int ldc) {
        
        final int OUTER_M = BLOCK_M * 2;
        final int OUTER_N = BLOCK_N * 2;
        
        for (int ii = 0; ii < m; ii += OUTER_M) {
            int iMax = Math.min(ii + OUTER_M, m);
            
            for (int jj = 0; jj < n; jj += OUTER_N) {
                int jMax = Math.min(jj + OUTER_N, n);
                
                for (int kk = 0; kk < k; kk += BLOCK_K) {
                    int kEnd = Math.min(kk + BLOCK_K, k);
                    
                    for (int i = ii; i < iMax; i += BLOCK_M) {
                        int iEnd2 = Math.min(i + BLOCK_M, iMax);
                        
                        for (int j = jj; j < jMax; j += BLOCK_N) {
                            int jEnd2 = Math.min(j + BLOCK_N, jMax);
                            
                            dgemmNTBlock(i, iEnd2, j, jEnd2, kk, kEnd, alpha,
                                         A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
                        }
                    }
                }
            }
        }
    }

    static void dgemmNTBlock(int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd,
                              double alpha, double[] A, int aOff, int lda,
                              double[] B, int bOff, int ldb,
                              double[] C, int cOff, int ldc) {
        
        int i4End = iStart + ((iEnd - iStart) / 4) * 4;
        int j4End = jStart + ((jEnd - jStart) / 4) * 4;
        
        for (int i = iStart; i < i4End; i += 4) {
            int aRow0 = aOff + i * lda;
            int aRow1 = aOff + (i + 1) * lda;
            int aRow2 = aOff + (i + 2) * lda;
            int aRow3 = aOff + (i + 3) * lda;
            int cRow0 = cOff + i * ldc;
            int cRow1 = cOff + (i + 1) * ldc;
            int cRow2 = cOff + (i + 2) * ldc;
            int cRow3 = cOff + (i + 3) * ldc;
            
            for (int j = jStart; j < j4End; j += 4) {
                double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
                double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
                double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
                double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;
                
                for (int l = kStart; l < kEnd; l++) {
                    int bColJ = bOff + j * ldb + l;
                    int bColJ1 = bOff + (j + 1) * ldb + l;
                    int bColJ2 = bOff + (j + 2) * ldb + l;
                    int bColJ3 = bOff + (j + 3) * ldb + l;
                    
                    double a0l = A[aRow0 + l];
                    double a1l = A[aRow1 + l];
                    double a2l = A[aRow2 + l];
                    double a3l = A[aRow3 + l];
                    double bj0 = B[bColJ];
                    double bj1 = B[bColJ1];
                    double bj2 = B[bColJ2];
                    double bj3 = B[bColJ3];
                    
                    c00 = FMA.op(a0l, bj0, c00);  c01 = FMA.op(a0l, bj1, c01);
                    c02 = FMA.op(a0l, bj2, c02); c03 = FMA.op(a0l, bj3, c03);
                    c10 = FMA.op(a1l, bj0, c10);  c11 = FMA.op(a1l, bj1, c11);
                    c12 = FMA.op(a1l, bj2, c12); c13 = FMA.op(a1l, bj3, c13);
                    c20 = FMA.op(a2l, bj0, c20);  c21 = FMA.op(a2l, bj1, c21);
                    c22 = FMA.op(a2l, bj2, c22); c23 = FMA.op(a2l, bj3, c23);
                    c30 = FMA.op(a3l, bj0, c30);  c31 = FMA.op(a3l, bj1, c31);
                    c32 = FMA.op(a3l, bj2, c32); c33 = FMA.op(a3l, bj3, c33);
                }
                
                C[cRow0 + j] += alpha * c00;     C[cRow0 + j + 1] += alpha * c01;
                C[cRow0 + j + 2] += alpha * c02; C[cRow0 + j + 3] += alpha * c03;
                C[cRow1 + j] += alpha * c10;     C[cRow1 + j + 1] += alpha * c11;
                C[cRow1 + j + 2] += alpha * c12; C[cRow1 + j + 3] += alpha * c13;
                C[cRow2 + j] += alpha * c20;     C[cRow2 + j + 1] += alpha * c21;
                C[cRow2 + j + 2] += alpha * c22; C[cRow2 + j + 3] += alpha * c23;
                C[cRow3 + j] += alpha * c30;     C[cRow3 + j + 1] += alpha * c31;
                C[cRow3 + j + 2] += alpha * c32; C[cRow3 + j + 3] += alpha * c33;
            }
            
            for (int j = j4End; j < jEnd; j++) {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                for (int l = kStart; l < kEnd; l++) {
                    double bj = B[bOff + j * ldb + l];
                    s0 += A[aRow0 + l] * bj;
                    s1 += A[aRow1 + l] * bj;
                    s2 += A[aRow2 + l] * bj;
                    s3 += A[aRow3 + l] * bj;
                }
                C[cRow0 + j] += alpha * s0;
                C[cRow1 + j] += alpha * s1;
                C[cRow2 + j] += alpha * s2;
                C[cRow3 + j] += alpha * s3;
            }
        }
        
        for (int i = i4End; i < iEnd; i++) {
            int aRow = aOff + i * lda;
            int cRow = cOff + i * ldc;
            for (int j = jStart; j < jEnd; j++) {
                double sum = 0.0;
                for (int l = kStart; l < kEnd; l++) {
                    sum += A[aRow + l] * B[bOff + j * ldb + l];
                }
                C[cRow + j] += alpha * sum;
            }
        }
    }

    // ==================== TT (A^T * B^T) Implementation ====================

    /**
     * Direct TT kernel: C := alpha * A^T * B^T + C.
     */
    static void dgemmTTDirect(int m, int n, int k, double alpha,
                              double[] A, int aOff, int lda,
                              double[] B, int bOff, int ldb,
                              double[] C, int cOff, int ldc) {
        
        int m4 = (m / 4) * 4;
        int n4 = (n / 4) * 4;
        
        for (int i = 0; i < m4; i += 4) {
            int cRow0 = cOff + i * ldc;
            int cRow1 = cOff + (i + 1) * ldc;
            int cRow2 = cOff + (i + 2) * ldc;
            int cRow3 = cOff + (i + 3) * ldc;
            
            for (int j = 0; j < n4; j += 4) {
                double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
                double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
                double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
                double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;
                
                for (int l = 0; l < k; l++) {
                    // A^T: A[l*lda + i], B^T: B[j*ldb + l]
                    double a0l = A[aOff + l * lda + i];
                    double a1l = A[aOff + l * lda + i + 1];
                    double a2l = A[aOff + l * lda + i + 2];
                    double a3l = A[aOff + l * lda + i + 3];
                    
                    double bj0 = B[bOff + j * ldb + l];
                    double bj1 = B[bOff + (j + 1) * ldb + l];
                    double bj2 = B[bOff + (j + 2) * ldb + l];
                    double bj3 = B[bOff + (j + 3) * ldb + l];
                    
                    c00 += a0l * bj0; c01 += a0l * bj1; c02 += a0l * bj2; c03 += a0l * bj3;
                    c10 += a1l * bj0; c11 += a1l * bj1; c12 += a1l * bj2; c13 += a1l * bj3;
                    c20 += a2l * bj0; c21 += a2l * bj1; c22 += a2l * bj2; c23 += a2l * bj3;
                    c30 += a3l * bj0; c31 += a3l * bj1; c32 += a3l * bj2; c33 += a3l * bj3;
                }
                
                C[cRow0 + j] += alpha * c00;     C[cRow0 + j + 1] += alpha * c01;
                C[cRow0 + j + 2] += alpha * c02; C[cRow0 + j + 3] += alpha * c03;
                C[cRow1 + j] += alpha * c10;     C[cRow1 + j + 1] += alpha * c11;
                C[cRow1 + j + 2] += alpha * c12; C[cRow1 + j + 3] += alpha * c13;
                C[cRow2 + j] += alpha * c20;     C[cRow2 + j + 1] += alpha * c21;
                C[cRow2 + j + 2] += alpha * c22; C[cRow2 + j + 3] += alpha * c23;
                C[cRow3 + j] += alpha * c30;     C[cRow3 + j + 1] += alpha * c31;
                C[cRow3 + j + 2] += alpha * c32; C[cRow3 + j + 3] += alpha * c33;
            }
            
            for (int j = n4; j < n; j++) {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                for (int l = 0; l < k; l++) {
                    double bj = B[bOff + j * ldb + l];
                    s0 += A[aOff + l * lda + i] * bj;
                    s1 += A[aOff + l * lda + i + 1] * bj;
                    s2 += A[aOff + l * lda + i + 2] * bj;
                    s3 += A[aOff + l * lda + i + 3] * bj;
                }
                C[cRow0 + j] += alpha * s0;
                C[cRow1 + j] += alpha * s1;
                C[cRow2 + j] += alpha * s2;
                C[cRow3 + j] += alpha * s3;
            }
        }
        
        for (int i = m4; i < m; i++) {
            int cRow = cOff + i * ldc;
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[aOff + l * lda + i] * B[bOff + j * ldb + l];
                }
                C[cRow + j] += alpha * sum;
            }
        }
    }

    /**
     * Blocked TT kernel: C := alpha * A^T * B^T + C.
     */
    static void dgemmTTBlocked(int m, int n, int k, double alpha,
                                double[] A, int aOff, int lda,
                                double[] B, int bOff, int ldb,
                                double[] C, int cOff, int ldc) {
        
        final int OUTER_M = BLOCK_M * 2;
        final int OUTER_N = BLOCK_N * 2;
        
        for (int ii = 0; ii < m; ii += OUTER_M) {
            int iMax = Math.min(ii + OUTER_M, m);
            
            for (int jj = 0; jj < n; jj += OUTER_N) {
                int jMax = Math.min(jj + OUTER_N, n);
                
                for (int kk = 0; kk < k; kk += BLOCK_K) {
                    int kEnd = Math.min(kk + BLOCK_K, k);
                    
                    for (int i = ii; i < iMax; i += BLOCK_M) {
                        int iEnd2 = Math.min(i + BLOCK_M, iMax);
                        
                        for (int j = jj; j < jMax; j += BLOCK_N) {
                            int jEnd2 = Math.min(j + BLOCK_N, jMax);
                            
                            dgemmTTBlock(i, iEnd2, j, jEnd2, kk, kEnd, alpha,
                                         A, aOff, lda, B, bOff, ldb, C, cOff, ldc);
                        }
                    }
                }
            }
        }
    }

    static void dgemmTTBlock(int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd,
                              double alpha, double[] A, int aOff, int lda,
                              double[] B, int bOff, int ldb,
                              double[] C, int cOff, int ldc) {
        
        int i4End = iStart + ((iEnd - iStart) / 4) * 4;
        int j4End = jStart + ((jEnd - jStart) / 4) * 4;
        
        for (int i = iStart; i < i4End; i += 4) {
            int cRow0 = cOff + i * ldc;
            int cRow1 = cOff + (i + 1) * ldc;
            int cRow2 = cOff + (i + 2) * ldc;
            int cRow3 = cOff + (i + 3) * ldc;
            
            for (int j = jStart; j < j4End; j += 4) {
                double c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0;
                double c10 = 0.0, c11 = 0.0, c12 = 0.0, c13 = 0.0;
                double c20 = 0.0, c21 = 0.0, c22 = 0.0, c23 = 0.0;
                double c30 = 0.0, c31 = 0.0, c32 = 0.0, c33 = 0.0;
                
                for (int l = kStart; l < kEnd; l++) {
                    double a0l = A[aOff + l * lda + i];
                    double a1l = A[aOff + l * lda + i + 1];
                    double a2l = A[aOff + l * lda + i + 2];
                    double a3l = A[aOff + l * lda + i + 3];
                    
                    double bj0 = B[bOff + j * ldb + l];
                    double bj1 = B[bOff + (j + 1) * ldb + l];
                    double bj2 = B[bOff + (j + 2) * ldb + l];
                    double bj3 = B[bOff + (j + 3) * ldb + l];
                    
                    c00 += a0l * bj0; c01 += a0l * bj1; c02 += a0l * bj2; c03 += a0l * bj3;
                    c10 += a1l * bj0; c11 += a1l * bj1; c12 += a1l * bj2; c13 += a1l * bj3;
                    c20 += a2l * bj0; c21 += a2l * bj1; c22 += a2l * bj2; c23 += a2l * bj3;
                    c30 += a3l * bj0; c31 += a3l * bj1; c32 += a3l * bj2; c33 += a3l * bj3;
                }
                
                C[cRow0 + j] += alpha * c00;     C[cRow0 + j + 1] += alpha * c01;
                C[cRow0 + j + 2] += alpha * c02; C[cRow0 + j + 3] += alpha * c03;
                C[cRow1 + j] += alpha * c10;     C[cRow1 + j + 1] += alpha * c11;
                C[cRow1 + j + 2] += alpha * c12; C[cRow1 + j + 3] += alpha * c13;
                C[cRow2 + j] += alpha * c20;     C[cRow2 + j + 1] += alpha * c21;
                C[cRow2 + j + 2] += alpha * c22; C[cRow2 + j + 3] += alpha * c23;
                C[cRow3 + j] += alpha * c30;     C[cRow3 + j + 1] += alpha * c31;
                C[cRow3 + j + 2] += alpha * c32; C[cRow3 + j + 3] += alpha * c33;
            }
            
            for (int j = j4End; j < jEnd; j++) {
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                for (int l = kStart; l < kEnd; l++) {
                    double bj = B[bOff + j * ldb + l];
                    s0 += A[aOff + l * lda + i] * bj;
                    s1 += A[aOff + l * lda + i + 1] * bj;
                    s2 += A[aOff + l * lda + i + 2] * bj;
                    s3 += A[aOff + l * lda + i + 3] * bj;
                }
                C[cRow0 + j] += alpha * s0;
                C[cRow1 + j] += alpha * s1;
                C[cRow2 + j] += alpha * s2;
                C[cRow3 + j] += alpha * s3;
            }
        }
        
        for (int i = i4End; i < iEnd; i++) {
            int cRow = cOff + i * ldc;
            for (int j = jStart; j < jEnd; j++) {
                double sum = 0.0;
                for (int l = kStart; l < kEnd; l++) {
                    sum += A[aOff + l * lda + i] * B[bOff + j * ldb + l];
                }
                C[cRow + j] += alpha * sum;
            }
        }
    }

}
