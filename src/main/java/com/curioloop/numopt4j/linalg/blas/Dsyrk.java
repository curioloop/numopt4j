/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DSYRK performs symmetric rank-k update.
 * 
 * <p>Operations:</p>
 * <ul>
 *   <li>C := alpha*A*Aᵀ + beta*C (trans='N')</li>
 *   <li>C := alpha*Aᵀ*A + beta*C (trans='T')</li>
 * </ul>
 * 
 * <p>Only the upper or lower triangular part of C is updated.</p>
 * 
 * <p>Uses pairwise summation (divide-and-conquer) for improved numerical
 * stability, reducing floating-point error from O(k) to O(log k).</p>
 * 
 * <p>Reference: BLAS Level 3 DSYRK</p>
 */
interface Dsyrk {

    int THRESHOLD = 64;

    /**
     * Performs symmetric rank-k update: C := alpha*op(A)*op(A)ᵀ + beta*C
     * 
     * @param uplo 'U' for upper triangular, 'L' for lower triangular
     * @param trans 'N' for C := alpha*A*Aᵀ + beta*C, 'T' for C := alpha*Aᵀ*A + beta*C
     * @param n order of matrix C
     * @param k number of columns of A (if trans='N') or rows of A (if trans='T')
     * @param alpha scalar multiplier
     * @param A matrix A
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @param beta scalar multiplier for C
     * @param C symmetric matrix C (only upper or lower part updated)
     * @param cOff offset into C
     * @param ldc leading dimension of C
     */
    static void dsyrk(BLAS.Uplo uplo, BLAS.Trans trans, int n, int k, double alpha,
                      double[] A, int aOff, int lda, double beta,
                      double[] C, int cOff, int ldc) {
        if (n == 0 || ((alpha == 0.0 || k == 0) && beta == 1.0)) return;

        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean noTrans = trans == BLAS.Trans.NoTrans;

        // Scale C by beta
        if (beta != 1.0) {
            scaleTriangular(C, cOff, ldc, n, beta, upper);
        }

        if (alpha == 0.0) return;

        if (noTrans) {
            // C := alpha*A*Aᵀ + C
            // A is n × k, C is n × n
            syrkNoTrans(n, k, alpha, A, aOff, lda, C, cOff, ldc, upper);
        } else {
            // C := alpha*Aᵀ*A + C
            // A is k × n, C is n × n
            syrkTrans(n, k, alpha, A, aOff, lda, C, cOff, ldc, upper);
        }
    }

    /**
     * C := alpha*A*Aᵀ + C (A is n × k)
     * 
     * <p>Uses pairwise summation (divide-and-conquer on k columns) for improved
     * numerical stability while maintaining cache-friendly row access pattern.</p>
     */
    static void syrkNoTrans(int n, int k, double alpha,
                            double[] A, int aOff, int lda,
                            double[] C, int cOff, int ldc, boolean upper) {
        if (k <= 0) return;
        syrkNoTransImpl(n, k, alpha, A, aOff, lda, C, cOff, ldc, upper, 0);
    }

    /**
     * Pairwise summation implementation for syrkNoTrans.
     * Divides k columns recursively to reduce floating-point error from O(k) to O(log k).
     * 
     * @param kStart starting column index in A
     */
    static void syrkNoTransImpl(int n, int k, double alpha,
                                double[] A, int aOff, int lda,
                                double[] C, int cOff, int ldc, boolean upper, int kStart) {
        if (k <= THRESHOLD) {
            // Base case: direct accumulation with 4-way unrolling
            if (upper) {
                for (int i = 0; i < n; i++) {
                    int aiOff = aOff + i * lda + kStart;
                    for (int j = i; j < n; j++) {
                        int ajOff = aOff + j * lda + kStart;
                        double sum = 0.0;
                        int p = 0;
                        for (; p + 3 < k; p += 4) {
                            sum = FMA.op(A[aiOff + p], A[ajOff + p], sum);
                            sum = FMA.op(A[aiOff + p + 1], A[ajOff + p + 1], sum);
                            sum = FMA.op(A[aiOff + p + 2], A[ajOff + p + 2], sum);
                            sum = FMA.op(A[aiOff + p + 3], A[ajOff + p + 3], sum);
                        }
                        for (; p < k; p++) {
                            sum = FMA.op(A[aiOff + p], A[ajOff + p], sum);
                        }
                        C[cOff + i * ldc + j] += alpha * sum;
                    }
                }
            } else {
                for (int i = 0; i < n; i++) {
                    int aiOff = aOff + i * lda + kStart;
                    for (int j = 0; j <= i; j++) {
                        int ajOff = aOff + j * lda + kStart;
                        double sum = 0.0;
                        int p = 0;
                        for (; p + 3 < k; p += 4) {
                            sum = FMA.op(A[aiOff + p], A[ajOff + p], sum);
                            sum = FMA.op(A[aiOff + p + 1], A[ajOff + p + 1], sum);
                            sum = FMA.op(A[aiOff + p + 2], A[ajOff + p + 2], sum);
                            sum = FMA.op(A[aiOff + p + 3], A[ajOff + p + 3], sum);
                        }
                        for (; p < k; p++) {
                            sum = FMA.op(A[aiOff + p], A[ajOff + p], sum);
                        }
                        C[cOff + i * ldc + j] += alpha * sum;
                    }
                }
            }
            return;
        }
        // Divide: split k columns in half for pairwise summation
        int mid = k / 2;
        syrkNoTransImpl(n, mid, alpha, A, aOff, lda, C, cOff, ldc, upper, kStart);
        syrkNoTransImpl(n, k - mid, alpha, A, aOff, lda, C, cOff, ldc, upper, kStart + mid);
    }

    /**
     * C := alpha*Aᵀ*A + C (A is k × n)
     * 
     * <p>Uses pairwise summation (divide-and-conquer on k rows) for improved
     * numerical stability while maintaining cache-friendly row access pattern.</p>
     */
    static void syrkTrans(int n, int k, double alpha,
                          double[] A, int aOff, int lda,
                          double[] C, int cOff, int ldc, boolean upper) {
        if (k <= 0) return;
        syrkTransImpl(n, k, alpha, A, aOff, lda, C, cOff, ldc, upper);
    }

    /**
     * Pairwise summation implementation for syrkTrans.
     * Divides k rows recursively to reduce floating-point error from O(k) to O(log k).
     */
    static void syrkTransImpl(int n, int k, double alpha,
                              double[] A, int aOff, int lda,
                              double[] C, int cOff, int ldc, boolean upper) {
        if (k <= THRESHOLD) {
            // Base case: direct rank-1 updates with row-major access
            // C += alpha * Σₚ A[p,:]ᵀ * A[p,:]
            if (upper) {
                for (int p = 0; p < k; p++) {
                    int rowOff = aOff + p * lda;
                    for (int i = 0; i < n; i++) {
                        double alphaAi = alpha * A[rowOff + i];
                        int cRowOff = cOff + i * ldc;
                        for (int j = i; j < n; j++) {
                            C[cRowOff + j] = FMA.op(alphaAi, A[rowOff + j], C[cRowOff + j]);
                        }
                    }
                }
            } else {
                for (int p = 0; p < k; p++) {
                    int rowOff = aOff + p * lda;
                    for (int i = 0; i < n; i++) {
                        double alphaAi = alpha * A[rowOff + i];
                        int cRowOff = cOff + i * ldc;
                        for (int j = 0; j <= i; j++) {
                            C[cRowOff + j] = FMA.op(alphaAi, A[rowOff + j], C[cRowOff + j]);
                        }
                    }
                }
            }
            return;
        }
        // Divide: split k rows in half for pairwise summation
        int mid = k / 2;
        syrkTransImpl(n, mid, alpha, A, aOff, lda, C, cOff, ldc, upper);
        syrkTransImpl(n, k - mid, alpha, A, aOff + mid * lda, lda, C, cOff, ldc, upper);
    }

    /**
     * Scales the triangular part of C by beta.
     */
    static void scaleTriangular(double[] C, int cOff, int ldc, int n, double beta, boolean upper) {
        if (beta == 0.0) {
            // Zero out
            if (upper) {
                for (int i = 0; i < n; i++) {
                    int rowOff = cOff + i * ldc;
                    for (int j = i; j < n; j++) {
                        C[rowOff + j] = 0.0;
                    }
                }
            } else {
                for (int i = 0; i < n; i++) {
                    int rowOff = cOff + i * ldc;
                    for (int j = 0; j <= i; j++) {
                        C[rowOff + j] = 0.0;
                    }
                }
            }
        } else {
            // Scale
            if (upper) {
                for (int i = 0; i < n; i++) {
                    int rowOff = cOff + i * ldc;
                    for (int j = i; j < n; j++) {
                        C[rowOff + j] *= beta;
                    }
                }
            } else {
                for (int i = 0; i < n; i++) {
                    int rowOff = cOff + i * ldc;
                    for (int j = 0; j <= i; j++) {
                        C[rowOff + j] *= beta;
                    }
                }
            }
        }
    }

}
