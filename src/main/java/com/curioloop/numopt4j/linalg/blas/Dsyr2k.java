/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * BLAS DSYR2K: Symmetric rank-2k update.
 * C := alpha*A*B^T + alpha*B*A^T + beta*C
 * 
 * Based on gonum implementation.
 */
interface Dsyr2k {

    int THRESHOLD = 64;

    static void dsyr2k(BLAS.Uplo uplo, BLAS.Transpose trans, int n, int k, double alpha,
                       double[] A, int lda, double[] B, int ldb, double beta,
                       double[] C, int ldc) {
        dsyr2k(uplo, trans, n, k, alpha, A, 0, lda, B, 0, ldb, beta, C, 0, ldc);
    }

    /**
     * DSYR2K performs the symmetric rank-2k update with offsets.
     * C := alpha*A*B^T + alpha*B*A^T + beta*C
     *
     * @param uplo  'U' for upper triangular, 'L' for lower triangular
     * @param trans 'N' for NoTrans (A*B^T), 'T' for Trans (A^T*B)
     * @param n     order of matrix C
     * @param k     number of columns of matrices A and B
     * @param alpha scalar
     * @param A     matrix (n x k for NoTrans, k x n for Trans)
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @param B     matrix (n x k for NoTrans, k x n for Trans)
     * @param bOff  offset into B
     * @param ldb   leading dimension of B
     * @param beta  scalar
     * @param C     symmetric matrix (n x n, row-major), overwritten
     * @param cOff  offset into C
     * @param ldc   leading dimension of C
     */
    static void dsyr2k(BLAS.Uplo uplo, BLAS.Transpose trans, int n, int k, double alpha,
                       double[] A, int aOff, int lda, double[] B, int bOff, int ldb, 
                       double beta, double[] C, int cOff, int ldc) {
        if (n == 0 || (alpha == 0.0 && beta == 1.0)) {
            return;
        }

        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean noTrans = trans == BLAS.Transpose.NoTrans;

        if (beta != 1.0) {
            scaleTriangular(C, cOff, ldc, n, beta, upper);
        }

        if (alpha == 0.0) {
            return;
        }

        if (noTrans) {
            syr2kNoTrans(n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, upper);
        } else {
            syr2kTrans(n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, upper);
        }
    }

    static void syr2kNoTrans(int n, int k, double alpha,
                             double[] A, int aOff, int lda,
                             double[] B, int bOff, int ldb,
                             double[] C, int cOff, int ldc, boolean upper) {
        if (k <= THRESHOLD) {
            syr2kNoTransBase(n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, upper);
            return;
        }
        int mid = k / 2;
        syr2kNoTrans(n, mid, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, upper);
        syr2kNoTrans(n, k - mid, alpha, A, aOff + mid, lda, B, bOff + mid, ldb, C, cOff, ldc, upper);
    }

    static void syr2kNoTransBase(int n, int k, double alpha,
                                 double[] A, int aOff, int lda,
                                 double[] B, int bOff, int ldb,
                                 double[] C, int cOff, int ldc, boolean upper) {
        if (upper) {
            for (int i = 0; i < n; i++) {
                int aiOff = aOff + i * lda;
                int biOff = bOff + i * ldb;
                int ciOff = cOff + i * ldc;
                for (int j = i; j < n; j++) {
                    int ajOff = aOff + j * lda;
                    int bjOff = bOff + j * ldb;
                    double sum1 = 0.0;
                    double sum2 = 0.0;
                    int l = 0;
                    for (; l + 3 < k; l += 4) {
                        sum1 = FMA.op(A[aiOff + l], B[bjOff + l], sum1);
                        sum1 = FMA.op(A[aiOff + l + 1], B[bjOff + l + 1], sum1);
                        sum1 = FMA.op(A[aiOff + l + 2], B[bjOff + l + 2], sum1);
                        sum1 = FMA.op(A[aiOff + l + 3], B[bjOff + l + 3], sum1);
                        sum2 = FMA.op(B[biOff + l], A[ajOff + l], sum2);
                        sum2 = FMA.op(B[biOff + l + 1], A[ajOff + l + 1], sum2);
                        sum2 = FMA.op(B[biOff + l + 2], A[ajOff + l + 2], sum2);
                        sum2 = FMA.op(B[biOff + l + 3], A[ajOff + l + 3], sum2);
                    }
                    for (; l < k; l++) {
                        sum1 = FMA.op(A[aiOff + l], B[bjOff + l], sum1);
                        sum2 = FMA.op(B[biOff + l], A[ajOff + l], sum2);
                    }
                    C[ciOff + j] += alpha * (sum1 + sum2);
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                int aiOff = aOff + i * lda;
                int biOff = bOff + i * ldb;
                int ciOff = cOff + i * ldc;
                for (int j = 0; j <= i; j++) {
                    int ajOff = aOff + j * lda;
                    int bjOff = bOff + j * ldb;
                    double sum1 = 0.0;
                    double sum2 = 0.0;
                    int l = 0;
                    for (; l + 3 < k; l += 4) {
                        sum1 = FMA.op(A[aiOff + l], B[bjOff + l], sum1);
                        sum1 = FMA.op(A[aiOff + l + 1], B[bjOff + l + 1], sum1);
                        sum1 = FMA.op(A[aiOff + l + 2], B[bjOff + l + 2], sum1);
                        sum1 = FMA.op(A[aiOff + l + 3], B[bjOff + l + 3], sum1);
                        sum2 = FMA.op(B[biOff + l], A[ajOff + l], sum2);
                        sum2 = FMA.op(B[biOff + l + 1], A[ajOff + l + 1], sum2);
                        sum2 = FMA.op(B[biOff + l + 2], A[ajOff + l + 2], sum2);
                        sum2 = FMA.op(B[biOff + l + 3], A[ajOff + l + 3], sum2);
                    }
                    for (; l < k; l++) {
                        sum1 = FMA.op(A[aiOff + l], B[bjOff + l], sum1);
                        sum2 = FMA.op(B[biOff + l], A[ajOff + l], sum2);
                    }
                    C[ciOff + j] += alpha * (sum1 + sum2);
                }
            }
        }
    }

    static void syr2kTrans(int n, int k, double alpha,
                           double[] A, int aOff, int lda,
                           double[] B, int bOff, int ldb,
                           double[] C, int cOff, int ldc, boolean upper) {
        if (k <= THRESHOLD) {
            syr2kTransBase(n, k, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, upper);
            return;
        }
        int mid = k / 2;
        syr2kTrans(n, mid, alpha, A, aOff, lda, B, bOff, ldb, C, cOff, ldc, upper);
        syr2kTrans(n, k - mid, alpha, A, aOff + mid * lda, lda, B, bOff + mid * ldb, ldb, C, cOff, ldc, upper);
    }

    static void syr2kTransBase(int n, int k, double alpha,
                               double[] A, int aOff, int lda,
                               double[] B, int bOff, int ldb,
                               double[] C, int cOff, int ldc, boolean upper) {
        if (upper) {
            for (int p = 0; p < k; p++) {
                int apOff = aOff + p * lda;
                int bpOff = bOff + p * ldb;
                for (int i = 0; i < n; i++) {
                    double alphaAi = alpha * A[apOff + i];
                    double alphaBi = alpha * B[bpOff + i];
                    int ciOff = cOff + i * ldc;
                    for (int j = i; j < n; j++) {
                        C[ciOff + j] = FMA.op(alphaAi, B[bpOff + j], C[ciOff + j]);
                        C[ciOff + j] = FMA.op(alphaBi, A[apOff + j], C[ciOff + j]);
                    }
                }
            }
        } else {
            for (int p = 0; p < k; p++) {
                int apOff = aOff + p * lda;
                int bpOff = bOff + p * ldb;
                for (int i = 0; i < n; i++) {
                    double alphaAi = alpha * A[apOff + i];
                    double alphaBi = alpha * B[bpOff + i];
                    int ciOff = cOff + i * ldc;
                    for (int j = 0; j <= i; j++) {
                        C[ciOff + j] = FMA.op(alphaAi, B[bpOff + j], C[ciOff + j]);
                        C[ciOff + j] = FMA.op(alphaBi, A[apOff + j], C[ciOff + j]);
                    }
                }
            }
        }
    }

    static void scaleTriangular(double[] C, int cOff, int ldc, int n, double beta, boolean upper) {
        if (beta == 0.0) {
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
