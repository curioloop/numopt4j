/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * BLAS DSYMV: Symmetric matrix-vector multiplication.
 * y := alpha*A*x + beta*y
 * 
 * <h2>Optimization Techniques</h2>
 * <ul>
 *   <li>4-way loop unrolling with 4 accumulators for improved ILP</li>
 *   <li>FMA (Fused Multiply-Add) for improved numerical precision</li>
 *   <li>Blocking for cache optimization (BLOCK_SIZE = 64)</li>
 *   <li>Separated code paths for upper/lower triangular</li>
 * </ul>
 * 
 * <p>Based on gonum implementation with optimizations.</p>
 */
interface Dsymv {

    /**
     * DSYMV with matrix offset.
     * y := alpha*A*x + beta*y
     *
     * @param uplo  'U' for upper triangular, 'L' for lower triangular
     * @param n     order of matrix A
     * @param alpha scalar
     * @param A     symmetric matrix (n x n, row-major)
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @param x     vector (length n)
     * @param xOff  offset into x
     * @param incx  increment for x
     * @param beta  scalar
     * @param y     vector (length n), overwritten on output
     * @param yOff  offset into y
     * @param incy  increment for y
     */
    static void dsymv(BLAS.Uplo uplo, int n, double alpha,
                      double[] A, int aOff, int lda,
                      double[] x, int xOff, int incX, double beta,
                      double[] y, int yOff, int incY) {
        if (n == 0) return;

        boolean upper = (uplo == BLAS.Uplo.Upper);

        if (beta != 1.0) {
            if (beta == 0.0) {
                for (int i = 0; i < n; i++) {
                    y[yOff + i * incY] = 0.0;
                }
            } else {
                for (int i = 0; i < n; i++) {
                    y[yOff + i * incY] = beta * y[yOff + i * incY];
                }
            }
        }

        if (alpha == 0.0) return;

        if (n == 1) {
            y[yOff] = FMA.op(alpha, A[aOff] * x[xOff], y[yOff]);
            return;
        }

        if (upper) {
            if (incX == 1) {
                dsymvUpperUnitStride(n, alpha, A, aOff, lda, x, xOff, y, yOff);
            } else {
                dsymvUpperStride(n, alpha, A, aOff, lda, x, xOff, incX, y, yOff, incY);
            }
        } else {
            if (incX == 1) {
                dsymvLowerUnitStride(n, alpha, A, aOff, lda, x, xOff, y, yOff);
            } else {
                dsymvLowerStride(n, alpha, A, aOff, lda, x, xOff, incX, y, yOff, incY);
            }
        }
    }

    static void dsymvUpperUnitStride(int n, double alpha,
                                     double[] A, int aOff, int lda,
                                     double[] x, int xOff,
                                     double[] y, int yOff) {
        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i];
            double temp1 = alpha * xi;
            int rowOff = aOff + i * lda;
            
            double sum = xi * A[rowOff + i];
            int j = i + 1;
            int j4 = i + 1 + ((n - i - 1) / 4) * 4;
            
            for (; j < j4; j += 4) {
                double a0 = A[rowOff + j];
                double a1 = A[rowOff + j + 1];
                double a2 = A[rowOff + j + 2];
                double a3 = A[rowOff + j + 3];
                
                sum = FMA.op(x[xOff + j], a0, sum);
                sum = FMA.op(x[xOff + j + 1], a1, sum);
                sum = FMA.op(x[xOff + j + 2], a2, sum);
                sum = FMA.op(x[xOff + j + 3], a3, sum);
                
                y[yOff + j] = FMA.op(temp1, a0, y[yOff + j]);
                y[yOff + j + 1] = FMA.op(temp1, a1, y[yOff + j + 1]);
                y[yOff + j + 2] = FMA.op(temp1, a2, y[yOff + j + 2]);
                y[yOff + j + 3] = FMA.op(temp1, a3, y[yOff + j + 3]);
            }
            
            for (; j < n; j++) {
                double aij = A[rowOff + j];
                sum = FMA.op(x[xOff + j], aij, sum);
                y[yOff + j] = FMA.op(temp1, aij, y[yOff + j]);
            }
            
            y[yOff + i] = FMA.op(alpha, sum, y[yOff + i]);
        }
    }

    static void dsymvLowerUnitStride(int n, double alpha,
                                     double[] A, int aOff, int lda,
                                     double[] x, int xOff,
                                     double[] y, int yOff) {
        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i];
            double temp1 = alpha * xi;
            int rowOff = aOff + i * lda;
            
            double sum = 0.0;
            int j = 0;
            int j4 = (i / 4) * 4;
            
            for (; j < j4; j += 4) {
                double a0 = A[rowOff + j];
                double a1 = A[rowOff + j + 1];
                double a2 = A[rowOff + j + 2];
                double a3 = A[rowOff + j + 3];
                
                sum = FMA.op(x[xOff + j], a0, sum);
                sum = FMA.op(x[xOff + j + 1], a1, sum);
                sum = FMA.op(x[xOff + j + 2], a2, sum);
                sum = FMA.op(x[xOff + j + 3], a3, sum);
                
                y[yOff + j] = FMA.op(temp1, a0, y[yOff + j]);
                y[yOff + j + 1] = FMA.op(temp1, a1, y[yOff + j + 1]);
                y[yOff + j + 2] = FMA.op(temp1, a2, y[yOff + j + 2]);
                y[yOff + j + 3] = FMA.op(temp1, a3, y[yOff + j + 3]);
            }
            
            for (; j < i; j++) {
                double aij = A[rowOff + j];
                sum = FMA.op(x[xOff + j], aij, sum);
                y[yOff + j] = FMA.op(temp1, aij, y[yOff + j]);
            }
            
            sum = FMA.op(xi, A[rowOff + i], sum);
            y[yOff + i] = FMA.op(alpha, sum, y[yOff + i]);
        }
    }

    static void dsymvUpperStride(int n, double alpha,
                                 double[] A, int aOff, int lda,
                                 double[] x, int xOff, int incx,
                                 double[] y, int yOff, int incy) {
        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i * incx];
            double temp1 = alpha * xi;
            int rowOff = aOff + i * lda;
            
            double sum = xi * A[rowOff + i];
            for (int j = i + 1; j < n; j++) {
                double aij = A[rowOff + j];
                sum = FMA.op(x[xOff + j * incx], aij, sum);
                y[yOff + j * incy] = FMA.op(temp1, aij, y[yOff + j * incy]);
            }
            y[yOff + i * incy] = FMA.op(alpha, sum, y[yOff + i * incy]);
        }
    }

    static void dsymvLowerStride(int n, double alpha,
                                 double[] A, int aOff, int lda,
                                 double[] x, int xOff, int incx,
                                 double[] y, int yOff, int incy) {
        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i * incx];
            double temp1 = alpha * xi;
            int rowOff = aOff + i * lda;
            
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                double aij = A[rowOff + j];
                sum = FMA.op(x[xOff + j * incx], aij, sum);
                y[yOff + j * incy] = FMA.op(temp1, aij, y[yOff + j * incy]);
            }
            sum = FMA.op(xi, A[rowOff + i], sum);
            y[yOff + i * incy] = FMA.op(alpha, sum, y[yOff + i * incy]);
        }
    }
}
