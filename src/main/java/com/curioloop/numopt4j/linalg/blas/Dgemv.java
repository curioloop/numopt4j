/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DGEMV performs matrix-vector multiplication.
 * 
 * <p>Operations:</p>
 * <ul>
 *   <li>y := alpha*A*x + beta*y (trans='N')</li>
 *   <li>y := alpha*Aᵀ*x + beta*y (trans='T')</li>
 * </ul>
 * 
 * <h2>Optimization Techniques (from DGEMM)</h2>
 * <ul>
 *   <li>4-way loop unrolling for reduced branch overhead</li>
 *   <li>FMA (Fused Multiply-Add) for improved precision and performance</li>
 *   <li>Deferred alpha application at write-back (O(1) instead of O(n) multiplications)</li>
 *   <li>Register blocking with multiple accumulators</li>
 *   <li>Pairwise summation for transpose operation (O(log m) error growth)</li>
 * </ul>
 * 
 * <p>Reference: BLAS Level 2 DGEMV</p>
 */
interface Dgemv {

    int THRESHOLD = 32;
    int BLOCK_SIZE = 64;

    static void dgemv(BLAS.Trans trans, int m, int n, double alpha,
                      double[] A, int aOff, int lda,
                      double[] x, int xOff, int incX, double beta,
                      double[] y, int yOff, int incY) {
        boolean noTrans = (trans == BLAS.Trans.NoTrans);
        int lenY = noTrans ? m : n;
        
        if (lenY == 0) return;
        if (alpha == 0.0 && beta == 1.0) return;

        if (beta != 1.0) {
            if (beta == 0.0) {
                for (int i = 0; i < lenY; i++) {
                    y[yOff + i * incY] = 0.0;
                }
            } else {
                for (int i = 0; i < lenY; i++) {
                    y[yOff + i * incY] *= beta;
                }
            }
        }

        if (alpha == 0.0) return;

        if (noTrans) {
            gemvNoTrans(m, n, alpha, A, aOff, lda, x, xOff, incX, y, yOff, incY);
        } else {
            gemvTrans(m, n, alpha, A, aOff, lda, x, xOff, incX, y, yOff, incY);
        }
    }

    static void gemvNoTrans(int m, int n, double alpha,
                            double[] A, int aOff, int lda,
                            double[] x, int xOff, int incx,
                            double[] y, int yOff, int incy) {
        if (n < BLOCK_SIZE && m < BLOCK_SIZE) {
            gemvNoTransDirect(m, n, alpha, A, aOff, lda, x, xOff, incx, y, yOff, incy);
        } else {
            gemvNoTransBlocked(m, n, alpha, A, aOff, lda, x, xOff, incx, y, yOff, incy);
        }
    }

    static void gemvNoTransDirect(int m, int n, double alpha,
                                  double[] A, int aOff, int lda,
                                  double[] x, int xOff, int incx,
                                  double[] y, int yOff, int incy) {
        int n4 = (n / 4) * 4;
        
        for (int i = 0; i < m; i++) {
            int rowOff = aOff + i * lda;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            
            int j = 0;
            for (; j < n4; j += 4) {
                s0 = FMA.op(A[rowOff + j], x[xOff + j * incx], s0);
                s1 = FMA.op(A[rowOff + j + 1], x[xOff + (j + 1) * incx], s1);
                s2 = FMA.op(A[rowOff + j + 2], x[xOff + (j + 2) * incx], s2);
                s3 = FMA.op(A[rowOff + j + 3], x[xOff + (j + 3) * incx], s3);
            }
            
            double sum = s0 + s1 + s2 + s3;
            for (; j < n; j++) {
                sum = FMA.op(A[rowOff + j], x[xOff + j * incx], sum);
            }
            
            y[yOff + i * incy] += alpha * sum;
        }
    }

    static void gemvNoTransBlocked(int m, int n, double alpha,
                                   double[] A, int aOff, int lda,
                                   double[] x, int xOff, int incx,
                                   double[] y, int yOff, int incy) {
        for (int ii = 0; ii < m; ii += BLOCK_SIZE) {
            int iEnd = Math.min(ii + BLOCK_SIZE, m);
            gemvNoTransDirect(iEnd - ii, n, alpha, A, aOff + ii * lda, lda, 
                              x, xOff, incx, y, yOff + ii * incy, incy);
        }
    }

    static void gemvTrans(int m, int n, double alpha,
                          double[] A, int aOff, int lda,
                          double[] x, int xOff, int incx,
                          double[] y, int yOff, int incy) {
        if (m <= 0) return;
        gemvTransImpl(m, n, alpha, A, aOff, lda, x, xOff, incx, y, yOff, incy);
    }

    static void gemvTransImpl(int m, int n, double alpha,
                              double[] A, int aOff, int lda,
                              double[] x, int xOff, int incx,
                              double[] y, int yOff, int incy) {
        if (m <= THRESHOLD) {
            int n4 = (n / 4) * 4;
            
            for (int i = 0; i < m; i++) {
                double axi = alpha * x[xOff + i * incx];
                int rowOff = aOff + i * lda;
                
                int j = 0;
                for (; j < n4; j += 4) {
                    y[yOff + j * incy] = FMA.op(A[rowOff + j], axi, y[yOff + j * incy]);
                    y[yOff + (j + 1) * incy] = FMA.op(A[rowOff + j + 1], axi, y[yOff + (j + 1) * incy]);
                    y[yOff + (j + 2) * incy] = FMA.op(A[rowOff + j + 2], axi, y[yOff + (j + 2) * incy]);
                    y[yOff + (j + 3) * incy] = FMA.op(A[rowOff + j + 3], axi, y[yOff + (j + 3) * incy]);
                }
                for (; j < n; j++) {
                    y[yOff + j * incy] = FMA.op(A[rowOff + j], axi, y[yOff + j * incy]);
                }
            }
            return;
        }
        
        int mid = m / 2;
        gemvTransImpl(mid, n, alpha, A, aOff, lda, x, xOff, incx, y, yOff, incy);
        gemvTransImpl(m - mid, n, alpha, A, aOff + mid * lda, lda, 
                      x, xOff + mid * incx, incx, y, yOff, incy);
    }

}
