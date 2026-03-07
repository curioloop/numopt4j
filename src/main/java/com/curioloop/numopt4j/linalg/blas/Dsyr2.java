/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * BLAS DSYR2: Symmetric rank-2 update.
 * A := alpha*x*y' + alpha*y*x' + A
 * 
 * <h2>Optimization Techniques</h2>
 * <ul>
 *   <li>4-way loop unrolling for improved ILP</li>
 *   <li>FMA (Fused Multiply-Add) for improved numerical precision</li>
 *   <li>Register accumulation to reduce memory access</li>
 *   <li>Separated code paths for upper/lower triangular and unit-stride</li>
 * </ul>
 * 
 */
interface Dsyr2 {

    /**
     * DSYR2 with matrix offset.
     * A := alpha*x*y' + alpha*y*x' + A
     *
     * @param uplo  'U' for upper triangular, 'L' for lower triangular
     * @param n     order of matrix A
     * @param alpha scalar
     * @param x     vector (length n)
     * @param xOff  offset into x
     * @param incx  increment for x
     * @param y     vector (length n)
     * @param yOff  offset into y
     * @param incy  increment for y
     * @param A     symmetric matrix (n x n, row-major), overwritten
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     */
    static void dsyr2(BLAS.Uplo uplo, int n, double alpha,
                      double[] x, int xOff, int incX,
                      double[] y, int yOff, int incY,
                      double[] A, int aOff, int lda) {
        if (n == 0 || alpha == 0.0) return;

        boolean upper = (uplo == BLAS.Uplo.Upper);

        if (incX == 1 && incY == 1) {
            if (upper) {
                dsyr2UpperUnitStride(n, alpha, x, xOff, y, yOff, A, aOff, lda);
            } else {
                dsyr2LowerUnitStride(n, alpha, x, xOff, y, yOff, A, aOff, lda);
            }
        } else {
            if (upper) {
                dsyr2UpperStride(n, alpha, x, xOff, incX, y, yOff, incY, A, aOff, lda);
            } else {
                dsyr2LowerStride(n, alpha, x, xOff, incX, y, yOff, incY, A, aOff, lda);
            }
        }
    }

    static void dsyr2UpperUnitStride(int n, double alpha,
                                     double[] x, int xOff,
                                     double[] y, int yOff,
                                     double[] A, int aOff, int lda) {
        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i];
            double yi = y[yOff + i];
            double temp1 = alpha * xi;
            double temp2 = alpha * yi;
            int rowOff = aOff + i * lda;
            
            int j = i;
            int j4 = i + ((n - i) / 4) * 4;
            
            for (; j < j4; j += 4) {
                double t0 = A[rowOff + j];
                double t1 = A[rowOff + j + 1];
                double t2 = A[rowOff + j + 2];
                double t3 = A[rowOff + j + 3];
                
                t0 = FMA.op(temp1, y[yOff + j], t0);
                t0 = FMA.op(temp2, x[xOff + j], t0);
                t1 = FMA.op(temp1, y[yOff + j + 1], t1);
                t1 = FMA.op(temp2, x[xOff + j + 1], t1);
                t2 = FMA.op(temp1, y[yOff + j + 2], t2);
                t2 = FMA.op(temp2, x[xOff + j + 2], t2);
                t3 = FMA.op(temp1, y[yOff + j + 3], t3);
                t3 = FMA.op(temp2, x[xOff + j + 3], t3);
                
                A[rowOff + j] = t0;
                A[rowOff + j + 1] = t1;
                A[rowOff + j + 2] = t2;
                A[rowOff + j + 3] = t3;
            }
            
            for (; j < n; j++) {
                double t = A[rowOff + j];
                t = FMA.op(temp1, y[yOff + j], t);
                t = FMA.op(temp2, x[xOff + j], t);
                A[rowOff + j] = t;
            }
        }
    }

    static void dsyr2LowerUnitStride(int n, double alpha,
                                     double[] x, int xOff,
                                     double[] y, int yOff,
                                     double[] A, int aOff, int lda) {
        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i];
            double yi = y[yOff + i];
            double temp1 = alpha * xi;
            double temp2 = alpha * yi;
            int rowOff = aOff + i * lda;
            
            int j = 0;
            int j4 = ((i + 1) / 4) * 4;
            
            for (; j < j4; j += 4) {
                double t0 = A[rowOff + j];
                double t1 = A[rowOff + j + 1];
                double t2 = A[rowOff + j + 2];
                double t3 = A[rowOff + j + 3];
                
                t0 = FMA.op(temp1, y[yOff + j], t0);
                t0 = FMA.op(temp2, x[xOff + j], t0);
                t1 = FMA.op(temp1, y[yOff + j + 1], t1);
                t1 = FMA.op(temp2, x[xOff + j + 1], t1);
                t2 = FMA.op(temp1, y[yOff + j + 2], t2);
                t2 = FMA.op(temp2, x[xOff + j + 2], t2);
                t3 = FMA.op(temp1, y[yOff + j + 3], t3);
                t3 = FMA.op(temp2, x[xOff + j + 3], t3);
                
                A[rowOff + j] = t0;
                A[rowOff + j + 1] = t1;
                A[rowOff + j + 2] = t2;
                A[rowOff + j + 3] = t3;
            }
            
            for (; j <= i; j++) {
                double t = A[rowOff + j];
                t = FMA.op(temp1, y[yOff + j], t);
                t = FMA.op(temp2, x[xOff + j], t);
                A[rowOff + j] = t;
            }
        }
    }

    static void dsyr2UpperStride(int n, double alpha,
                                 double[] x, int xOff, int incx,
                                 double[] y, int yOff, int incy,
                                 double[] A, int aOff, int lda) {
        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i * incx];
            double yi = y[yOff + i * incy];
            double temp1 = alpha * xi;
            double temp2 = alpha * yi;
            int rowOff = aOff + i * lda;
            
            for (int j = i; j < n; j++) {
                double t = A[rowOff + j];
                t = FMA.op(temp1, y[yOff + j * incy], t);
                t = FMA.op(temp2, x[xOff + j * incx], t);
                A[rowOff + j] = t;
            }
        }
    }

    static void dsyr2LowerStride(int n, double alpha,
                                 double[] x, int xOff, int incx,
                                 double[] y, int yOff, int incy,
                                 double[] A, int aOff, int lda) {
        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i * incx];
            double yi = y[yOff + i * incy];
            double temp1 = alpha * xi;
            double temp2 = alpha * yi;
            int rowOff = aOff + i * lda;
            
            for (int j = 0; j <= i; j++) {
                double t = A[rowOff + j];
                t = FMA.op(temp1, y[yOff + j * incy], t);
                t = FMA.op(temp2, x[xOff + j * incx], t);
                A[rowOff + j] = t;
            }
        }
    }
}
