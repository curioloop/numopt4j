/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

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

    int SIMD_MIN_N = 64;
    int SIMD_MIN_ROW = 32;

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
            if (n >= SIMD_MIN_N && SIMD.supportDsyr2()) {
                boolean computed = upper
                    ? Dsyr2SIMD.dsyr2UpperUnitStride(n, alpha, x, xOff, y, yOff, A, aOff, lda)
                    : Dsyr2SIMD.dsyr2LowerUnitStride(n, alpha, x, xOff, y, yOff, A, aOff, lda);
                if (computed) {
                    return;
                }
            }
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
            if (xi == 0.0 && yi == 0.0) {
                continue;
            }
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
                
                t0 = Math.fma(temp1, y[yOff + j], t0);
                t0 = Math.fma(temp2, x[xOff + j], t0);
                t1 = Math.fma(temp1, y[yOff + j + 1], t1);
                t1 = Math.fma(temp2, x[xOff + j + 1], t1);
                t2 = Math.fma(temp1, y[yOff + j + 2], t2);
                t2 = Math.fma(temp2, x[xOff + j + 2], t2);
                t3 = Math.fma(temp1, y[yOff + j + 3], t3);
                t3 = Math.fma(temp2, x[xOff + j + 3], t3);
                
                A[rowOff + j] = t0;
                A[rowOff + j + 1] = t1;
                A[rowOff + j + 2] = t2;
                A[rowOff + j + 3] = t3;
            }
            
            for (; j < n; j++) {
                double t = A[rowOff + j];
                t = Math.fma(temp1, y[yOff + j], t);
                t = Math.fma(temp2, x[xOff + j], t);
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
            if (xi == 0.0 && yi == 0.0) {
                continue;
            }
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
                
                t0 = Math.fma(temp1, y[yOff + j], t0);
                t0 = Math.fma(temp2, x[xOff + j], t0);
                t1 = Math.fma(temp1, y[yOff + j + 1], t1);
                t1 = Math.fma(temp2, x[xOff + j + 1], t1);
                t2 = Math.fma(temp1, y[yOff + j + 2], t2);
                t2 = Math.fma(temp2, x[xOff + j + 2], t2);
                t3 = Math.fma(temp1, y[yOff + j + 3], t3);
                t3 = Math.fma(temp2, x[xOff + j + 3], t3);
                
                A[rowOff + j] = t0;
                A[rowOff + j + 1] = t1;
                A[rowOff + j + 2] = t2;
                A[rowOff + j + 3] = t3;
            }
            
            for (; j <= i; j++) {
                double t = A[rowOff + j];
                t = Math.fma(temp1, y[yOff + j], t);
                t = Math.fma(temp2, x[xOff + j], t);
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
            if (xi == 0.0 && yi == 0.0) {
                continue;
            }
            double temp1 = alpha * xi;
            double temp2 = alpha * yi;
            int rowOff = aOff + i * lda;

            int j = i;
            int j4 = i + ((n - i) / 4) * 4;
            int xj = xOff + j * incx;
            int yj = yOff + j * incy;

            for (; j < j4; j += 4) {
                double t0 = A[rowOff + j];
                double t1 = A[rowOff + j + 1];
                double t2 = A[rowOff + j + 2];
                double t3 = A[rowOff + j + 3];

                t0 = Math.fma(temp1, y[yj], t0);
                t0 = Math.fma(temp2, x[xj], t0);
                t1 = Math.fma(temp1, y[yj + incy], t1);
                t1 = Math.fma(temp2, x[xj + incx], t1);
                t2 = Math.fma(temp1, y[yj + 2 * incy], t2);
                t2 = Math.fma(temp2, x[xj + 2 * incx], t2);
                t3 = Math.fma(temp1, y[yj + 3 * incy], t3);
                t3 = Math.fma(temp2, x[xj + 3 * incx], t3);

                A[rowOff + j] = t0;
                A[rowOff + j + 1] = t1;
                A[rowOff + j + 2] = t2;
                A[rowOff + j + 3] = t3;

                xj += 4 * incx;
                yj += 4 * incy;
            }

            for (; j < n; j++) {
                double t = A[rowOff + j];
                t = Math.fma(temp1, y[yj], t);
                t = Math.fma(temp2, x[xj], t);
                A[rowOff + j] = t;
                xj += incx;
                yj += incy;
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
            if (xi == 0.0 && yi == 0.0) {
                continue;
            }
            double temp1 = alpha * xi;
            double temp2 = alpha * yi;
            int rowOff = aOff + i * lda;

            int j = 0;
            int j4 = ((i + 1) / 4) * 4;
            int xj = xOff;
            int yj = yOff;

            for (; j < j4; j += 4) {
                double t0 = A[rowOff + j];
                double t1 = A[rowOff + j + 1];
                double t2 = A[rowOff + j + 2];
                double t3 = A[rowOff + j + 3];

                t0 = Math.fma(temp1, y[yj], t0);
                t0 = Math.fma(temp2, x[xj], t0);
                t1 = Math.fma(temp1, y[yj + incy], t1);
                t1 = Math.fma(temp2, x[xj + incx], t1);
                t2 = Math.fma(temp1, y[yj + 2 * incy], t2);
                t2 = Math.fma(temp2, x[xj + 2 * incx], t2);
                t3 = Math.fma(temp1, y[yj + 3 * incy], t3);
                t3 = Math.fma(temp2, x[xj + 3 * incx], t3);

                A[rowOff + j] = t0;
                A[rowOff + j + 1] = t1;
                A[rowOff + j + 2] = t2;
                A[rowOff + j + 3] = t3;

                xj += 4 * incx;
                yj += 4 * incy;
            }

            for (; j <= i; j++) {
                double t = A[rowOff + j];
                t = Math.fma(temp1, y[yj], t);
                t = Math.fma(temp2, x[xj], t);
                A[rowOff + j] = t;
                xj += incx;
                yj += incy;
            }
        }
    }
}

final class Dsyr2SIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;

    private Dsyr2SIMD() {
    }

    static boolean dsyr2UpperUnitStride(int n, double alpha,
                                        double[] x, int xOff,
                                        double[] y, int yOff,
                                        double[] A, int aOff, int lda) {
        if (LANES <= 1) {
            return false;
        }

        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i];
            double yi = y[yOff + i];
            if (xi == 0.0 && yi == 0.0) {
                continue;
            }

            int rowOff = aOff + i * lda + i;
            int len = n - i;
            double temp1 = alpha * xi;
            double temp2 = alpha * yi;
            if (len >= Dsyr2.SIMD_MIN_ROW) {
                updateRange(len, temp1, y, yOff + i, temp2, x, xOff + i, A, rowOff);
            } else {
                scalarRange(len, temp1, y, yOff + i, temp2, x, xOff + i, A, rowOff);
            }
        }
        return true;
    }

    static boolean dsyr2LowerUnitStride(int n, double alpha,
                                        double[] x, int xOff,
                                        double[] y, int yOff,
                                        double[] A, int aOff, int lda) {
        if (LANES <= 1) {
            return false;
        }

        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i];
            double yi = y[yOff + i];
            if (xi == 0.0 && yi == 0.0) {
                continue;
            }

            int rowOff = aOff + i * lda;
            int len = i + 1;
            double temp1 = alpha * xi;
            double temp2 = alpha * yi;
            if (len >= Dsyr2.SIMD_MIN_ROW) {
                updateRange(len, temp1, y, yOff, temp2, x, xOff, A, rowOff);
            } else {
                scalarRange(len, temp1, y, yOff, temp2, x, xOff, A, rowOff);
            }
        }
        return true;
    }

    static boolean probe() {
        if (LANES <= 1) {
            return false;
        }
        double[] x = new double[LANES];
        double[] y = new double[LANES];
        double[] a = new double[LANES];
        updateRange(LANES, 1.0, y, 0, 1.0, x, 0, a, 0);
        return true;
    }

    private static void updateRange(int len,
                                    double temp1, double[] y, int yOff,
                                    double temp2, double[] x, int xOff,
                                    double[] a, int aOff) {
        DoubleVector temp1Vec = DoubleVector.broadcast(SPECIES, temp1);
        DoubleVector temp2Vec = DoubleVector.broadcast(SPECIES, temp2);
        int limit = SPECIES.loopBound(len);
        int j = 0;
        for (; j < limit; j += LANES) {
            DoubleVector yVec = DoubleVector.fromArray(SPECIES, y, yOff + j);
            DoubleVector xVec = DoubleVector.fromArray(SPECIES, x, xOff + j);
            DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, aOff + j);
            temp1Vec.fma(yVec, temp2Vec.fma(xVec, aVec)).intoArray(a, aOff + j);
        }
        for (; j < len; j++) {
            a[aOff + j] = Math.fma(temp1, y[yOff + j], Math.fma(temp2, x[xOff + j], a[aOff + j]));
        }
    }

    private static void scalarRange(int len,
                                    double temp1, double[] y, int yOff,
                                    double temp2, double[] x, int xOff,
                                    double[] a, int aOff) {
        int j = 0;
        for (; j + 3 < len; j += 4) {
            a[aOff + j] = Math.fma(temp1, y[yOff + j], Math.fma(temp2, x[xOff + j], a[aOff + j]));
            a[aOff + j + 1] = Math.fma(temp1, y[yOff + j + 1], Math.fma(temp2, x[xOff + j + 1], a[aOff + j + 1]));
            a[aOff + j + 2] = Math.fma(temp1, y[yOff + j + 2], Math.fma(temp2, x[xOff + j + 2], a[aOff + j + 2]));
            a[aOff + j + 3] = Math.fma(temp1, y[yOff + j + 3], Math.fma(temp2, x[xOff + j + 3], a[aOff + j + 3]));
        }
        for (; j < len; j++) {
            a[aOff + j] = Math.fma(temp1, y[yOff + j], Math.fma(temp2, x[xOff + j], a[aOff + j]));
        }
    }
}
