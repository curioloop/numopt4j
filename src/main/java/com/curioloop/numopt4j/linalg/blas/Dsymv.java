/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

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
 */
interface Dsymv {

    int SIMD_MIN_N = 64;

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
            y[yOff] = Math.fma(alpha, A[aOff] * x[xOff], y[yOff]);
            return;
        }

        if (incX == 1 && incY == 1 && n >= SIMD_MIN_N && SIMD.supportDsymv()) {
            if (upper) {
                if (DsymvSIMD.dsymvUpperUnitStride(n, alpha, A, aOff, lda, x, xOff, incX, y, yOff)) {
                    return;
                }
            } else {
                if (DsymvSIMD.dsymvLowerUnitStride(n, alpha, A, aOff, lda, x, xOff, incX, y, yOff)) {
                    return;
                }
            }
        }

        if (upper) {
            if (incX == 1 && incY == 1) {
                dsymvUpperUnitStride(n, alpha, A, aOff, lda, x, xOff, y, yOff);
            } else {
                dsymvUpperStride(n, alpha, A, aOff, lda, x, xOff, incX, y, yOff, incY);
            }
        } else {
            if (incX == 1 && incY == 1) {
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
                
                sum = Math.fma(x[xOff + j], a0, sum);
                sum = Math.fma(x[xOff + j + 1], a1, sum);
                sum = Math.fma(x[xOff + j + 2], a2, sum);
                sum = Math.fma(x[xOff + j + 3], a3, sum);
                
                y[yOff + j] = Math.fma(temp1, a0, y[yOff + j]);
                y[yOff + j + 1] = Math.fma(temp1, a1, y[yOff + j + 1]);
                y[yOff + j + 2] = Math.fma(temp1, a2, y[yOff + j + 2]);
                y[yOff + j + 3] = Math.fma(temp1, a3, y[yOff + j + 3]);
            }
            
            for (; j < n; j++) {
                double aij = A[rowOff + j];
                sum = Math.fma(x[xOff + j], aij, sum);
                y[yOff + j] = Math.fma(temp1, aij, y[yOff + j]);
            }
            
            y[yOff + i] = Math.fma(alpha, sum, y[yOff + i]);
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
                
                sum = Math.fma(x[xOff + j], a0, sum);
                sum = Math.fma(x[xOff + j + 1], a1, sum);
                sum = Math.fma(x[xOff + j + 2], a2, sum);
                sum = Math.fma(x[xOff + j + 3], a3, sum);
                
                y[yOff + j] = Math.fma(temp1, a0, y[yOff + j]);
                y[yOff + j + 1] = Math.fma(temp1, a1, y[yOff + j + 1]);
                y[yOff + j + 2] = Math.fma(temp1, a2, y[yOff + j + 2]);
                y[yOff + j + 3] = Math.fma(temp1, a3, y[yOff + j + 3]);
            }
            
            for (; j < i; j++) {
                double aij = A[rowOff + j];
                sum = Math.fma(x[xOff + j], aij, sum);
                y[yOff + j] = Math.fma(temp1, aij, y[yOff + j]);
            }
            
            sum = Math.fma(xi, A[rowOff + i], sum);
            y[yOff + i] = Math.fma(alpha, sum, y[yOff + i]);
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

            double s0 = xi * A[rowOff + i];
            double s1 = 0.0;
            double s2 = 0.0;
            double s3 = 0.0;
            int j = i + 1;
            int j4 = i + 1 + ((n - i - 1) / 4) * 4;
            int xj = xOff + j * incx;
            int yj = yOff + j * incy;

            for (; j < j4; j += 4) {
                double a0 = A[rowOff + j];
                double a1 = A[rowOff + j + 1];
                double a2 = A[rowOff + j + 2];
                double a3 = A[rowOff + j + 3];
                double x0 = x[xj];
                double x1 = x[xj + incx];
                double x2 = x[xj + 2 * incx];
                double x3 = x[xj + 3 * incx];

                s0 = Math.fma(x0, a0, s0);
                s1 = Math.fma(x1, a1, s1);
                s2 = Math.fma(x2, a2, s2);
                s3 = Math.fma(x3, a3, s3);

                y[yj] = Math.fma(temp1, a0, y[yj]);
                y[yj + incy] = Math.fma(temp1, a1, y[yj + incy]);
                y[yj + 2 * incy] = Math.fma(temp1, a2, y[yj + 2 * incy]);
                y[yj + 3 * incy] = Math.fma(temp1, a3, y[yj + 3 * incy]);

                xj += 4 * incx;
                yj += 4 * incy;
            }

            double sum = (s0 + s1) + (s2 + s3);
            for (; j < n; j++) {
                double aij = A[rowOff + j];
                sum = Math.fma(x[xj], aij, sum);
                y[yj] = Math.fma(temp1, aij, y[yj]);
                xj += incx;
                yj += incy;
            }
            y[yOff + i * incy] = Math.fma(alpha, sum, y[yOff + i * incy]);
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

            double s0 = 0.0;
            double s1 = 0.0;
            double s2 = 0.0;
            double s3 = 0.0;
            int j = 0;
            int j4 = (i / 4) * 4;
            int xj = xOff;
            int yj = yOff;

            for (; j < j4; j += 4) {
                double a0 = A[rowOff + j];
                double a1 = A[rowOff + j + 1];
                double a2 = A[rowOff + j + 2];
                double a3 = A[rowOff + j + 3];
                double x0 = x[xj];
                double x1 = x[xj + incx];
                double x2 = x[xj + 2 * incx];
                double x3 = x[xj + 3 * incx];

                s0 = Math.fma(x0, a0, s0);
                s1 = Math.fma(x1, a1, s1);
                s2 = Math.fma(x2, a2, s2);
                s3 = Math.fma(x3, a3, s3);

                y[yj] = Math.fma(temp1, a0, y[yj]);
                y[yj + incy] = Math.fma(temp1, a1, y[yj + incy]);
                y[yj + 2 * incy] = Math.fma(temp1, a2, y[yj + 2 * incy]);
                y[yj + 3 * incy] = Math.fma(temp1, a3, y[yj + 3 * incy]);

                xj += 4 * incx;
                yj += 4 * incy;
            }

            double sum = (s0 + s1) + (s2 + s3);
            for (; j < i; j++) {
                double aij = A[rowOff + j];
                sum = Math.fma(x[xj], aij, sum);
                y[yj] = Math.fma(temp1, aij, y[yj]);
                xj += incx;
                yj += incy;
            }
            sum = Math.fma(xi, A[rowOff + i], sum);
            y[yOff + i * incy] = Math.fma(alpha, sum, y[yOff + i * incy]);
        }
    }
}

final class DsymvSIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;

    private DsymvSIMD() {
    }

    static boolean dsymvUpperUnitStride(int n, double alpha,
                                        double[] A, int aOff, int lda,
                                        double[] x, int xOff, int incX,
                                        double[] y, int yOff) {
        if (LANES <= 1 || incX != 1) {
            return false;
        }

        upperContiguousY(n, alpha, A, aOff, lda, x, xOff, y, yOff);
        return true;
    }

    static boolean dsymvLowerUnitStride(int n, double alpha,
                                        double[] A, int aOff, int lda,
                                        double[] x, int xOff, int incX,
                                        double[] y, int yOff) {
        if (LANES <= 1 || incX != 1) {
            return false;
        }

        lowerContiguousY(n, alpha, A, aOff, lda, x, xOff, y, yOff);
        return true;
    }

    private static void upperContiguousY(int n, double alpha,
                                         double[] A, int aOff, int lda,
                                         double[] x, int xOff,
                                         double[] y, int yOff) {
        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i];
            double temp1 = alpha * xi;
            DoubleVector tempVec = DoubleVector.broadcast(SPECIES, temp1);
            int rowOff = aOff + i * lda;

            double sum = xi * A[rowOff + i];
            DoubleVector acc0 = DoubleVector.zero(SPECIES);
            DoubleVector acc1 = DoubleVector.zero(SPECIES);
            int base = i + 1;
            int limit = base + SPECIES.loopBound(n - base);
            int j = base;

            for (; j + 2 * LANES <= limit; j += 2 * LANES) {
                DoubleVector a0 = DoubleVector.fromArray(SPECIES, A, rowOff + j);
                DoubleVector x0 = DoubleVector.fromArray(SPECIES, x, xOff + j);
                DoubleVector y0 = DoubleVector.fromArray(SPECIES, y, yOff + j);
                acc0 = a0.fma(x0, acc0);
                tempVec.fma(a0, y0).intoArray(y, yOff + j);

                DoubleVector a1 = DoubleVector.fromArray(SPECIES, A, rowOff + j + LANES);
                DoubleVector x1 = DoubleVector.fromArray(SPECIES, x, xOff + j + LANES);
                DoubleVector y1 = DoubleVector.fromArray(SPECIES, y, yOff + j + LANES);
                acc1 = a1.fma(x1, acc1);
                tempVec.fma(a1, y1).intoArray(y, yOff + j + LANES);
            }

            for (; j < limit; j += LANES) {
                DoubleVector a0 = DoubleVector.fromArray(SPECIES, A, rowOff + j);
                DoubleVector x0 = DoubleVector.fromArray(SPECIES, x, xOff + j);
                DoubleVector y0 = DoubleVector.fromArray(SPECIES, y, yOff + j);
                acc0 = a0.fma(x0, acc0);
                tempVec.fma(a0, y0).intoArray(y, yOff + j);
            }

            sum += acc0.add(acc1).reduceLanes(VectorOperators.ADD);
            for (; j < n; j++) {
                double aij = A[rowOff + j];
                sum = Math.fma(x[xOff + j], aij, sum);
                y[yOff + j] = Math.fma(temp1, aij, y[yOff + j]);
            }

            y[yOff + i] = Math.fma(alpha, sum, y[yOff + i]);
        }
    }

    private static void lowerContiguousY(int n, double alpha,
                                         double[] A, int aOff, int lda,
                                         double[] x, int xOff,
                                         double[] y, int yOff) {
        for (int i = 0; i < n; i++) {
            double xi = x[xOff + i];
            double temp1 = alpha * xi;
            DoubleVector tempVec = DoubleVector.broadcast(SPECIES, temp1);
            int rowOff = aOff + i * lda;

            DoubleVector acc0 = DoubleVector.zero(SPECIES);
            DoubleVector acc1 = DoubleVector.zero(SPECIES);
            int limit = SPECIES.loopBound(i);
            int j = 0;
            double sum = 0.0;

            for (; j + 2 * LANES <= limit; j += 2 * LANES) {
                DoubleVector a0 = DoubleVector.fromArray(SPECIES, A, rowOff + j);
                DoubleVector x0 = DoubleVector.fromArray(SPECIES, x, xOff + j);
                DoubleVector y0 = DoubleVector.fromArray(SPECIES, y, yOff + j);
                acc0 = a0.fma(x0, acc0);
                tempVec.fma(a0, y0).intoArray(y, yOff + j);

                DoubleVector a1 = DoubleVector.fromArray(SPECIES, A, rowOff + j + LANES);
                DoubleVector x1 = DoubleVector.fromArray(SPECIES, x, xOff + j + LANES);
                DoubleVector y1 = DoubleVector.fromArray(SPECIES, y, yOff + j + LANES);
                acc1 = a1.fma(x1, acc1);
                tempVec.fma(a1, y1).intoArray(y, yOff + j + LANES);
            }

            for (; j < limit; j += LANES) {
                DoubleVector a0 = DoubleVector.fromArray(SPECIES, A, rowOff + j);
                DoubleVector x0 = DoubleVector.fromArray(SPECIES, x, xOff + j);
                DoubleVector y0 = DoubleVector.fromArray(SPECIES, y, yOff + j);
                acc0 = a0.fma(x0, acc0);
                tempVec.fma(a0, y0).intoArray(y, yOff + j);
            }

            sum += acc0.add(acc1).reduceLanes(VectorOperators.ADD);
            for (; j < i; j++) {
                double aij = A[rowOff + j];
                sum = Math.fma(x[xOff + j], aij, sum);
                y[yOff + j] = Math.fma(temp1, aij, y[yOff + j]);
            }

            sum = Math.fma(xi, A[rowOff + i], sum);
            y[yOff + i] = Math.fma(alpha, sum, y[yOff + i]);
        }
    }

    static boolean probe() {
        if (LANES <= 1) {
            return false;
        }

        int n = Math.max(LANES, 8);
        double[] A = new double[n * n];
        double[] x = new double[n];
        double[] y = new double[n];
        dsymvUpperUnitStride(n, 1.0, A, 0, n, x, 0, 1, y, 0);
        dsymvLowerUnitStride(n, 1.0, A, 0, n, x, 0, 1, y, 0);
        return true;
    }
}
