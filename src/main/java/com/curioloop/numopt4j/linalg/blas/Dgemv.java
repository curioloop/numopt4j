/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

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
    int SIMD_MIN_N = 64;
    int PAIRED_ROW_MIN_N = 32;
    int TRANS_PAIRED_ROW_MIN_N = 16;

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
        if (incx == 1 && n >= SIMD_MIN_N && SIMD.supportDgemv()) {
            if (DgemvSIMD.gemvNoTrans(m, n, alpha, A, aOff, lda, x, xOff, y, yOff, incy)) {
                return;
            }
        }

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
        // Keep the adjacent-row direct kernel once n reaches this gate; the old probe variants converged here.
        if (incx == 1 && m > 1 && n >= PAIRED_ROW_MIN_N) {
            gemvNoTransPairedRows(m, n, alpha, A, aOff, lda, x, xOff, y, yOff, incy);
            return;
        }

        gemvNoTransDirectScalar(m, n, alpha, A, aOff, lda, x, xOff, incx, y, yOff, incy);
    }

    static void gemvNoTransDirectScalar(int m, int n, double alpha,
                                        double[] A, int aOff, int lda,
                                        double[] x, int xOff, int incx,
                                        double[] y, int yOff, int incy) {
        int n4 = (n / 4) * 4;
        
        for (int i = 0; i < m; i++) {
            int rowOff = aOff + i * lda;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            
            int j = 0;
            for (; j < n4; j += 4) {
                s0 = Math.fma(A[rowOff + j], x[xOff + j * incx], s0);
                s1 = Math.fma(A[rowOff + j + 1], x[xOff + (j + 1) * incx], s1);
                s2 = Math.fma(A[rowOff + j + 2], x[xOff + (j + 2) * incx], s2);
                s3 = Math.fma(A[rowOff + j + 3], x[xOff + (j + 3) * incx], s3);
            }
            
            double sum = s0 + s1 + s2 + s3;
            for (; j < n; j++) {
                sum = Math.fma(A[rowOff + j], x[xOff + j * incx], sum);
            }
            
            y[yOff + i * incy] += alpha * sum;
        }
    }

    static void gemvNoTransPairedRows(int m, int n, double alpha,
                                      double[] A, int aOff, int lda,
                                      double[] x, int xOff,
                                      double[] y, int yOff, int incy) {
        int n4 = (n / 4) * 4;
        int i = 0;

        for (; i + 1 < m; i += 2) {
            int row0 = aOff + i * lda;
            int row1 = row0 + lda;
            double s00 = 0.0, s01 = 0.0, s02 = 0.0, s03 = 0.0;
            double s10 = 0.0, s11 = 0.0, s12 = 0.0, s13 = 0.0;

            int j = 0;
            for (; j < n4; j += 4) {
                double x0 = x[xOff + j];
                double x1 = x[xOff + j + 1];
                double x2 = x[xOff + j + 2];
                double x3 = x[xOff + j + 3];

                s00 = Math.fma(A[row0 + j], x0, s00);
                s01 = Math.fma(A[row0 + j + 1], x1, s01);
                s02 = Math.fma(A[row0 + j + 2], x2, s02);
                s03 = Math.fma(A[row0 + j + 3], x3, s03);

                s10 = Math.fma(A[row1 + j], x0, s10);
                s11 = Math.fma(A[row1 + j + 1], x1, s11);
                s12 = Math.fma(A[row1 + j + 2], x2, s12);
                s13 = Math.fma(A[row1 + j + 3], x3, s13);
            }

            double sum0 = s00 + s01 + s02 + s03;
            double sum1 = s10 + s11 + s12 + s13;
            for (; j < n; j++) {
                double xj = x[xOff + j];
                sum0 = Math.fma(A[row0 + j], xj, sum0);
                sum1 = Math.fma(A[row1 + j], xj, sum1);
            }

            int yi = yOff + i * incy;
            y[yi] = Math.fma(alpha, sum0, y[yi]);
            y[yi + incy] = Math.fma(alpha, sum1, y[yi + incy]);
        }

        if (i < m) {
            gemvNoTransDirectScalar(1, n, alpha, A, aOff + i * lda, lda, x, xOff, 1, y, yOff + i * incy, incy);
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
            if (incy == 1 && n >= SIMD_MIN_N && SIMD.supportDgemv()) {
                if (DgemvSIMD.gemvTransLeaf(m, n, alpha, A, aOff, lda, x, xOff, incx, y, yOff)) {
                    return;
                }
            }

            // The transpose leaf keeps a paired-row update once the leaf is large enough to amortize y traffic.
            if (incy == 1 && m > 1 && n >= TRANS_PAIRED_ROW_MIN_N) {
                gemvTransPairedRows(m, n, alpha, A, aOff, lda, x, xOff, incx, y, yOff);
                return;
            }

            gemvTransScalarLeaf(m, n, alpha, A, aOff, lda, x, xOff, incx, y, yOff, incy);
            return;
        }

        int mid = m / 2;
        gemvTransImpl(mid, n, alpha, A, aOff, lda, x, xOff, incx, y, yOff, incy);
        gemvTransImpl(m - mid, n, alpha, A, aOff + mid * lda, lda,
                      x, xOff + mid * incx, incx, y, yOff, incy);
    }

    static void gemvTransScalarLeaf(int m, int n, double alpha,
                                    double[] A, int aOff, int lda,
                                    double[] x, int xOff, int incx,
                                    double[] y, int yOff, int incy) {
        int n4 = (n / 4) * 4;

        for (int i = 0; i < m; i++) {
            double axi = alpha * x[xOff + i * incx];
            int rowOff = aOff + i * lda;

            int j = 0;
            for (; j < n4; j += 4) {
                y[yOff + j * incy] = Math.fma(A[rowOff + j], axi, y[yOff + j * incy]);
                y[yOff + (j + 1) * incy] = Math.fma(A[rowOff + j + 1], axi, y[yOff + (j + 1) * incy]);
                y[yOff + (j + 2) * incy] = Math.fma(A[rowOff + j + 2], axi, y[yOff + (j + 2) * incy]);
                y[yOff + (j + 3) * incy] = Math.fma(A[rowOff + j + 3], axi, y[yOff + (j + 3) * incy]);
            }
            for (; j < n; j++) {
                y[yOff + j * incy] = Math.fma(A[rowOff + j], axi, y[yOff + j * incy]);
            }
        }
    }

    static void gemvTransPairedRows(int m, int n, double alpha,
                                    double[] A, int aOff, int lda,
                                    double[] x, int xOff, int incx,
                                    double[] y, int yOff) {
        int n4 = (n / 4) * 4;
        int i = 0;

        for (; i + 1 < m; i += 2) {
            int row0 = aOff + i * lda;
            int row1 = row0 + lda;
            double axi0 = alpha * x[xOff + i * incx];
            double axi1 = alpha * x[xOff + (i + 1) * incx];

            int j = 0;
            for (; j < n4; j += 4) {
                double y0 = y[yOff + j];
                double y1 = y[yOff + j + 1];
                double y2 = y[yOff + j + 2];
                double y3 = y[yOff + j + 3];

                y0 = Math.fma(A[row0 + j], axi0, y0);
                y1 = Math.fma(A[row0 + j + 1], axi0, y1);
                y2 = Math.fma(A[row0 + j + 2], axi0, y2);
                y3 = Math.fma(A[row0 + j + 3], axi0, y3);

                y0 = Math.fma(A[row1 + j], axi1, y0);
                y1 = Math.fma(A[row1 + j + 1], axi1, y1);
                y2 = Math.fma(A[row1 + j + 2], axi1, y2);
                y3 = Math.fma(A[row1 + j + 3], axi1, y3);

                y[yOff + j] = y0;
                y[yOff + j + 1] = y1;
                y[yOff + j + 2] = y2;
                y[yOff + j + 3] = y3;
            }
            for (; j < n; j++) {
                y[yOff + j] = Math.fma(A[row1 + j], axi1, Math.fma(A[row0 + j], axi0, y[yOff + j]));
            }
        }

        if (i < m) {
            gemvTransScalarLeaf(1, n, alpha, A, aOff + i * lda, lda, x, xOff + i * incx, incx, y, yOff, 1);
        }
    }

}

final class DgemvSIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;

    private DgemvSIMD() {
    }

    static boolean gemvNoTrans(int m, int n, double alpha,
                               double[] A, int aOff, int lda,
                               double[] x, int xOff,
                               double[] y, int yOff, int incy) {
        if (LANES <= 1) {
            return false;
        }

        int limit = SPECIES.loopBound(n);
        for (int i = 0; i < m; i++) {
            int rowOff = aOff + i * lda;
            DoubleVector acc0 = DoubleVector.zero(SPECIES);
            DoubleVector acc1 = DoubleVector.zero(SPECIES);

            int j = 0;
            for (; j + 2 * LANES <= limit; j += 2 * LANES) {
                acc0 = DoubleVector.fromArray(SPECIES, A, rowOff + j)
                    .fma(DoubleVector.fromArray(SPECIES, x, xOff + j), acc0);
                acc1 = DoubleVector.fromArray(SPECIES, A, rowOff + j + LANES)
                    .fma(DoubleVector.fromArray(SPECIES, x, xOff + j + LANES), acc1);
            }
            for (; j < limit; j += LANES) {
                acc0 = DoubleVector.fromArray(SPECIES, A, rowOff + j)
                    .fma(DoubleVector.fromArray(SPECIES, x, xOff + j), acc0);
            }

            double sum = acc0.add(acc1).reduceLanes(VectorOperators.ADD);
            for (; j < n; j++) {
                sum = Math.fma(A[rowOff + j], x[xOff + j], sum);
            }
            int yi = yOff + i * incy;
            y[yi] = Math.fma(alpha, sum, y[yi]);
        }
        return true;
    }

    static boolean gemvTransLeaf(int m, int n, double alpha,
                                 double[] A, int aOff, int lda,
                                 double[] x, int xOff, int incx,
                                 double[] y, int yOff) {
        if (LANES <= 1) {
            return false;
        }

        int limit = SPECIES.loopBound(n);
        for (int i = 0; i < m; i++) {
            double axi = alpha * x[xOff + i * incx];
            DoubleVector axiVec = DoubleVector.broadcast(SPECIES, axi);
            int rowOff = aOff + i * lda;

            int j = 0;
            for (; j < limit; j += LANES) {
                DoubleVector rowVec = DoubleVector.fromArray(SPECIES, A, rowOff + j);
                DoubleVector yVec = DoubleVector.fromArray(SPECIES, y, yOff + j);
                axiVec.fma(rowVec, yVec).intoArray(y, yOff + j);
            }
            for (; j < n; j++) {
                y[yOff + j] = Math.fma(A[rowOff + j], axi, y[yOff + j]);
            }
        }
        return true;
    }

    static boolean probe() {
        if (LANES <= 1) {
            return false;
        }

        int n = Math.max(LANES, 8);
        double[] A = new double[2 * n];
        double[] x = new double[n];
        double[] y = new double[n];
        gemvNoTrans(2, n, 1.0, A, 0, n, x, 0, y, 0, 1);
        gemvTransLeaf(2, n, 1.0, A, 0, n, new double[2], 0, 1, y, 0);
        return true;
    }
}
