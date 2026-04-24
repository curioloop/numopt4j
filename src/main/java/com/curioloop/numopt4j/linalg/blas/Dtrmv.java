/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * DTRMV — triangular matrix-vector multiply.
 *
 * <p>Performs one of the matrix-vector operations:
 * <pre>
 *   x = A * x    if trans == NoTrans
 *   x = Aᵀ * x   if trans == Trans or ConjTrans
 * </pre>
 * where {@code A} is an {@code n×n} triangular matrix stored in row-major order,
 * and {@code x} is a vector of length {@code n}.
 *
 * <p>Corresponds to BLAS routine {@code DTRMV} (Level 2).
 *
 * <p>Implementation notes:
 * <ul>
 *   <li>Unit-stride paths use FMA-based inner loops for improved numerical accuracy.</li>
 *   <li>General-stride paths support arbitrary positive or negative {@code incX}.</li>
 *   <li>Four dispatch paths: Upper/Lower × NoTrans/Trans.</li>
 * </ul>
 *
 * @see <a href="https://netlib.org/blas/dtrmv.f">BLAS DTRMV reference (Netlib)</a>
 */
interface Dtrmv {

    int SIMD_MIN_N = 64;
    int SIMD_MIN_ROW = 32;
    int COLUMN_VIEW_MIN_N = 16;

    static void dtrmv(BLAS.Uplo uplo, BLAS.Trans trans, BLAS.Diag diag, int n,
                      double[] A, int aOff, int lda,
                      double[] x, int xOff, int incX) {

        boolean upper = (uplo == BLAS.Uplo.Upper);
        boolean transA = (trans == BLAS.Trans.Trans || trans == BLAS.Trans.Conj);
        boolean unit = (diag == BLAS.Diag.Unit);

        if (n == 0) return;

        if (n == 1) {
            if (!unit) {
                x[xOff] *= A[aOff];
            }
            return;
        }

        if (incX == 1 && n >= SIMD_MIN_N && SIMD.supportDtrmv()) {
            boolean computed;
            if (!transA) {
                computed = upper
                    ? DtrmvSIMD.dtrmvUpperNoTrans(n, A, aOff, lda, x, xOff, unit)
                    : DtrmvSIMD.dtrmvLowerNoTrans(n, A, aOff, lda, x, xOff, unit);
            } else {
                computed = upper
                    ? DtrmvSIMD.dtrmvUpperTrans(n, A, aOff, lda, x, xOff, unit)
                    : DtrmvSIMD.dtrmvLowerTrans(n, A, aOff, lda, x, xOff, unit);
            }
            if (computed) {
                return;
            }
        }

        if (!transA) {
            // Keep the same-array adjacent-column specialization for Dlarft/Dtrtri/Dlahr2-style replays.
            if (upper) {
                    if (dtrmvUpperNoTransAdjacentColumnView(n, A, aOff, lda, x, xOff, incX, unit)) {
                    return;
                }
                dtrmvUpperNoTrans(n, A, aOff, lda, x, xOff, incX, unit);
            } else {
                    if (dtrmvLowerNoTransAdjacentColumnView(n, A, aOff, lda, x, xOff, incX, unit)) {
                    return;
                }
                dtrmvLowerNoTrans(n, A, aOff, lda, x, xOff, incX, unit);
            }
        } else {
            if (upper) {
                dtrmvUpperTrans(n, A, aOff, lda, x, xOff, incX, unit);
            } else {
                dtrmvLowerTrans(n, A, aOff, lda, x, xOff, incX, unit);
            }
        }
    }

    private static boolean dtrmvUpperNoTransAdjacentColumnView(int n,
                                                               double[] A, int aOff, int lda,
                                                               double[] x, int xOff, int incx,
                                                               boolean unit) {
        return dtrmvNoTransAdjacentColumnView(n, A, aOff, lda, x, xOff, incx, unit, n, true);
    }

    private static boolean dtrmvLowerNoTransAdjacentColumnView(int n,
                                                               double[] A, int aOff, int lda,
                                                               double[] x, int xOff, int incx,
                                                               boolean unit) {
        return dtrmvNoTransAdjacentColumnView(n, A, aOff, lda, x, xOff, incx, unit, -1, false);
    }

    private static boolean dtrmvNoTransAdjacentColumnView(int n,
                                                          double[] A, int aOff, int lda,
                                                          double[] x, int xOff, int incx,
                                                          boolean unit,
                                                          int vectorColOffset,
                                                          boolean upper) {
        if (!isSameArrayAdjacentColumnView(n, A, aOff, lda, x, xOff, incx, vectorColOffset)) {
            return false;
        }

        if (upper) {
            dtrmvUpperNoTransAdjacentColumn(n, A, aOff, lda, vectorColOffset, unit);
        } else {
            dtrmvLowerNoTransAdjacentColumn(n, A, aOff, lda, vectorColOffset, unit);
        }
        return true;
    }

    private static boolean isSameArrayAdjacentColumnView(int n,
                                                         double[] A, int aOff, int lda,
                                                         double[] x, int xOff, int incx,
                                                         int vectorColOffset) {
        return n >= COLUMN_VIEW_MIN_N
            && A == x
            && incx == lda
            && incx > 1
            && xOff == aOff + vectorColOffset;
    }

    // Reuse each strided column-view load across four row dots for Dlarft/Dtrtri-style shapes.
    private static void dtrmvUpperNoTransAdjacentColumn(int n, double[] A, int aOff, int lda,
                                                        int vectorColOffset, boolean unit) {
        int i = 0;
        for (; i + 3 < n; i += 4) {
            int row0 = aOff + i * lda;
            int row1 = row0 + lda;
            int row2 = row1 + lda;
            int row3 = row2 + lda;

            double x0 = A[row0 + vectorColOffset];
            double x1 = A[row1 + vectorColOffset];
            double x2 = A[row2 + vectorColOffset];
            double x3 = A[row3 + vectorColOffset];

            double s0 = unit ? x0 : A[row0 + i] * x0;
            double s1 = unit ? x1 : A[row1 + i + 1] * x1;
            double s2 = unit ? x2 : A[row2 + i + 2] * x2;
            double s3 = unit ? x3 : A[row3 + i + 3] * x3;

            s0 = Math.fma(A[row0 + i + 1], x1, s0);
            s0 = Math.fma(A[row0 + i + 2], x2, s0);
            s0 = Math.fma(A[row0 + i + 3], x3, s0);
            s1 = Math.fma(A[row1 + i + 2], x2, s1);
            s1 = Math.fma(A[row1 + i + 3], x3, s1);
            s2 = Math.fma(A[row2 + i + 3], x3, s2);

            for (int j = i + 4; j < n; j++) {
                double xj = A[aOff + j * lda + vectorColOffset];
                s0 = Math.fma(A[row0 + j], xj, s0);
                s1 = Math.fma(A[row1 + j], xj, s1);
                s2 = Math.fma(A[row2 + j], xj, s2);
                s3 = Math.fma(A[row3 + j], xj, s3);
            }

            A[row0 + vectorColOffset] = s0;
            A[row1 + vectorColOffset] = s1;
            A[row2 + vectorColOffset] = s2;
            A[row3 + vectorColOffset] = s3;
        }

        for (; i < n; i++) {
            int rowOff = aOff + i * lda;
            double xi = A[rowOff + vectorColOffset];
            double tmp = unit ? xi : A[rowOff + i] * xi;
            for (int j = i + 1; j < n; j++) {
                tmp = Math.fma(A[rowOff + j], A[aOff + j * lda + vectorColOffset], tmp);
            }
            A[rowOff + vectorColOffset] = tmp;
        }
    }

    private static void dtrmvLowerNoTransAdjacentColumn(int n, double[] A, int aOff, int lda,
                                                        int vectorColOffset, boolean unit) {
        int rem = n & 3;
        for (int i = n - 1; i >= rem + 3; i -= 4) {
            int base = i - 3;
            int row0 = aOff + base * lda;
            int row1 = row0 + lda;
            int row2 = row1 + lda;
            int row3 = row2 + lda;

            double x0 = A[row0 + vectorColOffset];
            double x1 = A[row1 + vectorColOffset];
            double x2 = A[row2 + vectorColOffset];
            double x3 = A[row3 + vectorColOffset];

            double s0 = unit ? x0 : A[row0 + base] * x0;
            double s1 = unit ? x1 : A[row1 + base + 1] * x1;
            double s2 = unit ? x2 : A[row2 + base + 2] * x2;
            double s3 = unit ? x3 : A[row3 + base + 3] * x3;

            for (int j = 0; j < base; j++) {
                double xj = A[aOff + j * lda + vectorColOffset];
                s0 = Math.fma(A[row0 + j], xj, s0);
                s1 = Math.fma(A[row1 + j], xj, s1);
                s2 = Math.fma(A[row2 + j], xj, s2);
                s3 = Math.fma(A[row3 + j], xj, s3);
            }

            s1 = Math.fma(A[row1 + base], x0, s1);
            s2 = Math.fma(A[row2 + base], x0, s2);
            s2 = Math.fma(A[row2 + base + 1], x1, s2);
            s3 = Math.fma(A[row3 + base], x0, s3);
            s3 = Math.fma(A[row3 + base + 1], x1, s3);
            s3 = Math.fma(A[row3 + base + 2], x2, s3);

            A[row0 + vectorColOffset] = s0;
            A[row1 + vectorColOffset] = s1;
            A[row2 + vectorColOffset] = s2;
            A[row3 + vectorColOffset] = s3;
        }

        for (int i = rem - 1; i >= 0; i--) {
            int rowOff = aOff + i * lda;
            double xi = A[rowOff + vectorColOffset];
            double tmp = unit ? xi : A[rowOff + i] * xi;
            for (int j = 0; j < i; j++) {
                tmp = Math.fma(A[rowOff + j], A[aOff + j * lda + vectorColOffset], tmp);
            }
            A[rowOff + vectorColOffset] = tmp;
        }
    }

    static void dtrmvUpperNoTrans(int n, double[] A, int aOff, int lda,
                                  double[] x, int xOff, int incx, boolean unit) {
        if (incx == 1) {
            for (int i = 0; i < n; i++) {
                int rowOff = aOff + i * lda;
                double s0 = unit ? x[xOff + i] : A[rowOff + i] * x[xOff + i];
                double s1 = 0.0;
                double s2 = 0.0;
                double s3 = 0.0;
                int j = i + 1;
                int j4 = i + 1 + ((n - i - 1) / 4) * 4;
                for (; j < j4; j += 4) {
                    s0 = Math.fma(A[rowOff + j], x[xOff + j], s0);
                    s1 = Math.fma(A[rowOff + j + 1], x[xOff + j + 1], s1);
                    s2 = Math.fma(A[rowOff + j + 2], x[xOff + j + 2], s2);
                    s3 = Math.fma(A[rowOff + j + 3], x[xOff + j + 3], s3);
                }
                double tmp = (s0 + s1) + (s2 + s3);
                for (; j < n; j++) {
                    tmp = Math.fma(A[rowOff + j], x[xOff + j], tmp);
                }
                x[xOff + i] = tmp;
            }
        } else {
            int kx = incx > 0 ? 0 : -(n - 1) * incx;
            int ix = kx;
            for (int i = 0; i < n; i++) {
                int rowOff = aOff + i * lda;
                double s0 = unit ? x[xOff + ix] : A[rowOff + i] * x[xOff + ix];
                double s1 = 0.0;
                double s2 = 0.0;
                double s3 = 0.0;
                int j = i + 1;
                int j4 = i + 1 + ((n - i - 1) / 4) * 4;
                int jx = ix + incx;
                for (; j < j4; j += 4) {
                    s0 = Math.fma(A[rowOff + j], x[xOff + jx], s0);
                    s1 = Math.fma(A[rowOff + j + 1], x[xOff + jx + incx], s1);
                    s2 = Math.fma(A[rowOff + j + 2], x[xOff + jx + 2 * incx], s2);
                    s3 = Math.fma(A[rowOff + j + 3], x[xOff + jx + 3 * incx], s3);
                    jx += 4 * incx;
                }
                double tmp = (s0 + s1) + (s2 + s3);
                for (; j < n; j++) {
                    tmp = Math.fma(A[rowOff + j], x[xOff + jx], tmp);
                    jx += incx;
                }
                x[xOff + ix] = tmp;
                ix += incx;
            }
        }
    }

    static void dtrmvLowerNoTrans(int n, double[] A, int aOff, int lda,
                                  double[] x, int xOff, int incx, boolean unit) {
        if (incx == 1) {
            for (int i = n - 1; i >= 0; i--) {
                int rowOff = aOff + i * lda;
                double s0 = unit ? x[xOff + i] : A[rowOff + i] * x[xOff + i];
                double s1 = 0.0;
                double s2 = 0.0;
                double s3 = 0.0;
                int j = 0;
                int j4 = (i / 4) * 4;
                for (; j < j4; j += 4) {
                    s0 = Math.fma(A[rowOff + j], x[xOff + j], s0);
                    s1 = Math.fma(A[rowOff + j + 1], x[xOff + j + 1], s1);
                    s2 = Math.fma(A[rowOff + j + 2], x[xOff + j + 2], s2);
                    s3 = Math.fma(A[rowOff + j + 3], x[xOff + j + 3], s3);
                }
                double tmp = (s0 + s1) + (s2 + s3);
                for (; j < i; j++) {
                    tmp = Math.fma(A[rowOff + j], x[xOff + j], tmp);
                }
                x[xOff + i] = tmp;
            }
        } else {
            int kx = incx > 0 ? 0 : -(n - 1) * incx;
            int ix = kx + (n - 1) * incx;
            for (int i = n - 1; i >= 0; i--) {
                int rowOff = aOff + i * lda;
                double s0 = unit ? x[xOff + ix] : A[rowOff + i] * x[xOff + ix];
                double s1 = 0.0;
                double s2 = 0.0;
                double s3 = 0.0;
                int j = 0;
                int j4 = (i / 4) * 4;
                int jx = kx;
                for (; j < j4; j += 4) {
                    s0 = Math.fma(A[rowOff + j], x[xOff + jx], s0);
                    s1 = Math.fma(A[rowOff + j + 1], x[xOff + jx + incx], s1);
                    s2 = Math.fma(A[rowOff + j + 2], x[xOff + jx + 2 * incx], s2);
                    s3 = Math.fma(A[rowOff + j + 3], x[xOff + jx + 3 * incx], s3);
                    jx += 4 * incx;
                }
                double tmp = (s0 + s1) + (s2 + s3);
                for (; j < i; j++) {
                    tmp = Math.fma(A[rowOff + j], x[xOff + jx], tmp);
                    jx += incx;
                }
                x[xOff + ix] = tmp;
                ix -= incx;
            }
        }
    }

    static void dtrmvUpperTrans(int n, double[] A, int aOff, int lda,
                                double[] x, int xOff, int incx, boolean unit) {
        if (incx == 1) {
            for (int i = n - 1; i >= 0; i--) {
                int rowOff = aOff + i * lda;
                double xi = x[xOff + i];
                if (xi != 0.0) {
                    int j = i + 1;
                    int j4 = i + 1 + ((n - i - 1) / 4) * 4;
                    for (; j < j4; j += 4) {
                        x[xOff + j] = Math.fma(xi, A[rowOff + j], x[xOff + j]);
                        x[xOff + j + 1] = Math.fma(xi, A[rowOff + j + 1], x[xOff + j + 1]);
                        x[xOff + j + 2] = Math.fma(xi, A[rowOff + j + 2], x[xOff + j + 2]);
                        x[xOff + j + 3] = Math.fma(xi, A[rowOff + j + 3], x[xOff + j + 3]);
                    }
                    for (; j < n; j++) {
                        x[xOff + j] = Math.fma(xi, A[rowOff + j], x[xOff + j]);
                    }
                }
                if (!unit) {
                    x[xOff + i] *= A[rowOff + i];
                }
            }
        } else {
            int kx = incx > 0 ? 0 : -(n - 1) * incx;
            int ix = kx + (n - 1) * incx;
            for (int i = n - 1; i >= 0; i--) {
                int rowOff = aOff + i * lda;
                double xi = x[xOff + ix];
                if (xi != 0.0) {
                    int j = i + 1;
                    int j4 = i + 1 + ((n - i - 1) / 4) * 4;
                    int jx = kx + (i + 1) * incx;
                    for (; j < j4; j += 4) {
                        x[xOff + jx] = Math.fma(xi, A[rowOff + j], x[xOff + jx]);
                        x[xOff + jx + incx] = Math.fma(xi, A[rowOff + j + 1], x[xOff + jx + incx]);
                        x[xOff + jx + 2 * incx] = Math.fma(xi, A[rowOff + j + 2], x[xOff + jx + 2 * incx]);
                        x[xOff + jx + 3 * incx] = Math.fma(xi, A[rowOff + j + 3], x[xOff + jx + 3 * incx]);
                        jx += 4 * incx;
                    }
                    for (; j < n; j++) {
                        x[xOff + jx] = Math.fma(xi, A[rowOff + j], x[xOff + jx]);
                        jx += incx;
                    }
                }
                if (!unit) {
                    x[xOff + ix] *= A[rowOff + i];
                }
                ix -= incx;
            }
        }
    }

    static void dtrmvLowerTrans(int n, double[] A, int aOff, int lda,
                                double[] x, int xOff, int incx, boolean unit) {
        if (incx == 1) {
            for (int i = 0; i < n; i++) {
                int rowOff = aOff + i * lda;
                double xi = x[xOff + i];
                if (xi != 0.0) {
                    int j = 0;
                    int j4 = (i / 4) * 4;
                    for (; j < j4; j += 4) {
                        x[xOff + j] = Math.fma(xi, A[rowOff + j], x[xOff + j]);
                        x[xOff + j + 1] = Math.fma(xi, A[rowOff + j + 1], x[xOff + j + 1]);
                        x[xOff + j + 2] = Math.fma(xi, A[rowOff + j + 2], x[xOff + j + 2]);
                        x[xOff + j + 3] = Math.fma(xi, A[rowOff + j + 3], x[xOff + j + 3]);
                    }
                    for (; j < i; j++) {
                        x[xOff + j] = Math.fma(xi, A[rowOff + j], x[xOff + j]);
                    }
                }
                if (!unit) {
                    x[xOff + i] *= A[rowOff + i];
                }
            }
        } else {
            int kx = incx > 0 ? 0 : -(n - 1) * incx;
            int ix = kx;
            for (int i = 0; i < n; i++) {
                int rowOff = aOff + i * lda;
                double xi = x[xOff + ix];
                if (xi != 0.0) {
                    int j = 0;
                    int j4 = (i / 4) * 4;
                    int jx = kx;
                    for (; j < j4; j += 4) {
                        x[xOff + jx] = Math.fma(xi, A[rowOff + j], x[xOff + jx]);
                        x[xOff + jx + incx] = Math.fma(xi, A[rowOff + j + 1], x[xOff + jx + incx]);
                        x[xOff + jx + 2 * incx] = Math.fma(xi, A[rowOff + j + 2], x[xOff + jx + 2 * incx]);
                        x[xOff + jx + 3 * incx] = Math.fma(xi, A[rowOff + j + 3], x[xOff + jx + 3 * incx]);
                        jx += 4 * incx;
                    }
                    for (; j < i; j++) {
                        x[xOff + jx] = Math.fma(xi, A[rowOff + j], x[xOff + jx]);
                        jx += incx;
                    }
                }
                if (!unit) {
                    x[xOff + ix] *= A[rowOff + i];
                }
                ix += incx;
            }
        }
    }
}

final class DtrmvSIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;

    private DtrmvSIMD() {
    }

    static boolean dtrmvUpperNoTrans(int n, double[] A, int aOff, int lda,
                                     double[] x, int xOff, boolean unit) {
        if (LANES <= 1) {
            return false;
        }

        for (int i = 0; i < n; i++) {
            int rowOff = aOff + i * lda + i;
            double tmp;
            if (unit) {
                tmp = x[xOff + i];
                int len = n - i - 1;
                if (len >= Dtrmv.SIMD_MIN_ROW) {
                    tmp += dotContiguous(A, rowOff + 1, x, xOff + i + 1, len);
                } else {
                    tmp += scalarDot(A, rowOff + 1, x, xOff + i + 1, len);
                }
            } else {
                int len = n - i;
                if (len >= Dtrmv.SIMD_MIN_ROW) {
                    tmp = dotContiguous(A, rowOff, x, xOff + i, len);
                } else {
                    tmp = scalarDot(A, rowOff, x, xOff + i, len);
                }
            }
            x[xOff + i] = tmp;
        }
        return true;
    }

    static boolean dtrmvLowerNoTrans(int n, double[] A, int aOff, int lda,
                                     double[] x, int xOff, boolean unit) {
        if (LANES <= 1) {
            return false;
        }

        for (int i = n - 1; i >= 0; i--) {
            int rowOff = aOff + i * lda;
            double tmp;
            if (unit) {
                tmp = x[xOff + i];
                int len = i;
                if (len >= Dtrmv.SIMD_MIN_ROW) {
                    tmp += dotContiguous(A, rowOff, x, xOff, len);
                } else {
                    tmp += scalarDot(A, rowOff, x, xOff, len);
                }
            } else {
                int len = i + 1;
                if (len >= Dtrmv.SIMD_MIN_ROW) {
                    tmp = dotContiguous(A, rowOff, x, xOff, len);
                } else {
                    tmp = scalarDot(A, rowOff, x, xOff, len);
                }
            }
            x[xOff + i] = tmp;
        }
        return true;
    }

    static boolean dtrmvUpperTrans(int n, double[] A, int aOff, int lda,
                                   double[] x, int xOff, boolean unit) {
        if (LANES <= 1) {
            return false;
        }

        for (int i = n - 1; i >= 0; i--) {
            int rowOff = aOff + i * lda;
            double xi = x[xOff + i];
            int len = n - i - 1;
            if (xi != 0.0) {
                if (len >= Dtrmv.SIMD_MIN_ROW) {
                    axpyContiguous(len, xi, A, rowOff + i + 1, x, xOff + i + 1);
                } else {
                    scalarAxpy(len, xi, A, rowOff + i + 1, x, xOff + i + 1);
                }
            }
            if (!unit) {
                x[xOff + i] *= A[rowOff + i];
            }
        }
        return true;
    }

    static boolean dtrmvLowerTrans(int n, double[] A, int aOff, int lda,
                                   double[] x, int xOff, boolean unit) {
        if (LANES <= 1) {
            return false;
        }

        for (int i = 0; i < n; i++) {
            int rowOff = aOff + i * lda;
            double xi = x[xOff + i];
            int len = i;
            if (xi != 0.0) {
                if (len >= Dtrmv.SIMD_MIN_ROW) {
                    axpyContiguous(len, xi, A, rowOff, x, xOff);
                } else {
                    scalarAxpy(len, xi, A, rowOff, x, xOff);
                }
            }
            if (!unit) {
                x[xOff + i] *= A[rowOff + i];
            }
        }
        return true;
    }

    static boolean probe() {
        if (LANES <= 1) {
            return false;
        }
        double[] a = new double[LANES];
        double[] x = new double[LANES];
        dotContiguous(a, 0, x, 0, LANES);
        axpyContiguous(LANES, 1.0, a, 0, x, 0);
        return true;
    }

    private static double dotContiguous(double[] a, int aOff, double[] x, int xOff, int len) {
        DoubleVector acc = DoubleVector.zero(SPECIES);
        int limit = SPECIES.loopBound(len);
        int j = 0;
        for (; j < limit; j += LANES) {
            acc = DoubleVector.fromArray(SPECIES, a, aOff + j)
                .fma(DoubleVector.fromArray(SPECIES, x, xOff + j), acc);
        }
        double sum = acc.reduceLanes(VectorOperators.ADD);
        for (; j < len; j++) {
            sum = Math.fma(a[aOff + j], x[xOff + j], sum);
        }
        return sum;
    }

    private static double scalarDot(double[] a, int aOff, double[] x, int xOff, int len) {
        double s0 = 0.0;
        double s1 = 0.0;
        double s2 = 0.0;
        double s3 = 0.0;
        int j = 0;
        for (; j + 3 < len; j += 4) {
            s0 = Math.fma(a[aOff + j], x[xOff + j], s0);
            s1 = Math.fma(a[aOff + j + 1], x[xOff + j + 1], s1);
            s2 = Math.fma(a[aOff + j + 2], x[xOff + j + 2], s2);
            s3 = Math.fma(a[aOff + j + 3], x[xOff + j + 3], s3);
        }
        for (; j < len; j++) {
            s0 = Math.fma(a[aOff + j], x[xOff + j], s0);
        }
        return (s0 + s1) + (s2 + s3);
    }

    private static void axpyContiguous(int len, double alpha,
                                       double[] a, int aOff,
                                       double[] x, int xOff) {
        DoubleVector alphaVec = DoubleVector.broadcast(SPECIES, alpha);
        int limit = SPECIES.loopBound(len);
        int j = 0;
        for (; j < limit; j += LANES) {
            DoubleVector xVec = DoubleVector.fromArray(SPECIES, x, xOff + j);
            DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, aOff + j);
            alphaVec.fma(aVec, xVec).intoArray(x, xOff + j);
        }
        for (; j < len; j++) {
            x[xOff + j] = Math.fma(alpha, a[aOff + j], x[xOff + j]);
        }
    }

    private static void scalarAxpy(int len, double alpha,
                                   double[] a, int aOff,
                                   double[] x, int xOff) {
        int j = 0;
        for (; j + 3 < len; j += 4) {
            x[xOff + j] = Math.fma(alpha, a[aOff + j], x[xOff + j]);
            x[xOff + j + 1] = Math.fma(alpha, a[aOff + j + 1], x[xOff + j + 1]);
            x[xOff + j + 2] = Math.fma(alpha, a[aOff + j + 2], x[xOff + j + 2]);
            x[xOff + j + 3] = Math.fma(alpha, a[aOff + j + 3], x[xOff + j + 3]);
        }
        for (; j < len; j++) {
            x[xOff + j] = Math.fma(alpha, a[aOff + j], x[xOff + j]);
        }
    }
}
