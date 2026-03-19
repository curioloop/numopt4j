/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

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

        if (!transA) {
            if (upper) {
                dtrmvUpperNoTrans(n, A, aOff, lda, x, xOff, incX, unit);
            } else {
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

    static void dtrmvUpperNoTrans(int n, double[] A, int aOff, int lda,
                                  double[] x, int xOff, int incx, boolean unit) {
        if (incx == 1) {
            for (int i = 0; i < n; i++) {
                int rowOff = aOff + i * lda;
                double tmp = unit ? x[xOff + i] : A[rowOff + i] * x[xOff + i];
                for (int j = i + 1; j < n; j++) {
                    tmp = FMA.op(A[rowOff + j], x[xOff + j], tmp);
                }
                x[xOff + i] = tmp;
            }
        } else {
            int kx = incx > 0 ? 0 : -(n - 1) * incx;
            int ix = kx;
            for (int i = 0; i < n; i++) {
                int rowOff = aOff + i * lda;
                double tmp = unit ? x[xOff + ix] : A[rowOff + i] * x[xOff + ix];
                int jx = ix + incx;
                for (int j = i + 1; j < n; j++) {
                    tmp = FMA.op(A[rowOff + j], x[xOff + jx], tmp);
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
                double tmp = unit ? x[xOff + i] : A[rowOff + i] * x[xOff + i];
                for (int j = 0; j < i; j++) {
                    tmp = FMA.op(A[rowOff + j], x[xOff + j], tmp);
                }
                x[xOff + i] = tmp;
            }
        } else {
            int kx = incx > 0 ? 0 : -(n - 1) * incx;
            int ix = kx + (n - 1) * incx;
            for (int i = n - 1; i >= 0; i--) {
                int rowOff = aOff + i * lda;
                double tmp = unit ? x[xOff + ix] : A[rowOff + i] * x[xOff + ix];
                int jx = kx;
                for (int j = 0; j < i; j++) {
                    tmp = FMA.op(A[rowOff + j], x[xOff + jx], tmp);
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
                for (int j = i + 1; j < n; j++) {
                    x[xOff + j] = FMA.op(xi, A[rowOff + j], x[xOff + j]);
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
                int jx = kx + (i + 1) * incx;
                for (int j = i + 1; j < n; j++) {
                    x[xOff + jx] = FMA.op(xi, A[rowOff + j], x[xOff + jx]);
                    jx += incx;
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
                for (int j = 0; j < i; j++) {
                    x[xOff + j] = FMA.op(xi, A[rowOff + j], x[xOff + j]);
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
                int jx = kx;
                for (int j = 0; j < i; j++) {
                    x[xOff + jx] = FMA.op(xi, A[rowOff + j], x[xOff + jx]);
                    jx += incx;
                }
                if (!unit) {
                    x[xOff + ix] *= A[rowOff + i];
                }
                ix += incx;
            }
        }
    }
}
