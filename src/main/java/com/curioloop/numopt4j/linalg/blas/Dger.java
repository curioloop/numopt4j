/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * DGER performs rank-1 update: A := alpha*x*yᵀ + A
 * 
 * <p>Reference: BLAS Level 2 DGER</p>
 */
interface Dger {

    int SIMD_MIN_N = 32;

    static void dger(int m, int n, double alpha, double[] x, int xOff, int incX,
                     double[] y, int yOff, int incY, double[] A, int aOff, int lda) {
        if (m == 0 || n == 0) return;
        if (alpha == 0.0) return;

        if (incY == 1) {
            dgerIncYUnit(m, n, alpha, x, xOff, incX, y, yOff, A, aOff, lda);
            return;
        }

        int ky = (incY < 0) ? (-(n - 1) * incY) : 0;
        int kx = (incX < 0) ? (-(m - 1) * incX) : 0;

        int ix = kx;
        for (int i = 0; i < m; i++) {
            double xi = alpha * x[xOff + ix];
            if (xi != 0.0) {
                dgerRowUpdateStridedY(n, xi, y, yOff + ky, incY, A, aOff + i * lda);
            }
            ix += incX;
        }
    }

    static void dgerIncYUnit(int m, int n, double alpha, double[] x, int xOff, int incX,
                             double[] y, int yOff, double[] a, int aOff, int lda) {
        // The contiguous-y update settled on the 32-column gate; wider cutoffs only increased scalar tail work.
        boolean simd = n >= SIMD_MIN_N && SIMD.supportDger();
        if (incX == 1) {
            for (int i = 0; i < m; i++) {
                double xi = alpha * x[xOff + i];
                if (xi != 0.0) {
                    if (simd) {
                        DgerSIMD.dgerRowUpdateContiguous(n, xi, y, yOff, a, aOff + i * lda);
                    } else {
                        dgerRowUpdateContiguous(n, xi, y, yOff, a, aOff + i * lda);
                    }
                }
            }
            return;
        }

        int kx = (incX < 0) ? (-(m - 1) * incX) : 0;
        int ix = kx;
        for (int i = 0; i < m; i++) {
            double xi = alpha * x[xOff + ix];
            if (xi != 0.0) {
                if (simd) {
                    DgerSIMD.dgerRowUpdateContiguous(n, xi, y, yOff, a, aOff + i * lda);
                } else {
                    dgerRowUpdateContiguous(n, xi, y, yOff, a, aOff + i * lda);
                }
            }
            ix += incX;
        }
    }

    static void dgerRowUpdateContiguous(int n, double alphaXi, double[] y, int yOff,
                                        double[] a, int rowOff) {
        int j = 0;
        int unrolled = n & ~3;
        for (; j < unrolled; j += 4) {
            a[rowOff + j] = Math.fma(alphaXi, y[yOff + j], a[rowOff + j]);
            a[rowOff + j + 1] = Math.fma(alphaXi, y[yOff + j + 1], a[rowOff + j + 1]);
            a[rowOff + j + 2] = Math.fma(alphaXi, y[yOff + j + 2], a[rowOff + j + 2]);
            a[rowOff + j + 3] = Math.fma(alphaXi, y[yOff + j + 3], a[rowOff + j + 3]);
        }
        for (; j < n; j++) {
            a[rowOff + j] = Math.fma(alphaXi, y[yOff + j], a[rowOff + j]);
        }
    }

    static void dgerRowUpdateStridedY(int n, double alphaXi, double[] y, int yOff, int incY,
                                      double[] a, int rowOff) {
        int iy = yOff;
        int j = 0;
        int unrolled = n & ~3;
        for (; j < unrolled; j += 4) {
            a[rowOff + j] = Math.fma(alphaXi, y[iy], a[rowOff + j]);
            iy += incY;
            a[rowOff + j + 1] = Math.fma(alphaXi, y[iy], a[rowOff + j + 1]);
            iy += incY;
            a[rowOff + j + 2] = Math.fma(alphaXi, y[iy], a[rowOff + j + 2]);
            iy += incY;
            a[rowOff + j + 3] = Math.fma(alphaXi, y[iy], a[rowOff + j + 3]);
            iy += incY;
        }
        for (; j < n; j++) {
            a[rowOff + j] = Math.fma(alphaXi, y[iy], a[rowOff + j]);
            iy += incY;
        }
    }
}

final class DgerSIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;

    private DgerSIMD() {
    }

    static void dgerRowUpdateContiguous(int n, double alphaXi, double[] y, int yOff,
                                        double[] a, int rowOff) {
        DoubleVector alphaVec = DoubleVector.broadcast(SPECIES, alphaXi);
        int limit = SPECIES.loopBound(n);
        int j = 0;
        for (; j < limit; j += LANES) {
            DoubleVector yVec = DoubleVector.fromArray(SPECIES, y, yOff + j);
            DoubleVector aVec = DoubleVector.fromArray(SPECIES, a, rowOff + j);
            alphaVec.fma(yVec, aVec).intoArray(a, rowOff + j);
        }
        for (; j < n; j++) {
            a[rowOff + j] = Math.fma(alphaXi, y[yOff + j], a[rowOff + j]);
        }
    }

    static boolean probe() {
        if (LANES <= 1) {
            return false;
        }
        double[] y = new double[LANES];
        double[] a = new double[LANES];
        dgerRowUpdateContiguous(LANES, 1.0, y, 0, a, 0);
        return true;
    }
}
