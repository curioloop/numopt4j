/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DGER performs rank-1 update: A := alpha*x*yᵀ + A
 * 
 * <p>Reference: BLAS Level 2 DGER</p>
 */
interface Dger {

    static void dger(int m, int n, double alpha, double[] x, int xOff, int incX,
                     double[] y, int yOff, int incY, double[] A, int aOff, int lda) {
        if (m == 0 || n == 0) return;
        if (alpha == 0.0) return;

        if (incX == 1 && incY == 1) {
            for (int i = 0; i < m; i++) {
                double xi = alpha * x[xOff + i];
                int rowOff = aOff + i * lda;
                for (int j = 0; j < n; j++) {
                    A[rowOff + j] = FMA.op(xi, y[yOff + j], A[rowOff + j]);
                }
            }
            return;
        }

        int ky = (incY < 0) ? (-(n - 1) * incY) : 0;
        int kx = (incX < 0) ? (-(m - 1) * incX) : 0;

        int ix = kx;
        for (int i = 0; i < m; i++) {
            double xi = alpha * x[xOff + ix];
            int rowOff = aOff + i * lda;
            int iy = ky;
            for (int j = 0; j < n; j++) {
                A[rowOff + j] = FMA.op(xi, y[yOff + iy], A[rowOff + j]);
                iy += incY;
            }
            ix += incX;
        }
    }
}
