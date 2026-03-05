/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DAXPY computes y += α × x.
 * 
 * <p>Mathematical operation: y[yOff + i*incY] += α × x[xOff + i*incX]</p>
 * 
 * <p>Reference: BLAS DAXPY</p>
 */
interface Daxpy {

    static void daxpy(int n, double alpha,
                      double[] x, int xOff, int incX,
                      double[] y, int yOff, int incY) {
        if (n <= 0 || alpha == 0.0) return;

        if (incX == 1 && incY == 1) {
            int k = 0;
            for (; k + 3 < n; k += 4) {
                y[yOff + k] = FMA.op(alpha, x[xOff + k], y[yOff + k]);
                y[yOff + k + 1] = FMA.op(alpha, x[xOff + k + 1], y[yOff + k + 1]);
                y[yOff + k + 2] = FMA.op(alpha, x[xOff + k + 2], y[yOff + k + 2]);
                y[yOff + k + 3] = FMA.op(alpha, x[xOff + k + 3], y[yOff + k + 3]);
            }
            for (; k < n; k++) {
                y[yOff + k] = FMA.op(alpha, x[xOff + k], y[yOff + k]);
            }
        } else {
            int xi = xOff, yi = yOff;
            for (int k = 0; k < n; k++) {
                y[yi] = FMA.op(alpha, x[xi], y[yi]);
                xi += incX;
                yi += incY;
            }
        }
    }

}
