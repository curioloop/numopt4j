/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * DAXPY computes y += α × x.
 * 
 * <p>Mathematical operation: y[yOff + i*incY] += α × x[xOff + i*incX]</p>
 * 
 * <p>Reference: BLAS DAXPY</p>
 */
interface Daxpy {

    int SIMD_MIN_N = 64;

    static void daxpy(int n, double alpha,
                      double[] x, int xOff, int incX,
                      double[] y, int yOff, int incY) {
        if (n <= 0 || alpha == 0.0) return;

        if (incX == 1 && incY == 1 && n >= SIMD_MIN_N && SIMD.supportDaxpy()) {
            if (DaxpySIMD.daxpy(n, alpha, x, xOff, y, yOff)) {
                return;
            }
        }

        if (incX == 1 && incY == 1) {
            int k = 0;
            for (; k + 3 < n; k += 4) {
                y[yOff + k] = Math.fma(alpha, x[xOff + k], y[yOff + k]);
                y[yOff + k + 1] = Math.fma(alpha, x[xOff + k + 1], y[yOff + k + 1]);
                y[yOff + k + 2] = Math.fma(alpha, x[xOff + k + 2], y[yOff + k + 2]);
                y[yOff + k + 3] = Math.fma(alpha, x[xOff + k + 3], y[yOff + k + 3]);
            }
            for (; k < n; k++) {
                y[yOff + k] = Math.fma(alpha, x[xOff + k], y[yOff + k]);
            }
        } else {
            int xi = xOff, yi = yOff;
            for (int k = 0; k < n; k++) {
                y[yi] = Math.fma(alpha, x[xi], y[yi]);
                xi += incX;
                yi += incY;
            }
        }
    }

}

final class DaxpySIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;

    private DaxpySIMD() {
    }

    static boolean daxpy(int n, double alpha,
                         double[] x, int xOff,
                         double[] y, int yOff) {
        if (LANES <= 1) {
            return false;
        }

        DoubleVector alphaVec = DoubleVector.broadcast(SPECIES, alpha);
        int limit = SPECIES.loopBound(n);
        int k = 0;
        for (; k < limit; k += LANES) {
            DoubleVector xVec = DoubleVector.fromArray(SPECIES, x, xOff + k);
            DoubleVector yVec = DoubleVector.fromArray(SPECIES, y, yOff + k);
            alphaVec.fma(xVec, yVec).intoArray(y, yOff + k);
        }
        for (; k < n; k++) {
            y[yOff + k] = Math.fma(alpha, x[xOff + k], y[yOff + k]);
        }
        return true;
    }

    static boolean probe() {
        if (LANES <= 1) {
            return false;
        }
        double[] x = new double[LANES];
        double[] y = new double[LANES];
        return daxpy(LANES, 1.0, x, 0, y, 0);
    }
}
