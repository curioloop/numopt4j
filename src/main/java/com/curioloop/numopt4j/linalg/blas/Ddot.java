/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * DDOT computes dot product of two vectors with pairwise summation.
 * 
 * <p>Mathematical operation: result = Σ x[i] × y[i]</p>
 * 
 * <p>Pairwise summation reduces floating-point error from O(n) to O(log n).</p>
 * 
 * <p>Reference: BLAS DDOT</p>
 */
interface Ddot {

    int THRESHOLD = 64;

    /**
     * Computes dot product with stride support.
     * result = Σ x[xOff + i*incX] × y[yOff + i*incY]
     */
    static double ddot(int n, double[] x, int xOff, int incX,
                       double[] y, int yOff, int incY) {
        if (n <= 0) return 0.0;
        
        if (incX == 1 && incY == 1) {
            return ddotContiguous(x, xOff, y, yOff, n);
        }
        return ddotStrided(x, xOff, incX, y, yOff, incY, n);
    }

    static double ddotContiguous(double[] x, int xOff, double[] y, int yOff, int n) {
        if (n <= THRESHOLD) {
            if (n == THRESHOLD && SIMD.supportDdot()) {
                return DdotSIMD.ddot(n, x, xOff, y, yOff);
            }

            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int k = 0;
            for (; k + 3 < n; k += 4) {
                s0 = Math.fma(x[xOff + k], y[yOff + k], s0);
                s1 = Math.fma(x[xOff + k + 1], y[yOff + k + 1], s1);
                s2 = Math.fma(x[xOff + k + 2], y[yOff + k + 2], s2);
                s3 = Math.fma(x[xOff + k + 3], y[yOff + k + 3], s3);
            }
            for (; k < n; k++) {
                s0 = Math.fma(x[xOff + k], y[yOff + k], s0);
            }
            return (s0 + s1) + (s2 + s3);
        }
        int mid = n / 2;
        return ddotContiguous(x, xOff, y, yOff, mid) +
               ddotContiguous(x, xOff + mid, y, yOff + mid, n - mid);
    }

    static double ddotStrided(double[] x, int xOff, int incX,
                              double[] y, int yOff, int incY, int n) {
        if (n <= THRESHOLD) {
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int xi = xOff, yi = yOff;
            int k = 0;
            for (; k + 3 < n; k += 4) {
                s0 = Math.fma(x[xi], y[yi], s0); xi += incX; yi += incY;
                s1 = Math.fma(x[xi], y[yi], s1); xi += incX; yi += incY;
                s2 = Math.fma(x[xi], y[yi], s2); xi += incX; yi += incY;
                s3 = Math.fma(x[xi], y[yi], s3); xi += incX; yi += incY;
            }
            for (; k < n; k++) {
                s0 = Math.fma(x[xi], y[yi], s0); xi += incX; yi += incY;
            }
            return (s0 + s1) + (s2 + s3);
        }
        int mid = n / 2;
        return ddotStrided(x, xOff, incX, y, yOff, incY, mid) +
               ddotStrided(x, xOff + mid * incX, incX, y, yOff + mid * incY, incY, n - mid);
    }

}

final class DdotSIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;

    private DdotSIMD() {
    }

    static double ddot(int n, double[] x, int xOff, double[] y, int yOff) {
        if (LANES == 4) {
            DoubleVector acc = DoubleVector.zero(SPECIES);
            int k = 0;
            for (; k + 3 < n; k += 4) {
                acc = DoubleVector.fromArray(SPECIES, x, xOff + k)
                    .fma(DoubleVector.fromArray(SPECIES, y, yOff + k), acc);
            }
            double s0 = acc.lane(0);
            double s1 = acc.lane(1);
            double s2 = acc.lane(2);
            double s3 = acc.lane(3);
            for (; k < n; k++) {
                s0 = Math.fma(x[xOff + k], y[yOff + k], s0);
            }
            return (s0 + s1) + (s2 + s3);
        }

        if (LANES == 2) {
            DoubleVector acc01 = DoubleVector.zero(SPECIES);
            DoubleVector acc23 = DoubleVector.zero(SPECIES);
            int k = 0;
            for (; k + 3 < n; k += 4) {
                acc01 = DoubleVector.fromArray(SPECIES, x, xOff + k)
                    .fma(DoubleVector.fromArray(SPECIES, y, yOff + k), acc01);
                acc23 = DoubleVector.fromArray(SPECIES, x, xOff + k + 2)
                    .fma(DoubleVector.fromArray(SPECIES, y, yOff + k + 2), acc23);
            }
            double s0 = acc01.lane(0);
            double s1 = acc01.lane(1);
            double s2 = acc23.lane(0);
            double s3 = acc23.lane(1);
            for (; k < n; k++) {
                s0 = Math.fma(x[xOff + k], y[yOff + k], s0);
            }
            return (s0 + s1) + (s2 + s3);
        }

        double s0 = 0.0;
        double s1 = 0.0;
        double s2 = 0.0;
        double s3 = 0.0;
        int k = 0;
        for (; k + 3 < n; k += 4) {
            s0 = Math.fma(x[xOff + k], y[yOff + k], s0);
            s1 = Math.fma(x[xOff + k + 1], y[yOff + k + 1], s1);
            s2 = Math.fma(x[xOff + k + 2], y[yOff + k + 2], s2);
            s3 = Math.fma(x[xOff + k + 3], y[yOff + k + 3], s3);
        }
        for (; k < n; k++) {
            s0 = Math.fma(x[xOff + k], y[yOff + k], s0);
        }
        return (s0 + s1) + (s2 + s3);
    }

    static boolean probe() {
        double[] x = {1.0, 2.0, 3.0, 4.0};
        double[] y = {5.0, 6.0, 7.0, 8.0};
        ddot(4, x, 0, y, 0);
        return true;
    }
}
