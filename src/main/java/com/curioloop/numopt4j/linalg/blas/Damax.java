/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DAMAX computes the infinity norm (maximum absolute value) of a vector.
 * 
 * <p>Mathematical operation: ‖x‖∞ = max|xᵢ|</p>
 *
 * IDAMAX finds the index of the element with maximum absolute value.
 *
 * <p>Mathematical operation: k = argmax|xᵢ|</p>
 */
interface Damax {

    /**
     * Computes the infinity norm of vector x.
     * 
     * @param n    number of elements
     * @param x    input vector
     * @param xOff offset in x
     * @param incX storage spacing
     * @return ‖x‖∞ = max|xᵢ|
     */
    static double damax(int n, double[] x, int xOff, int incX) {
        if (n <= 0) return 0.0;

        // ---- fast path: contiguous ----
        if (incX == 1) {
            int i = xOff;
            int end = xOff + n;

            double max0 = 0.0;
            double max1 = 0.0;

            int limit = i + ((end - i) & ~7); // 8-way

            for (; i < limit; i += 8) {
                double a0 = Math.abs(x[i]);
                double a1 = Math.abs(x[i+1]);
                double a2 = Math.abs(x[i+2]);
                double a3 = Math.abs(x[i+3]);
                double a4 = Math.abs(x[i+4]);
                double a5 = Math.abs(x[i+5]);
                double a6 = Math.abs(x[i+6]);
                double a7 = Math.abs(x[i+7]);

                // two independent chains → better ILP
                if (a0 > max0) max0 = a0;
                if (a1 > max1) max1 = a1;
                if (a2 > max0) max0 = a2;
                if (a3 > max1) max1 = a3;
                if (a4 > max0) max0 = a4;
                if (a5 > max1) max1 = a5;
                if (a6 > max0) max0 = a6;
                if (a7 > max1) max1 = a7;
            }

            double result = Math.max(max0, max1);

            for (; i < end; i++) {
                double a = Math.abs(x[i]);
                if (a > result) result = a;
            }

            return result;
        }

        double result = 0.0;
        int ix = xOff + ((incX > 0) ? 0 : (n - 1) * (-incX));

        for (int i = 0; i < n; i++) {
            double a = Math.abs(x[ix]);
            if (a > result) result = a;
            ix += incX;
        }

        return result;
    }

    /**
     * Finds the index of the element with maximum absolute value.
     *
     * @param n    number of elements
     * @param x    input vector
     * @param xOff offset in x
     * @param incX storage spacing
     * @return 0-based index of max |xᵢ|, or -1 if n ≤ 0
     */
    static int idamax(int n, double[] x, int xOff, int incX) {
        if (n < 1 || incX <= 0) return -1;
        if (n == 1) return 0;

        int idmax = 0;
        if (incX == 1) {
            double dmax = Math.abs(x[xOff]);
            int i = 1;
            int limit = n - 3;
            for (; i < limit; i += 4) {
                double a1 = Math.abs(x[xOff + i]);
                double a2 = Math.abs(x[xOff + i + 1]);
                double a3 = Math.abs(x[xOff + i + 2]);
                double a4 = Math.abs(x[xOff + i + 3]);
                if (a1 > dmax) { dmax = a1; idmax = i; }
                if (a2 > dmax) { dmax = a2; idmax = i + 1; }
                if (a3 > dmax) { dmax = a3; idmax = i + 2; }
                if (a4 > dmax) { dmax = a4; idmax = i + 3; }
            }
            for (; i < n; i++) {
                double absxi = Math.abs(x[xOff + i]);
                if (absxi > dmax) {
                    dmax = absxi;
                    idmax = i;
                }
            }
        } else {
            int ix = xOff;
            double dmax = Math.abs(x[ix]);
            ix += incX;
            for (int i = 1; i < n; i++) {
                double absxi = Math.abs(x[ix]);
                if (absxi > dmax) {
                    dmax = absxi;
                    idmax = i;
                }
                ix += incX;
            }
        }
        return idmax;
    }
}
