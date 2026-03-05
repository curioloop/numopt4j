/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.abs;

/**
 * DASUM computes the sum of absolute values (L1 norm) of a vector.
 *
 * <p>Mathematical operation: ‖x‖₁ = Σ|xᵢ|</p>
 */
interface Dasum {

    /**
     * Computes the sum of absolute values of vector x.
     *
     * @param n    number of elements
     * @param x    input vector
     * @param xOff offset in x
     * @param incX storage spacing
     * @return ‖x‖₁ = Σ|xᵢ|
     */
    static double dasum(int n, double[] x, int xOff, int incX) {
        if (n <= 0) return 0.0;

        if (incX == 1) {
            int i = xOff;
            int end = xOff + n;

            double sum0 = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            double sum3 = 0.0;

            int limit = i + ((end - i) & ~15);

            for (; i < limit; i += 16) {
                sum0 += abs(x[i]) + abs(x[i + 4]) + abs(x[i + 8]) + abs(x[i + 12]);
                sum1 += abs(x[i + 1]) + abs(x[i + 5]) + abs(x[i + 9]) + abs(x[i + 13]);
                sum2 += abs(x[i + 2]) + abs(x[i + 6]) + abs(x[i + 10]) + abs(x[i + 14]);
                sum3 += abs(x[i + 3]) + abs(x[i + 7]) + abs(x[i + 11]) + abs(x[i + 15]);
            }

            double result = sum0 + sum1 + sum2 + sum3;

            for (; i < end; i++) {
                result += abs(x[i]);
            }

            return result;
        }

        double result = 0.0;
        int ix = xOff + ((incX > 0) ? 0 : (n - 1) * (-incX));

        for (int i = 0; i < n; i++) {
            result += abs(x[ix]);
            ix += incX;
        }

        return result;
    }
}
