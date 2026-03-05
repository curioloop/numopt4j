/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DSCAL scales a vector by a constant.
 * 
 * <p>Mathematical operation: x *= α</p>
 * 
 * <p>Reference: BLAS DSCAL</p>
 */
interface Dscal {

    /**
     * Scales vector x by constant alpha.
     */
    static void dscal(int n, double alpha, double[] x, int xOff, int incX) {
        if (n <= 0) return;

        if (incX == 1) {
            int k = 0;
            for (; k + 3 < n; k += 4) {
                x[xOff + k] *= alpha;
                x[xOff + k + 1] *= alpha;
                x[xOff + k + 2] *= alpha;
                x[xOff + k + 3] *= alpha;
            }
            for (; k < n; k++) {
                x[xOff + k] *= alpha;
            }
        } else {
            int xi = xOff;
            for (int k = 0; k < n; k++) {
                x[xi] *= alpha;
                xi += incX;
            }
        }
    }
}
