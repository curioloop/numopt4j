/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * DSCAL scales a vector by a constant.
 * 
 * <p>Mathematical operation: x *= α</p>
 * 
 * <p>Reference: BLAS DSCAL</p>
 */
interface Dscal {

    int SIMD_MIN_N = 80;

    /**
     * Scales vector x by constant alpha.
     */
    static void dscal(int n, double alpha, double[] x, int xOff, int incX) {
        if (n <= 0) return;

        if (incX == 1 && n >= SIMD_MIN_N && SIMD.supportDscal()) {
            if (DscalSIMD.dscal(n, alpha, x, xOff)) {
                return;
            }
        }

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

final class DscalSIMD {

    private static final VectorSpecies<Double> SPECIES = Gate.SPECIES;
    private static final int LANES = Gate.LANES;

    private DscalSIMD() {
    }

    static boolean dscal(int n, double alpha, double[] x, int xOff) {
        if (LANES <= 1) {
            return false;
        }

        DoubleVector alphaVec = DoubleVector.broadcast(SPECIES, alpha);
        int limit = SPECIES.loopBound(n);
        int k = 0;
        for (; k < limit; k += LANES) {
            alphaVec.mul(DoubleVector.fromArray(SPECIES, x, xOff + k)).intoArray(x, xOff + k);
        }
        for (; k < n; k++) {
            x[xOff + k] *= alpha;
        }
        return true;
    }

    static boolean probe() {
        if (LANES <= 1) {
            return false;
        }
        double[] x = new double[LANES];
        return dscal(LANES, 1.0, x, 0);
    }
}
