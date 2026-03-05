/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.abs;
import static java.lang.Math.sqrt;

/**
 * DROT applies a plane rotation, and DROTG constructs a Givens plane rotation.
 * 
 * <p>DROTG mathematical operation:</p>
 * <pre>
 *   [ c  s] [a]   [r]
 *   [-s  c] [b] = [0]
 * </pre>
 * 
 * <p>where r = ±Sqrt(a² + b²), c = a/r, s = b/r</p>
 * 
 * <p>DROT mathematical operation:</p>
 * <pre>
 *   [xᵢ]   [ c  s] [xᵢ]
 *   [yᵢ] ← [-s  c] [yᵢ]
 * </pre>
 * 
 * <p>Reference: BLAS DROT, DROTG</p>
 */
interface Drot {

    static void drotg(double a, double b, double[] out, int off) {
        double c, s, r, z;

        if (b == 0) {
            c = 1.0;
            s = 0.0;
            r = abs(a);
            z = (a < 0) ? -1.0 : 1.0;
        } else if (a == 0) {
            c = 0.0;
            s = (b < 0) ? -1.0 : 1.0;
            r = abs(b);
            z = 1.0;
        } else {
            double absA = abs(a);
            double absB = abs(b);
            
            if (absA > absB) {
                r = absA * sqrt(1 + (b / a) * (b / a));
                c = a / r;
                s = b / r;
                z = s;
            } else {
                r = absB * sqrt(1 + (a / b) * (a / b));
                c = a / r;
                s = b / r;
                z = (c != 0) ? 1.0 / c : 1.0;
            }
            
            if (r < 0) {
                r = -r;
                c = -c;
                s = -s;
            }
        }

        out[off] = c;
        out[off + 1] = s;
        out[off + 2] = r;
        out[off + 3] = z;
    }

    static void drot(int n, double[] x, int xOff, int incX,
                     double[] y, int yOff, int incY, double c, double s) {
        if (n <= 0) return;

        if (incX == 1 && incY == 1) {
            int k = 0;
            for (; k + 3 < n; k += 4) {
                int i0 = xOff + k, j0 = yOff + k;
                
                double t0 = FMA.op(c, x[i0], s * y[j0]);
                y[j0] = FMA.op(c, y[j0], -s * x[i0]);
                x[i0] = t0;

                double t1 = FMA.op(c, x[i0 + 1], s * y[j0 + 1]);
                y[j0 + 1] = FMA.op(c, y[j0 + 1], -s * x[i0 + 1]);
                x[i0 + 1] = t1;

                double t2 = FMA.op(c, x[i0 + 2], s * y[j0 + 2]);
                y[j0 + 2] = FMA.op(c, y[j0 + 2], -s * x[i0 + 2]);
                x[i0 + 2] = t2;

                double t3 = FMA.op(c, x[i0 + 3], s * y[j0 + 3]);
                y[j0 + 3] = FMA.op(c, y[j0 + 3], -s * x[i0 + 3]);
                x[i0 + 3] = t3;
            }
            for (; k < n; k++) {
                int i = xOff + k, j = yOff + k;
                double temp = FMA.op(c, x[i], s * y[j]);
                y[j] = FMA.op(c, y[j], -s * x[i]);
                x[i] = temp;
            }
        } else {
            int ix = xOff + ((incX > 0) ? 0 : (n - 1) * (-incX));
            int iy = yOff + ((incY > 0) ? 0 : (n - 1) * (-incY));
            for (int i = 0; i < n; i++) {
                double temp = FMA.op(c, x[ix], s * y[iy]);
                y[iy] = FMA.op(c, y[iy], -s * x[ix]);
                x[ix] = temp;
                ix += incX;
                iy += incY;
            }
        }
    }
}
