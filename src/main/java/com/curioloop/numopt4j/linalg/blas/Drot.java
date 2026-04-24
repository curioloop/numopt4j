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
        if (c == 1.0 && s == 0.0) return;

        if (incX == 1 && incY == 1) {
            drotContiguous(n, x, xOff, y, yOff, c, s);
            return;
        }

        if (incX == incY && incX > 0) {
            drotEqualPositiveStride(n, x, xOff, y, yOff, incX, c, s);
            return;
        }

        int ix = xOff + ((incX > 0) ? 0 : (n - 1) * (-incX));
        int iy = yOff + ((incY > 0) ? 0 : (n - 1) * (-incY));
        for (int i = 0; i < n; i++) {
            rotatePair(x, ix, y, iy, c, s);
            ix += incX;
            iy += incY;
        }
    }

    private static void drotContiguous(int n, double[] x, int xOff,
                                       double[] y, int yOff, double c, double s) {
        int k = 0;
        for (; k + 3 < n; k += 4) {
            int i0 = xOff + k;
            int j0 = yOff + k;

            rotatePair(x, i0, y, j0, c, s);
            rotatePair(x, i0 + 1, y, j0 + 1, c, s);
            rotatePair(x, i0 + 2, y, j0 + 2, c, s);
            rotatePair(x, i0 + 3, y, j0 + 3, c, s);
        }
        for (; k < n; k++) {
            rotatePair(x, xOff + k, y, yOff + k, c, s);
        }
    }

    private static void drotEqualPositiveStride(int n, double[] x, int xOff,
                                                double[] y, int yOff, int inc,
                                                double c, double s) {
        int ix = xOff;
        int iy = yOff;
        int i = 0;
        for (; i + 3 < n; i += 4) {
            rotatePair(x, ix, y, iy, c, s);
            ix += inc;
            iy += inc;

            rotatePair(x, ix, y, iy, c, s);
            ix += inc;
            iy += inc;

            rotatePair(x, ix, y, iy, c, s);
            ix += inc;
            iy += inc;

            rotatePair(x, ix, y, iy, c, s);
            ix += inc;
            iy += inc;
        }
        for (; i < n; i++) {
            rotatePair(x, ix, y, iy, c, s);
            ix += inc;
            iy += inc;
        }
    }

    private static void rotatePair(double[] x, int xIndex, double[] y, int yIndex, double c, double s) {
        double temp = Math.fma(c, x[xIndex], s * y[yIndex]);
        y[yIndex] = Math.fma(c, y[yIndex], -s * x[xIndex]);
        x[xIndex] = temp;
    }
}
