/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

interface Dlasq4 {

    double CNST1 = 0.563;
    double CNST2 = 1.01;
    double CNST3 = 1.05;
    double CNST_THIRD = 0.333;

    /**
     * Computes an approximation to the smallest eigenvalue using values of d
     * from the previous dqds transform. This shift estimate (tau) is used by
     * {@link Dlasq5} to accelerate convergence of the dqds iteration.
     *
     * @param i0       zero-based first index of the unreduced submatrix
     * @param n0       zero-based last index of the unreduced submatrix
     * @param z        qd array (length ≥ 4*(n0+1))
     * @param zOff     offset into z
     * @param pp       ping-pong flag (0 or 1)
     * @param n0in     original value of n0 before deflation
     * @param dmin     minimum diagonal value from previous transform
     * @param dmin1    second minimum diagonal value
     * @param dmin2    third minimum diagonal value
     * @param dn       last diagonal value
     * @param dn1      second-to-last diagonal value
     * @param dn2      third-to-last diagonal value
     * @param ttype    shift type array (length ≥ ttypeOff+1); updated on exit
     * @param ttypeOff offset into ttype
     * @param g        shift parameter array (length ≥ gOff+1); updated on exit
     * @param gOff     offset into g
     * @return tau, the computed shift estimate
     */
    static double dlasq4(int i0, int n0, double[] z, int zOff, int pp, int n0in,
                         double dmin, double dmin1, double dmin2, double dn, double dn1, double dn2,
                         int[] ttype, int ttypeOff, double[] g, int gOff) {
        if (dmin <= 0) {
            ttype[ttypeOff] = -1;
            return -dmin;
        }

        int nn = 4 * (n0 + 1) + pp - 1;
        double s;

        if (n0in == n0) {
            if (dmin == dn || dmin == dn1) {
                double b1 = sqrt(z[zOff + nn - 3]) * sqrt(z[zOff + nn - 5]);
                double b2 = sqrt(z[zOff + nn - 7]) * sqrt(z[zOff + nn - 9]);
                double a2 = z[zOff + nn - 7] + z[zOff + nn - 5];

                if (dmin == dn && dmin1 == dn1) {
                    double gap2 = dmin2 - a2 - dmin2 / 4;
                    double gap1;
                    if (gap2 > 0 && gap2 > b2) {
                        gap1 = a2 - dn - (b2 / gap2) * b2;
                    } else {
                        gap1 = a2 - dn - (b1 + b2);
                    }
                    if (gap1 > 0 && gap1 > b1) {
                        s = max(dn - (b1 / gap1) * b1, 0.5 * dmin);
                        ttype[ttypeOff] = -2;
                    } else {
                        s = 0;
                        if (dn > b1) s = dn - b1;
                        if (a2 > b1 + b2) s = min(s, a2 - (b1 + b2));
                        s = max(s, CNST_THIRD * dmin);
                        ttype[ttypeOff] = -3;
                    }
                } else {
                    ttype[ttypeOff] = -4;
                    s = dmin / 4;
                    double gam;
                    int np;
                    if (dmin == dn) {
                        gam = dn;
                        a2 = 0;
                        if (z[zOff + nn - 5] > z[zOff + nn - 7]) {
                            return s;
                        }
                        b2 = z[zOff + nn - 5] / z[zOff + nn - 7];
                        np = nn - 9;
                    } else {
                        np = nn - 2 * pp;
                        gam = dn1;
                        if (z[zOff + np - 4] > z[zOff + np - 2]) {
                            return s;
                        }
                        a2 = z[zOff + np - 4] / z[zOff + np - 2];
                        if (z[zOff + nn - 9] > z[zOff + nn - 11]) {
                            return s;
                        }
                        b2 = z[zOff + nn - 9] / z[zOff + nn - 11];
                        np = nn - 13;
                    }

                    a2 += b2;
                    for (int i4loop = np + 1; i4loop >= 4 * (i0 + 1) - 1 + pp; i4loop -= 4) {
                        int i4 = i4loop - 1;
                        if (b2 == 0) break;
                        b1 = b2;
                        if (z[zOff + i4] > z[zOff + i4 - 2]) {
                            return s;
                        }
                        b2 *= z[zOff + i4] / z[zOff + i4 - 2];
                        a2 += b2;
                        if (100 * max(b2, b1) < a2 || CNST1 < a2) break;
                    }
                    a2 *= CNST3;
                    if (a2 < CNST1) {
                        s = gam * (1 - sqrt(a2)) / (1 + a2);
                    }
                }
            } else if (dmin == dn2) {
                ttype[ttypeOff] = -5;
                s = dmin / 4;

                int np = nn - 2 * pp;
                double b1 = z[zOff + np - 2];
                double b2 = z[zOff + np - 6];
                double gam = dn2;
                if (z[zOff + np - 8] > b2 || z[zOff + np - 4] > b1) {
                    return s;
                }
                double a2 = (z[zOff + np - 8] / b2) * (1 + z[zOff + np - 4] / b1);

                if (n0 - i0 > 2) {
                    b2 = z[zOff + nn - 13] / z[zOff + nn - 15];
                    a2 += b2;
                    for (int i4loop = (nn + 1) - 17; i4loop >= 4 * (i0 + 1) - 1 + pp; i4loop -= 4) {
                        int i4 = i4loop - 1;
                        if (b2 == 0) break;
                        b1 = b2;
                        if (z[zOff + i4] > z[zOff + i4 - 2]) {
                            return s;
                        }
                        b2 *= z[zOff + i4] / z[zOff + i4 - 2];
                        a2 += b2;
                        if (100 * max(b2, b1) < a2 || CNST1 < a2) break;
                    }
                    a2 *= CNST3;
                }

                if (a2 < CNST1) {
                    s = gam * (1 - sqrt(a2)) / (1 + a2);
                }
            } else {
                if (ttype[ttypeOff] == -6) {
                    g[gOff] += CNST_THIRD * (1 - g[gOff]);
                } else if (ttype[ttypeOff] == -18) {
                    g[gOff] = CNST_THIRD / 4;
                } else {
                    g[gOff] = 0.25;
                }
                s = g[gOff] * dmin;
                ttype[ttypeOff] = -6;
            }
        } else if (n0in == (n0 + 1)) {
            if (dmin1 == dn1 && dmin2 == dn2) {
                ttype[ttypeOff] = -7;
                s = CNST_THIRD * dmin1;
                if (z[zOff + nn - 5] > z[zOff + nn - 7]) {
                    return s;
                }
                double b1 = z[zOff + nn - 5] / z[zOff + nn - 7];
                double b2 = b1;
                if (b2 != 0) {
                    for (int i4loop = 4 * (n0 + 1) - 9 + pp; i4loop >= 4 * (i0 + 1) - 1 + pp; i4loop -= 4) {
                        int i4 = i4loop - 1;
                        double a2 = b1;
                        if (z[zOff + i4] > z[zOff + i4 - 2]) {
                            return s;
                        }
                        b1 *= z[zOff + i4] / z[zOff + i4 - 2];
                        b2 += b1;
                        if (100 * max(b1, a2) < b2) break;
                    }
                }
                b2 = sqrt(CNST3 * b2);
                double a2 = dmin1 / (1 + b2 * b2);
                double gap2 = 0.5 * dmin2 - a2;
                if (gap2 > 0 && gap2 > b2 * a2) {
                    s = max(s, a2 * (1 - CNST2 * a2 * (b2 / gap2) * b2));
                } else {
                    s = max(s, a2 * (1 - CNST2 * b2));
                    ttype[ttypeOff] = -8;
                }
            } else {
                s = dmin1 / 4;
                if (dmin1 == dn1) s = 0.5 * dmin1;
                ttype[ttypeOff] = -9;
            }
        } else if (n0in == (n0 + 2)) {
            if (dmin2 == dn2 && 2 * z[zOff + nn - 5] < z[zOff + nn - 7]) {
                ttype[ttypeOff] = -10;
                s = CNST_THIRD * dmin2;
                if (z[zOff + nn - 5] > z[zOff + nn - 7]) {
                    return s;
                }
                double b1 = z[zOff + nn - 5] / z[zOff + nn - 7];
                double b2 = b1;
                if (b2 != 0) {
                    for (int i4loop = 4 * (n0 + 1) - 9 + pp; i4loop >= 4 * (i0 + 1) - 1 + pp; i4loop -= 4) {
                        int i4 = i4loop - 1;
                        if (z[zOff + i4] > z[zOff + i4 - 2]) {
                            return s;
                        }
                        b1 *= z[zOff + i4] / z[zOff + i4 - 2];
                        b2 += b1;
                        if (100 * b1 < b2) break;
                    }
                }
                b2 = sqrt(CNST3 * b2);
                double a2 = dmin2 / (1 + b2 * b2);
                double gap2 = z[zOff + nn - 7] + z[zOff + nn - 9] -
                              sqrt(z[zOff + nn - 11]) * sqrt(z[zOff + nn - 9]) - a2;
                if (gap2 > 0 && gap2 > b2 * a2) {
                    s = max(s, a2 * (1 - CNST2 * a2 * (b2 / gap2) * b2));
                } else {
                    s = max(s, a2 * (1 - CNST2 * b2));
                }
            } else {
                s = dmin2 / 4;
                ttype[ttypeOff] = -11;
            }
        } else if (n0in > n0 + 2) {
            s = 0;
            ttype[ttypeOff] = -12;
        } else {
            s = 0;
        }

        return s;
    }
}
