/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

interface Dlasq5 {

    /**
     * Computes one dqds transform in ping-pong form. This is the shifted variant
     * that subtracts tau from each diagonal element before the transform.
     * When tau is zero, small diagonal values are set to zero to avoid underflow.
     *
     * <p>Output values (dmin, dmin1, dmin2, dn, dnm1, dnm2) are written to
     * {@code state[stateOff..stateOff+5]}.
     *
     * @param i0       zero-based first index of the unreduced submatrix
     * @param n0       zero-based last index of the unreduced submatrix
     * @param z        qd array (length ≥ 4*(n0+1)); updated in-place
     * @param zOff     offset into z
     * @param pp       ping-pong flag (0 or 1)
     * @param tau      shift value
     * @param sigma    accumulated shift (used for threshold computation)
     * @param state    output array (length ≥ stateOff+6): [dmin, dmin1, dmin2, dn, dnm1, dnm2]
     * @param stateOff offset into state
     */
    static void dlasq5(int i0, int n0, double[] z, int zOff, int pp, double tau, double sigma,
                       double[] state, int stateOff) {
        if (n0 - i0 - 1 <= 0) {
            return;
        }

        double eps = Dlamch.dlamch('P');
        double dthresh = eps * (sigma + tau);
        if (tau < dthresh * 0.5) {
            tau = 0;
        }

        double dmin, dmin1, dmin2, dn, dnm1, dnm2;
        int j4;
        double emin;

        if (tau != 0) {
            j4 = 4 * i0 + pp;
            emin = z[zOff + j4 + 4];
            double d = z[zOff + j4] - tau;
            dmin = d;

            if (pp == 0) {
                for (int j4loop = 4 * (i0 + 1); j4loop <= 4 * ((n0 + 1) - 3); j4loop += 4) {
                    j4 = j4loop - 1;
                    z[zOff + j4 - 2] = d + z[zOff + j4 - 1];
                    double tmp = z[zOff + j4 + 1] / z[zOff + j4 - 2];
                    d = d * tmp - tau;
                    dmin = min(dmin, d);
                    z[zOff + j4] = z[zOff + j4 - 1] * tmp;
                    emin = min(emin, z[zOff + j4]);
                }
            } else {
                for (int j4loop = 4 * (i0 + 1); j4loop <= 4 * ((n0 + 1) - 3); j4loop += 4) {
                    j4 = j4loop - 1;
                    z[zOff + j4 - 3] = d + z[zOff + j4];
                    double tmp = z[zOff + j4 + 2] / z[zOff + j4 - 3];
                    d = d * tmp - tau;
                    dmin = min(dmin, d);
                    z[zOff + j4 - 1] = z[zOff + j4] * tmp;
                    emin = min(emin, z[zOff + j4 - 1]);
                }
            }

            dnm2 = d;
            dmin2 = dmin;
            j4 = 4 * ((n0 + 1) - 2) - pp - 1;
            int j4p2 = j4 + 2 * pp - 1;
            z[zOff + j4 - 2] = dnm2 + z[zOff + j4p2];
            z[zOff + j4] = z[zOff + j4p2 + 2] * (z[zOff + j4p2] / z[zOff + j4 - 2]);
            dnm1 = z[zOff + j4p2 + 2] * (dnm2 / z[zOff + j4 - 2]) - tau;
            dmin = min(dmin, dnm1);

            dmin1 = dmin;
            j4 += 4;
            j4p2 = j4 + 2 * pp - 1;
            z[zOff + j4 - 2] = dnm1 + z[zOff + j4p2];
            z[zOff + j4] = z[zOff + j4p2 + 2] * (z[zOff + j4p2] / z[zOff + j4 - 2]);
            dn = z[zOff + j4p2 + 2] * (dnm1 / z[zOff + j4 - 2]) - tau;
            dmin = min(dmin, dn);
        } else {
            j4 = 4 * (i0 + 1) + pp - 4;
            emin = z[zOff + j4 + 4];
            double d = z[zOff + j4] - tau;
            dmin = d;

            if (pp == 0) {
                for (int j4loop = 4 * (i0 + 1); j4loop <= 4 * ((n0 + 1) - 3); j4loop += 4) {
                    j4 = j4loop - 1;
                    z[zOff + j4 - 2] = d + z[zOff + j4 - 1];
                    double tmp = z[zOff + j4 + 1] / z[zOff + j4 - 2];
                    d = d * tmp - tau;
                    if (d < dthresh) {
                        d = 0;
                    }
                    dmin = min(dmin, d);
                    z[zOff + j4] = z[zOff + j4 - 1] * tmp;
                    emin = min(emin, z[zOff + j4]);
                }
            } else {
                for (int j4loop = 4 * (i0 + 1); j4loop <= 4 * ((n0 + 1) - 3); j4loop += 4) {
                    j4 = j4loop - 1;
                    z[zOff + j4 - 3] = d + z[zOff + j4];
                    double tmp = z[zOff + j4 + 2] / z[zOff + j4 - 3];
                    d = d * tmp - tau;
                    if (d < dthresh) {
                        d = 0;
                    }
                    dmin = min(dmin, d);
                    z[zOff + j4 - 1] = z[zOff + j4] * tmp;
                    emin = min(emin, z[zOff + j4 - 1]);
                }
            }

            dnm2 = d;
            dmin2 = dmin;
            j4 = 4 * ((n0 + 1) - 2) - pp - 1;
            int j4p2 = j4 + 2 * pp - 1;
            z[zOff + j4 - 2] = dnm2 + z[zOff + j4p2];
            z[zOff + j4] = z[zOff + j4p2 + 2] * (z[zOff + j4p2] / z[zOff + j4 - 2]);
            dnm1 = z[zOff + j4p2 + 2] * (dnm2 / z[zOff + j4 - 2]) - tau;
            dmin = min(dmin, dnm1);

            dmin1 = dmin;
            j4 += 4;
            j4p2 = j4 + 2 * pp - 1;
            z[zOff + j4 - 2] = dnm1 + z[zOff + j4p2];
            z[zOff + j4] = z[zOff + j4p2 + 2] * (z[zOff + j4p2] / z[zOff + j4 - 2]);
            dn = z[zOff + j4p2 + 2] * (dnm1 / z[zOff + j4 - 2]) - tau;
            dmin = min(dmin, dn);
        }

        z[zOff + j4 + 2] = dn;
        z[zOff + 4 * (n0 + 1) - pp - 1] = emin;

        state[stateOff] = dmin;
        state[stateOff + 1] = dmin1;
        state[stateOff + 2] = dmin2;
        state[stateOff + 3] = dn;
        state[stateOff + 4] = dnm1;
        state[stateOff + 5] = dnm2;
    }
}
