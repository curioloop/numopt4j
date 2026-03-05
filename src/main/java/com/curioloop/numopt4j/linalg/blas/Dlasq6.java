/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

interface Dlasq6 {

    /**
     * Computes one dqd transform in ping-pong form with protection against
     * overflow and underflow. Unlike {@link Dlasq5}, this routine does not
     * apply a shift (tau = 0) and uses safmin-based guards to avoid division
     * by zero or overflow.
     *
     * <p>Output values (dmin, dmin1, dmin2, dn, dnm1, dnm2) are written to
     * {@code out[0..5]}.
     *
     *
     * @param i0   zero-based first index of the unreduced submatrix
     * @param n0   zero-based last index of the unreduced submatrix
     * @param z    qd array (length ≥ 4*(n0+1)); updated in-place
     * @param zOff offset into z
     * @param pp   ping-pong flag (0 or 1)
     * @param out  output array (length ≥ 6): [dmin, dmin1, dmin2, dn, dnm1, dnm2]
     */
    static void dlasq6(int i0, int n0, double[] z, int zOff, int pp, double[] out) {
        if (n0 - i0 - 1 <= 0) {
            return;
        }

        double safmin = Dlamch.dlamch('S');
        int j4 = 4 * (i0 + 1) + pp - 4;
        double emin = z[zOff + j4 + 4];
        double d = z[zOff + j4];
        double dmin = d;

        if (pp == 0) {
            for (int j4loop = 4 * (i0 + 1); j4loop <= 4 * ((n0 + 1) - 3); j4loop += 4) {
                j4 = j4loop - 1;
                z[zOff + j4 - 2] = d + z[zOff + j4 - 1];
                if (z[zOff + j4 - 2] == 0) {
                    z[zOff + j4] = 0;
                    d = z[zOff + j4 + 1];
                    dmin = d;
                    emin = 0;
                } else if (safmin * z[zOff + j4 + 1] < z[zOff + j4 - 2] &&
                           safmin * z[zOff + j4 - 2] < z[zOff + j4 + 1]) {
                    double tmp = z[zOff + j4 + 1] / z[zOff + j4 - 2];
                    z[zOff + j4] = z[zOff + j4 - 1] * tmp;
                    d = d * tmp;
                } else {
                    z[zOff + j4] = z[zOff + j4 + 1] * (z[zOff + j4 - 1] / z[zOff + j4 - 2]);
                    d = z[zOff + j4 + 1] * (d / z[zOff + j4 - 2]);
                }
                dmin = min(dmin, d);
                emin = min(emin, z[zOff + j4]);
            }
        } else {
            for (int j4loop = 4 * (i0 + 1); j4loop <= 4 * ((n0 + 1) - 3); j4loop += 4) {
                j4 = j4loop - 1;
                z[zOff + j4 - 3] = d + z[zOff + j4];
                if (z[zOff + j4 - 3] == 0) {
                    z[zOff + j4 - 1] = 0;
                    d = z[zOff + j4 + 2];
                    dmin = d;
                    emin = 0;
                } else if (safmin * z[zOff + j4 + 2] < z[zOff + j4 - 3] &&
                           safmin * z[zOff + j4 - 3] < z[zOff + j4 + 2]) {
                    double tmp = z[zOff + j4 + 2] / z[zOff + j4 - 3];
                    z[zOff + j4 - 1] = z[zOff + j4] * tmp;
                    d = d * tmp;
                } else {
                    z[zOff + j4 - 1] = z[zOff + j4 + 2] * (z[zOff + j4] / z[zOff + j4 - 3]);
                    d = z[zOff + j4 + 2] * (d / z[zOff + j4 - 3]);
                }
                dmin = min(dmin, d);
                emin = min(emin, z[zOff + j4 - 1]);
            }
        }

        double dnm2 = d;
        double dmin2 = dmin;
        j4 = 4 * (n0 - 1) - pp - 1;
        int j4p2 = j4 + 2 * pp - 1;
        z[zOff + j4 - 2] = dnm2 + z[zOff + j4p2];
        double dnm1;
        if (z[zOff + j4 - 2] == 0) {
            z[zOff + j4] = 0;
            dnm1 = z[zOff + j4p2 + 2];
            dmin = dnm1;
            emin = 0;
        } else if (safmin * z[zOff + j4p2 + 2] < z[zOff + j4 - 2] &&
                   safmin * z[zOff + j4 - 2] < z[zOff + j4p2 + 2]) {
            double tmp = z[zOff + j4p2 + 2] / z[zOff + j4 - 2];
            z[zOff + j4] = z[zOff + j4p2] * tmp;
            dnm1 = dnm2 * tmp;
        } else {
            z[zOff + j4] = z[zOff + j4p2 + 2] * (z[zOff + j4p2] / z[zOff + j4 - 2]);
            dnm1 = z[zOff + j4p2 + 2] * (dnm2 / z[zOff + j4 - 2]);
        }
        dmin = min(dmin, dnm1);

        double dmin1 = dmin;
        j4 += 4;
        j4p2 = j4 + 2 * pp - 1;
        z[zOff + j4 - 2] = dnm1 + z[zOff + j4p2];
        double dn;
        if (z[zOff + j4 - 2] == 0) {
            z[zOff + j4] = 0;
            dn = z[zOff + j4p2 + 2];
            dmin = dn;
            emin = 0;
        } else if (safmin * z[zOff + j4p2 + 2] < z[zOff + j4 - 2] &&
                   safmin * z[zOff + j4 - 2] < z[zOff + j4p2 + 2]) {
            double tmp = z[zOff + j4p2 + 2] / z[zOff + j4 - 2];
            z[zOff + j4] = z[zOff + j4p2] * tmp;
            dn = dnm1 * tmp;
        } else {
            z[zOff + j4] = z[zOff + j4p2 + 2] * (z[zOff + j4p2] / z[zOff + j4 - 2]);
            dn = z[zOff + j4p2 + 2] * (dnm1 / z[zOff + j4 - 2]);
        }
        dmin = min(dmin, dn);

        z[zOff + j4 + 2] = dn;
        z[zOff + 4 * (n0 + 1) - pp - 1] = emin;

        out[0] = dmin;
        out[1] = dmin1;
        out[2] = dmin2;
        out[3] = dn;
        out[4] = dnm1;
        out[5] = dnm2;
    }
}
