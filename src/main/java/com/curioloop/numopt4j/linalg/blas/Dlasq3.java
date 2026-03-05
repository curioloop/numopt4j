/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

interface Dlasq3 {

    double CBIAS = 1.5;

    /**
     * DLASQ3 checks for deflation, computes a shift (tau) and calls dqds.
     * In case of failure it changes shifts and tries again until output is positive.
     *
     * <p>State is passed in/out via {@code istate} and {@code dstate} arrays:
     * <ul>
     *   <li>istate[iOff+0..5]: i0, n0, pp, iter, nFail, nDiv</li>
     *   <li>istate[iOff+6]:    ttype</li>
     *   <li>dstate[dOff+0..4]: dmin1, dmin2, dn, dn1, dn2</li>
     *   <li>dstate[dOff+5]:    g (shift parameter, persistent across calls)</li>
     *   <li>dstate[dOff+6]:    sigma (accumulated shift)</li>
     *   <li>dstate[dOff+7]:    desig (compensated summation residual)</li>
     *   <li>dstate[dOff+8]:    qmax</li>
     *   <li>dstate[dOff+9]:    dmin (minimum diagonal, passed in from Dlasq2 and updated each call)</li>
     *   <li>dstate[dOff+10..15]: Dlasq5 output (dmin, dmin1, dmin2, dn, dnm1, dnm2)</li>
     * </ul>
     *
     * <p>LAPACK routine: DLASQ3
     * <p>Go reference: gonum/lapack/gonum/dlasq3.go
     */
    static void dlasq3(int[] istate, int iOff, double[] dstate, int dOff, double[] z, int zOff) {
        double eps = Dlamch.dlamch('P');
        double tol = eps * 100;
        double tol2 = tol * tol;

        int i0 = istate[iOff];
        int n0 = istate[iOff + 1];
        int pp = istate[iOff + 2];
        int iter = istate[iOff + 3];
        int nFail = istate[iOff + 4];
        int nDiv = istate[iOff + 5];

        double dmin1 = dstate[dOff];
        double dmin2 = dstate[dOff + 1];
        double dn = dstate[dOff + 2];
        double dn1 = dstate[dOff + 3];
        double dn2 = dstate[dOff + 4];
        // g is passed to Dlasq4 via dstate[dOff+5] directly; no local variable needed
        double sigma = dstate[dOff + 6];
        double desig = dstate[dOff + 7];
        double qmax = dstate[dOff + 8];

        int n0in = n0;
        double dmin = dstate[dOff + 9]; // dmin passed from Dlasq2 via dstate[dOff+9]

        while (true) {
            if (n0 < i0) {
                break;
            }
            if (n0 == i0) {
                z[zOff + 4 * (n0 + 1) - 4] = z[zOff + 4 * (n0 + 1) + pp - 4] + sigma;
                n0--;
                continue;
            }

            int nn = 4 * (n0 + 1) + pp - 1;
            if (n0 != i0 + 1) {
                if (z[zOff + nn - 5] > tol2 * (sigma + z[zOff + nn - 3]) &&
                    z[zOff + nn - 2 * pp - 4] > tol2 * z[zOff + nn - 7]) {
                    if (z[zOff + nn - 9] > tol2 * sigma &&
                        z[zOff + nn - 2 * pp - 8] > tol2 * z[zOff + nn - 11]) {
                        break;
                    }
                } else {
                    z[zOff + 4 * (n0 + 1) - 4] = z[zOff + 4 * (n0 + 1) + pp - 4] + sigma;
                    n0--;
                    continue;
                }
            }

            if (z[zOff + nn - 3] > z[zOff + nn - 7]) {
                double tmp = z[zOff + nn - 3];
                z[zOff + nn - 3] = z[zOff + nn - 7];
                z[zOff + nn - 7] = tmp;
            }
            double t = 0.5 * (z[zOff + nn - 7] - z[zOff + nn - 3] + z[zOff + nn - 5]);
            if (z[zOff + nn - 5] > z[zOff + nn - 3] * tol2 && t != 0) {
                double s = z[zOff + nn - 3] * (z[zOff + nn - 5] / t);
                if (s <= t) {
                    s = z[zOff + nn - 3] * (z[zOff + nn - 5] / (t * (1 + sqrt(1 + s / t))));
                } else {
                    s = z[zOff + nn - 3] * (z[zOff + nn - 5] / (t + sqrt(t) * sqrt(t + s)));
                }
                t = z[zOff + nn - 7] + (s + z[zOff + nn - 5]);
                z[zOff + nn - 3] *= z[zOff + nn - 7] / t;
                z[zOff + nn - 7] = t;
            }
            z[zOff + 4 * (n0 + 1) - 8] = z[zOff + nn - 7] + sigma;
            z[zOff + 4 * (n0 + 1) - 4] = z[zOff + nn - 3] + sigma;
            n0 -= 2;
        }

        if (pp == 2) {
            pp = 0;
        }

        if ((dmin <= 0 || n0 < n0in) && n0 >= i0) {
            if (CBIAS * z[zOff + 4 * (i0 + 1) + pp - 4] < z[zOff + 4 * (n0 + 1) + pp - 4]) {
                int ipn4Out = 4 * (i0 + n0 + 2);
                for (int j4loop = 4 * (i0 + 1); j4loop <= 2 * (i0 + n0 + 1); j4loop += 4) {
                    int ipn4 = ipn4Out - 1;
                    int j4 = j4loop - 1;

                    double tmp = z[zOff + j4 - 3];
                    z[zOff + j4 - 3] = z[zOff + ipn4 - j4 - 4];
                    z[zOff + ipn4 - j4 - 4] = tmp;
                    tmp = z[zOff + j4 - 2];
                    z[zOff + j4 - 2] = z[zOff + ipn4 - j4 - 3];
                    z[zOff + ipn4 - j4 - 3] = tmp;
                    tmp = z[zOff + j4 - 1];
                    z[zOff + j4 - 1] = z[zOff + ipn4 - j4 - 6];
                    z[zOff + ipn4 - j4 - 6] = tmp;
                    tmp = z[zOff + j4];
                    z[zOff + j4] = z[zOff + ipn4 - j4 - 5];
                    z[zOff + ipn4 - j4 - 5] = tmp;
                }
                if (n0 - i0 <= 4) {
                    z[zOff + 4 * (n0 + 1) + pp - 2] = z[zOff + 4 * (i0 + 1) + pp - 2];
                    z[zOff + 4 * (n0 + 1) - pp - 1] = z[zOff + 4 * (i0 + 1) - pp - 1];
                }
                dn2 = min(dn2, z[zOff + 4 * (i0 + 1) - pp - 2]);
                z[zOff + 4 * (n0 + 1) + pp - 2] = min(min(z[zOff + 4 * (n0 + 1) + pp - 2],
                        z[zOff + 4 * (i0 + 1) + pp - 2]), z[zOff + 4 * (i0 + 1) + pp + 2]);
                z[zOff + 4 * (n0 + 1) - pp - 1] = min(min(z[zOff + 4 * (n0 + 1) - pp - 1],
                        z[zOff + 4 * (i0 + 1) - pp - 1]), z[zOff + 4 * (i0 + 1) - pp + 3]);
                qmax = max(max(qmax, z[zOff + 4 * (i0 + 1) + pp - 4]), z[zOff + 4 * (i0 + 1) + pp]);
                dmin = -0.0;
            }
        }

        double tau = Dlasq4.dlasq4(i0, n0, z, zOff, pp, n0in, dmin, dmin1, dmin2, dn, dn1, dn2, istate, iOff + 6, dstate, dOff + 5);

        boolean underflow = false;
        while (true) {
            Dlasq5.dlasq5(i0, n0, z, zOff, pp, tau, sigma, dstate, dOff + 10);
            dmin = dstate[dOff + 10];
            dmin1 = dstate[dOff + 11];
            dmin2 = dstate[dOff + 12];
            dn = dstate[dOff + 13];
            dn1 = dstate[dOff + 14];
            dn2 = dstate[dOff + 15];

            nDiv += n0 - i0 + 2;
            iter++;

            if (dmin >= 0 && dmin1 >= 0) {
                // Success.
                break;
            } else if (dmin < 0 && dmin1 > 0 &&
                    z[zOff + 4 * n0 - pp - 1] < tol * (sigma + dn1) &&
                    abs(dn2) < tol * sigma) {
                // Convergence hidden by negative dn.
                z[zOff + 4 * n0 - pp + 1] = 0;
                dmin = 0;
                break;
            } else if (dmin < 0) {
                // Tau too big. Select new tau and try again.
                nFail++;
                if (istate[iOff + 6] < -22) {
                    tau = 0;
                } else if (dmin1 > 0) {
                    tau = (tau + dmin) * (1 - 2 * eps);
                    istate[iOff + 6] -= 11;
                } else {
                    tau = tau / 4;
                    istate[iOff + 6] -= 12;
                }
                continue;
            } else if (Double.isNaN(dmin)) {
                if (tau == 0) {
                    break;
                }
                tau = 0;
                continue;
            } else {
                // Possible underflow. Play it safe.
                underflow = true;
                break;
            }
        }

        if (underflow) {
            // Risk of underflow: fall back to Dlasq6 (safe dqd without shifts).
            double[] lasq6Out = new double[6];
            Dlasq6.dlasq6(i0, n0, z, zOff, pp, lasq6Out);
            dmin  = lasq6Out[0];
            dmin1 = lasq6Out[1];
            dmin2 = lasq6Out[2];
            dn    = lasq6Out[3];
            dn1   = lasq6Out[4];
            dn2   = lasq6Out[5];
            nDiv += n0 - i0 + 2;
            iter++;
            tau = 0;
        }

        if (tau < sigma) {
            desig += tau;
            double t = sigma + desig;
            desig -= t - sigma;
            sigma = t;
        } else {
            double t = sigma + tau;
            desig += sigma - (t - tau);
            sigma = t;
        }

        istate[iOff] = i0;
        istate[iOff + 1] = n0;
        istate[iOff + 2] = pp;
        istate[iOff + 3] = iter;
        istate[iOff + 4] = nFail;
        istate[iOff + 5] = nDiv;

        dstate[dOff] = dmin1;
        dstate[dOff + 1] = dmin2;
        dstate[dOff + 2] = dn;
        dstate[dOff + 3] = dn1;
        dstate[dOff + 4] = dn2;
        dstate[dOff + 6] = sigma;
        dstate[dOff + 7] = desig;
        dstate[dOff + 8] = qmax;
        dstate[dOff + 9] = dmin; // write back dmin for next iteration in Dlasq2
    }
}
