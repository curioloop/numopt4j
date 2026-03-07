/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

interface Dlasq2 {

    double CBIAS = 1.5;

    /**
     * DLASQ2 computes all the eigenvalues of the symmetric positive definite
     * tridiagonal matrix associated with the qd array Z to high relative accuracy,
     * avoiding denormalization, underflow and overflow.
     *
     * <p>The tridiagonal matrix is L*U where L is a unit lower bidiagonal matrix
     * with sub-diagonals Z(2,4,6,...) and U is an upper bidiagonal matrix with
     * 1's above and diagonal Z(1,3,5,...).
     *
     * <p>Return codes:
     * <ul>
     *   <li>0 – algorithm completed successfully</li>
     *   <li>1 – a split was marked by a positive value in e</li>
     *   <li>2 – current block not diagonalized after 100*n iterations; Z holds
     *           a qd array with the same eigenvalues as the input</li>
     *   <li>3 – termination criterion of outer while loop not met (more than n
     *           unreduced blocks created)</li>
     * </ul>
     *
     * <p>LAPACK routine: DLASQ2
     *
     * @param n       order of the matrix (number of eigenvalues)
     * @param z       qd array of length at least 4*n; must not contain negative elements
     * @param zOff    offset into z
     * @param istate  integer state array (length ≥ iOff+7): [i0, n0, pp, iter, nFail, nDiv, ttype]
     * @param iOff    offset into istate
     * @param dstate  double state array (length ≥ dOff+16): [dmin1, dmin2, dn, dn1, dn2, g, sigma, desig, qmax, ...]
     * @param dOff    offset into dstate
     * @return info status code (0 = success)
     */
    static int dlasq2(int n, double[] z, int zOff, int[] istate, int iOff, double[] dstate, int dOff) {
        if (n == 0) {
            return 0;
        }

        double eps = Dlamch.dlamch('P');
        double safmin = Dlamch.dlamch('S');
        double tol = eps * 100;
        double tol2 = tol * tol;

        if (n == 1) {
            if (z[zOff] < 0) {
                return -1;
            }
            return 0;
        }

        if (n == 2) {
            if (z[zOff + 1] < 0 || z[zOff + 2] < 0) {
                return -1;
            }
            if (z[zOff + 2] > z[zOff]) {
                double tmp = z[zOff];
                z[zOff] = z[zOff + 2];
                z[zOff + 2] = tmp;
            }
            z[zOff + 4] = z[zOff] + z[zOff + 1] + z[zOff + 2];
            if (z[zOff + 1] > z[zOff + 2] * tol2) {
                double t = 0.5 * (z[zOff] - z[zOff + 2] + z[zOff + 1]);
                double s = z[zOff + 2] * (z[zOff + 1] / t);
                if (s <= t) {
                    s = z[zOff + 2] * (z[zOff + 1] / (t * (1 + sqrt(1 + s / t))));
                } else {
                    s = z[zOff + 2] * (z[zOff + 1] / (t + sqrt(t) * sqrt(t + s)));
                }
                t = z[zOff] + s + z[zOff + 1];
                z[zOff + 2] *= z[zOff] / t;
                z[zOff] = t;
            }
            z[zOff + 1] = z[zOff + 2];
            z[zOff + 5] = z[zOff + 1] + z[zOff];
            return 0;
        }

        z[zOff + 2 * n - 1] = 0;
        double emin = z[zOff + 1];
        double d = 0, e = 0, qmax = 0;
        for (int k = 0; k < 2 * (n - 1); k += 2) {
            if (z[zOff + k] < 0 || z[zOff + k + 1] < 0) {
                return -1;
            }
            d += z[zOff + k];
            e += z[zOff + k + 1];
            qmax = max(qmax, z[zOff + k]);
            emin = min(emin, z[zOff + k + 1]);
        }
        if (z[zOff + 2 * (n - 1)] < 0) {
            return -1;
        }
        d += z[zOff + 2 * (n - 1)];

        if (e == 0) {
            for (int k = 1; k < n; k++) {
                z[zOff + k] = z[zOff + 2 * k];
            }
            Dlasrt.dlasrt('D', n, z, zOff);
            z[zOff + 2 * (n - 1)] = d;
            return 0;
        }

        double trace = d + e;
        if (trace == 0) {
            z[zOff + 2 * (n - 1)] = 0;
            return 0;
        }

        for (int k = 2 * n; k >= 2; k -= 2) {
            z[zOff + 2 * k - 1] = 0;
            z[zOff + 2 * k - 2] = z[zOff + k - 1];
            z[zOff + 2 * k - 3] = 0;
            z[zOff + 2 * k - 4] = z[zOff + k - 2];
        }

        int i0 = 0;
        int n0 = n - 1;

        if (CBIAS * z[zOff + 4 * i0] < z[zOff + 4 * n0]) {
            int ipn4Out = 4 * (i0 + n0 + 2);
            for (int i4loop = 4 * (i0 + 1); i4loop <= 2 * (i0 + n0 + 1); i4loop += 4) {
                int i4 = i4loop - 1;
                int ipn4 = ipn4Out - 1;
                double tmp = z[zOff + i4 - 3];
                z[zOff + i4 - 3] = z[zOff + ipn4 - i4 - 4];
                z[zOff + ipn4 - i4 - 4] = tmp;
                tmp = z[zOff + i4 - 1];
                z[zOff + i4 - 1] = z[zOff + ipn4 - i4 - 6];
                z[zOff + ipn4 - i4 - 6] = tmp;
            }
        }

        int pp = 0;
        for (int k = 0; k < 2; k++) {
            d = z[zOff + 4 * n0 + pp];
            for (int i4loop = 4 * n0 + pp; i4loop >= 4 * (i0 + 1) + pp; i4loop -= 4) {
                int i4 = i4loop - 1;
                if (z[zOff + i4 - 1] <= tol2 * d) {
                    z[zOff + i4 - 1] = -0.0;
                    d = z[zOff + i4 - 3];
                } else {
                    d = z[zOff + i4 - 3] * (d / (d + z[zOff + i4 - 1]));
                }
            }

            emin = z[zOff + 4 * (i0 + 1) + pp];
            d = z[zOff + 4 * i0 + pp];
            for (int i4loop = 4 * (i0 + 1) + pp; i4loop <= 4 * n0 + pp; i4loop += 4) {
                int i4 = i4loop - 1;
                z[zOff + i4 - 2 * pp - 2] = d + z[zOff + i4 - 1];
                if (z[zOff + i4 - 1] <= tol2 * d) {
                    z[zOff + i4 - 1] = -0.0;
                    z[zOff + i4 - 2 * pp - 2] = d;
                    z[zOff + i4 - 2 * pp] = 0;
                    d = z[zOff + i4 + 1];
                } else if (safmin * z[zOff + i4 + 1] < z[zOff + i4 - 2 * pp - 2] &&
                        safmin * z[zOff + i4 - 2 * pp - 2] < z[zOff + i4 + 1]) {
                    double tmp = z[zOff + i4 + 1] / z[zOff + i4 - 2 * pp - 2];
                    z[zOff + i4 - 2 * pp] = z[zOff + i4 - 1] * tmp;
                    d *= tmp;
                } else {
                    z[zOff + i4 - 2 * pp] = z[zOff + i4 + 1] * (z[zOff + i4 - 1] / z[zOff + i4 - 2 * pp - 2]);
                    d = z[zOff + i4 + 1] * (d / z[zOff + i4 - 2 * pp - 2]);
                }
                emin = min(emin, z[zOff + i4 - 2 * pp]);
            }
            z[zOff + 4 * (n0 + 1) - pp - 3] = d;

            qmax = z[zOff + 4 * (i0 + 1) - pp - 3];
            for (int i4loop = 4 * (i0 + 1) - pp + 2; i4loop <= 4 * (n0 + 1) + pp - 2; i4loop += 4) {
                int i4 = i4loop - 1;
                qmax = max(qmax, z[zOff + i4]);
            }
            pp = 1 - pp;
        }

        istate[iOff] = i0;
        istate[iOff + 1] = n0;
        istate[iOff + 2] = pp;
        istate[iOff + 3] = 2;
        istate[iOff + 4] = 0;
        istate[iOff + 5] = 2 * (n0 - i0);
        istate[iOff + 6] = 0;

        dstate[dOff] = 0;
        dstate[dOff + 1] = 0;
        dstate[dOff + 2] = 0;
        dstate[dOff + 3] = 0;
        dstate[dOff + 4] = 0;
        dstate[dOff + 5] = 0;
        dstate[dOff + 6] = 0;
        dstate[dOff + 7] = 0;
        dstate[dOff + 8] = qmax;

        int i4 = 0;

        outer:
        for (int iwhila = 1; iwhila <= n + 1; iwhila++) {
            if (istate[iOff + 1] < 0) {
                for (int k = 1; k < n; k++) {
                    z[zOff + k] = z[zOff + 4 * k];
                }
                Dlasrt.dlasrt('D', n, z, zOff);
                e = 0;
                for (int k = n - 1; k >= 0; k--) {
                    e += z[zOff + k];
                }
                z[zOff + 2 * n] = trace;
                z[zOff + 2 * n + 1] = e;
                z[zOff + 2 * n + 2] = istate[iOff + 3];
                z[zOff + 2 * n + 3] = istate[iOff + 5] / (n * n);
                z[zOff + 2 * n + 4] = 100.0 * istate[iOff + 4] / istate[iOff + 3];
                return 0;
            }

            double sigma = 0;
            if (istate[iOff + 1] != n - 1) {
                sigma = -z[zOff + 4 * (istate[iOff + 1] + 1) - 2];
            }
            if (sigma < 0) {
                return 1;
            }

            double emax = 0;
            double eminTmp;
            if (istate[iOff + 1] > istate[iOff]) {
                eminTmp = abs(z[zOff + 4 * (istate[iOff + 1] + 1) - 6]);
            } else {
                eminTmp = 0;
            }
            double qmin = z[zOff + 4 * (istate[iOff + 1] + 1) - 4];
            qmax = qmin;
            boolean zSmall = false;
            for (int i4loop = 4 * (istate[iOff + 1] + 1); i4loop >= 8; i4loop -= 4) {
                i4 = i4loop - 1;
                if (z[zOff + i4 - 5] <= 0) {
                    zSmall = true;
                    break;
                }
                if (qmin >= 4 * emax) {
                    qmin = min(qmin, z[zOff + i4 - 3]);
                    emax = max(emax, z[zOff + i4 - 5]);
                }
                qmax = max(qmax, z[zOff + i4 - 7] + z[zOff + i4 - 5]);
                eminTmp = min(eminTmp, z[zOff + i4 - 5]);
            }
            if (!zSmall) {
                i4 = 3;
            }
            istate[iOff] = (i4 + 1) / 4 - 1;
            istate[iOff + 2] = 0;

            if (istate[iOff + 1] - istate[iOff] > 1) {
                double dee = z[zOff + 4 * istate[iOff]];
                double deemin = dee;
                int kmin = istate[iOff];
                for (int i4loop = 4 * (istate[iOff] + 1) + 1; i4loop <= 4 * (istate[iOff + 1] + 1) - 3; i4loop += 4) {
                    int i4tmp = i4loop - 1;
                    dee = z[zOff + i4tmp] * (dee / (dee + z[zOff + i4tmp - 2]));
                    if (dee <= deemin) {
                        deemin = dee;
                        kmin = (i4tmp + 4) / 4 - 1;
                    }
                }
                if ((kmin - istate[iOff]) * 2 < istate[iOff + 1] - kmin && deemin <= 0.5 * z[zOff + 4 * istate[iOff + 1]]) {
                    int ipn4Out = 4 * (istate[iOff] + istate[iOff + 1] + 2);
                    istate[iOff + 2] = 2;
                    for (int j4loop = 4 * (istate[iOff] + 1); j4loop <= 2 * (istate[iOff] + istate[iOff + 1] + 1); j4loop += 4) {
                        int j4 = j4loop - 1;
                        int ipn4 = ipn4Out - 1;
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
                }
            }

            // Put -(initial shift) into dmin.
            // PP = 0 for ping, PP = 1 for pong.
            // PP = 2 indicates that flipping was applied to the Z array and
            //      that the tests for deflation upon entry in Dlasq3 should
            //      not be performed.
            // dmin is passed to Dlasq3 via dstate[dOff+9] and updated each iteration.
            double dmin = -max(0, qmin - 2 * sqrt(qmin) * sqrt(emax));
            dstate[dOff + 9] = dmin;

            int nbig = 100 * (istate[iOff + 1] - istate[iOff] + 1);
            for (int iwhilb = 0; iwhilb < nbig; iwhilb++) {
                if (istate[iOff] > istate[iOff + 1]) {
                    continue outer;
                }

                dstate[dOff + 6] = sigma;
                Dlasq3.dlasq3(istate, iOff, dstate, dOff, z, zOff);
                sigma = dstate[dOff + 6];

                istate[iOff + 2] = 1 - istate[iOff + 2];

                if (istate[iOff + 2] == 0 && istate[iOff + 1] - istate[iOff] >= 3) {
                    if (z[zOff + 4 * (istate[iOff + 1] + 1) - 1] <= tol2 * qmax ||
                            z[zOff + 4 * (istate[iOff + 1] + 1) - 2] <= tol2 * sigma) {
                        int splt = istate[iOff] - 1;
                        qmax = z[zOff + 4 * istate[iOff]];
                        eminTmp = z[zOff + 4 * (istate[iOff] + 1) - 2];
                        double oldemn = z[zOff + 4 * (istate[iOff] + 1) - 1];
                        for (int i4loop = 4 * (istate[iOff] + 1); i4loop <= 4 * (istate[iOff + 1] - 2); i4loop += 4) {
                            int i4tmp = i4loop - 1;
                            if (z[zOff + i4tmp] <= tol2 * z[zOff + i4tmp - 3] || z[zOff + i4tmp - 1] <= tol2 * sigma) {
                                z[zOff + i4tmp - 1] = -sigma;
                                splt = i4tmp / 4;
                                qmax = 0;
                                eminTmp = z[zOff + i4tmp + 3];
                                oldemn = z[zOff + i4tmp + 4];
                            } else {
                                qmax = max(qmax, z[zOff + i4tmp + 1]);
                                eminTmp = min(eminTmp, z[zOff + i4tmp - 1]);
                                oldemn = min(oldemn, z[zOff + i4tmp]);
                            }
                        }
                        z[zOff + 4 * (istate[iOff + 1] + 1) - 2] = eminTmp;
                        z[zOff + 4 * (istate[iOff + 1] + 1) - 1] = oldemn;
                        istate[iOff] = splt + 1;
                    }
                }
            }

            // Maximum number of iterations exceeded, restore the shift sigma
            // and place the new d's and e's in a qd array.
            // This might need to be done for several blocks.
            int i0cur = istate[iOff];
            int n0cur = istate[iOff + 1];
            int i1 = i0cur;
            int n1;
            while (true) {
                double tempq = z[zOff + 4 * i0cur];
                z[zOff + 4 * i0cur] += sigma;
                for (int k = i0cur + 1; k <= n0cur; k++) {
                    double tempe = z[zOff + 4 * (k + 1) - 6];
                    z[zOff + 4 * (k + 1) - 6] *= tempq / z[zOff + 4 * (k + 1) - 8];
                    tempq = z[zOff + 4 * k];
                    z[zOff + 4 * k] += sigma + tempe - z[zOff + 4 * (k + 1) - 6];
                }
                // Prepare to do this on the previous block if there is one.
                if (i1 <= 0) {
                    break;
                }
                n1 = i1 - 1;
                while (i1 >= 1 && z[zOff + 4 * (i1 + 1) - 6] >= 0) {
                    i1--;
                }
                sigma = -z[zOff + 4 * (n1 + 1) - 2];
            }
            for (int k = 0; k < n; k++) {
                z[zOff + 2 * k] = z[zOff + 4 * k];
                // Only the block i0..n0 is unfinished. The rest of the e's
                // must be essentially zero, although sometimes other data
                // has been stored in them.
                if (k < n0cur) {
                    z[zOff + 2 * (k + 1) - 1] = z[zOff + 4 * (k + 1) - 1];
                } else {
                    z[zOff + 2 * (k + 1)] = 0;
                }
            }
            return 2;
        }

        return 3;
    }
}