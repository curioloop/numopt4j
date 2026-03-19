/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.abs;
import static java.lang.Math.max;

/**
 * DLAEXC: Swaps two adjacent diagonal blocks of order 1 or 2 in an n×n upper
 * quasi-triangular matrix T by an orthogonal similarity transformation.
 * T must be in Schur canonical form. Returns false if the swap would make T
 * too far from Schur form.
 *
 */
interface Dlaexc {

    static final double DLAMCH_P = Dlamch.dlamch('P');
    static final double DLAMCH_S = Dlamch.dlamch('S');

    static boolean dlaexc(boolean wantq, int n,
                          double[] t, int tOff, int ldt,
                          double[] q, int qOff, int ldq,
                          int j1, int n1, int n2, double[] work, int wOff) {
        if (n == 0 || n1 == 0 || n2 == 0) {
            return true;
        }

        if (j1 + n1 >= n) {
            return true;
        }

        int j2 = j1 + 1;
        int j3 = j1 + 2;

        int dOff = wOff;
        int xOff = wOff + 16;
        int uOff = wOff + 22;
        int tauOff = wOff + 25;
        int givensOff = wOff + 26;
        int larfxOff = wOff + 29;

        if (n1 == 1 && n2 == 1) {
            Dlartg.dlartg(t[tOff + j1 * ldt + j2], t[tOff + j2 * ldt + j2] - t[tOff + j1 * ldt + j1], work, givensOff);
            double cs = work[givensOff];
            double sn = work[givensOff + 1];

            if (n - j3 > 0) {
                Drot.drot(n - j3, t, tOff + j1 * ldt + j3, 1, t, tOff + j2 * ldt + j3, 1, cs, sn);
            }
            if (j1 > 0) {
                Drot.drot(j1, t, tOff + j1, ldt, t, tOff + j2, ldt, cs, sn);
            }

            double t11 = t[tOff + j1 * ldt + j1];
            double t22 = t[tOff + j2 * ldt + j2];
            t[tOff + j1 * ldt + j1] = t22;
            t[tOff + j2 * ldt + j2] = t11;

            if (wantq) {
                Drot.drot(n, q, qOff + j1, ldq, q, qOff + j2, ldq, cs, sn);
            }

            return true;
        }

        int nd = n1 + n2;
        final int ldd = 4;

        Dlamv.dlacpy('A', nd, nd, t, tOff + j1 * ldt + j1, ldt, work, dOff, ldd);
        double dnorm = Dlange.dlange('M', nd, nd, work, dOff, ldd);

        double eps = DLAMCH_P;
        double thresh = max(10 * eps * dnorm, DLAMCH_S / eps);

        final int ldx = 2;

        Dlasy2.dlasy2(false, false, -1, n1, n2,
                work, dOff, ldd, work, dOff + n1 * ldd + n1, ldd, work, dOff + n1, ldd, work, xOff, ldx);
        double scale = work[xOff + 4];

        if (n1 == 1 && n2 == 2) {
            work[uOff] = scale;
            work[uOff + 1] = work[xOff];
            work[uOff + 2] = 1;

            Dlarfg.dlarfg(3, work[xOff + 1], work, uOff, 1, work, tauOff);

            double t11 = t[tOff + j1 * ldt + j1];

            Dlarfx.dlarfx(true, 3, 3, work, uOff, work[tauOff], work, dOff, ldd, work, larfxOff);
            Dlarfx.dlarfx(false, 3, 3, work, uOff, work[tauOff], work, dOff, ldd, work, larfxOff);

            if (max(abs(work[dOff + 2 * ldd]), max(abs(work[dOff + 2 * ldd + 1]), abs(work[dOff + 2 * ldd + 2] - t11))) > thresh) {
                return false;
            }

            Dlarfx.dlarfx(true, 3, n - j1, work, uOff, work[tauOff], t, tOff + j1 * ldt + j1, ldt, work, larfxOff);
            Dlarfx.dlarfx(false, j2 + 1, 3, work, uOff, work[tauOff], t, tOff + j1, ldt, work, larfxOff);

            t[tOff + j3 * ldt + j1] = 0;
            t[tOff + j3 * ldt + j2] = 0;
            t[tOff + j3 * ldt + j3] = t11;

            if (wantq) {
                Dlarfx.dlarfx(false, n, 3, work, uOff, work[tauOff], q, qOff + j1, ldq, work, larfxOff);
            }

        } else if (n1 == 2 && n2 == 1) {
            work[uOff] = 1;
            work[uOff + 1] = -work[xOff + ldx];
            work[uOff + 2] = scale;

            Dlarfg.dlarfg(3, -work[xOff], work, uOff + 1, 1, work, tauOff);

            double t33 = t[tOff + j3 * ldt + j3];

            Dlarfx.dlarfx(true, 3, 3, work, uOff, work[tauOff], work, dOff, ldd, work, larfxOff);
            Dlarfx.dlarfx(false, 3, 3, work, uOff, work[tauOff], work, dOff, ldd, work, larfxOff);

            if (max(abs(work[dOff + ldd]), max(abs(work[dOff + 2 * ldd]), abs(work[dOff] - t33))) > thresh) {
                return false;
            }

            Dlarfx.dlarfx(false, j3 + 1, 3, work, uOff, work[tauOff], t, tOff + j1, ldt, work, larfxOff);
            Dlarfx.dlarfx(true, 3, n - j1 - 1, work, uOff, work[tauOff], t, tOff + j1 * ldt + j2, ldt, work, larfxOff);

            t[tOff + j1 * ldt + j1] = t33;
            t[tOff + j2 * ldt + j1] = 0;
            t[tOff + j3 * ldt + j1] = 0;

            if (wantq) {
                Dlarfx.dlarfx(false, n, 3, work, uOff, work[tauOff], q, qOff + j1, ldq, work, larfxOff);
            }

        } else {
            int u1Off = uOff;
            int tau1Off = tauOff;
            int u2Off = uOff + 3;
            int tau2Off = tauOff + 1;

            work[u1Off] = 1;
            work[u1Off + 1] = -work[xOff + ldx];
            work[u1Off + 2] = scale;

            Dlarfg.dlarfg(3, -work[xOff], work, u1Off + 1, 1, work, tau1Off);

            double temp = -work[tau1Off] * (work[xOff + 1] + work[u1Off + 1] * work[xOff + ldx + 1]);

            work[u2Off] = 1;
            work[u2Off + 1] = -temp * work[u1Off + 2];
            work[u2Off + 2] = scale;

            Dlarfg.dlarfg(3, -temp * work[u1Off + 1] - work[xOff + ldx + 1], work, u2Off + 1, 1, work, tau2Off);

            Dlarfx.dlarfx(true, 3, 4, work, u1Off, work[tau1Off], work, dOff, ldd, work, larfxOff);
            Dlarfx.dlarfx(false, 4, 3, work, u1Off, work[tau1Off], work, dOff, ldd, work, larfxOff);
            Dlarfx.dlarfx(true, 3, 4, work, u2Off, work[tau2Off], work, dOff + ldd, ldd, work, larfxOff);
            Dlarfx.dlarfx(false, 4, 3, work, u2Off, work[tau2Off], work, dOff + 1, ldd, work, larfxOff);

            double m1 = max(abs(work[dOff + 2 * ldd]), abs(work[dOff + 2 * ldd + 1]));
            double m2 = max(abs(work[dOff + 3 * ldd]), abs(work[dOff + 3 * ldd + 1]));
            if (max(m1, m2) > thresh) {
                return false;
            }

            int j4 = j1 + 3;
            Dlarfx.dlarfx(true, 3, n - j1, work, u1Off, work[tau1Off], t, tOff + j1 * ldt + j1, ldt, work, larfxOff);
            Dlarfx.dlarfx(false, j4 + 1, 3, work, u1Off, work[tau1Off], t, tOff + j1, ldt, work, larfxOff);
            Dlarfx.dlarfx(true, 3, n - j1, work, u2Off, work[tau2Off], t, tOff + j2 * ldt + j1, ldt, work, larfxOff);
            Dlarfx.dlarfx(false, j4 + 1, 3, work, u2Off, work[tau2Off], t, tOff + j2, ldt, work, larfxOff);

            t[tOff + j3 * ldt + j1] = 0;
            t[tOff + j3 * ldt + j2] = 0;
            t[tOff + j4 * ldt + j1] = 0;
            t[tOff + j4 * ldt + j2] = 0;

            if (wantq) {
                Dlarfx.dlarfx(false, n, 3, work, u1Off, work[tau1Off], q, qOff + j1, ldq, work, larfxOff);
                Dlarfx.dlarfx(false, n, 3, work, u2Off, work[tau2Off], q, qOff + j2, ldq, work, larfxOff);
            }
        }

        if (n2 == 2) {
            double a = t[tOff + j1 * ldt + j1];
            double b = t[tOff + j1 * ldt + j2];
            double c = t[tOff + j2 * ldt + j1];
            double dval = t[tOff + j2 * ldt + j2];

            Dlanv2.dlanv2(a, b, c, dval, work, dOff);
            t[tOff + j1 * ldt + j1] = work[dOff];
            t[tOff + j1 * ldt + j2] = work[dOff + 1];
            t[tOff + j2 * ldt + j1] = work[dOff + 2];
            t[tOff + j2 * ldt + j2] = work[dOff + 3];
            double cs = work[dOff + 8];
            double sn = work[dOff + 9];

            if (n - j1 - 2 > 0) {
                Drot.drot(n - j1 - 2, t, tOff + j1 * ldt + j1 + 2, 1, t, tOff + j2 * ldt + j1 + 2, 1, cs, sn);
            }
            if (j1 > 0) {
                Drot.drot(j1, t, tOff + j1, ldt, t, tOff + j2, ldt, cs, sn);
            }
            if (wantq) {
                Drot.drot(n, q, qOff + j1, ldq, q, qOff + j2, ldq, cs, sn);
            }
        }

        if (n1 == 2) {
            int j3b = j1 + n2;
            int j4 = j3b + 1;
            double a = t[tOff + j3b * ldt + j3b];
            double b = t[tOff + j3b * ldt + j4];
            double c = t[tOff + j4 * ldt + j3b];
            double dval = t[tOff + j4 * ldt + j4];

            Dlanv2.dlanv2(a, b, c, dval, work, dOff);
            t[tOff + j3b * ldt + j3b] = work[dOff];
            t[tOff + j3b * ldt + j4] = work[dOff + 1];
            t[tOff + j4 * ldt + j3b] = work[dOff + 2];
            t[tOff + j4 * ldt + j4] = work[dOff + 3];
            double cs = work[dOff + 8];
            double sn = work[dOff + 9];

            if (n - j3b - 2 > 0) {
                Drot.drot(n - j3b - 2, t, tOff + j3b * ldt + j3b + 2, 1, t, tOff + j4 * ldt + j3b + 2, 1, cs, sn);
            }
            Drot.drot(j3b, t, tOff + j3b, ldt, t, tOff + j4, ldt, cs, sn);
            if (wantq) {
                Drot.drot(n, q, qOff + j3b, ldq, q, qOff + j4, ldq, cs, sn);
            }
        }

        return true;
    }
}
