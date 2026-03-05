/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

interface Dlanv2 {

    static final double DLAMCH_P = Dlamch.dlamch('P');
    static final double DLAMCH_S = Dlamch.dlamch('S');
    static final double DLAMCH_E = Dlamch.dlamch('E');
    static final double DLAMCH_B = Dlamch.dlamch('B');

    static void dlanv2(double a, double b, double c, double d, double[] out, int outOff) {
        double aa, bb, cc, dd;
        double rt1r, rt1i, rt2r, rt2i;
        double cs, sn;

        if (c == 0) {
            aa = a;
            bb = b;
            cc = 0;
            dd = d;
            cs = 1;
            sn = 0;
        } else if (b == 0) {
            aa = d;
            bb = -c;
            cc = 0;
            dd = a;
            cs = 0;
            sn = 1;
        } else if (a == d && Math.signum(b) != Math.signum(c)) {
            aa = a;
            bb = b;
            cc = c;
            dd = d;
            cs = 1;
            sn = 0;
        } else {
            double temp = a - d;
            double p = temp / 2;
            double bcmax = Math.max(Math.abs(b), Math.abs(c));
            double bcmis = Math.min(Math.abs(b), Math.abs(c));
            if (b * c < 0) {
                bcmis = -bcmis;
            }
            double scale = Math.max(Math.abs(p), bcmax);
            double z = p / scale * p + bcmax / scale * bcmis;

            if (z >= 4 * DLAMCH_P) {
                if (p > 0) {
                    z = p + Math.sqrt(scale) * Math.sqrt(z);
                } else {
                    z = p - Math.sqrt(scale) * Math.sqrt(z);
                }
                aa = d + z;
                dd = d - bcmax / z * bcmis;
                double tau = Math.hypot(c, z);
                cs = z / tau;
                sn = c / tau;
                bb = b - c;
                cc = 0;
            } else {
                double safmn2 = Math.pow(DLAMCH_B, Math.log(DLAMCH_S / DLAMCH_E) / Math.log(DLAMCH_B) / 2);
                double safmx2 = 1 / safmn2;
                double sigma = b + c;

                for (int iter = 0; iter < 20; iter++) {
                    scale = Math.max(Math.abs(temp), Math.abs(sigma));
                    if (scale >= safmx2) {
                        sigma *= safmn2;
                        temp *= safmn2;
                    } else if (scale <= safmn2) {
                        sigma *= safmx2;
                        temp *= safmx2;
                    } else {
                        break;
                    }
                }

                p = temp / 2;
                double tau = Math.hypot(sigma, temp);
                cs = Math.sqrt((1 + Math.abs(sigma) / tau) / 2);
                sn = -p / (tau * cs);
                if (sigma < 0) {
                    sn = -sn;
                }

                aa = a * cs + b * sn;
                bb = -a * sn + b * cs;
                cc = c * cs + d * sn;
                dd = -c * sn + d * cs;

                a = aa * cs + cc * sn;
                b = bb * cs + dd * sn;
                c = -aa * sn + cc * cs;
                d = -bb * sn + dd * cs;

                temp = (a + d) / 2;
                aa = temp;
                bb = b;
                cc = c;
                dd = temp;

                if (cc != 0) {
                    if (bb != 0) {
                        if (Math.signum(bb) == Math.signum(cc)) {
                            double sab = Math.sqrt(Math.abs(bb));
                            double sac = Math.sqrt(Math.abs(cc));
                            p = sab * sac;
                            if (cc < 0) {
                                p = -p;
                            }
                            tau = 1 / Math.sqrt(Math.abs(bb + cc));
                            aa = temp + p;
                            bb = bb - cc;
                            cc = 0;
                            dd = temp - p;
                            double cs1 = sab * tau;
                            double sn1 = sac * tau;
                            double csOld = cs;
                            cs = csOld * cs1 - sn * sn1;
                            sn = csOld * sn1 + sn * cs1;
                        }
                    } else {
                        bb = -cc;
                        cc = 0;
                        double csOld = cs;
                        cs = -sn;
                        sn = csOld;
                    }
                }
            }
        }

        rt1r = aa;
        rt2r = dd;
        if (cc != 0) {
            rt1i = Math.sqrt(Math.abs(bb)) * Math.sqrt(Math.abs(cc));
            rt2i = -rt1i;
        } else {
            rt1i = 0;
            rt2i = 0;
        }

        out[outOff] = aa;
        out[outOff + 1] = bb;
        out[outOff + 2] = cc;
        out[outOff + 3] = dd;
        out[outOff + 4] = rt1r;
        out[outOff + 5] = rt1i;
        out[outOff + 6] = rt2r;
        out[outOff + 7] = rt2i;
        out[outOff + 8] = cs;
        out[outOff + 9] = sn;
    }
}
