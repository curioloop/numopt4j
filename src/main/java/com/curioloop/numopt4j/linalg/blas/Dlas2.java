/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * DLAS2: Computes the singular values of the 2×2 matrix [F G; 0 H].
 * Returns ssmin (smaller) and ssmax (larger) singular values.
 * Also provides DLASV2 which additionally computes the singular vectors.
 *
 */
interface Dlas2 {

    static void dlas2(double f, double g, double h, double[] out, int off) {
        double fa = abs(f);
        double ga = abs(g);
        double ha = abs(h);
        double fhmin = min(fa, ha);
        double fhmax = max(fa, ha);

        if (fhmin == 0) {
            if (fhmax == 0) {
                out[off] = 0;
                out[off + 1] = ga;
            } else {
                double v = min(fhmax, ga) / max(fhmax, ga);
                out[off] = 0;
                out[off + 1] = max(fhmax, ga) * sqrt(1 + v * v);
            }
            return;
        }

        if (ga < fhmax) {
            double as = 1 + fhmin / fhmax;
            double at = (fhmax - fhmin) / fhmax;
            double au = (ga / fhmax) * (ga / fhmax);
            double c = 2 / (sqrt(as * as + au) + sqrt(at * at + au));
            out[off] = fhmin * c;
            out[off + 1] = fhmax / c;
        } else {
            double au = fhmax / ga;
            if (au == 0) {
                out[off] = fhmin * fhmax / ga;
                out[off + 1] = ga;
            } else {
                double as = 1 + fhmin / fhmax;
                double at = (fhmax - fhmin) / fhmax;
                double c = 1 / (sqrt(1 + (as * au) * (as * au)) + sqrt(1 + (at * au) * (at * au)));
                out[off] = 2 * (fhmin * c) * au;
                out[off + 1] = ga / (c + c);
            }
        }
    }

    static void dlasv2(double f, double g, double h, double[] out, int off) {
        double ft = f;
        double fa = abs(ft);
        double ht = h;
        double ha = abs(ht);
        int pmax = 1;
        boolean swap = ha > fa;
        if (swap) {
            pmax = 3;
            double tmp = ft;
            ft = ht;
            ht = tmp;
            tmp = fa;
            fa = ha;
            ha = tmp;
        }
        double gt = g;
        double ga = abs(gt);
        double clt = 0, crt = 0, slt = 0, srt = 0;
        double ssmin = 0, ssmax = 0;

        if (ga == 0) {
            ssmin = ha;
            ssmax = fa;
            clt = 1;
            crt = 1;
            slt = 0;
            srt = 0;
        } else {
            boolean gasmall = true;
            if (ga > fa) {
                pmax = 2;
                if ((fa / ga) < Dlamch.EPSILON) {
                    gasmall = false;
                    ssmax = ga;
                    if (ha > 1) {
                        ssmin = fa / (ga / ha);
                    } else {
                        ssmin = (fa / ga) * ha;
                    }
                    clt = 1;
                    slt = ht / gt;
                    srt = 1;
                    crt = ft / gt;
                }
            }
            if (gasmall) {
                double d = fa - ha;
                double l = d / fa;
                if (d == fa) {
                    l = 1;
                }
                double m = gt / ft;
                double t = 2 - l;
                double s = hypot(t, m);
                double r;
                if (l == 0) {
                    r = abs(m);
                } else {
                    r = hypot(l, m);
                }
                double a = 0.5 * (s + r);
                ssmin = ha / a;
                ssmax = fa * a;
                if (m == 0) {
                    if (l == 0) {
                        t = copySign(2, ft) * copySign(1, gt);
                    } else {
                        t = gt / copySign(d, ft) + m / t;
                    }
                } else {
                    t = (m / (s + t) + m / (r + l)) * (1 + a);
                }
                l = hypot(t, 2);
                crt = 2 / l;
                srt = t / l;
                clt = (crt + srt * m) / a;
                slt = (ht / ft) * srt / a;
            }
        }
        double csl, snl, csr, snr;
        if (swap) {
            csl = srt;
            snl = crt;
            csr = slt;
            snr = clt;
        } else {
            csl = clt;
            snl = slt;
            csr = crt;
            snr = srt;
        }
        double tsign;
        switch (pmax) {
            case 1:
                tsign = copySign(1, csr) * copySign(1, csl) * copySign(1, f);
                break;
            case 2:
                tsign = copySign(1, snr) * copySign(1, csl) * copySign(1, g);
                break;
            case 3:
                tsign = copySign(1, snr) * copySign(1, snl) * copySign(1, h);
                break;
            default:
                tsign = 1;
        }
        ssmax = copySign(ssmax, tsign);
        ssmin = copySign(ssmin, tsign * copySign(1, f) * copySign(1, h));

        out[off] = ssmin;
        out[off + 1] = ssmax;
        out[off + 2] = snr;
        out[off + 3] = csr;
        out[off + 4] = snl;
        out[off + 5] = csl;
    }

}
