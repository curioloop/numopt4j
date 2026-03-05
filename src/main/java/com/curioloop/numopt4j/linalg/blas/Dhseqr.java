/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

interface Dhseqr {

    static final double DLAMCH_P = Dlamch.dlamch('P');
    static final double DLAMCH_S = Dlamch.dlamch('S');
    static final double DLAMCH_E = Dlamch.dlamch('E');

    static int dhseqrQuery(int n, int ilo, int ihi) {
        if (n == 0) {
            return 10;
        }
        // Do a proper workspace query through dlaqr04 so that dlaqr23's extra scratch is included.
        double[] tmp = new double[1];
        boolean wt = true, wz = true;
        Dlaqr.dlaqr04(wt, wz, n, ilo, ihi, null, 0, Math.max(1, n),
                null, 0, null, 0, ilo, ihi, null, 0, 1, tmp, 0, -1, 1);
        return Math.max(10, (int) tmp[0]);
    }

    static int dhseqr(char job, char compz, int n, int ilo, int ihi,
                      double[] H, int ldh, double[] wr, double[] wi,
                      double[] z, int ldz, double[] work, int workOff, int lwork) {

        if (n == 0) {
            work[workOff] = 1;  // Go returns work[0]=1 for n==0
            return 0;
        }

        if (lwork == -1) {
            boolean wt = Character.toUpperCase(job) == 'S';
            boolean wz = Character.toUpperCase(compz) == 'V' || Character.toUpperCase(compz) == 'I';
            Dlaqr.dlaqr04(wt, wz, n, ilo, ihi, null, 0, Math.max(1, n),
                    null, 0, null, 0, ilo, ihi, null, 0, 1, work, workOff, -1, 1);
            work[workOff] = Math.max(Math.max(10, n), work[workOff]);
            return 0;
        }

        boolean wantt = Character.toUpperCase(job) == 'S';
        boolean wantz = Character.toUpperCase(compz) == 'V' || Character.toUpperCase(compz) == 'I';

        for (int i = 0; i < ilo; i++) {
            wr[i] = H[i * ldh + i];
            wi[i] = 0;
        }
        for (int i = ihi + 1; i < n; i++) {
            wr[i] = H[i * ldh + i];
            wi[i] = 0;
        }

        if (Character.toUpperCase(compz) == 'I') {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    z[i * ldz + j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }

        if (ilo == ihi) {
            wr[ilo] = H[ilo * ldh + ilo];
            wi[ilo] = 0;
            work[workOff] = Math.max(10, n);
            return 0;
        }

        // Dlahqr/Dlaqr04 crossover point (matches Go: nmin from ilaenv(12), default 75)
        final int ntiny = 15;
        final int nl = 49;
        int nmin = Ilaenv.ilaenv(12, "DHSEQR", "" + job + compz, n, ilo, ihi, lwork);
        nmin = Math.max(ntiny, nmin);

        int unconverged;
        if (n > nmin) {
            // Large matrices: use Dlaqr04 (multi-shift QR, Level 3 BLAS)
            unconverged = Dlaqr.dlaqr04(wantt, wantz, n, ilo, ihi,
                    H, 0, ldh, wr, 0, wi, 0, ilo, ihi, z, 0, ldz,
                    work, workOff, lwork, 1);
        } else {
            // Small matrices: use Dlahqr (single-shift QR)
            unconverged = dlahqr(wantt, wantz, n, ilo, ihi, H, 0, ldh, wr, 0, wi, 0, ilo, ihi, z, 0, ldz, work, workOff);
            if (unconverged > 0) {
                // A rare Dlahqr failure! Dlaqr04 sometimes succeeds when Dlahqr fails.
                int kbot = unconverged;
                if (n >= nl) {
                    // Larger matrices have enough subdiagonal scratch space to call Dlaqr04 directly.
                    unconverged = Dlaqr.dlaqr04(wantt, wantz, n, ilo, kbot,
                            H, 0, ldh, wr, 0, wi, 0, ilo, ihi, z, 0, ldz,
                            work, workOff, lwork, 1);
                } else {
                    // Tiny matrices: copy into a larger nl×nl array before calling Dlaqr04.
                    double[] hl = new double[nl * nl];
                    Dlamv.dlacpy('A', n, n, H, 0, ldh, hl, 0, nl);
                    double[] workl = new double[nl];
                    unconverged = Dlaqr.dlaqr04(wantt, wantz, nl, ilo, kbot,
                            hl, 0, nl, wr, 0, wi, 0, ilo, ihi, z, 0, ldz,
                            workl, 0, nl, 1);
                    work[workOff] = workl[0];
                    if (wantt || unconverged > 0) {
                        Dlamv.dlacpy('A', n, n, hl, 0, nl, H, 0, ldh);
                    }
                }
            }
        }

        if ((wantt || unconverged > 0) && n > 2) {
            for (int i = 2; i < n; i++) {
                for (int j = 0; j < i - 1; j++) {
                    H[i * ldh + j] = 0;
                }
            }
        }

        work[workOff] = Math.max(10, n);
        return unconverged;
    }

    static int dlahqr(boolean wantt, boolean wantz, int n, int ilo, int ihi,
                      double[] H, int hOff, int ldh, double[] wr, int wrOff, double[] wi, int wiOff,
                      int iloz, int ihiz, double[] z, int zOff, int ldz,
                      double[] work, int workOff) {

        if (n == 0) {
            return 0;
        }

        for (int j = ilo; j < ihi - 2; j++) {
            H[hOff + (j + 2) * ldh + j] = 0;
            H[hOff + (j + 3) * ldh + j] = 0;
        }
        if (ilo <= ihi - 2) {
            H[hOff + ihi * ldh + ihi - 2] = 0;
        }

        int nh = ihi - ilo + 1;
        int nz = ihiz - iloz + 1;

        double ulp = DLAMCH_P;
        double smlnum = (double) nh / ulp * DLAMCH_S;

        int i1 = 0, i2 = 0;
        if (wantt) {
            i1 = 0;
            i2 = n - 1;
        }

        int itmax = 30 * Math.max(10, nh);

        int kdefl = 0;

        int vOff = workOff;
        int dlarfgOff = workOff + 3;

        int i = ihi;
        while (i >= ilo) {
            int l = ilo;

            boolean converged = false;
            for (int its = 0; its <= itmax; its++) {
                int k;
                for (k = i; k > l; k--) {
                    if (Math.abs(H[hOff + k * ldh + k - 1]) <= smlnum) {
                        break;
                    }
                    double tst = Math.abs(H[hOff + (k - 1) * ldh + k - 1]) + Math.abs(H[hOff + k * ldh + k]);
                    if (tst == 0) {
                        if (k - 2 >= ilo) {
                            tst += Math.abs(H[hOff + (k - 1) * ldh + k - 2]);
                        }
                        if (k + 1 <= ihi) {
                            tst += Math.abs(H[hOff + (k + 1) * ldh + k]);
                        }
                    }
                    if (Math.abs(H[hOff + k * ldh + k - 1]) <= ulp * tst) {
                        double ab = Math.max(Math.abs(H[hOff + k * ldh + k - 1]), Math.abs(H[hOff + (k - 1) * ldh + k]));
                        double ba = Math.min(Math.abs(H[hOff + k * ldh + k - 1]), Math.abs(H[hOff + (k - 1) * ldh + k]));
                        double aa = Math.max(Math.abs(H[hOff + k * ldh + k]), Math.abs(H[hOff + (k - 1) * ldh + k - 1] - H[hOff + k * ldh + k]));
                        double bb = Math.min(Math.abs(H[hOff + k * ldh + k]), Math.abs(H[hOff + (k - 1) * ldh + k - 1] - H[hOff + k * ldh + k]));
                        double s = aa + ab;
                        if (ab / s * ba <= Math.max(smlnum, aa / s * bb * ulp)) {
                            break;
                        }
                    }
                }
                l = k;
                if (l > ilo) {
                    H[hOff + l * ldh + l - 1] = 0;
                }
                if (l >= i - 1) {
                    converged = true;
                    break;
                }
                kdefl++;

                if (!wantt) {
                    i1 = l;
                    i2 = i;
                }

                final double dat1 = 0.75;
                final double dat2 = -0.4375;
                final int kexsh = 10;

                double h11, h21, h12, h22;
                if (kdefl % (2 * kexsh) == 0) {
                    double s = Math.abs(H[hOff + i * ldh + i - 1]) + Math.abs(H[hOff + (i - 1) * ldh + i - 2]);
                    h11 = dat1 * s + H[hOff + i * ldh + i];
                    h12 = dat2 * s;
                    h21 = s;
                    h22 = h11;
                } else if (kdefl % kexsh == 0) {
                    double s = Math.abs(H[hOff + (l + 1) * ldh + l]) + Math.abs(H[hOff + (l + 2) * ldh + l + 1]);
                    h11 = dat1 * s + H[hOff + l * ldh + l];
                    h12 = dat2 * s;
                    h21 = s;
                    h22 = h11;
                } else {
                    h11 = H[hOff + (i - 1) * ldh + i - 1];
                    h21 = H[hOff + i * ldh + i - 1];
                    h12 = H[hOff + (i - 1) * ldh + i];
                    h22 = H[hOff + i * ldh + i];
                }

                double s = Math.abs(h11) + Math.abs(h12) + Math.abs(h21) + Math.abs(h22);
                double rt1r = 0, rt1i = 0, rt2r = 0, rt2i = 0;
                if (s != 0) {
                    h11 /= s;
                    h21 /= s;
                    h12 /= s;
                    h22 /= s;
                    double tr = (h11 + h22) / 2;
                    double det = (h11 - tr) * (h22 - tr) - h12 * h21;
                    double rtdisc = Math.sqrt(Math.abs(det));
                    if (det >= 0) {
                        rt1r = tr * s;
                        rt2r = rt1r;
                        rt1i = rtdisc * s;
                        rt2i = -rt1i;
                    } else {
                        rt1r = tr + rtdisc;
                        rt2r = tr - rtdisc;
                        if (Math.abs(rt1r - h22) <= Math.abs(rt2r - h22)) {
                            rt1r *= s;
                            rt2r = rt1r;
                        } else {
                            rt2r *= s;
                            rt1r = rt2r;
                        }
                        rt1i = 0;
                        rt2i = 0;
                    }
                }

                int m;
                for (m = i - 2; m >= l; m--) {
                    double h21s = H[hOff + (m + 1) * ldh + m];
                    s = Math.abs(H[hOff + m * ldh + m] - rt2r) + Math.abs(rt2i) + Math.abs(h21s);
                    if (s == 0) {
                        s = 1;
                    }
                    h21s /= s;
                    work[vOff] = h21s * H[hOff + m * ldh + m + 1] + (H[hOff + m * ldh + m] - rt1r) * ((H[hOff + m * ldh + m] - rt2r) / s) - rt2i / s * rt1i;
                    work[vOff + 1] = h21s * (H[hOff + m * ldh + m] + H[hOff + (m + 1) * ldh + m + 1] - rt1r - rt2r);
                    work[vOff + 2] = h21s * H[hOff + (m + 2) * ldh + m + 1];
                    s = Math.abs(work[vOff]) + Math.abs(work[vOff + 1]) + Math.abs(work[vOff + 2]);
                    if (s == 0) {
                        s = 1;
                    }
                    work[vOff] /= s;
                    work[vOff + 1] /= s;
                    work[vOff + 2] /= s;
                    if (m == l) {
                        break;
                    }
                    double dsum = Math.abs(H[hOff + (m - 1) * ldh + m - 1]) + Math.abs(H[hOff + m * ldh + m]) + Math.abs(H[hOff + (m + 1) * ldh + m + 1]);
                    if (Math.abs(H[hOff + m * ldh + m - 1]) * (Math.abs(work[vOff + 1]) + Math.abs(work[vOff + 2])) <= ulp * Math.abs(work[vOff]) * dsum) {
                        break;
                    }
                }

                for (int kk = m; kk < i; kk++) {
                    int nr = Math.min(3, i - kk + 1);
                    if (kk > m) {
                        for (int jj = 0; jj < nr; jj++) {
                            work[vOff + jj] = H[hOff + (kk + jj) * ldh + kk - 1];
                        }
                    }

                    dlarfg(nr, work[vOff], work, vOff + 1, 1, work, dlarfgOff);
                    double t0 = work[dlarfgOff + 1];
                    double beta = work[dlarfgOff];

                    if (kk > m) {
                        H[hOff + kk * ldh + kk - 1] = beta;
                        H[hOff + (kk + 1) * ldh + kk - 1] = 0;
                        if (kk < i - 1) {
                            H[hOff + (kk + 2) * ldh + kk - 1] = 0;
                        }
                    } else if (m > l) {
                        H[hOff + kk * ldh + kk - 1] *= 1 - t0;
                    }

                    double t1 = t0 * work[vOff + 1];
                    if (nr == 3) {
                        double t2 = t0 * work[vOff + 2];

                        for (int j = kk; j <= i2; j++) {
                            double sum = H[hOff + kk * ldh + j] + work[vOff + 1] * H[hOff + (kk + 1) * ldh + j] + work[vOff + 2] * H[hOff + (kk + 2) * ldh + j];
                            H[hOff + kk * ldh + j] -= sum * t0;
                            H[hOff + (kk + 1) * ldh + j] -= sum * t1;
                            H[hOff + (kk + 2) * ldh + j] -= sum * t2;
                        }

                        for (int j = i1; j <= Math.min(kk + 3, i); j++) {
                            double sum = H[hOff + j * ldh + kk] + work[vOff + 1] * H[hOff + j * ldh + kk + 1] + work[vOff + 2] * H[hOff + j * ldh + kk + 2];
                            H[hOff + j * ldh + kk] -= sum * t0;
                            H[hOff + j * ldh + kk + 1] -= sum * t1;
                            H[hOff + j * ldh + kk + 2] -= sum * t2;
                        }

                        if (wantz) {
                            for (int j = iloz; j <= ihiz; j++) {
                                double sum = z[zOff + j * ldz + kk] + work[vOff + 1] * z[zOff + j * ldz + kk + 1] + work[vOff + 2] * z[zOff + j * ldz + kk + 2];
                                z[zOff + j * ldz + kk] -= sum * t0;
                                z[zOff + j * ldz + kk + 1] -= sum * t1;
                                z[zOff + j * ldz + kk + 2] -= sum * t2;
                            }
                        }
                    } else if (nr == 2) {
                        for (int j = kk; j <= i2; j++) {
                            double sum = H[hOff + kk * ldh + j] + work[vOff + 1] * H[hOff + (kk + 1) * ldh + j];
                            H[hOff + kk * ldh + j] -= sum * t0;
                            H[hOff + (kk + 1) * ldh + j] -= sum * t1;
                        }

                        for (int j = i1; j <= i; j++) {
                            double sum = H[hOff + j * ldh + kk] + work[vOff + 1] * H[hOff + j * ldh + kk + 1];
                            H[hOff + j * ldh + kk] -= sum * t0;
                            H[hOff + j * ldh + kk + 1] -= sum * t1;
                        }

                        if (wantz) {
                            for (int j = iloz; j <= ihiz; j++) {
                                double sum = z[zOff + j * ldz + kk] + work[vOff + 1] * z[zOff + j * ldz + kk + 1];
                                z[zOff + j * ldz + kk] -= sum * t0;
                                z[zOff + j * ldz + kk + 1] -= sum * t1;
                            }
                        }
                    }
                }
            }

            if (!converged) {
                return i + 1;
            }

            if (l == i) {
                wr[wrOff + i] = H[hOff + i * ldh + i];
                wi[wiOff + i] = 0;
            } else if (l == i - 1) {
                double a = H[hOff + (i - 1) * ldh + i - 1];
                double b = H[hOff + (i - 1) * ldh + i];
                double c = H[hOff + i * ldh + i - 1];
                double d = H[hOff + i * ldh + i];

                Dlanv2.dlanv2(a, b, c, d, work, vOff);
                H[hOff + (i - 1) * ldh + i - 1] = work[vOff];
                H[hOff + (i - 1) * ldh + i] = work[vOff + 1];
                H[hOff + i * ldh + i - 1] = work[vOff + 2];
                H[hOff + i * ldh + i] = work[vOff + 3];
                wr[wrOff + i - 1] = work[vOff + 4];
                wi[wiOff + i - 1] = work[vOff + 5];
                wr[wrOff + i] = work[vOff + 6];
                wi[wiOff + i] = work[vOff + 7];

                if (wantt) {
                    if (i2 > i) {
                        Drot.drot(i2 - i, H, (i - 1) * ldh + i + 1, 1, H, i * ldh + i + 1, 1, work[vOff + 8], work[vOff + 9]);
                    }
                    Drot.drot(i - i1 - 1, H, i1 * ldh + i - 1, ldh, H, i1 * ldh + i, ldh, work[vOff + 8], work[vOff + 9]);
                }

                if (wantz) {
                    Drot.drot(nz, z, iloz * ldz + i - 1, ldz, z, iloz * ldz + i, ldz, work[vOff + 8], work[vOff + 9]);
                }
            }

            kdefl = 0;
            i = l - 1;
        }

        return 0;
    }

    static void dlarfg(int n, double alpha, double[] x, int xOff, int incx, double[] result, int resultOff) {
        if (n <= 1) {
            result[resultOff] = alpha;
            result[resultOff + 1] = 0;
            return;
        }

        double xnorm = 0;
        for (int i = 1; i < n; i++) {
            xnorm += x[xOff + (i - 1) * incx] * x[xOff + (i - 1) * incx];
        }
        xnorm = Math.sqrt(xnorm);

        if (xnorm == 0) {
            result[resultOff] = alpha;
            result[resultOff + 1] = 0;
            return;
        }

        double beta = -Math.copySign(Math.hypot(alpha, xnorm), alpha);
        double safmin = DLAMCH_S / DLAMCH_E;
        int knt = 0;

        if (Math.abs(beta) < safmin) {
            double rsafmn = 1 / safmin;
            while (Math.abs(beta) < safmin) {
                knt++;
                for (int i = 1; i < n; i++) {
                    x[xOff + (i - 1) * incx] *= rsafmn;
                }
                beta *= rsafmn;
                alpha *= rsafmn;
            }
            xnorm = 0;
            for (int i = 1; i < n; i++) {
                xnorm += x[xOff + (i - 1) * incx] * x[xOff + (i - 1) * incx];
            }
            xnorm = Math.sqrt(xnorm);
            beta = -Math.copySign(Math.hypot(alpha, xnorm), alpha);
        }

        double tau = (beta - alpha) / beta;
        double scale = 1.0 / (alpha - beta);
        for (int i = 1; i < n; i++) {
            x[xOff + (i - 1) * incx] *= scale;
        }

        for (int j = 0; j < knt; j++) {
            beta *= safmin;
        }

        result[resultOff] = beta;
        result[resultOff + 1] = tau;
    }

}
