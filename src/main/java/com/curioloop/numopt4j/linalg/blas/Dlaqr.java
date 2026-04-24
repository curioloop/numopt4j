/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * Dlaqr — merged implementation of the small-bulge multi-shift QR family.
 *
 * <p>Contains four routines that were previously maintained in separate files:
 * <ul>
 *   <li>{@link #dlaqr1}  — starting vector for a Francis double-shift step (LAPACK DLAQR1)</li>
 *   <li>{@link #dlaqr04} — multi-shift QR with AED, DLAQR0 / DLAQR4 behaviour</li>
 *   <li>{@link #dlaqr23} — aggressive early deflation window, DLAQR2 / DLAQR3 behaviour</li>
 *   <li>{@link #dlaqr5}  — small-bulge multi-shift QR sweep (LAPACK DLAQR5)</li>
 * </ul>
 */
interface Dlaqr {

    // =========================================================================
    // Shared machine-epsilon constants
    // =========================================================================

    double DLAMCH_P = Dlamch.dlamch('P');
    double DLAMCH_S = Dlamch.dlamch('S');
    double DLAMCH_E = Dlamch.dlamch('E');

    // =========================================================================
    // dlaqr04 tuning constants
    // =========================================================================

    /** Matrices of order ≤ NTINY must use Dlahqr (hard limit due to subdiagonal scratch). */
    int NTINY = 15;
    /** Vary deflation window size after this many iterations without deflation. */
    int KEXNW = 5;
    /** Use exceptional shifts after this many iterations without deflation. */
    int KEXSH = 6;
    /** Wilkinson shift constant 1 (used for exceptional shifts). */
    double WILK1 = 0.75;
    /** Wilkinson shift constant 2 (used for exceptional shifts). */
    double WILK2 = -0.4375;

    // =========================================================================
    // dlaqr23 scratch-layout constants
    // =========================================================================

    /**
     * Extra scratch slots appended to the base lwkopt so callers allocate enough space.
     *
     * <pre>
     *   tauTmp       : jw doubles        — elementary reflectors from Dgehrd
     *   trexcWork    : 29 + n doubles    — Dlaexc needs [0..28], Dlarfx needs [29..29+n-1]
     *   dlanv2Result : 10 doubles        — output array for Dlanv2 (indices 4–7 = eigenvalues)
     * </pre>
     */
    int TREXC_WORK_BASE = 29;
    int DLANV2_RESULT   = 10;


    // =========================================================================
    // dlaqr1 — starting vector for a Francis double-shift step
    // =========================================================================

    /**
     * Sets v to a scalar multiple of the first column of the product
     *   (H - (sr1 + i*si1)*I) * (H - (sr2 + i*si2)*I)
     * where H is a 2×2 or 3×3 upper Hessenberg matrix.
     *
     * <p>Corresponds to LAPACK DLAQR1.
     *
     *
     * @param n    order of H; must be 2 or 3
     * @param h    upper Hessenberg matrix (row-major, stride ldh)
     * @param hOff offset into h
     * @param ldh  leading dimension of h
     * @param sr1  real part of first shift
     * @param si1  imaginary part of first shift
     * @param sr2  real part of second shift
     * @param si2  imaginary part of second shift
     * @param v    output vector of length n
     * @param vOff offset into v
     */
    static void dlaqr1(int n, double[] h, int hOff, int ldh,
                       double sr1, double si1, double sr2, double si2,
                       double[] v, int vOff) {

        if (n == 2) {
            double s = abs(h[hOff] - sr2) + abs(si2) + abs(h[hOff + ldh]);
            if (s == 0) {
                v[vOff] = 0;
                v[vOff + 1] = 0;
            } else {
                double h21s = h[hOff + ldh] / s;
                v[vOff]     = h21s * h[hOff + 1] + (h[hOff] - sr1) * ((h[hOff] - sr2) / s) - si1 * (si2 / s);
                v[vOff + 1] = h21s * (h[hOff] + h[hOff + ldh + 1] - sr1 - sr2);
            }
            return;
        }

        // n == 3
        double s = abs(h[hOff] - sr2) + abs(si2) + abs(h[hOff + ldh]) + abs(h[hOff + 2 * ldh]);
        if (s == 0) {
            v[vOff] = 0;
            v[vOff + 1] = 0;
            v[vOff + 2] = 0;
        } else {
            double h21s = h[hOff + ldh] / s;
            double h31s = h[hOff + 2 * ldh] / s;
            v[vOff]     = (h[hOff] - sr1) * ((h[hOff] - sr2) / s) - si1 * (si2 / s) + h[hOff + 1] * h21s + h[hOff + 2] * h31s;
            v[vOff + 1] = h21s * (h[hOff] + h[hOff + ldh + 1] - sr1 - sr2) + h[hOff + ldh + 2] * h31s;
            v[vOff + 2] = h31s * (h[hOff] + h[hOff + 2 * ldh + 2] - sr1 - sr2) + h21s * h[hOff + 2 * ldh + 1];
        }
    }


    // =========================================================================
    // dlaqr04 — multi-shift QR with aggressive early deflation
    // =========================================================================

    /**
     * Returns the optimal lwork for {@link #dlaqr04} without accessing H or Z.
     *
     * @param n matrix order
     * @return optimal workspace size
     */
    static int dlaqr04Query(int n) {
        return dlaqr04Query(true, true, n, 0, n - 1, 1);
    }

    static int dlaqr04Query(boolean wantt, boolean wantz, int n, int ilo, int ihi, int recur) {
        if (n == 0) return 1;
        if (n <= NTINY) return 1;

        String jbcmpz = (wantt ? "S" : "E") + (wantz ? "V" : "N");
        String fname = recur > 0 ? "DLAQR0" : "DLAQR4";

        int nwr = Ilaenv.ilaenv(13, fname, jbcmpz, n, ilo, ihi, -1);
        nwr = max(2, nwr);
        nwr = min(ihi - ilo + 1, min((n - 1) / 3, nwr));

        int nsr = Ilaenv.ilaenv(15, fname, jbcmpz, n, ilo, ihi, -1);
        nsr = min(nsr, min((n - 3) / 6, ihi - ilo));
        nsr = max(2, nsr & ~1);

        int lwkopt = dlaqr23Query(n, nwr + 1, recur);
        lwkopt = max(3 * nsr / 2, lwkopt);
        return max(max(10, n), lwkopt);
    }

    /**
     * Returns the optimal lwork for {@link #dlaqr23} as called from {@link #dlaqr04},
     * including extra scratch for tauTmp, trexcWork, and dlanv2Result.
     *
     * @param n     matrix order
     * @param nw    deflation window size
     * @param recur recursion depth
     * @return optimal workspace size
     */
    static int dlaqr23Query(int n, int nw, int recur) {
        if (n == 0) return 1;
        int lwkopt = max(1, 2 * nw);
        if (nw > 2) {
            lwkopt = n + lwkopt;
            if (recur > 0) {
                lwkopt = max(lwkopt, dlaqr04Query(true, true, nw, 0, nw - 1, recur - 1));
            }
        }
        lwkopt += nw + (TREXC_WORK_BASE + n) + DLANV2_RESULT;
        return lwkopt;
    }

    /**
     * Performs the multi-shift QR algorithm with aggressive early deflation.
     *
     * <p>Corresponds to LAPACK DLAQR0 (recur &gt; 0) / DLAQR4 (recur == 0).
     *
     * {@link #dlaqr23}; Dlanv2 scratch reuses work[workOff..+9]; no input validation.
     *
     * @param wantt  if true, compute the full Schur form T
     * @param wantz  if true, accumulate Schur vectors into Z
     * @param n      order of H
     * @param ilo    first row/column of the active block (0-based)
     * @param ihi    last  row/column of the active block (0-based, inclusive)
     * @param h      upper Hessenberg matrix (row-major, stride ldh)
     * @param hOff   offset into h
     * @param ldh    leading dimension of h
     * @param wr     on return, real parts of eigenvalues wr[ilo..ihi]
     * @param wrOff  offset into wr
     * @param wi     on return, imaginary parts of eigenvalues wi[ilo..ihi]
     * @param wiOff  offset into wi
     * @param iloz   first row of Z to update (only used when wantz)
     * @param ihiz   last  row of Z to update (only used when wantz)
     * @param z      Schur vector matrix (row-major, stride ldz); may be null when !wantz
     * @param zOff   offset into z
     * @param ldz    leading dimension of z
     * @param work   workspace; work[workOff] = optimal lwork on return
     * @param workOff offset into work
     * @param lwork  workspace length; use -1 for a workspace query
     * @param recur  recursion depth (0 = DLAQR4 behaviour, &gt;0 = DLAQR0 behaviour)
     * @return number of unconverged eigenvalues (0 = all converged)
     */
    static int dlaqr04(boolean wantt, boolean wantz, int n, int ilo, int ihi,
                       double[] h, int hOff, int ldh,
                       double[] wr, int wrOff, double[] wi, int wiOff,
                       int iloz, int ihiz, double[] z, int zOff, int ldz,
                       double[] work, int workOff, int lwork, int recur) {

        if (n == 0) {
            work[workOff] = 1;
            return 0;
        }

        if (lwork == -1) {
            work[workOff] = dlaqr04Query(wantt, wantz, n, ilo, ihi, recur);
            return 0;
        }

        if (n <= NTINY) {
            return Dhseqr.dlahqr(wantt, wantz, n, ilo, ihi, h, hOff, ldh,
                                  wr, wrOff, wi, wiOff, iloz, ihiz, z, zOff, ldz, work, workOff);
        }

        String jbcmpz = (wantt ? "S" : "E") + (wantz ? "V" : "N");
        String fname  = recur > 0 ? "DLAQR0" : "DLAQR4";

        int nwr = Ilaenv.ilaenv(13, fname, jbcmpz, n, ilo, ihi, lwork);
        nwr = max(2, nwr);
        nwr = min(ihi - ilo + 1, min((n - 1) / 3, nwr));

        int nsr = Ilaenv.ilaenv(15, fname, jbcmpz, n, ilo, ihi, lwork);
        nsr = min(nsr, min((n - 3) / 6, ihi - ilo));
        nsr = max(2, nsr & ~1);

        int lwkopt = dlaqr23Query(n, nwr + 1, recur);
        lwkopt = max(3 * nsr / 2, lwkopt);

        int nmin   = Ilaenv.ilaenv(12, fname, jbcmpz, n, ilo, ihi, lwork);
        nmin   = max(NTINY, nmin);
        int nibble = Ilaenv.ilaenv(14, fname, jbcmpz, n, ilo, ihi, lwork);
        nibble = max(0, nibble);
        int kacc22 = Ilaenv.ilaenv(16, fname, jbcmpz, n, ilo, ihi, lwork);
        kacc22 = max(0, min(kacc22, 2));

        int nwmax = min((n - 1) / 3, lwork / 2);
        int nw    = nwmax;
        int nsmax = min((n - 3) / 6, 2 * lwork / 3) & ~1;

        int ndfl = 1, ndec = 0;
        int itmax = max(30, 2 * KEXSH) * max(10, (ihi - ilo + 1));
        int it = 0, unconverged = 0;

        for (int kbot = ihi; kbot >= ilo; ) {
            if (it == itmax) {
                unconverged = kbot + 1;
                break;
            }
            it++;

            int ktop = ilo;
            for (int k = kbot; k >= ilo + 1; k--) {
                if (h[hOff + k * ldh + k - 1] == 0) { ktop = k; break; }
            }

            int nh     = kbot - ktop + 1;
            int nwupbd = min(nh, nwmax);
            if (ndfl < KEXNW) {
                nw = min(nwupbd, nwr);
            } else {
                nw = min(nwupbd, 2 * nw);
            }
            if (nw < nwmax) {
                if (nw >= nh - 1) {
                    nw = nh;
                } else {
                    int kwtop = kbot - nw + 1;
                    if (kwtop > ilo && abs(h[hOff + kwtop * ldh + kwtop - 1]) > abs(h[hOff + (kwtop - 1) * ldh + kwtop - 2])) {
                        nw++;
                    }
                }
            }
            if (ndfl < KEXNW) {
                ndec = -1;
            } else if (ndec >= 0 || nw >= nwupbd) {
                ndec++;
                if (nw - ndec < 2) ndec = 0;
                nw -= ndec;
            }

            int kv  = n - nw;
            int kt  = nw;
            int kwv = nw + 1;
            int nhv = n - kwv - kt;

            long lsld = dlaqr23(wantt, wantz, n, ktop, kbot, nw,
                                 h, hOff, ldh, iloz, ihiz, z, zOff, ldz,
                                 wr, wrOff, wi, wiOff,
                                 h, hOff + kv * ldh, ldh, nhv,
                                 h, hOff + kv * ldh + kt, ldh, nhv,
                                 h, hOff + kwv * ldh, ldh,
                                 work, workOff, lwork, recur);
            int ls = (int) (lsld >>> 32);
            int ld = (int) lsld;

            kbot -= ld;
            int ks = kbot - ls + 1;

            if (ld > 0 && (100 * ld > nw * nibble || kbot - ktop + 1 <= min(nmin, nwmax))) {
                ndfl = 1;
                continue;
            }

            int ns = min(min(nsmax, nsr), max(2, kbot - ktop)) & ~1;

            if (ndfl % KEXSH == 0) {
                ks = kbot - ns + 1;
                for (int i = kbot; i > max(ks, ktop + 1); i -= 2) {
                    double ss = abs(h[hOff + i * ldh + i - 1]) + abs(h[hOff + (i - 1) * ldh + i - 2]);
                    double aa = WILK1 * ss + h[hOff + i * ldh + i];
                    Dlanv2.dlanv2(aa, ss, WILK2 * ss, aa, work, workOff);
                    wr[wrOff + i - 1] = work[workOff + 4];
                    wi[wiOff + i - 1] = work[workOff + 5];
                    wr[wrOff + i]     = work[workOff + 6];
                    wi[wiOff + i]     = work[workOff + 7];
                }
                if (ks == ktop) {
                    wr[wrOff + ks + 1] = h[hOff + (ks + 1) * ldh + ks + 1];
                    wi[wiOff + ks + 1] = 0;
                    wr[wrOff + ks]     = wr[wrOff + ks + 1];
                    wi[wiOff + ks]     = wi[wiOff + ks + 1];
                }
            } else {
                if (kbot - ks + 1 <= ns / 2) {
                    ks = kbot - ns + 1;
                    kt = n - ns;
                    Dlamv.dlacpy('A', ns, ns, h, hOff + ks * ldh + ks, ldh, h, hOff + kt * ldh, ldh);
                    if (ns > nmin && recur > 0) {
                        ks += dlaqr04(false, false, ns, 1, ns - 1, h, hOff + kt * ldh, ldh,
                                       wr, wrOff + ks, wi, wiOff + ks,
                                       0, 0, null, 0, 1, work, workOff, lwork, recur - 1);
                    } else {
                        ks += Dhseqr.dlahqr(false, false, ns, 0, ns - 1, h, hOff + kt * ldh, ldh,
                                             wr, wrOff + ks, wi, wiOff + ks,
                                             0, 0, null, 0, 1, work, workOff);
                    }
                    if (ks >= kbot) {
                        double aa = h[hOff + (kbot - 1) * ldh + kbot - 1];
                        double bb = h[hOff + (kbot - 1) * ldh + kbot];
                        double cc = h[hOff + kbot * ldh + kbot - 1];
                        double dd = h[hOff + kbot * ldh + kbot];
                        Dlanv2.dlanv2(aa, bb, cc, dd, work, workOff);
                        wr[wrOff + kbot - 1] = work[workOff + 4];
                        wi[wiOff + kbot - 1] = work[workOff + 5];
                        wr[wrOff + kbot]     = work[workOff + 6];
                        wi[wiOff + kbot]     = work[workOff + 7];
                        ks = kbot - 1;
                    }
                }

                if (kbot - ks + 1 > ns) {
                    boolean sorted = false;
                    for (int k = kbot; k > ks; k--) {
                        if (sorted) break;
                        sorted = true;
                        for (int i = ks; i < k; i++) {
                            if (abs(wr[wrOff + i]) + abs(wi[wiOff + i]) >= abs(wr[wrOff + i + 1]) + abs(wi[wiOff + i + 1])) continue;
                            sorted = false;
                            double tmp = wr[wrOff + i]; wr[wrOff + i] = wr[wrOff + i + 1]; wr[wrOff + i + 1] = tmp;
                            tmp = wi[wiOff + i]; wi[wiOff + i] = wi[wiOff + i + 1]; wi[wiOff + i + 1] = tmp;
                        }
                    }
                }

                for (int i = kbot; i > ks + 1; i -= 2) {
                    if (wi[wiOff + i] == -wi[wiOff + i - 1]) continue;
                    double tmp = wr[wrOff + i]; wr[wrOff + i] = wr[wrOff + i - 1]; wr[wrOff + i - 1] = wr[wrOff + i - 2]; wr[wrOff + i - 2] = tmp;
                    tmp = wi[wiOff + i]; wi[wiOff + i] = wi[wiOff + i - 1]; wi[wiOff + i - 1] = wi[wiOff + i - 2]; wi[wiOff + i - 2] = tmp;
                }
            }

            if (kbot - ks + 1 == 2 && wi[wiOff + kbot] == 0) {
                if (abs(wr[wrOff + kbot] - h[hOff + kbot * ldh + kbot]) < abs(wr[wrOff + kbot - 1] - h[hOff + kbot * ldh + kbot])) {
                    wr[wrOff + kbot - 1] = wr[wrOff + kbot];
                } else {
                    wr[wrOff + kbot] = wr[wrOff + kbot - 1];
                }
            }

            ns = min(ns, kbot - ks + 1) & ~1;
            ks = kbot - ns + 1;

            int kdu = 2 * ns;
            int ku  = n - kdu;
            int kwh = kdu;
            kwv = kdu + 3;
            nhv = n - kwv - kdu;

            dlaqr5(wantt, wantz, kacc22, n, ktop, kbot, ns,
                   wr, wrOff + ks, wi, wiOff + ks,
                   h, hOff, ldh, iloz, ihiz, z, zOff, ldz,
                     work, workOff + 3, 3,
                   h, hOff + ku * ldh, ldh,
                   nhv, h, hOff + kwv * ldh, ldh,
                   nhv, h, hOff + ku * ldh + kwh, ldh);

            ndfl = (ld > 0) ? 1 : ndfl + 1;
        }

        work[workOff] = lwkopt;
        return unconverged;
    }


    // =========================================================================
    // dlaqr23 — aggressive early deflation window
    // =========================================================================

    /**
     * Performs aggressive early deflation on H[ktop:kbot+1, ktop:kbot+1].
     *
     * <p>Corresponds to LAPACK DLAQR2 (recur == 0) / DLAQR3 (recur &gt; 0).
     *
     * <ul>
     *   <li>Returns a packed {@code long}: high 32 bits = ns, low 32 bits = nd.</li>
     *   <li>Extra scratch (tauTmp, trexcWork, dlanv2Result) carved from the tail of work[].</li>
     *   <li>Dtrexc writes updated ilst into work[trexcOff+1] and returns boolean ok.</li>
     *   <li>Dlarfg writes tau into work[workOff] and returns beta.</li>
     *   <li>Dlanv2 writes 10 outputs into work[dlanv2Off..+9]; eigenvalues at indices 4–7.</li>
     * </ul>
     *
     * @param wantt   if true, update the full H to quasi-triangular Schur form
     * @param wantz   if true, accumulate orthogonal transforms into Z
     * @param n       order of H
     * @param ktop    first row/col of the isolated block
     * @param kbot    last  row/col of the isolated block
     * @param nw      deflation window size
     * @param h       Hessenberg matrix, row-major
     * @param hOff    base offset into h[]
     * @param ldh     leading dimension of h
     * @param iloz    first row of Z to update
     * @param ihiz    last  row of Z to update
     * @param z       Schur vector matrix Z (or unused)
     * @param zOff    base offset into z[]
     * @param ldz     leading dimension of z
     * @param sr      real parts of eigenvalue estimates
     * @param srOff   base offset into sr[]
     * @param si      imaginary parts of eigenvalue estimates
     * @param siOff   base offset into si[]
     * @param v       nw×nw work matrix
     * @param vOff    base offset into v[]
     * @param ldv     leading dimension of v
     * @param nh      column count of t
     * @param t       nw×nh work matrix
     * @param tOff    base offset into t[]
     * @param ldt     leading dimension of t
     * @param nv      row count of wv
     * @param wv      nv×nw work matrix
     * @param wvOff   base offset into wv[]
     * @param ldwv    leading dimension of wv
     * @param work    workspace array
     * @param workOff base offset into work[]
     * @param lwork   workspace size (pass -1 for a size query)
     * @param recur   recursion depth; 0 → use Dlahqr, &gt;0 → use dlaqr04
     * @return packed long: high 32 bits = ns (unconverged shifts), low 32 bits = nd (deflated)
     */
    static long dlaqr23(boolean wantt, boolean wantz, int n, int ktop, int kbot, int nw,
                        double[] h, int hOff, int ldh,
                        int iloz, int ihiz, double[] z, int zOff, int ldz,
                        double[] sr, int srOff, double[] si, int siOff,
                        double[] v, int vOff, int ldv, int nh, double[] t, int tOff, int ldt,
                        int nv, double[] wv, int wvOff, int ldwv,
                        double[] work, int workOff, int lwork, int recur) {

        if (nw == 0) {
            work[workOff] = 1;
            return 0L;
        }

        int jw = nw;

        int trexcLen    = TREXC_WORK_BASE + n;
        int extraScratch = jw + trexcLen + DLANV2_RESULT;

        int lwkopt = max(1, 2 * nw);
        if (jw > 2) {
            Dgehrd.dgehrd(jw, 0, jw - 2, t, tOff, ldt, work, workOff, work, workOff, -1);
            int lwk1 = (int) work[workOff];
            Dgehrd.dormhr(BLAS.Side.Right, BLAS.Trans.NoTrans, jw, jw, 0, jw - 2,
                          t, tOff, ldt, work, workOff, v, vOff, ldv, work, workOff, -1);
            int lwk2 = (int) work[workOff];
            if (recur > 0) {
                int lwk3 = dlaqr04Query(jw);
                lwkopt = max(jw + max(lwk1, lwk2), lwk3);
            } else {
                lwkopt = jw + max(lwk1, lwk2);
            }
        }
        lwkopt += extraScratch;

        if (lwork == -1) {
            work[workOff] = lwkopt;
            return 0L;
        }

        // Scratch layout: tail of work[workOff..workOff+lwork-1]
        int tauOff    = workOff + lwork - extraScratch;
        int trexcOff  = tauOff + jw;
        int dlanv2Off = trexcOff + trexcLen;
        int baseWork  = lwork - extraScratch;

        double ulp    = DLAMCH_P;
        double smlnum = (double) n / ulp * DLAMCH_S;

        double s = 0;
        int kwtop = kbot - jw + 1;
        if (kwtop != ktop) {
            s = h[hOff + kwtop * ldh + kwtop - 1];
        }

        // Special case: 1×1 deflation window
        if (kwtop == kbot) {
            sr[srOff + kwtop] = h[hOff + kwtop * ldh + kwtop];
            si[siOff + kwtop] = 0;
            int ns = 1, nd = 0;
            if (abs(s) <= max(smlnum, ulp * abs(h[hOff + kwtop * ldh + kwtop]))) {
                ns = 0; nd = 1;
                if (kwtop > ktop) h[hOff + kwtop * ldh + kwtop - 1] = 0;
            }
            work[workOff] = 1;
            return ((long) ns << 32) | (nd & 0xFFFFFFFFL);
        }

        // Copy window into T and initialize V = I
        Dlamv.dlacpy('U', jw, jw, h, hOff + kwtop * ldh + kwtop, ldh, t, tOff, ldt);
        for (int i = 0; i < jw - 1; i++) {
            t[tOff + (i + 1) * ldt + i] = h[hOff + (kwtop + 1 + i) * ldh + kwtop + i];
        }
        Dlamv.dlaset('A', jw, jw, 0, 1, v, vOff, ldv);

        int nmin = Ilaenv.ilaenv(12, "DLAQR3", "SV", jw, 0, jw - 1, lwork);
        int infqr;
        if (recur > 0 && jw > nmin) {
            infqr = dlaqr04(true, true, jw, 0, jw - 1, t, tOff, ldt,
                             sr, srOff + kwtop, si, siOff + kwtop,
                             0, jw - 1, v, vOff, ldv,
                             work, workOff, baseWork, recur - 1);
        } else {
            infqr = Dhseqr.dlahqr(true, true, jw, 0, jw - 1, t, tOff, ldt,
                                   sr, srOff + kwtop, si, siOff + kwtop,
                                   0, jw - 1, v, vOff, ldv, work, workOff);
        }

        // Clean margin below second subdiagonal
        for (int j = 0; j < jw - 3; j++) {
            t[tOff + (j + 2) * ldt + j] = 0;
            t[tOff + (j + 3) * ldt + j] = 0;
        }
        if (jw >= 3) t[tOff + (jw - 1) * ldt + jw - 3] = 0;

        // Deflation detection loop
        int ns   = jw;
        int ilst = infqr;

        while (ilst < ns) {
            boolean bulge = ns >= 2 && t[tOff + (ns - 1) * ldt + ns - 2] != 0;
            if (!bulge) {
                double abst = abs(t[tOff + (ns - 1) * ldt + ns - 1]);
                if (abst == 0) abst = abs(s);
                if (abs(s * v[vOff + ns - 1]) <= max(smlnum, ulp * abst)) {
                    ns--;
                } else {
                    Dtrexc.dtrexc(true, jw, t, tOff, ldt, v, vOff, ldv, ns - 1, ilst, work, trexcOff);
                    ilst = (int) work[trexcOff + 1] + 1;
                }
                continue;
            }
            double abst = abs(t[tOff + (ns - 1) * ldt + ns - 1])
                    + sqrt(abs(t[tOff + (ns - 1) * ldt + ns - 2]))
                    * sqrt(abs(t[tOff + (ns - 2) * ldt + ns - 1]));
            if (abst == 0) abst = abs(s);
            if (max(abs(s * v[vOff + ns - 1]), abs(s * v[vOff + ns - 2])) <= max(smlnum, ulp * abst)) {
                ns -= 2;
            } else {
                Dtrexc.dtrexc(true, jw, t, tOff, ldt, v, vOff, ldv, ns - 1, ilst, work, trexcOff);
                ilst = (int) work[trexcOff + 1] + 2;
            }
        }

        if (ns == 0) s = 0;

        // Sort diagonal blocks by decreasing magnitude
        if (ns < jw) {
            boolean sorted = false;
            int i = ns;
            while (!sorted) {
                sorted = true;
                int kend = i - 1;
                i = infqr;
                int k;
                if (i == ns - 1 || t[tOff + (i + 1) * ldt + i] == 0) { k = i + 1; } else { k = i + 2; }
                while (k <= kend) {
                    double evi = (k == i + 1) ? abs(t[tOff + i * ldt + i])
                            : abs(t[tOff + i * ldt + i]) + sqrt(abs(t[tOff + (i + 1) * ldt + i])) * sqrt(abs(t[tOff + i * ldt + i + 1]));
                    double evk = (k == kend || t[tOff + (k + 1) * ldt + k] == 0) ? abs(t[tOff + k * ldt + k])
                            : abs(t[tOff + k * ldt + k]) + sqrt(abs(t[tOff + (k + 1) * ldt + k])) * sqrt(abs(t[tOff + k * ldt + k + 1]));
                    if (evi >= evk) {
                        i = k;
                    } else {
                        sorted = false;
                        boolean ok = Dtrexc.dtrexc(true, jw, t, tOff, ldt, v, vOff, ldv, i, k, work, trexcOff);
                        i = ok ? (int) work[trexcOff + 1] : k;
                    }
                    if (i == kend || t[tOff + (i + 1) * ldt + i] == 0) { k = i + 1; } else { k = i + 2; }
                }
            }
        }

        // Restore eigenvalue arrays from Schur form T
        for (int i = jw - 1; i >= infqr; ) {
            if (i == infqr || t[tOff + i * ldt + i - 1] == 0) {
                sr[srOff + kwtop + i] = t[tOff + i * ldt + i];
                si[siOff + kwtop + i] = 0;
                i--;
                continue;
            }
            double aa = t[tOff + (i - 1) * ldt + i - 1];
            double bb = t[tOff + (i - 1) * ldt + i];
            double cc = t[tOff + i * ldt + i - 1];
            double dd = t[tOff + i * ldt + i];
            Dlanv2.dlanv2(aa, bb, cc, dd, work, dlanv2Off);
            sr[srOff + kwtop + i - 1] = work[dlanv2Off + 4];
            si[siOff + kwtop + i - 1] = work[dlanv2Off + 5];
            sr[srOff + kwtop + i]     = work[dlanv2Off + 6];
            si[siOff + kwtop + i]     = work[dlanv2Off + 7];
            i -= 2;
        }

        // Return to Hessenberg form and update H (and optionally Z)
        if (ns < jw || s == 0) {
            if (ns > 1 && s != 0) {
                Dlamv.dcopy(ns, v, vOff, 1, work, workOff, 1);
                double alpha = work[workOff];
                Dlarfg.dlarfg(ns, alpha, work, workOff + 1, 1, work, workOff);
                double tau = work[workOff];
                work[workOff] = 1;

                Dlamv.dlaset('L', jw - 2, jw - 2, 0, 0, t, tOff + 2 * ldt, ldt);

                Dlarf.dlarf(BLAS.Side.Left,  ns, jw, work, workOff, 1, tau, t, tOff, ldt, work, workOff + jw);
                Dlarf.dlarf(BLAS.Side.Right, ns, ns, work, workOff, 1, tau, t, tOff, ldt, work, workOff + jw);
                Dlarf.dlarf(BLAS.Side.Right, jw, ns, work, workOff, 1, tau, v, vOff,  ldv, work, workOff + jw);

                Dgehrd.dgehrd(jw, 0, ns - 1, t, tOff, ldt, work, tauOff, work, workOff + jw, baseWork - jw);
            }

            if (kwtop > 0) h[hOff + kwtop * ldh + kwtop - 1] = s * v[vOff];
            Dlamv.dlacpy('U', jw, jw, t, tOff, ldt, h, hOff + kwtop * ldh + kwtop, ldh);
            for (int i = 0; i < jw - 1; i++) {
                h[hOff + (kwtop + 1 + i) * ldh + kwtop + i] = t[tOff + (i + 1) * ldt + i];
            }

            if (ns > 1 && s != 0) {
                Dgehrd.dormhr(BLAS.Side.Right, BLAS.Trans.NoTrans, jw, ns, 0, ns - 1,
                              t, tOff, ldt, work, tauOff, v, vOff, ldv,
                              work, workOff + jw, baseWork - jw);
            }

            int ltop = wantt ? 0 : ktop;
            for (int krow = ltop; krow < kwtop; krow += nv) {
                int kln = min(nv, kwtop - krow);
                Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kln, jw, jw,
                            1, h, hOff + krow * ldh + kwtop, ldh, v, vOff, ldv,
                            0, wv, wvOff, ldwv);
                Dlamv.dlacpy('A', kln, jw, wv, wvOff, ldwv, h, hOff + krow * ldh + kwtop, ldh);
            }

            if (wantt) {
                for (int kcol = kbot + 1; kcol < n; kcol += nh) {
                    int kln = min(nh, n - kcol);
                    Dgemm.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, jw, kln, jw,
                                1, v, vOff, ldv, h, hOff + kwtop * ldh + kcol, ldh,
                                0, t, tOff, ldt);
                    Dlamv.dlacpy('A', jw, kln, t, tOff, ldt, h, hOff + kwtop * ldh + kcol, ldh);
                }
            }

            if (wantz) {
                for (int krow = iloz; krow <= ihiz; krow += nv) {
                    int kln = min(nv, ihiz - krow + 1);
                    Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, kln, jw, jw,
                                1, z, zOff + krow * ldz + kwtop, ldz, v, vOff, ldv,
                                0, wv, wvOff, ldwv);
                    Dlamv.dlacpy('A', kln, jw, wv, wvOff, ldwv, z, zOff + krow * ldz + kwtop, ldz);
                }
            }
        }

        int nd = jw - ns;
        ns -= infqr;
        work[workOff] = lwkopt;
        return ((long) ns << 32) | (nd & 0xFFFFFFFFL);
    }


    // =========================================================================
    // dlaqr5 — small-bulge multi-shift QR sweep
    // =========================================================================

    /**
     * Performs a single small-bulge multi-shift QR sweep on H[ktop:kbot+1, ktop:kbot+1].
     *
     * <p>Corresponds to LAPACK DLAQR5.
     *
     * into the supplied output slot and returns beta; Dlaqr1 called with explicit offset;
     * no input validation.
     *
     * @param wantt  if true, update the full H to quasi-triangular Schur form
     * @param wantz  if true, accumulate orthogonal transforms into Z
     * @param kacc22 far-from-diagonal update mode: 0 = none, 1 or 2 = accumulate via U
     * @param n      order of H
     * @param ktop   first row/col of the isolated block
     * @param kbot   last  row/col of the isolated block
     * @param nshfts number of simultaneous shifts (must be positive and even)
     * @param sr     real parts of shifts (may be reordered)
     * @param srOff  base offset into sr[]
     * @param si     imaginary parts of shifts (may be reordered)
     * @param siOff  base offset into si[]
     * @param h      Hessenberg matrix, row-major
     * @param hOff   base offset into h[]
     * @param ldh    leading dimension of h
     * @param iloz   first row of Z to update
     * @param ihiz   last  row of Z to update
     * @param z      Schur vector matrix Z (or unused if !wantz)
     * @param zOff   base offset into z[]
     * @param ldz    leading dimension of z
     * @param v      (nshfts/2)×3 auxiliary matrix for Householder vectors
     * @param vOff   base offset into v[]
     * @param ldv    leading dimension of v (≥ 3)
     * @param u      (2*nshfts)×(2*nshfts) accumulation matrix
     * @param uOff   base offset into u[]
     * @param ldu    leading dimension of u
     * @param nv     row count of wv
     * @param wv     nv×(2*nshfts) work matrix for vertical far-from-diagonal updates
     * @param wvOff  base offset into wv[]
     * @param ldwv   leading dimension of wv
     * @param nh     column count of wh
     * @param wh     (2*nshfts-1)×nh work matrix for horizontal far-from-diagonal updates
     * @param whOff  base offset into wh[]
     * @param ldwh   leading dimension of wh
     */
    static void dlaqr5(boolean wantt, boolean wantz, int kacc22, int n, int ktop, int kbot, int nshfts,
                       double[] sr, int srOff, double[] si, int siOff,
                       double[] h, int hOff, int ldh,
                       int iloz, int ihiz, double[] z, int zOff, int ldz,
                       double[] v, int vOff, int ldv,
                       double[] u, int uOff, int ldu,
                       int nv, double[] wv, int wvOff, int ldwv,
                       int nh, double[] wh, int whOff, int ldwh) {

        if (nshfts < 2) return;
        if (ktop >= kbot) return;

        // Shuffle shifts into adjacent conjugate pairs
        for (int i = 0; i < nshfts - 2; i += 2) {
            if (si[siOff + i] == -si[siOff + i + 1]) continue;
            double tmpSr = sr[srOff + i]; sr[srOff + i] = sr[srOff + i + 1]; sr[srOff + i + 1] = sr[srOff + i + 2]; sr[srOff + i + 2] = tmpSr;
            double tmpSi = si[siOff + i]; si[siOff + i] = si[siOff + i + 1]; si[siOff + i + 1] = si[siOff + i + 2]; si[siOff + i + 2] = tmpSi;
        }

        int ns = nshfts;
        double safmin = DLAMCH_S;
        double ulp    = DLAMCH_P;
        double smlnum = safmin * n / ulp;
        boolean accum = kacc22 == 1 || kacc22 == 2;

        if (ktop + 2 <= kbot) h[hOff + (ktop + 2) * ldh + ktop] = 0;

        int nbmps = ns / 2;
        int kdu   = 4 * nbmps;

        for (int incol = ktop - 2 * nbmps + 1; incol <= kbot - 2; incol += 2 * nbmps) {

            int jtop;
            if (accum) { jtop = max(ktop, incol); }
            else if (wantt) { jtop = 0; }
            else { jtop = ktop; }

            int ndcol = incol + kdu;

            if (accum) Dlamv.dlaset('A', kdu, kdu, 0, 1, u, uOff, ldu);

            for (int krcol = incol; krcol <= min(incol + 2 * nbmps - 1, kbot - 2); krcol++) {

                int mtop  = max(0, (ktop - krcol) / 2);
                int mbot  = min(nbmps, (kbot - krcol - 1) / 2) - 1;
                int m22   = mbot + 1;
                boolean bmp22 = (mbot < nbmps - 1) && (krcol + 2 * m22 == kbot - 2);

                if (bmp22) {
                    int k = krcol + 2 * m22;
                    if (k == ktop - 1) {
                        dlaqr1(2, h, hOff + (k + 1) * ldh + k + 1, ldh,
                               sr[srOff + 2 * m22], si[siOff + 2 * m22],
                               sr[srOff + 2 * m22 + 1], si[siOff + 2 * m22 + 1],
                               v, vOff + m22 * ldv);
                        double beta = v[vOff + m22 * ldv];
                        Dlarfg.dlarfg(2, beta, v, vOff + m22 * ldv + 1, 1, v, vOff + m22 * ldv);
                    } else {
                        double beta = h[hOff + (k + 1) * ldh + k];
                        v[vOff + m22 * ldv + 1] = h[hOff + (k + 2) * ldh + k];
                        beta = Dlarfg.dlarfg(2, beta, v, vOff + m22 * ldv + 1, 1, v, vOff + m22 * ldv);
                        h[hOff + (k + 1) * ldh + k] = beta;
                        h[hOff + (k + 2) * ldh + k] = 0;
                    }

                    double t1 = v[vOff + m22 * ldv];
                    double t2 = t1 * v[vOff + m22 * ldv + 1];
                    for (int j = jtop; j <= min(kbot, k + 3); j++) {
                        double refsum = h[hOff + j * ldh + k + 1] + v[vOff + m22 * ldv + 1] * h[hOff + j * ldh + k + 2];
                        h[hOff + j * ldh + k + 1] -= refsum * t1;
                        h[hOff + j * ldh + k + 2] -= refsum * t2;
                    }

                    int jbot;
                    if (accum) { jbot = min(ndcol, kbot); }
                    else if (wantt) { jbot = n - 1; }
                    else { jbot = kbot; }
                    for (int j = k + 1; j <= jbot; j++) {
                        double refsum = h[hOff + (k + 1) * ldh + j] + v[vOff + m22 * ldv + 1] * h[hOff + (k + 2) * ldh + j];
                        h[hOff + (k + 1) * ldh + j] -= refsum * t1;
                        h[hOff + (k + 2) * ldh + j] -= refsum * t2;
                    }

                    if (k >= ktop && h[hOff + (k + 1) * ldh + k] != 0) {
                        double tst1 = abs(h[hOff + k * ldh + k]) + abs(h[hOff + (k + 1) * ldh + k + 1]);
                        if (tst1 == 0) {
                            if (k >= ktop + 1) tst1 += abs(h[hOff + k * ldh + k - 1]);
                            if (k >= ktop + 2) tst1 += abs(h[hOff + k * ldh + k - 2]);
                            if (k <= kbot - 2) tst1 += abs(h[hOff + (k + 2) * ldh + k + 1]);
                            if (k <= kbot - 3) tst1 += abs(h[hOff + (k + 3) * ldh + k + 1]);
                            if (k <= kbot - 4) tst1 += abs(h[hOff + (k + 4) * ldh + k + 1]);
                        }
                        if (abs(h[hOff + (k + 1) * ldh + k]) <= max(smlnum, ulp * tst1)) {
                            double h12 = max(abs(h[hOff + (k + 1) * ldh + k]), abs(h[hOff + k * ldh + k + 1]));
                            double h21 = min(abs(h[hOff + (k + 1) * ldh + k]), abs(h[hOff + k * ldh + k + 1]));
                            double h11 = max(abs(h[hOff + (k + 1) * ldh + k + 1]), abs(h[hOff + k * ldh + k] - h[hOff + (k + 1) * ldh + k + 1]));
                            double h22 = min(abs(h[hOff + (k + 1) * ldh + k + 1]), abs(h[hOff + k * ldh + k] - h[hOff + (k + 1) * ldh + k + 1]));
                            double scl  = h11 + h12;
                            double tst2 = h22 * (h11 / scl);
                            if (tst2 == 0 || h21 * (h12 / scl) <= max(smlnum, ulp * tst2)) {
                                h[hOff + (k + 1) * ldh + k] = 0;
                            }
                        }
                    }

                    if (accum) {
                        int kms = k - incol - 1;
                        for (int j = max(0, ktop - incol - 1); j < kdu; j++) {
                            double refsum = u[uOff + j * ldu + kms + 1] + v[vOff + m22 * ldv + 1] * u[uOff + j * ldu + kms + 2];
                            u[uOff + j * ldu + kms + 1] -= refsum * t1;
                            u[uOff + j * ldu + kms + 2] -= refsum * t2;
                        }
                    } else if (wantz) {
                        for (int j = iloz; j <= ihiz; j++) {
                            double refsum = z[zOff + j * ldz + k + 1] + v[vOff + m22 * ldv + 1] * z[zOff + j * ldz + k + 2];
                            z[zOff + j * ldz + k + 1] -= refsum * t1;
                            z[zOff + j * ldz + k + 2] -= refsum * t2;
                        }
                    }
                }

                for (int m = mbot; m >= mtop; m--) {
                    int k = krcol + 2 * m;
                    double t1, t2, t3;
                    if (k == ktop - 1) {
                        dlaqr1(3, h, hOff + ktop * ldh + ktop, ldh,
                               sr[srOff + 2 * m], si[siOff + 2 * m],
                               sr[srOff + 2 * m + 1], si[siOff + 2 * m + 1],
                               v, vOff + m * ldv);
                        Dlarfg.dlarfg(3, v[vOff + m * ldv], v, vOff + m * ldv + 1, 1, v, vOff + m * ldv);
                    } else {
                        t1 = v[vOff + m * ldv];
                        t2 = t1 * v[vOff + m * ldv + 1];
                        t3 = t1 * v[vOff + m * ldv + 2];
                        double refsum = v[vOff + m * ldv + 2] * h[hOff + (k + 3) * ldh + k + 2];
                        h[hOff + (k + 3) * ldh + k]     = -refsum * t1;
                        h[hOff + (k + 3) * ldh + k + 1] = -refsum * t2;
                        h[hOff + (k + 3) * ldh + k + 2] -= refsum * t3;

                        double beta = h[hOff + (k + 1) * ldh + k];
                        v[vOff + m * ldv + 1] = h[hOff + (k + 2) * ldh + k];
                        v[vOff + m * ldv + 2] = h[hOff + (k + 3) * ldh + k];
                        beta = Dlarfg.dlarfg(3, beta, v, vOff + m * ldv + 1, 1, v, vOff + m * ldv);

                        if (h[hOff + (k + 3) * ldh + k] != 0 || h[hOff + (k + 3) * ldh + k + 1] != 0 || h[hOff + (k + 3) * ldh + k + 2] == 0) {
                            h[hOff + (k + 1) * ldh + k] = beta;
                            h[hOff + (k + 2) * ldh + k] = 0;
                            h[hOff + (k + 3) * ldh + k] = 0;
                        } else {
                            double[] vt = new double[3];
                            dlaqr1(3, h, hOff + (k + 1) * ldh + k + 1, ldh,
                                   sr[srOff + 2 * m], si[siOff + 2 * m],
                                   sr[srOff + 2 * m + 1], si[siOff + 2 * m + 1],
                                   vt, 0);
                            Dlarfg.dlarfg(3, vt[0], vt, 1, 1, vt, 0);
                            t1 = vt[0]; t2 = t1 * vt[1]; t3 = t1 * vt[2];
                            refsum = h[hOff + (k + 1) * ldh + k] + vt[1] * h[hOff + (k + 2) * ldh + k];
                            double dsum = abs(h[hOff + k * ldh + k]) + abs(h[hOff + (k + 1) * ldh + k + 1]) + abs(h[hOff + (k + 2) * ldh + k + 2]);
                            if (abs(h[hOff + (k + 2) * ldh + k] - refsum * t2) + abs(refsum * t3) > ulp * dsum) {
                                h[hOff + (k + 1) * ldh + k] = beta;
                                h[hOff + (k + 2) * ldh + k] = 0;
                                h[hOff + (k + 3) * ldh + k] = 0;
                            } else {
                                h[hOff + (k + 1) * ldh + k] -= refsum * t1;
                                h[hOff + (k + 2) * ldh + k] = 0;
                                h[hOff + (k + 3) * ldh + k] = 0;
                                v[vOff + m * ldv]     = vt[0];
                                v[vOff + m * ldv + 1] = vt[1];
                                v[vOff + m * ldv + 2] = vt[2];
                            }
                        }
                    }

                    t1 = v[vOff + m * ldv];
                    t2 = t1 * v[vOff + m * ldv + 1];
                    t3 = t1 * v[vOff + m * ldv + 2];
                    for (int j = jtop; j <= min(kbot, k + 3); j++) {
                        double refsum = h[hOff + j * ldh + k + 1] + v[vOff + m * ldv + 1] * h[hOff + j * ldh + k + 2] + v[vOff + m * ldv + 2] * h[hOff + j * ldh + k + 3];
                        h[hOff + j * ldh + k + 1] -= refsum * t1;
                        h[hOff + j * ldh + k + 2] -= refsum * t2;
                        h[hOff + j * ldh + k + 3] -= refsum * t3;
                    }

                    double refsum = h[hOff + (k + 1) * ldh + k + 1] + v[vOff + m * ldv + 1] * h[hOff + (k + 2) * ldh + k + 1] + v[vOff + m * ldv + 2] * h[hOff + (k + 3) * ldh + k + 1];
                    h[hOff + (k + 1) * ldh + k + 1] -= refsum * t1;
                    h[hOff + (k + 2) * ldh + k + 1] -= refsum * t2;
                    h[hOff + (k + 3) * ldh + k + 1] -= refsum * t3;

                    if (k < ktop) continue;
                    if (h[hOff + (k + 1) * ldh + k] != 0) {
                        double tst1 = abs(h[hOff + k * ldh + k]) + abs(h[hOff + (k + 1) * ldh + k + 1]);
                        if (tst1 == 0) {
                            if (k >= ktop + 1) tst1 += abs(h[hOff + k * ldh + k - 1]);
                            if (k >= ktop + 2) tst1 += abs(h[hOff + k * ldh + k - 2]);
                            if (k >= ktop + 3) tst1 += abs(h[hOff + k * ldh + k - 3]);
                            if (k <= kbot - 2) tst1 += abs(h[hOff + (k + 2) * ldh + k + 1]);
                            if (k <= kbot - 3) tst1 += abs(h[hOff + (k + 3) * ldh + k + 1]);
                            if (k <= kbot - 4) tst1 += abs(h[hOff + (k + 4) * ldh + k + 1]);
                        }
                        if (abs(h[hOff + (k + 1) * ldh + k]) <= max(smlnum, ulp * tst1)) {
                            double h12 = max(abs(h[hOff + (k + 1) * ldh + k]), abs(h[hOff + k * ldh + k + 1]));
                            double h21 = min(abs(h[hOff + (k + 1) * ldh + k]), abs(h[hOff + k * ldh + k + 1]));
                            double h11 = max(abs(h[hOff + (k + 1) * ldh + k + 1]), abs(h[hOff + k * ldh + k] - h[hOff + (k + 1) * ldh + k + 1]));
                            double h22 = min(abs(h[hOff + (k + 1) * ldh + k + 1]), abs(h[hOff + k * ldh + k] - h[hOff + (k + 1) * ldh + k + 1]));
                            double scl  = h11 + h12;
                            double tst2 = h22 * (h11 / scl);
                            if (tst2 == 0 || h21 * (h12 / scl) <= max(smlnum, ulp * tst2)) {
                                h[hOff + (k + 1) * ldh + k] = 0;
                            }
                        }
                    }
                }

                int jbot;
                if (accum) { jbot = min(ndcol, kbot); }
                else if (wantt) { jbot = n - 1; }
                else { jbot = kbot; }

                for (int m = mbot; m >= mtop; m--) {
                    int k  = krcol + 2 * m;
                    double t1 = v[vOff + m * ldv];
                    double t2 = t1 * v[vOff + m * ldv + 1];
                    double t3 = t1 * v[vOff + m * ldv + 2];
                    for (int j = max(ktop, krcol + 2 * (m + 1)); j <= jbot; j++) {
                        double refsum = h[hOff + (k + 1) * ldh + j] + v[vOff + m * ldv + 1] * h[hOff + (k + 2) * ldh + j] + v[vOff + m * ldv + 2] * h[hOff + (k + 3) * ldh + j];
                        h[hOff + (k + 1) * ldh + j] -= refsum * t1;
                        h[hOff + (k + 2) * ldh + j] -= refsum * t2;
                        h[hOff + (k + 3) * ldh + j] -= refsum * t3;
                    }
                }

                if (accum) {
                    for (int m = mbot; m >= mtop; m--) {
                        int k   = krcol + 2 * m;
                        int kms = k - incol - 1;
                        int i2  = max(0, ktop - incol - 1);
                        i2 = max(i2, kms - (krcol - incol));
                        int i4  = min(kdu, krcol + 2 * mbot - incol + 5);
                        double t1 = v[vOff + m * ldv];
                        double t2 = t1 * v[vOff + m * ldv + 1];
                        double t3 = t1 * v[vOff + m * ldv + 2];
                        for (int j = i2; j < i4; j++) {
                            double refsum = u[uOff + j * ldu + kms + 1] + v[vOff + m * ldv + 1] * u[uOff + j * ldu + kms + 2] + v[vOff + m * ldv + 2] * u[uOff + j * ldu + kms + 3];
                            u[uOff + j * ldu + kms + 1] -= refsum * t1;
                            u[uOff + j * ldu + kms + 2] -= refsum * t2;
                            u[uOff + j * ldu + kms + 3] -= refsum * t3;
                        }
                    }
                } else if (wantz) {
                    for (int m = mbot; m >= mtop; m--) {
                        int k  = krcol + 2 * m;
                        double t1 = v[vOff + m * ldv];
                        double t2 = t1 * v[vOff + m * ldv + 1];
                        double t3 = t1 * v[vOff + m * ldv + 2];
                        for (int j = iloz; j <= ihiz; j++) {
                            double refsum = z[zOff + j * ldz + k + 1] + v[vOff + m * ldv + 1] * z[zOff + j * ldz + k + 2] + v[vOff + m * ldv + 2] * z[zOff + j * ldz + k + 3];
                            z[zOff + j * ldz + k + 1] -= refsum * t1;
                            z[zOff + j * ldz + k + 2] -= refsum * t2;
                            z[zOff + j * ldz + k + 3] -= refsum * t3;
                        }
                    }
                }
            }

            if (!accum) continue;

            int jt0 = ktop, jb0 = kbot;
            if (wantt) { jt0 = 0; jb0 = n - 1; }

            int k1 = max(0, ktop - incol - 1);
            int nu = kdu - max(0, ndcol - kbot) - k1;

            for (int jcol = min(ndcol, kbot) + 1; jcol <= jb0; jcol += nh) {
                int jlen = min(nh, jb0 - jcol + 1);
                Dgemm.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, nu, jlen, nu,
                            1, u, uOff + k1 * ldu + k1, ldu,
                            h, hOff + (incol + k1 + 1) * ldh + jcol, ldh,
                            0, wh, whOff, ldwh);
                Dlamv.dlacpy('A', nu, jlen, wh, whOff, ldwh, h, hOff + (incol + k1 + 1) * ldh + jcol, ldh);
            }

            for (int jrow = jt0; jrow < max(ktop, incol); jrow += nv) {
                int jlen = min(nv, max(ktop, incol) - jrow);
                Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, jlen, nu, nu,
                            1, h, hOff + jrow * ldh + incol + k1 + 1, ldh,
                            u, uOff + k1 * ldu + k1, ldu,
                            0, wv, wvOff, ldwv);
                Dlamv.dlacpy('A', jlen, nu, wv, wvOff, ldwv, h, hOff + jrow * ldh + incol + k1 + 1, ldh);
            }

            if (wantz) {
                for (int jrow = iloz; jrow <= ihiz; jrow += nv) {
                    int jlen = min(nv, ihiz - jrow + 1);
                    Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, jlen, nu, nu,
                                1, z, zOff + jrow * ldz + incol + k1 + 1, ldz,
                                u, uOff + k1 * ldu + k1, ldu,
                                0, wv, wvOff, ldwv);
                    Dlamv.dlacpy('A', jlen, nu, wv, wvOff, ldwv, z, zOff + jrow * ldz + incol + k1 + 1, ldz);
                }
            }
        }
    }

}
