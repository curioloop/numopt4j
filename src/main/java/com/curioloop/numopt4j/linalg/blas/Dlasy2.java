/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.abs;
import static java.lang.Math.max;

/**
 * Dlasy2 solves the Sylvester matrix equation where the matrices are of order 1
 * or 2. It computes the unknown n1×n2 matrix X so that
 *
 * <pre>
 * TL*X   + sgn*X*TR  = scale*B  if tranl == false and tranr == false,
 * TLᵀ*X + sgn*X*TR   = scale*B  if tranl == true  and tranr == false,
 * TL*X   + sgn*X*TRᵀ = scale*B  if tranl == false and tranr == true,
 * TLᵀ*X + sgn*X*TRᵀ  = scale*B  if tranl == true  and tranr == true,
 * </pre>
 *
 * <p>where TL is n1×n1, TR is n2×n2, B is n1×n2, and 1 &lt;= n1,n2 &lt;= 2.</p>
 *
 * <p>isgn must be 1 or -1, and n1 and n2 must be 0, 1, or 2.</p>
 *
 * <p>Reference: gonum/lapack/gonum/dlasy2.go</p>
 *
 * <p>x array layout (size &gt;= 6):</p>
 * <ul>
 *   <li>x[0:4] - solution matrix X (max 2x2)</li>
 *   <li>x[4] - scale factor (output)</li>
 *   <li>x[5] - xnorm (output)</li>
 * </ul>
 */
interface Dlasy2 {

    double EPS = Dlamch.dlamch('P');
    double SMLNUM = Dlamch.dlamch('S') / EPS;

    int[] LOC_U12 = {1, 0, 3, 2};
    int[] LOC_L21 = {2, 3, 0, 1};
    int[] LOC_U22 = {3, 2, 1, 0};

    static boolean dlasy2(boolean tranl, boolean tranr, int isgn, int n1, int n2,
                          double[] tl, int tlOff, int ldtl,
                          double[] tr, int trOff, int ldtr,
                          double[] b, int bOff, int ldb,
                          double[] x, int xOff, int ldx) {
        boolean ok = true;
        double scale = 1.0;
        double xnorm = 0.0;

        if (n1 == 0 || n2 == 0) {
            x[xOff + 4] = scale;
            x[xOff + 5] = xnorm;
            return ok;
        }

        double sgn = isgn;

        if (n1 == 1 && n2 == 1) {
            double tau1 = tl[tlOff] + sgn * tr[trOff];
            double bet = abs(tau1);
            if (bet <= SMLNUM) {
                tau1 = SMLNUM;
                bet = SMLNUM;
                ok = false;
            }
            double gam = abs(b[bOff]);
            if (SMLNUM * gam > bet) {
                scale = 1 / gam;
            }
            x[xOff] = b[bOff] * scale / tau1;
            xnorm = abs(x[xOff]);
            x[xOff + 4] = scale;
            x[xOff + 5] = xnorm;
            return ok;
        }

        if (n1 + n2 == 3) {
            double[] tmp = new double[4];
            double smin, btmp0, btmp1;

            if (n1 == 1 && n2 == 2) {
                smin = abs(tl[tlOff]);
                smin = max(smin, max(abs(tr[trOff]), abs(tr[trOff + 1])));
                smin = max(smin, max(abs(tr[trOff + ldtr]), abs(tr[trOff + ldtr + 1])));
                smin = max(EPS * smin, SMLNUM);

                tmp[0] = tl[tlOff] + sgn * tr[trOff];
                tmp[3] = tl[tlOff] + sgn * tr[trOff + ldtr + 1];
                if (tranr) {
                    tmp[1] = sgn * tr[trOff + 1];
                    tmp[2] = sgn * tr[trOff + ldtr];
                } else {
                    tmp[1] = sgn * tr[trOff + ldtr];
                    tmp[2] = sgn * tr[trOff + 1];
                }
                btmp0 = b[bOff];
                btmp1 = b[bOff + 1];
            } else {
                smin = abs(tr[trOff]);
                smin = max(smin, max(abs(tl[tlOff]), abs(tl[tlOff + 1])));
                smin = max(smin, max(abs(tl[tlOff + ldtl]), abs(tl[tlOff + ldtl + 1])));
                smin = max(EPS * smin, SMLNUM);

                tmp[0] = tl[tlOff] + sgn * tr[trOff];
                tmp[3] = tl[tlOff + ldtl + 1] + sgn * tr[trOff];
                if (tranl) {
                    tmp[1] = tl[tlOff + ldtl];
                    tmp[2] = tl[tlOff + 1];
                } else {
                    tmp[1] = tl[tlOff + 1];
                    tmp[2] = tl[tlOff + ldtl];
                }
                btmp0 = b[bOff];
                btmp1 = b[bOff + ldb];
            }

            int ipiv = BLAS.idamax(4, tmp, 0, 1);

            double u11 = tmp[ipiv];
            if (abs(u11) <= smin) {
                ok = false;
                u11 = smin;
            }

            double u12 = tmp[LOC_U12[ipiv]];
            double l21 = tmp[LOC_L21[ipiv]] / u11;
            double u22 = tmp[LOC_U22[ipiv]] - l21 * u12;

            if (abs(u22) <= smin) {
                ok = false;
                u22 = smin;
            }

            if ((ipiv & 2) != 0) {
                double t = btmp0;
                btmp0 = btmp1;
                btmp1 = t - l21 * btmp1;
            } else {
                btmp1 -= l21 * btmp0;
            }

            scale = 1;
            if (2 * SMLNUM * abs(btmp1) > abs(u22) || 2 * SMLNUM * abs(btmp0) > abs(u11)) {
                scale = 0.5 / max(abs(btmp0), abs(btmp1));
                btmp0 *= scale;
                btmp1 *= scale;
            }

            double x22 = btmp1 / u22;
            double x21 = btmp0 / u11 - (u12 / u11) * x22;

            if ((ipiv & 1) != 0) {
                double t = x21;
                x21 = x22;
                x22 = t;
            }

            x[xOff] = x21;
            if (n1 == 1) {
                x[xOff + 1] = x22;
                xnorm = abs(x[xOff]) + abs(x[xOff + 1]);
            } else {
                x[xOff + ldx] = x22;
                xnorm = max(abs(x[xOff]), abs(x[xOff + ldx]));
            }
            x[xOff + 4] = scale;
            x[xOff + 5] = xnorm;
            return ok;
        }

        double smin = max(abs(tr[trOff]), abs(tr[trOff + 1]));
        smin = max(smin, max(abs(tr[trOff + ldtr]), abs(tr[trOff + ldtr + 1])));
        smin = max(smin, max(abs(tl[tlOff]), abs(tl[tlOff + 1])));
        smin = max(smin, max(abs(tl[tlOff + ldtl]), abs(tl[tlOff + ldtl + 1])));
        smin = max(EPS * smin, SMLNUM);

        double[] t = new double[16];
        t[0] = tl[tlOff] + sgn * tr[trOff];
        t[5] = tl[tlOff] + sgn * tr[trOff + ldtr + 1];
        t[10] = tl[tlOff + ldtl + 1] + sgn * tr[trOff];
        t[15] = tl[tlOff + ldtl + 1] + sgn * tr[trOff + ldtr + 1];

        if (tranl) {
            t[2] = tl[tlOff + ldtl];
            t[7] = tl[tlOff + ldtl];
            t[8] = tl[tlOff + 1];
            t[13] = tl[tlOff + 1];
        } else {
            t[2] = tl[tlOff + 1];
            t[7] = tl[tlOff + 1];
            t[8] = tl[tlOff + ldtl];
            t[13] = tl[tlOff + ldtl];
        }

        if (tranr) {
            t[1] = sgn * tr[trOff + 1];
            t[4] = sgn * tr[trOff + ldtr];
            t[11] = sgn * tr[trOff + 1];
            t[14] = sgn * tr[trOff + ldtr];
        } else {
            t[1] = sgn * tr[trOff + ldtr];
            t[4] = sgn * tr[trOff + 1];
            t[11] = sgn * tr[trOff + ldtr];
            t[14] = sgn * tr[trOff + 1];
        }

        double[] btmp = new double[4];
        btmp[0] = b[bOff];
        btmp[1] = b[bOff + 1];
        btmp[2] = b[bOff + ldb];
        btmp[3] = b[bOff + ldb + 1];

        // Compact 4x4 LU decomposition with indirect row/column indexing
        // iRow tracks row permutation, jCol tracks column permutation (no physical swap)
        int[] iRow = {0, 4, 8, 12};
        int[] jCol = {0, 1, 2, 3};

        // Step 0: Find pivot in rows 0-3, columns 0-3
        double xmax = 0;
        int ipsv = 0, jpsv = 0;
        for (int idx = 0; idx < 16; idx++) {
            double tij = abs(t[idx]);
            if (tij >= xmax) {
                xmax = tij;
                ipsv = idx >> 2;
                jpsv = idx & 3;
            }
        }

        // Row swap (swap index)
        if (ipsv != 0) {
            int tmp = iRow[0]; iRow[0] = iRow[ipsv]; iRow[ipsv] = tmp;
            double tmpB = btmp[0]; btmp[0] = btmp[ipsv]; btmp[ipsv] = tmpB;
        }

        // Column swap (swap index only, no physical move)
        if (jpsv != 0) {
            int tmp = jCol[0]; jCol[0] = jCol[jpsv]; jCol[jpsv] = tmp;
        }

        // Check diagonal and eliminate column 0
        int d0 = iRow[0] + jCol[0];
        if (abs(t[d0]) < smin) { ok = false; t[d0] = smin; }
        for (int i = 1; i < 4; i++) {
            int diag = iRow[i] + jCol[0];
            double mult = t[diag] / t[d0];
            for (int j = 1; j < 4; j++) {
                t[iRow[i] + jCol[j]] -= mult * t[iRow[0] + jCol[j]];
            }
            t[diag] = mult;
            btmp[i] -= mult * btmp[0];
        }

        // Step 1: Find pivot in rows 1-3, columns 1-3
        xmax = 0; ipsv = 1; jpsv = 1;
        for (int i = 1; i < 4; i++) {
            for (int j = 1; j < 4; j++) {
                double tij = abs(t[iRow[i] + jCol[j]]);
                if (tij >= xmax) {
                    xmax = tij;
                    ipsv = i;
                    jpsv = j;
                }
            }
        }

        if (ipsv != 1) {
            int tmp = iRow[1]; iRow[1] = iRow[ipsv]; iRow[ipsv] = tmp;
            double tmpB = btmp[1]; btmp[1] = btmp[ipsv]; btmp[ipsv] = tmpB;
        }

        if (jpsv != 1) {
            int tmp = jCol[1]; jCol[1] = jCol[jpsv]; jCol[jpsv] = tmp;
        }

        int d1 = iRow[1] + jCol[1];
        if (abs(t[d1]) < smin) { ok = false; t[d1] = smin; }
        for (int i = 2; i < 4; i++) {
            int diag = iRow[i] + jCol[1];
            double mult = t[diag] / t[d1];
            for (int j = 2; j < 4; j++) {
                t[iRow[i] + jCol[j]] -= mult * t[iRow[1] + jCol[j]];
            }
            t[diag] = mult;
            btmp[i] -= mult * btmp[1];
        }

        // Step 2: Find pivot in rows 2-3, columns 2-3
        xmax = 0; ipsv = 2; jpsv = 2;
        for (int i = 2; i < 4; i++) {
            for (int j = 2; j < 4; j++) {
                double tij = abs(t[iRow[i] + jCol[j]]);
                if (tij >= xmax) {
                    xmax = tij;
                    ipsv = i;
                    jpsv = j;
                }
            }
        }

        if (ipsv != 2) {
            int tmp = iRow[2]; iRow[2] = iRow[3]; iRow[3] = tmp;
            double tmpB = btmp[2]; btmp[2] = btmp[3]; btmp[3] = tmpB;
        }

        if (jpsv != 2) {
            int tmp = jCol[2]; jCol[2] = jCol[jpsv]; jCol[jpsv] = tmp;
        }

        int d2 = iRow[2] + jCol[2];
        if (abs(t[d2]) < smin) { ok = false; t[d2] = smin; }
        int diag3 = iRow[3] + jCol[2];
        double mult = t[diag3] / t[d2];
        t[iRow[3] + jCol[3]] -= mult * t[iRow[2] + jCol[3]];
        t[diag3] = mult;
        btmp[3] -= mult * btmp[2];

        // Step 3: Check final diagonal
        int d3 = iRow[3] + jCol[3];
        if (abs(t[d3]) < smin) { ok = false; t[d3] = smin; }

        if (8 * SMLNUM * abs(btmp[0]) > abs(t[d0]) ||
            8 * SMLNUM * abs(btmp[1]) > abs(t[d1]) ||
            8 * SMLNUM * abs(btmp[2]) > abs(t[d2]) ||
            8 * SMLNUM * abs(btmp[3]) > abs(t[d3])) {

            double maxbtmp = max(max(abs(btmp[0]), abs(btmp[1])), max(abs(btmp[2]), abs(btmp[3])));
            scale = (1.0 / 8.0) / maxbtmp;
            btmp[0] *= scale;
            btmp[1] *= scale;
            btmp[2] *= scale;
            btmp[3] *= scale;
        }

        // Back substitution with indirect indexing
        btmp[3] /= t[d3];
        btmp[2] = (btmp[2] - t[iRow[2] + jCol[3]] * btmp[3]) / t[d2];
        btmp[1] = (btmp[1] - t[iRow[1] + jCol[2]] * btmp[2] - t[iRow[1] + jCol[3]] * btmp[3]) / t[d1];
        btmp[0] = (btmp[0] - t[iRow[0] + jCol[1]] * btmp[1] - t[iRow[0] + jCol[2]] * btmp[2] - t[iRow[0] + jCol[3]] * btmp[3]) / t[d0];

        // Apply column permutations (no allocation)
        double t0 = btmp[0], t1 = btmp[1], t2 = btmp[2], t3 = btmp[3];
        btmp[jCol[0]] = t0;
        btmp[jCol[1]] = t1;
        btmp[jCol[2]] = t2;
        btmp[jCol[3]] = t3;

        x[xOff] = btmp[0];
        x[xOff + 1] = btmp[1];
        x[xOff + ldx] = btmp[2];
        x[xOff + ldx + 1] = btmp[3];
        xnorm = max(abs(btmp[0]) + abs(btmp[1]), abs(btmp[2]) + abs(btmp[3]));

        x[xOff + 4] = scale;
        x[xOff + 5] = xnorm;
        return ok;
    }
}
