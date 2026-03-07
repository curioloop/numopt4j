/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Solves a linear equation or a system of 2 linear equations of the form
 * <pre>
 *   (ca A   - w D) X = scale B  if trans == false,
 *   (ca Aᵀ - w D) X = scale B   if trans == true,
 * </pre>
 * where A is a na×na real matrix, ca is a real scalar, D is a na×na diagonal
 * real matrix, w is a scalar, real if nw == 1, complex if nw == 2, and X and B
 * are na×1 matrices, real if w is real, complex if w is complex.
 *
 *
 * <p>x array layout (size >= 6):</p>
 * <ul>
 *   <li>x[0:4] - solution vector (max 2x2 for complex case)</li>
 *   <li>x[4] - scale factor (output)</li>
 *   <li>x[5] - xnorm (output)</li>
 * </ul>
 */
interface Dlaln2 {

    double SMLNUM = 2 * Dlamch.safmin();
    double BIGNUM = 1.0 / SMLNUM;

    int[] PIVOT = {
            0, 1, 2, 3,
            1, 0, 3, 2,
            2, 3, 0, 1,
            3, 2, 1, 0
    };

    static boolean dlaln2(boolean trans, int na, int nw, double smin, double ca,
                          double[] a, int aOff, int lda, double d1, double d2,
                          double[] b, int bOff, int ldb, double wr, double wi,
                          double[] x, int xOff, int ldx) {
        if (na != 1 && na != 2) {
            throw new IllegalArgumentException("na must be 1 or 2");
        }
        if (nw != 1 && nw != 2) {
            throw new IllegalArgumentException("nw must be 1 or 2");
        }

        double smini = Math.max(smin, SMLNUM);
        boolean ok = true;
        double scale = 1.0;

        if (na == 1) {
            if (nw == 1) {
                double csr = ca * a[aOff] - wr * d1;
                double cnorm = Math.abs(csr);

                if (cnorm < smini) {
                    csr = smini;
                    cnorm = smini;
                    ok = false;
                }

                double bnorm = Math.abs(b[bOff]);
                if (cnorm < 1 && bnorm > Math.max(1, BIGNUM * cnorm)) {
                    scale = 1.0 / bnorm;
                }

                x[xOff] = b[bOff] * scale / csr;
                double xnorm = Math.abs(x[xOff]);

                x[xOff + 4] = scale;
                x[xOff + 5] = xnorm;
                return ok;
            }

            double csr = ca * a[aOff] - wr * d1;
            double csi = -wi * d1;
            double cnorm = Math.abs(csr) + Math.abs(csi);

            if (cnorm < smini) {
                csr = smini;
                csi = 0;
                cnorm = smini;
                ok = false;
            }

            double bnorm = Math.abs(b[bOff]) + Math.abs(b[bOff + 1]);
            if (cnorm < 1 && bnorm > Math.max(1, BIGNUM * cnorm)) {
                scale = 1.0 / bnorm;
            }

            double cxReal = scale * b[bOff];
            double cxImag = scale * b[bOff + 1];
            double denom = csr * csr + csi * csi;
            x[xOff] = (cxReal * csr + cxImag * csi) / denom;
            x[xOff + 1] = (cxImag * csr - cxReal * csi) / denom;
            double xnorm = Math.abs(x[xOff]) + Math.abs(x[xOff + 1]);

            x[xOff + 4] = scale;
            x[xOff + 5] = xnorm;
            return ok;
        }

        double[] crv = {
            ca * a[aOff] - wr * d1,
            ca * a[aOff + 1],
            ca * a[aOff + lda],
            ca * a[aOff + lda + 1] - wr * d2
        };

        if (trans) {
            double tmp = crv[1];
            crv[1] = crv[2];
            crv[2] = tmp;
        }

        if (nw == 1) {
            double cmax = 0;
            int icmax = 0;
            for (int j = 0; j < 4; j++) {
                double v = Math.abs(crv[j]);
                if (v > cmax) {
                    cmax = v;
                    icmax = j;
                }
            }

            if (cmax < smini) {
                double bnorm = Math.max(Math.abs(b[bOff]), Math.abs(b[bOff + ldb]));
                if (smini < 1 && bnorm > Math.max(1, BIGNUM * smini)) {
                    scale = 1.0 / bnorm;
                }
                double temp = scale / smini;
                x[xOff] = temp * b[bOff];
                x[xOff + ldx] = temp * b[bOff + ldb];
                double xnorm = temp * bnorm;
                x[xOff + 4] = scale;
                x[xOff + 5] = xnorm;
                return false;
            }

            int pivRow = icmax << 2;
            double ur11 = crv[icmax];
            double ur12 = crv[PIVOT[pivRow + 1]];
            double cr21 = crv[PIVOT[pivRow + 2]];
            double cr22 = crv[PIVOT[pivRow + 3]];
            double ur11r = 1.0 / ur11;
            double lr21 = ur11r * cr21;
            double ur22 = cr22 - ur12 * lr21;

            if (Math.abs(ur22) < smini) {
                ur22 = smini;
                ok = false;
            }

            double br1, br2;
            if (icmax > 1) {
                br1 = b[bOff + ldb];
                br2 = b[bOff];
            } else {
                br1 = b[bOff];
                br2 = b[bOff + ldb];
            }
            br2 -= lr21 * br1;

            double bbnd = Math.max(Math.abs(ur22 * ur11r * br1), Math.abs(br2));
            if (bbnd > 1 && Math.abs(ur22) < 1 && bbnd >= BIGNUM * Math.abs(ur22)) {
                scale = 1.0 / bbnd;
            }

            double xr2 = br2 * scale / ur22;
            double xr1 = scale * br1 * ur11r - ur11r * ur12 * xr2;

            if ((icmax & 1) != 0) {
                x[xOff] = xr2;
                x[xOff + ldx] = xr1;
            } else {
                x[xOff] = xr1;
                x[xOff + ldx] = xr2;
            }
            double xnorm = Math.max(Math.abs(xr1), Math.abs(xr2));

            if (xnorm > 1 && cmax > 1 && xnorm > BIGNUM / cmax) {
                double temp = cmax / BIGNUM;
                x[xOff] *= temp;
                x[xOff + ldx] *= temp;
                xnorm *= temp;
                scale *= temp;
            }

            x[xOff + 4] = scale;
            x[xOff + 5] = xnorm;
            return ok;
        }

        double[] civ = {
            -wi * d1,
            0,
            0,
            -wi * d2
        };

        double cmax = 0;
        int icmax = 0;
        for (int j = 0; j < 4; j++) {
            double v = Math.abs(crv[j]) + Math.abs(civ[j]);
            if (v > cmax) {
                cmax = v;
                icmax = j;
            }
        }

        if (cmax < smini) {
            double br1 = Math.abs(b[bOff]) + Math.abs(b[bOff + 1]);
            double br2 = Math.abs(b[bOff + ldb]) + Math.abs(b[bOff + ldb + 1]);
            double bnorm = Math.max(br1, br2);
            if (smini < 1 && bnorm > 1 && bnorm > BIGNUM * smini) {
                scale = 1.0 / bnorm;
            }
            double temp = scale / smini;
            x[xOff] = temp * b[bOff];
            x[xOff + 1] = temp * b[bOff + 1];
            x[xOff + ldx] = temp * b[bOff + ldb];
            x[xOff + ldx + 1] = temp * b[bOff + ldb + 1];
            double xnorm = temp * bnorm;
            x[xOff + 4] = scale;
            x[xOff + 5] = xnorm;
            return false;
        }

        int pivRow = icmax << 2;
        double ur11 = crv[icmax];
        double ui11 = civ[icmax];
        double ur12 = crv[PIVOT[pivRow + 1]];
        double ui12 = civ[PIVOT[pivRow + 1]];
        double cr21 = crv[PIVOT[pivRow + 2]];
        double ci21 = civ[PIVOT[pivRow + 2]];
        double cr22 = crv[PIVOT[pivRow + 3]];
        double ci22 = civ[PIVOT[pivRow + 3]];

        double ur11r, ui11r;
        double lr21, li21;
        double ur12s, ui12s;
        double ur22, ui22;

        if (icmax == 0 || icmax == 3) {
            if (Math.abs(ur11) > Math.abs(ui11)) {
                double temp = ui11 / ur11;
                ur11r = 1.0 / (ur11 * (1 + temp * temp));
                ui11r = -temp * ur11r;
            } else {
                double temp = ur11 / ui11;
                ui11r = -1.0 / (ui11 * (1 + temp * temp));
                ur11r = -temp * ui11r;
            }
            lr21 = cr21 * ur11r;
            li21 = cr21 * ui11r;
            ur12s = ur12 * ur11r;
            ui12s = ur12 * ui11r;
            ur22 = cr22 - ur12 * lr21;
            ui22 = ci22 - ur12 * li21;
        } else {
            ur11r = 1.0 / ur11;
            ui11r = 0;
            lr21 = cr21 * ur11r;
            li21 = ci21 * ur11r;
            ur12s = ur12 * ur11r;
            ui12s = ui12 * ur11r;
            ur22 = cr22 - ur12 * lr21 + ui12 * li21;
            ui22 = -ur12 * li21 - ui12 * lr21;
        }

        double u22abs = Math.abs(ur22) + Math.abs(ui22);

        if (u22abs < smini) {
            ur22 = smini;
            ui22 = 0;
            ok = false;
        }

        double br1, bi1, br2, bi2;
        if (icmax > 1) {
            br1 = b[bOff + ldb];
            bi1 = b[bOff + ldb + 1];
            br2 = b[bOff];
            bi2 = b[bOff + 1];
        } else {
            br1 = b[bOff];
            bi1 = b[bOff + 1];
            br2 = b[bOff + ldb];
            bi2 = b[bOff + ldb + 1];
        }
        br2 += -lr21 * br1 + li21 * bi1;
        bi2 += -li21 * br1 - lr21 * bi1;

        double bbnd1 = u22abs * (Math.abs(ur11r) + Math.abs(ui11r)) * (Math.abs(br1) + Math.abs(bi1));
        double bbnd2 = Math.abs(br2) + Math.abs(bi2);
        double bbnd = Math.max(bbnd1, bbnd2);
        if (bbnd > 1 && u22abs < 1 && bbnd >= BIGNUM * u22abs) {
            scale = 1.0 / bbnd;
            br1 *= scale;
            bi1 *= scale;
            br2 *= scale;
            bi2 *= scale;
        }

        double denom = ur22 * ur22 + ui22 * ui22;
        double xr2 = (br2 * ur22 + bi2 * ui22) / denom;
        double xi2 = (bi2 * ur22 - br2 * ui22) / denom;
        double xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
        double xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;

        if ((icmax & 1) != 0) {
            x[xOff] = xr2;
            x[xOff + 1] = xi2;
            x[xOff + ldx] = xr1;
            x[xOff + ldx + 1] = xi1;
        } else {
            x[xOff] = xr1;
            x[xOff + 1] = xi1;
            x[xOff + ldx] = xr2;
            x[xOff + ldx + 1] = xi2;
        }
        double xnorm = Math.max(Math.abs(xr1) + Math.abs(xi1), Math.abs(xr2) + Math.abs(xi2));

        if (xnorm > 1 && cmax > 1 && xnorm > BIGNUM / cmax) {
            double temp = cmax / BIGNUM;
            x[xOff] *= temp;
            x[xOff + 1] *= temp;
            x[xOff + ldx] *= temp;
            x[xOff + ldx + 1] *= temp;
            xnorm *= temp;
            scale *= temp;
        }

        x[xOff + 4] = scale;
        x[xOff + 5] = xnorm;
        return ok;
    }
}
