/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

interface Dbdsqr {

    /**
     * Computes the singular value decomposition (SVD) of a real n×n bidiagonal matrix B.
     *
     * <p>Computes B = Q * S * Pᵀ where S is diagonal with singular values in decreasing order,
     * Q is orthogonal (left singular vectors), and P is orthogonal (right singular vectors).
     * Optionally computes U*Q, Pᵀ*VT, and Qᵀ*C.
     *
     *
     * @param uplo  blas.Upper if B is upper bidiagonal, blas.Lower if lower bidiagonal
     * @param n     order of the bidiagonal matrix B
     * @param ncvt  number of columns of VT (right singular vectors)
     * @param nru   number of rows of U (left singular vectors)
     * @param ncc   number of columns of C
     * @param d     diagonal elements of B (length n, overwritten with singular values)
     * @param dOff  offset into d
     * @param e     off-diagonal elements of B (length n-1, overwritten)
     * @param eOff  offset into e
     * @param vt    matrix VT (nru×n, row-major); updated to Pᵀ*VT on exit
     * @param vtOff offset into vt
     * @param ldvt  leading dimension of vt
     * @param u     matrix U (nru×n, row-major); updated to U*Q on exit
     * @param uOff  offset into u
     * @param ldu   leading dimension of u
     * @param c     matrix C (n×ncc, row-major); updated to Qᵀ*C on exit
     * @param cOff  offset into c
     * @param ldc   leading dimension of c
     * @param work  workspace (length >= 4*n)
     * @param workOff offset into work
     * @return true on success, false if convergence failure
     */
    static boolean dbdsqr(BLAS.Uplo uplo, int n, int ncvt, int nru, int ncc,
                          double[] d, int dOff, double[] e, int eOff,
                          double[] vt, int vtOff, int ldvt,
                          double[] u, int uOff, int ldu,
                          double[] c, int cOff, int ldc,
                          double[] work, int workOff) {
        if (n == 0) {
            return true;
        }

        final int maxIter = 6;
        double eps = Dlamch.EPSILON;
        double unfl = Dlamch.SAFE_MIN;

        int nm1 = n - 1;
        int nm12 = nm1 + nm1;
        int nm13 = nm12 + nm1;
        int idir = 0;

        // Netlib keeps these as local scalars; tmp is only needed because the
        // Java helper ports expose array-based outputs.
        double[] tmp = new double[6];

        if (n > 1 && ncvt == 0 && nru == 0 && ncc == 0) {
            int info = Dlasq1.dlasq1(n, d, dOff, e, eOff, work, workOff);
            if (info != 2) {
                return info == 0;
            }
        }

        boolean lower = uplo == BLAS.Uplo.Lower;
        if (lower) {
            for (int i = 0; i < n - 1; i++) {
                Dlartg.dlartg(d[dOff + i], e[eOff + i], tmp, 0);
                double cs = tmp[0], sn = tmp[1], r = tmp[2];
                d[dOff + i] = r;
                e[eOff + i] = sn * d[dOff + i + 1];
                d[dOff + i + 1] *= cs;
                work[workOff + i] = cs;
                work[workOff + nm1 + i] = sn;
            }
            if (nru > 0) {
                Dlasr.dlasr(BLAS.Side.Right, 'V', 'F', nru, n, work, workOff, work, workOff + nm1, u, uOff, ldu);
            }
            if (ncc > 0) {
                Dlasr.dlasr(BLAS.Side.Left, 'V', 'F', n, ncc, work, workOff, work, workOff + nm1, c, cOff, ldc);
            }
        }

        double tolmul = max(10, min(100, pow(eps, -1.0 / 8)));
        double tol = tolmul * eps;

        double smax = 0;
        for (int i = 0; i < n; i++) {
            smax = max(smax, abs(d[dOff + i]));
        }
        for (int i = 0; i < n - 1; i++) {
            smax = max(smax, abs(e[eOff + i]));
        }

        double smin = 0, thresh;
        if (tol >= 0) {
            double sminoa = abs(d[dOff]);
            if (sminoa != 0) {
                double mu = sminoa;
                for (int i = 1; i < n; i++) {
                    mu = abs(d[dOff + i]) * (mu / (mu + abs(e[eOff + i - 1])));
                    sminoa = min(sminoa, mu);
                    if (sminoa == 0) break;
                }
            }
            sminoa = sminoa / sqrt(n);
            thresh = max(tol * sminoa, maxIter * n * n * unfl);
        } else {
            thresh = max(abs(tol) * smax, maxIter * n * n * unfl);
        }

        int maxIt = maxIter * n * n;
        int iter = 0;
        int oldl2 = -1;
        int oldm = -1;
        int m = n;

        outer:
        while (m > 1) {
            if (iter > maxIt) {
                int info = 0;
                for (int i = 0; i < n - 1; i++) {
                    if (e[eOff + i] != 0) info++;
                }
                return info == 0;
            }

            if (tol < 0 && abs(d[dOff + m - 1]) <= thresh) {
                d[dOff + m - 1] = 0;
            }

            smax = abs(d[dOff + m - 1]);
            int l2 = 0;
            boolean broke = false;
            for (int l3 = 0; l3 < m - 1; l3++) {
                l2 = m - l3 - 2;
                double abss = abs(d[dOff + l2]);
                double abse = abs(e[eOff + l2]);
                if (tol < 0 && abss <= thresh) {
                    d[dOff + l2] = 0;
                }
                if (abse <= thresh) {
                    broke = true;
                    break;
                }
                smax = max(max(smax, abss), abse);
            }

            if (broke) {
                e[eOff + l2] = 0;
                if (l2 == m - 2) {
                    m--;
                    continue;
                }
                l2++;
            } else {
                l2 = 0;
            }

            if (l2 == m - 2) {
                Dlas2.dlasv2(d[dOff + m - 2], e[eOff + m - 2], d[dOff + m - 1], tmp, 0);
                double ssmin = tmp[0], ssmax = tmp[1];
                double sinr = tmp[2], cosr = tmp[3], sinl = tmp[4], cosl = tmp[5];
                d[dOff + m - 1] = ssmin;
                d[dOff + m - 2] = ssmax;
                e[eOff + m - 2] = 0;
                if (ncvt > 0) {
                    BLAS.drot(ncvt, vt, vtOff + (m - 2) * ldvt, 1, vt, vtOff + (m - 1) * ldvt, 1, cosr, sinr);
                }
                if (nru > 0) {
                    BLAS.drot(nru, u, uOff + m - 2, ldu, u, uOff + m - 1, ldu, cosl, sinl);
                }
                if (ncc > 0) {
                    BLAS.drot(ncc, c, cOff + (m - 2) * ldc, 1, c, cOff + (m - 1) * ldc, 1, cosl, sinl);
                }
                m -= 2;
                continue;
            }

            if (l2 > oldm - 1 || m - 1 < oldl2) {
                if (abs(d[dOff + l2]) >= abs(d[dOff + m - 1])) {
                    idir = 1;
                } else {
                    idir = 2;
                }
            }

            if (idir == 1) {
                if (abs(e[eOff + m - 2]) <= abs(tol) * abs(d[dOff + m - 1]) || (tol < 0 && abs(e[eOff + m - 2]) <= thresh)) {
                    e[eOff + m - 2] = 0;
                    continue;
                }
                if (tol >= 0) {
                    double mu = abs(d[dOff + l2]);
                    smin = mu;
                    for (int l3 = l2; l3 < m - 1; l3++) {
                        if (abs(e[eOff + l3]) <= tol * mu) {
                            e[eOff + l3] = 0;
                            continue outer;
                        }
                        mu = abs(d[dOff + l3 + 1]) * (mu / (mu + abs(e[eOff + l3])));
                        smin = min(smin, mu);
                    }
                }
            } else {
                if (abs(e[eOff + l2]) <= abs(tol) * abs(d[dOff + l2]) || (tol < 0 && abs(e[eOff + l2]) <= thresh)) {
                    e[eOff + l2] = 0;
                    continue;
                }
                if (tol >= 0) {
                    double mu = abs(d[dOff + m - 1]);
                    smin = mu;
                    for (int l3 = m - 2; l3 >= l2; l3--) {
                        if (abs(e[eOff + l3]) <= tol * mu) {
                            e[eOff + l3] = 0;
                            continue outer;
                        }
                        mu = abs(d[dOff + l3]) * (mu / (mu + abs(e[eOff + l3])));
                        smin = min(smin, mu);
                    }
                }
            }

            oldl2 = l2;
            oldm = m;

            double shift;
            if (tol >= 0 && n * tol * (smin / smax) <= max(eps, 0.01 * tol)) {
                shift = 0;
            } else {
                double sl2;
                if (idir == 1) {
                    sl2 = abs(d[dOff + l2]);
                    Dlas2.dlas2(d[dOff + m - 2], e[eOff + m - 2], d[dOff + m - 1], tmp, 0);
                    shift = tmp[0];
                } else {
                    sl2 = abs(d[dOff + m - 1]);
                    Dlas2.dlas2(d[dOff + l2], e[eOff + l2], d[dOff + l2 + 1], tmp, 0);
                    shift = tmp[0];
                }
                if (sl2 > 0 && (shift / sl2) * (shift / sl2) < eps) {
                    shift = 0;
                }
            }

            iter += m - l2 + 1;

            if (shift == 0) {
                if (idir == 1) {
                    double cs = 1, oldcs = 1;
                    double sn = 0, r = 0, oldsn = 0;
                    for (int i = l2; i < m - 1; i++) {
                        Dlartg.dlartg(d[dOff + i] * cs, e[eOff + i], tmp, 0);
                        cs = tmp[0]; sn = tmp[1]; r = tmp[2];
                        if (i > l2) {
                            e[eOff + i - 1] = oldsn * r;
                        }
                        Dlartg.dlartg(oldcs * r, d[dOff + i + 1] * sn, tmp, 0);
                        oldcs = tmp[0]; oldsn = tmp[1]; d[dOff + i] = tmp[2];
                        work[workOff + i - l2] = cs;
                        work[workOff + nm1 + i - l2] = sn;
                        work[workOff + nm12 + i - l2] = oldcs;
                        work[workOff + nm13 + i - l2] = oldsn;
                    }
                    double h = d[dOff + m - 1] * cs;
                    d[dOff + m - 1] = h * oldcs;
                    e[eOff + m - 2] = h * oldsn;
                    if (ncvt > 0) {
                        Dlasr.dlasr(BLAS.Side.Left, 'V', 'F', m - l2, ncvt, work, workOff, work, workOff + nm1, vt, vtOff + l2 * ldvt, ldvt);
                    }
                    if (nru > 0) {
                        Dlasr.dlasr(BLAS.Side.Right, 'V', 'F', nru, m - l2, work, workOff + nm12, work, workOff + nm13, u, uOff + l2, ldu);
                    }
                    if (ncc > 0) {
                        Dlasr.dlasr(BLAS.Side.Left, 'V', 'F', m - l2, ncc, work, workOff + nm12, work, workOff + nm13, c, cOff + l2 * ldc, ldc);
                    }
                    if (abs(e[eOff + m - 2]) <= thresh) {
                        e[eOff + m - 2] = 0;
                    }
                } else {
                    double cs = 1, oldcs = 1;
                    double sn = 0, r = 0, oldsn = 0;
                    for (int i = m - 1; i >= l2 + 1; i--) {
                        Dlartg.dlartg(d[dOff + i] * cs, e[eOff + i - 1], tmp, 0);
                        cs = tmp[0]; sn = tmp[1]; r = tmp[2];
                        if (i < m - 1) {
                            e[eOff + i] = oldsn * r;
                        }
                        Dlartg.dlartg(oldcs * r, d[dOff + i - 1] * sn, tmp, 0);
                        oldcs = tmp[0]; oldsn = tmp[1]; d[dOff + i] = tmp[2];
                        work[workOff + i - l2 - 1] = cs;
                        work[workOff + nm1 + i - l2 - 1] = -sn;
                        work[workOff + nm12 + i - l2 - 1] = oldcs;
                        work[workOff + nm13 + i - l2 - 1] = -oldsn;
                    }
                    double h = d[dOff + l2] * cs;
                    d[dOff + l2] = h * oldcs;
                    e[eOff + l2] = h * oldsn;
                    if (ncvt > 0) {
                        Dlasr.dlasr(BLAS.Side.Left, 'V', 'B', m - l2, ncvt, work, workOff + nm12, work, workOff + nm13, vt, vtOff + l2 * ldvt, ldvt);
                    }
                    if (nru > 0) {
                        Dlasr.dlasr(BLAS.Side.Right, 'V', 'B', nru, m - l2, work, workOff, work, workOff + nm1, u, uOff + l2, ldu);
                    }
                    if (ncc > 0) {
                        Dlasr.dlasr(BLAS.Side.Left, 'V', 'B', m - l2, ncc, work, workOff, work, workOff + nm1, c, cOff + l2 * ldc, ldc);
                    }
                    if (abs(e[eOff + l2]) <= thresh) {
                        e[eOff + l2] = 0;
                    }
                }
            } else {
                if (idir == 1) {
                    double f = (abs(d[dOff + l2]) - shift) * (copySign(1, d[dOff + l2]) + shift / d[dOff + l2]);
                    double g = e[eOff + l2];
                    double cosl = 0, sinl = 0, r = 0;
                    for (int i = l2; i < m - 1; i++) {
                        Dlartg.dlartg(f, g, tmp, 0);
                        double cosr = tmp[0], sinr = tmp[1];
                        r = tmp[2];
                        if (i > l2) {
                            e[eOff + i - 1] = r;
                        }
                        f = cosr * d[dOff + i] + sinr * e[eOff + i];
                        e[eOff + i] = cosr * e[eOff + i] - sinr * d[dOff + i];
                        g = sinr * d[dOff + i + 1];
                        d[dOff + i + 1] *= cosr;
                        Dlartg.dlartg(f, g, tmp, 0);
                        cosl = tmp[0]; sinl = tmp[1]; r = tmp[2];
                        d[dOff + i] = r;
                        f = cosl * e[eOff + i] + sinl * d[dOff + i + 1];
                        d[dOff + i + 1] = cosl * d[dOff + i + 1] - sinl * e[eOff + i];
                        if (i < m - 2) {
                            g = sinl * e[eOff + i + 1];
                            e[eOff + i + 1] = cosl * e[eOff + i + 1];
                        }
                        work[workOff + i - l2] = cosr;
                        work[workOff + nm1 + i - l2] = sinr;
                        work[workOff + nm12 + i - l2] = cosl;
                        work[workOff + nm13 + i - l2] = sinl;
                    }
                    e[eOff + m - 2] = f;
                    if (ncvt > 0) {
                        Dlasr.dlasr(BLAS.Side.Left, 'V', 'F', m - l2, ncvt, work, workOff, work, workOff + nm1, vt, vtOff + l2 * ldvt, ldvt);
                    }
                    if (nru > 0) {
                        Dlasr.dlasr(BLAS.Side.Right, 'V', 'F', nru, m - l2, work, workOff + nm12, work, workOff + nm13, u, uOff + l2, ldu);
                    }
                    if (ncc > 0) {
                        Dlasr.dlasr(BLAS.Side.Left, 'V', 'F', m - l2, ncc, work, workOff + nm12, work, workOff + nm13, c, cOff + l2 * ldc, ldc);
                    }
                    if (abs(e[eOff + m - 2]) <= thresh) {
                        e[eOff + m - 2] = 0;
                    }
                } else {
                    double f = (abs(d[dOff + m - 1]) - shift) * (copySign(1, d[dOff + m - 1]) + shift / d[dOff + m - 1]);
                    double g = e[eOff + m - 2];
                    for (int i = m - 1; i > l2; i--) {
                        Dlartg.dlartg(f, g, tmp, 0);
                        double cosr = tmp[0], sinr = tmp[1], r = tmp[2];
                        if (i < m - 1) {
                            e[eOff + i] = r;
                        }
                        f = cosr * d[dOff + i] + sinr * e[eOff + i - 1];
                        e[eOff + i - 1] = cosr * e[eOff + i - 1] - sinr * d[dOff + i];
                        g = sinr * d[dOff + i - 1];
                        d[dOff + i - 1] *= cosr;
                        Dlartg.dlartg(f, g, tmp, 0);
                        double cosl = tmp[0], sinl = tmp[1];
                        r = tmp[2];
                        d[dOff + i] = r;
                        f = cosl * e[eOff + i - 1] + sinl * d[dOff + i - 1];
                        d[dOff + i - 1] = cosl * d[dOff + i - 1] - sinl * e[eOff + i - 1];
                        if (i > l2 + 1) {
                            g = sinl * e[eOff + i - 2];
                            e[eOff + i - 2] *= cosl;
                        }
                        work[workOff + i - l2 - 1] = cosr;
                        work[workOff + nm1 + i - l2 - 1] = -sinr;
                        work[workOff + nm12 + i - l2 - 1] = cosl;
                        work[workOff + nm13 + i - l2 - 1] = -sinl;
                    }
                    e[eOff + l2] = f;
                    if (abs(e[eOff + l2]) <= thresh) {
                        e[eOff + l2] = 0;
                    }
                    if (ncvt > 0) {
                        Dlasr.dlasr(BLAS.Side.Left, 'V', 'B', m - l2, ncvt, work, workOff + nm12, work, workOff + nm13, vt, vtOff + l2 * ldvt, ldvt);
                    }
                    if (nru > 0) {
                        Dlasr.dlasr(BLAS.Side.Right, 'V', 'B', nru, m - l2, work, workOff, work, workOff + nm1, u, uOff + l2, ldu);
                    }
                    if (ncc > 0) {
                        Dlasr.dlasr(BLAS.Side.Left, 'V', 'B', m - l2, ncc, work, workOff, work, workOff + nm1, c, cOff + l2 * ldc, ldc);
                    }
                }
            }
        }

        for (int i = 0; i < n; i++) {
            if (d[dOff + i] < 0) {
                d[dOff + i] *= -1;
                if (ncvt > 0) {
                    BLAS.dscal(ncvt, -1, vt, vtOff + i * ldvt, 1);
                }
            }
        }

        for (int i = 0; i < n - 1; i++) {
            int isub = 0;
            double sminVal = d[dOff];
            for (int j = 1; j < n - i; j++) {
                if (d[dOff + j] <= sminVal) {
                    isub = j;
                    sminVal = d[dOff + j];
                }
            }
            if (isub != n - i - 1) {
                d[dOff + isub] = d[dOff + n - i - 1];
                d[dOff + n - i - 1] = sminVal;
                if (ncvt > 0) {
                    BLAS.dswap(ncvt, vt, vtOff + isub * ldvt, 1, vt, vtOff + (n - i - 1) * ldvt, 1);
                }
                if (nru > 0) {
                    BLAS.dswap(nru, u, uOff + isub, ldu, u, uOff + n - i - 1, ldu);
                }
                if (ncc > 0) {
                    BLAS.dswap(ncc, c, cOff + isub * ldc, 1, c, cOff + (n - i - 1) * ldc, 1);
                }
            }
        }

        int info = 0;
        for (int i = 0; i < n - 1; i++) {
            if (e[eOff + i] != 0) info++;
        }
        return info == 0;
    }
}
