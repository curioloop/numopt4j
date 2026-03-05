/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Computes some or all of the right and/or left eigenvectors of an n×n
 * upper quasi-triangular matrix T in Schur canonical form.
 *
 * <p>Reference: gonum/lapack/gonum/dtrevc3.go</p>
 */
interface Dtrevc3 {

    int NBMIN = 8;
    int NBMAX = 128;

    static int dtrevc3Query(char side, char howmny, int n) {
        if (n == 0) {
            return 1;
        }
        int nb = Ilaenv.ilaenv(1, "DTREVC", String.valueOf(side) + howmny, n, -1, -1, -1);
        return n + 2 * n * nb;
    }

    static int dtrevc3(char side, char howmny, boolean[] selected, int n,
                              double[] T, int tOff, int ldt,
                              double[] VL, int vlOff, int ldvl,
                              double[] VR, int vrOff, int ldvr,
                              int mm, double[] work, int workOff, int lwork) {
        boolean bothv = side == 'B';
        boolean rightv = side == 'R' || bothv;
        boolean leftv = side == 'L' || bothv;

        boolean allVec = howmny == 'A';
        boolean allMulQ = howmny == 'B';

        if (n == 0) {
            work[workOff] = 1;
            return 0;
        }

        int m;
        if (howmny == 'S') {
            if (selected == null || selected.length != n) {
                throw new IllegalArgumentException("selected must have length n");
            }
            m = 0;
            for (int j = 0; j < n; ) {
                if (j == n - 1 || T[tOff + (j + 1) * ldt + j] == 0) {
                    if (selected[j]) m++;
                    j++;
                } else {
                    if (selected[j] || selected[j + 1]) {
                        selected[j] = true;
                        selected[j + 1] = false;
                        m += 2;
                    }
                    j += 2;
                }
            }
        } else {
            m = n;
        }

        if (mm < m) {
            throw new IllegalArgumentException("mm < m");
        }

        int nb = Ilaenv.ilaenv(1, "DTREVC", String.valueOf(side) + howmny, n, -1, -1, -1);
        if (lwork == -1) {
            work[workOff] = n + 2 * n * nb;
            return m;
        }

        if (m == 0) {
            return 0;
        }

        if (howmny == 'B' && lwork >= n + 2 * n * NBMIN) {
            nb = Math.min((lwork - n) / (2 * n), NBMAX);
            Dlamv.dlaset('A', n, 1 + 2 * nb, 0, 0, work, workOff, 1 + 2 * nb);
        } else {
            nb = 1;
        }

        double ulp = Dlamch.eps();
        double smlnum = (double) n / ulp * Dlamch.safmin();
        double bignum = (1 - ulp) / smlnum;

        double[] norms = work;
        int ldb = 2 * nb;
        int bOff = workOff + n;

        norms[0] = 0;
        for (int j = 1; j < n; j++) {
            double cn = 0;
            for (int i = 0; i < j; i++) {
                cn += Math.abs(T[tOff + i * ldt + j]);
            }
            norms[j] = cn;
        }

        double[] x = new double[4+2];
        byte[] iscomplex = new byte[NBMAX];

        int iv = Math.max(2, nb) - 1;
        int ip = 0;
        int is = m - 1;

        if (rightv) {
            for (int ki = n - 1; ki >= 0; ki--) {
                if (ip == -1) {
                    ip = 1;
                    continue;
                }

                if (ki == 0 || T[tOff + ki * ldt + ki - 1] == 0) {
                    ip = 0;
                } else {
                    ip = -1;
                }

                if (howmny == 'S') {
                    if (ip == 0) {
                        if (!selected[ki]) continue;
                    } else if (!selected[ki - 1]) {
                        continue;
                    }
            }

            double wr = T[tOff + ki * ldt + ki];
            double wi = 0;
            if (ip != 0) {
                wi = Math.sqrt(Math.abs(T[tOff + ki * ldt + ki - 1])) *
                     Math.sqrt(Math.abs(T[tOff + (ki - 1) * ldt + ki]));
            }
            double smin = Math.max(ulp * (Math.abs(wr) + Math.abs(wi)), smlnum);

            if (ip == 0) {
                work[bOff + ki * ldb + iv] = 1;
                for (int k = 0; k < ki; k++) {
                    work[bOff + k * ldb + iv] = -T[tOff + k * ldt + ki];
                }

                for (int j = ki - 1; j >= 0; ) {
                    if (j == 0 || T[tOff + j * ldt + j - 1] == 0) {
                        Dlaln2.dlaln2(false, 1, 1, smin, 1,
                                T, tOff + j * ldt + j, ldt, 1, 1,
                                work, bOff + j * ldb + iv, ldb, wr, 0, x, 0, 2);
                        double scale = x[4];
                        double xnorm = x[5];

                        if (xnorm > 1 && norms[j] > bignum / xnorm) {
                            x[0] /= xnorm;
                            scale /= xnorm;
                        }
                        if (scale != 1) {
                            for (int k = 0; k <= ki; k++) {
                                work[bOff + k * ldb + iv] *= scale;
                            }
                        }
                        work[bOff + j * ldb + iv] = x[0];
                        for (int k = 0; k < j; k++) {
                            work[bOff + k * ldb + iv] -= x[0] * T[tOff + k * ldt + j];
                        }
                        j--;
                    } else {
                        Dlaln2.dlaln2(false, 2, 1, smin, 1,
                                T, tOff + (j - 1) * ldt + j - 1, ldt, 1, 1,
                                work, bOff + (j - 1) * ldb + iv, ldb, wr, 0, x, 0, 2);
                        double scale = x[4];
                        double xnorm = x[5];

                        if (xnorm > 1) {
                            double beta = Math.max(norms[j - 1], norms[j]);
                            if (beta > bignum / xnorm) {
                                x[0] /= xnorm;
                                x[2] /= xnorm;
                                scale /= xnorm;
                            }
                        }
                        if (scale != 1) {
                            for (int k = 0; k <= ki; k++) {
                                work[bOff + k * ldb + iv] *= scale;
                            }
                        }
                        work[bOff + (j - 1) * ldb + iv] = x[0];
                        work[bOff + j * ldb + iv] = x[2];
                        for (int k = 0; k < j - 1; k++) {
                            work[bOff + k * ldb + iv] -= x[0] * T[tOff + k * ldt + j - 1];
                            work[bOff + k * ldb + iv] -= x[2] * T[tOff + k * ldt + j];
                        }
                        j -= 2;
                    }
                }

                if (howmny != 'B') {
                    for (int k = 0; k <= ki; k++) {
                        VR[vrOff + k * ldvr + is] = work[bOff + k * ldb + iv];
                    }
                    int ii = 0;
                    double maxVal = Math.abs(VR[vrOff + ldvr + is]);
                    for (int k = 1; k <= ki; k++) {
                        double v = Math.abs(VR[vrOff + k * ldvr + is]);
                        if (v > maxVal) {
                            maxVal = v;
                            ii = k;
                        }
                    }
                    double remax = 1.0 / Math.abs(VR[vrOff + ii * ldvr + is]);
                    for (int k = 0; k <= ki; k++) {
                        VR[vrOff + k * ldvr + is] *= remax;
                    }
                    for (int k = ki + 1; k < n; k++) {
                        VR[vrOff + k * ldvr + is] = 0;
                    }
                } else if (nb == 1) {
                    if (ki > 0) {
                        BLAS.dgemv(BLAS.Transpose.NoTrans, n, ki, 1, VR, vrOff, ldvr,
                                work, bOff + iv, ldb, work[bOff + ki * ldb + iv],
                                VR, vrOff + ki, ldvr);
                    }
                    int ii = 0;
                    double maxVal = Math.abs(VR[vrOff + ldvr + ki]);
                    for (int k = 1; k < n; k++) {
                        double v = Math.abs(VR[vrOff + k * ldvr + ki]);
                        if (v > maxVal) {
                            maxVal = v;
                            ii = k;
                        }
                    }
                    double remax = 1.0 / Math.abs(VR[vrOff + ii * ldvr + ki]);
                    BLAS.dscal(n, remax, VR, vrOff + ki, ldvr);
                } else {
                    for (int k = ki + 1; k < n; k++) {
                        work[bOff + k * ldb + iv] = 0;
                    }
                    iscomplex[iv] = (byte) ip;
                }
            } else {
                if (Math.abs(T[tOff + (ki - 1) * ldt + ki]) >= Math.abs(T[tOff + ki * ldt + ki - 1])) {
                    work[bOff + (ki - 1) * ldb + iv - 1] = 1;
                    work[bOff + ki * ldb + iv] = wi / T[tOff + (ki - 1) * ldt + ki];
                } else {
                    work[bOff + (ki - 1) * ldb + iv - 1] = -wi / T[tOff + ki * ldt + ki - 1];
                    work[bOff + ki * ldb + iv] = 1;
                }
                work[bOff + ki * ldb + iv - 1] = 0;
                work[bOff + (ki - 1) * ldb + iv] = 0;

                for (int k = 0; k < ki - 1; k++) {
                    work[bOff + k * ldb + iv - 1] = -work[bOff + (ki - 1) * ldb + iv - 1] * T[tOff + k * ldt + ki - 1];
                    work[bOff + k * ldb + iv] = -work[bOff + ki * ldb + iv] * T[tOff + k * ldt + ki];
                }

                for (int j = ki - 2; j >= 0; ) {
                    if (j == 0 || T[tOff + j * ldt + j - 1] == 0) {
                        Dlaln2.dlaln2(false, 1, 2, smin, 1,
                                T, tOff + j * ldt + j, ldt, 1, 1,
                                work, bOff + j * ldb + iv - 1, ldb, wr, wi, x, 0, 2);
                        double scale = x[4];
                        double xnorm = x[5];

                        if (xnorm > 1 && norms[j] > bignum / xnorm) {
                            x[0] /= xnorm;
                            x[1] /= xnorm;
                            scale /= xnorm;
                        }
                        if (scale != 1) {
                            for (int k = 0; k <= ki; k++) {
                                work[bOff + k * ldb + iv - 1] *= scale;
                                work[bOff + k * ldb + iv] *= scale;
                            }
                        }
                        work[bOff + j * ldb + iv - 1] = x[0];
                        work[bOff + j * ldb + iv] = x[1];
                        for (int k = 0; k < j; k++) {
                            work[bOff + k * ldb + iv - 1] -= x[0] * T[tOff + k * ldt + j];
                            work[bOff + k * ldb + iv] -= x[1] * T[tOff + k * ldt + j];
                        }
                        j--;
                    } else {
                        Dlaln2.dlaln2(false, 2, 2, smin, 1,
                                T, tOff + (j - 1) * ldt + j - 1, ldt, 1, 1,
                                work, bOff + (j - 1) * ldb + iv - 1, ldb, wr, wi, x, 0, 2);
                        double scale = x[4];
                        double xnorm = x[5];

                        if (xnorm > 1) {
                            double beta = Math.max(norms[j - 1], norms[j]);
                            if (beta > bignum / xnorm) {
                                double rec = 1.0 / xnorm;
                                x[0] *= rec;
                                x[1] *= rec;
                                x[2] *= rec;
                                x[3] *= rec;
                                scale *= rec;
                            }
                        }
                        if (scale != 1) {
                            for (int k = 0; k <= ki; k++) {
                                work[bOff + k * ldb + iv - 1] *= scale;
                                work[bOff + k * ldb + iv] *= scale;
                            }
                        }
                        work[bOff + (j - 1) * ldb + iv - 1] = x[0];
                        work[bOff + (j - 1) * ldb + iv] = x[1];
                        work[bOff + j * ldb + iv - 1] = x[2];
                        work[bOff + j * ldb + iv] = x[3];
                        for (int k = 0; k < j - 1; k++) {
                            work[bOff + k * ldb + iv - 1] -= x[0] * T[tOff + k * ldt + j - 1];
                            work[bOff + k * ldb + iv] -= x[1] * T[tOff + k * ldt + j - 1];
                            work[bOff + k * ldb + iv - 1] -= x[2] * T[tOff + k * ldt + j];
                            work[bOff + k * ldb + iv] -= x[3] * T[tOff + k * ldt + j];
                        }
                        j -= 2;
                    }
                }

                if (howmny != 'B') {
                    for (int k = 0; k <= ki; k++) {
                        VR[vrOff + k * ldvr + is - 1] = work[bOff + k * ldb + iv - 1];
                        VR[vrOff + k * ldvr + is] = work[bOff + k * ldb + iv];
                    }
                    double emax = 0;
                    for (int k = 0; k <= ki; k++) {
                        emax = Math.max(emax,
                                Math.abs(VR[vrOff + k * ldvr + is - 1]) + Math.abs(VR[vrOff + k * ldvr + is]));
                    }
                    double remax = 1.0 / emax;
                    for (int k = 0; k <= ki; k++) {
                        VR[vrOff + k * ldvr + is - 1] *= remax;
                        VR[vrOff + k * ldvr + is] *= remax;
                    }
                    for (int k = ki + 1; k < n; k++) {
                        VR[vrOff + k * ldvr + is - 1] = 0;
                        VR[vrOff + k * ldvr + is] = 0;
                    }
                } else if (nb == 1) {
                    if (ki - 1 > 0) {
                        BLAS.dgemv(BLAS.Transpose.NoTrans, n, ki - 1, 1, VR, vrOff, ldvr,
                                work, bOff + iv - 1, ldb, work[bOff + (ki - 1) * ldb + iv - 1],
                                VR, vrOff + ki - 1, ldvr);
                        BLAS.dgemv(BLAS.Transpose.NoTrans, n, ki - 1, 1, VR, vrOff, ldvr,
                                work, bOff + iv, ldb, work[bOff + ki * ldb + iv],
                                VR, vrOff + ki, ldvr);
                    } else {
                        BLAS.dscal(n, work[bOff + (ki - 1) * ldb + iv - 1], VR, vrOff + ki - 1, ldvr);
                        BLAS.dscal(n, work[bOff + ki * ldb + iv], VR, vrOff + ki, ldvr);
                    }
                    double emax = 0;
                    for (int k = 0; k < n; k++) {
                        emax = Math.max(emax,
                                Math.abs(VR[vrOff + k * ldvr + ki - 1]) + Math.abs(VR[vrOff + k * ldvr + ki]));
                    }
                    double remax = 1.0 / emax;
                    BLAS.dscal(n, remax, VR, vrOff + ki - 1, ldvr);
                    BLAS.dscal(n, remax, VR, vrOff + ki, ldvr);
                } else {
                    for (int k = ki + 1; k < n; k++) {
                        work[bOff + k * ldb + iv - 1] = 0;
                        work[bOff + k * ldb + iv] = 0;
                    }
                    iscomplex[iv - 1] = (byte) -ip;
                    iscomplex[iv] = (byte) ip;
                    iv--;
                }
            }

            if (nb > 1) {
                int ki2 = ki;
                if (ip != 0) ki2--;
                if (iv < 2 || ki2 == 0) {
                    int nbActual = nb - iv;
                    BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, n, nbActual, ki2 + nbActual,
                            1, VR, vrOff, ldvr, work, bOff + iv, ldb,
                            0, work, bOff + nb + iv, ldb);

                    for (int k = iv; k < nb; k++) {
                        double remax;
                        if (iscomplex[k] == 0) {
                            int ii = 0;
                            double maxVal = Math.abs(work[bOff + nb + k]);
                            for (int ii2 = 1; ii2 < n; ii2++) {
                                double v = Math.abs(work[bOff + ii2 * ldb + nb + k]);
                                if (v > maxVal) {
                                    maxVal = v;
                                    ii = ii2;
                                }
                            }
                            remax = 1.0 / Math.abs(work[bOff + ii * ldb + nb + k]);
                        } else if (iscomplex[k] == 1) {
                            double emax = 0;
                            for (int ii2 = 0; ii2 < n; ii2++) {
                                emax = Math.max(emax,
                                        Math.abs(work[bOff + ii2 * ldb + nb + k]) +
                                        Math.abs(work[bOff + ii2 * ldb + nb + k + 1]));
                            }
                            remax = 1.0 / emax;
                        } else {
                            remax = 1;
                        }
                        BLAS.dscal(n, remax, work, bOff + nb + k, ldb);
                    }

                    for (int i = 0; i < n; i++) {
                        for (int j = iv; j < nb; j++) {
                            VR[vrOff + i * ldvr + ki2 + j - iv] = work[bOff + i * ldb + nb + j];
                        }
                    }
                    iv = nb - 1;
                } else {
                    iv--;
                }
            }

            is--;
            if (ip != 0) is--;
        }
        }

        if (!leftv) {
            return m;
        }

        iv = 0;
        ip = 0;
        is = 0;

        for (int ki = 0; ki < n; ki++) {
            if (ip == 1) {
                ip = -1;
                continue;
            }

            if (ki == n - 1 || T[tOff + (ki + 1) * ldt + ki] == 0) {
                ip = 0;
            } else {
                ip = 1;
            }

            if (howmny == 'S' && !selected[ki]) {
                continue;
            }

            double wr = T[tOff + ki * ldt + ki];
            double wi = 0;
            if (ip != 0) {
                wi = Math.sqrt(Math.abs(T[tOff + ki * ldt + ki + 1])) *
                     Math.sqrt(Math.abs(T[tOff + (ki + 1) * ldt + ki]));
            }
            double smin = Math.max(ulp * (Math.abs(wr) + Math.abs(wi)), smlnum);

            if (ip == 0) {
                work[bOff + ki * ldb + iv] = 1;
                for (int k = ki + 1; k < n; k++) {
                    work[bOff + k * ldb + iv] = -T[tOff + ki * ldt + k];
                }

                double vmax = 1.0;
                double vcrit = bignum;

                for (int j = ki + 1; j < n; ) {
                    if (j == n - 1 || T[tOff + (j + 1) * ldt + j] == 0) {
                        if (norms[j] > vcrit) {
                            double rec = 1.0 / vmax;
                            for (int k = ki; k < n; k++) {
                                work[bOff + k * ldb + iv] *= rec;
                            }
                            vmax = 1;
                        }

                        double dot = 0;
                        for (int k = ki + 1; k < j; k++) {
                            dot += T[tOff + (ki + 1 + (k - ki - 1)) * ldt + j] * work[bOff + (ki + 1 + (k - ki - 1)) * ldb + iv];
                        }
                        work[bOff + j * ldb + iv] -= dot;

                        Dlaln2.dlaln2(false, 1, 1, smin, 1,
                                T, tOff + j * ldt + j, ldt, 1, 1,
                                work, bOff + j * ldb + iv, ldb, wr, 0, x, 0, 2);
                        double scale = x[4];

                        if (scale != 1) {
                            for (int k = ki; k < n; k++) {
                                work[bOff + k * ldb + iv] *= scale;
                            }
                        }
                        work[bOff + j * ldb + iv] = x[0];
                        vmax = Math.max(Math.abs(work[bOff + j * ldb + iv]), vmax);
                        vcrit = bignum / vmax;
                        j++;
                    } else {
                        double beta = Math.max(norms[j], norms[j + 1]);
                        if (beta > vcrit) {
                            for (int k = ki; k < n; k++) {
                                work[bOff + k * ldb + iv] /= vmax;
                            }
                            vmax = 1;
                        }

                        double dot1 = 0, dot2 = 0;
                        for (int k = ki + 1; k < j; k++) {
                            dot1 += T[tOff + (ki + 1 + (k - ki - 1)) * ldt + j] * work[bOff + (ki + 1 + (k - ki - 1)) * ldb + iv];
                            dot2 += T[tOff + (ki + 1 + (k - ki - 1)) * ldt + j + 1] * work[bOff + (ki + 1 + (k - ki - 1)) * ldb + iv];
                        }
                        work[bOff + j * ldb + iv] -= dot1;
                        work[bOff + (j + 1) * ldb + iv] -= dot2;

                        Dlaln2.dlaln2(true, 2, 1, smin, 1,
                                T, tOff + j * ldt + j, ldt, 1, 1,
                                work, bOff + j * ldb + iv, ldb, wr, 0, x, 0, 2);
                        double scale = x[4];

                        if (scale != 1) {
                            for (int k = ki; k < n; k++) {
                                work[bOff + k * ldb + iv] *= scale;
                            }
                        }
                        work[bOff + j * ldb + iv] = x[0];
                        work[bOff + (j + 1) * ldb + iv] = x[2];
                        vmax = Math.max(Math.abs(x[0]), Math.abs(x[2]));
                        if (vmax > 1) {
                            vcrit = bignum / vmax;
                        }
                        j += 2;
                    }
                }

                if (howmny != 'B') {
                    for (int k = ki; k < n; k++) {
                        VL[vlOff + k * ldvl + is] = work[bOff + k * ldb + iv];
                    }
                    int ii = ki;
                    double maxVal = Math.abs(VL[vlOff + ki * ldvl + is]);
                    for (int k = ki + 1; k < n; k++) {
                        double v = Math.abs(VL[vlOff + k * ldvl + is]);
                        if (v > maxVal) {
                            maxVal = v;
                            ii = k;
                        }
                    }
                    double remax = 1.0 / Math.abs(VL[vlOff + ii * ldvl + is]);
                    for (int k = ki; k < n; k++) {
                        VL[vlOff + k * ldvl + is] *= remax;
                    }
                    for (int k = 0; k < ki; k++) {
                        VL[vlOff + k * ldvl + is] = 0;
                    }
                } else if (nb == 1) {
                    if (n - ki - 1 > 0) {
                        BLAS.dgemv(BLAS.Transpose.NoTrans, n, n - ki - 1, 1, VL, vlOff + ki + 1, ldvl,
                                work, bOff + (ki + 1) * ldb + iv, ldb, work[bOff + ki * ldb + iv],
                                VL, vlOff + ki, ldvl);
                    }
                    int ii = ki;
                    double maxVal = Math.abs(VL[vlOff + ki * ldvl + ki]);
                    for (int k = ki + 1; k < n; k++) {
                        double v = Math.abs(VL[vlOff + k * ldvl + ki]);
                        if (v > maxVal) {
                            maxVal = v;
                            ii = k;
                        }
                    }
                    double remax = 1.0 / Math.abs(VL[vlOff + ii * ldvl + ki]);
                    BLAS.dscal(n, remax, VL, vlOff + ki, ldvl);
                } else {
                    for (int k = 0; k < ki; k++) {
                        work[bOff + k * ldb + iv] = 0;
                    }
                    iscomplex[iv] = (byte) ip;
                }
            } else {
                if (Math.abs(T[tOff + ki * ldt + ki + 1]) >= Math.abs(T[tOff + (ki + 1) * ldt + ki])) {
                    work[bOff + ki * ldb + iv] = wi / T[tOff + ki * ldt + ki + 1];
                    work[bOff + (ki + 1) * ldb + iv + 1] = 1;
                } else {
                    work[bOff + ki * ldb + iv] = 1;
                    work[bOff + (ki + 1) * ldb + iv + 1] = -wi / T[tOff + (ki + 1) * ldt + ki];
                }
                work[bOff + (ki + 1) * ldb + iv] = 0;
                work[bOff + ki * ldb + iv + 1] = 0;

                for (int k = ki + 2; k < n; k++) {
                    work[bOff + k * ldb + iv] = -work[bOff + ki * ldb + iv] * T[tOff + ki * ldt + k];
                    work[bOff + k * ldb + iv + 1] = -work[bOff + (ki + 1) * ldb + iv + 1] * T[tOff + (ki + 1) * ldt + k];
                }

                double vmax = Math.max(Math.abs(work[bOff + ki * ldb + iv]), 1.0);
                double vcrit = bignum / vmax;

                for (int j = ki + 2; j < n; ) {
                    if (j == n - 1 || T[tOff + (j + 1) * ldt + j] == 0) {
                        if (norms[j] > vcrit) {
                            double rec = 1.0 / vmax;
                            for (int k = ki; k < n; k++) {
                                work[bOff + k * ldb + iv] *= rec;
                                work[bOff + k * ldb + iv + 1] *= rec;
                            }
                            vmax = 1;
                        }

                        double dot1 = 0, dot2 = 0;
                        for (int k = ki + 2; k < j; k++) {
                            dot1 += T[tOff + k * ldt + j] * work[bOff + k * ldb + iv];
                            dot2 += T[tOff + k * ldt + j] * work[bOff + k * ldb + iv + 1];
                        }
                        work[bOff + j * ldb + iv] -= dot1;
                        work[bOff + j * ldb + iv + 1] -= dot2;

                        Dlaln2.dlaln2(false, 1, 2, smin, 1,
                                T, tOff + j * ldt + j, ldt, 1, 1,
                                work, bOff + j * ldb + iv, ldb, wr, -wi, x, 0, 2);
                        double scale = x[4];

                        if (scale != 1) {
                            for (int k = ki; k < n; k++) {
                                work[bOff + k * ldb + iv] *= scale;
                                work[bOff + k * ldb + iv + 1] *= scale;
                            }
                        }
                        work[bOff + j * ldb + iv] = x[0];
                        work[bOff + j * ldb + iv + 1] = x[1];
                        vmax = Math.max(Math.abs(x[0]) + Math.abs(x[1]), vmax);
                        vcrit = bignum / vmax;
                        j++;
                    } else {
                        double beta = Math.max(norms[j], norms[j + 1]);
                        if (beta > vcrit) {
                            double rec = 1.0 / vmax;
                            for (int k = ki; k < n; k++) {
                                work[bOff + k * ldb + iv] *= rec;
                                work[bOff + k * ldb + iv + 1] *= rec;
                            }
                            vmax = 1;
                        }

                        double dot1 = 0, dot2 = 0, dot3 = 0, dot4 = 0;
                        for (int k = ki + 2; k < j; k++) {
                            dot1 += T[tOff + k * ldt + j] * work[bOff + k * ldb + iv];
                            dot2 += T[tOff + k * ldt + j] * work[bOff + k * ldb + iv + 1];
                            dot3 += T[tOff + k * ldt + j + 1] * work[bOff + k * ldb + iv];
                            dot4 += T[tOff + k * ldt + j + 1] * work[bOff + k * ldb + iv + 1];
                        }
                        work[bOff + j * ldb + iv] -= dot1;
                        work[bOff + j * ldb + iv + 1] -= dot2;
                        work[bOff + (j + 1) * ldb + iv] -= dot3;
                        work[bOff + (j + 1) * ldb + iv + 1] -= dot4;

                        Dlaln2.dlaln2(true, 2, 2, smin, 1,
                                T, tOff + j * ldt + j, ldt, 1, 1,
                                work, bOff + j * ldb + iv, ldb, wr, -wi, x, 0, 2);
                        double scale = x[4];

                        if (scale != 1) {
                            for (int k = ki; k < n; k++) {
                                work[bOff + k * ldb + iv] *= scale;
                                work[bOff + k * ldb + iv + 1] *= scale;
                            }
                        }
                        work[bOff + j * ldb + iv] = x[0];
                        work[bOff + j * ldb + iv + 1] = x[1];
                        work[bOff + (j + 1) * ldb + iv] = x[2];
                        work[bOff + (j + 1) * ldb + iv + 1] = x[3];
                        vmax = Math.max(
                                Math.abs(x[0]) + Math.abs(x[1]),
                                Math.abs(x[2]) + Math.abs(x[3]));
                        if (vmax > 1) {
                            vcrit = bignum / vmax;
                        }
                        j += 2;
                    }
                }

                if (howmny != 'B') {
                    for (int k = ki; k < n; k++) {
                        VL[vlOff + k * ldvl + is] = work[bOff + k * ldb + iv];
                        VL[vlOff + k * ldvl + is + 1] = work[bOff + k * ldb + iv + 1];
                    }
                    double emax = 0;
                    for (int k = ki; k < n; k++) {
                        emax = Math.max(emax,
                                Math.abs(VL[vlOff + k * ldvl + is]) + Math.abs(VL[vlOff + k * ldvl + is + 1]));
                    }
                    double remax = 1.0 / emax;
                    for (int k = ki; k < n; k++) {
                        VL[vlOff + k * ldvl + is] *= remax;
                        VL[vlOff + k * ldvl + is + 1] *= remax;
                    }
                    for (int k = 0; k < ki; k++) {
                        VL[vlOff + k * ldvl + is] = 0;
                        VL[vlOff + k * ldvl + is + 1] = 0;
                    }
                } else if (nb == 1) {
                    if (n - ki - 2 > 0) {
                        BLAS.dgemv(BLAS.Transpose.NoTrans, n, n - ki - 2, 1, VL, vlOff + ki + 2, ldvl,
                                work, bOff + (ki + 2) * ldb + iv, ldb, work[bOff + ki * ldb + iv],
                                VL, vlOff + ki, ldvl);
                        BLAS.dgemv(BLAS.Transpose.NoTrans, n, n - ki - 2, 1, VL, vlOff + ki + 2, ldvl,
                                work, bOff + (ki + 2) * ldb + iv + 1, ldb, work[bOff + (ki + 1) * ldb + iv + 1],
                                VL, vlOff + ki + 1, ldvl);
                    } else {
                        BLAS.dscal(n, work[bOff + ki * ldb + iv], VL, vlOff + ki, ldvl);
                        BLAS.dscal(n, work[bOff + (ki + 1) * ldb + iv + 1], VL, vlOff + ki + 1, ldvl);
                    }
                    double emax = 0;
                    for (int k = 0; k < n; k++) {
                        emax = Math.max(emax,
                                Math.abs(VL[vlOff + k * ldvl + ki]) + Math.abs(VL[vlOff + k * ldvl + ki + 1]));
                    }
                    double remax = 1.0 / emax;
                    BLAS.dscal(n, remax, VL, vlOff + ki, ldvl);
                    BLAS.dscal(n, remax, VL, vlOff + ki + 1, ldvl);
                } else {
                    for (int k = 0; k < ki; k++) {
                        work[bOff + k * ldb + iv] = 0;
                        work[bOff + k * ldb + iv + 1] = 0;
                    }
                    iscomplex[iv] = (byte) ip;
                    iscomplex[iv + 1] = (byte) -ip;
                    iv++;
                }
            }

            if (nb > 1 && howmny == 'B') {
                int ki2 = ki;
                if (ip != 0) ki2++;
                if (iv >= nb - 2 || ki2 == n - 1) {
                    int nbActual = iv + 1;
                    BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, n, nbActual, n - ki2 + iv,
                            1, VL, vlOff + (ki2 - iv) * ldvl, ldvl, work, bOff + (ki2 - iv) * ldb, ldb,
                            0, work, bOff + nb, ldb);

                    for (int k = 0; k <= iv; k++) {
                        double remax;
                        if (iscomplex[k] == 0) {
                            int ii = 0;
                            double maxVal = Math.abs(work[bOff + nb + k]);
                            for (int ii2 = 1; ii2 < n; ii2++) {
                                double v = Math.abs(work[bOff + ii2 * ldb + nb + k]);
                                if (v > maxVal) {
                                    maxVal = v;
                                    ii = ii2;
                                }
                            }
                            remax = 1.0 / Math.abs(work[bOff + ii * ldb + nb + k]);
                        } else if (iscomplex[k] == 1) {
                            double emax = 0;
                            for (int ii2 = 0; ii2 < n; ii2++) {
                                emax = Math.max(emax,
                                        Math.abs(work[bOff + ii2 * ldb + nb + k]) +
                                        Math.abs(work[bOff + ii2 * ldb + nb + k + 1]));
                            }
                            remax = 1.0 / emax;
                        } else {
                            remax = 1;
                        }
                        BLAS.dscal(n, remax, work, bOff + nb + k, ldb);
                    }

                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j <= iv; j++) {
                            VL[vlOff + i * ldvl + ki2 - iv + j] = work[bOff + i * ldb + nb + j];
                        }
                    }
                    iv = 0;
                } else {
                    iv++;
                }
            }

            is++;
            if (ip != 0) is++;
        }

        return m;
    }

}
