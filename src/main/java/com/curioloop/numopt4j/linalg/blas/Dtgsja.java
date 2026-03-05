/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

interface Dtgsja {

    static final int MAXIT = 40;

    static void dlags2(boolean upper, double a1, double a2, double a3,
                       double b1, double b2, double b3,
                       double[] out, int off) {
        double csu, snu, csv, snv, csq, snq;

        if (upper) {
            double a = a1 * b3;
            double d = a3 * b1;
            double b = a2 * b1 - a1 * b2;

            Dlas2.dlasv2(a, b, d, out, off);
            double snr = out[off + 2];
            double csr = out[off + 3];
            double snl = out[off + 4];
            double csl = out[off + 5];

            if (abs(csl) >= abs(snl) || abs(csr) >= abs(snr)) {
                double ua11r = csl * a1;
                double ua12 = csl * a2 + snl * a3;

                double vb11r = csr * b1;
                double vb12 = csr * b2 + snr * b3;

                double aua12 = abs(csl) * abs(a2) + abs(snl) * abs(a3);
                double avb12 = abs(csr) * abs(b2) + abs(snr) * abs(b3);

                if (abs(ua11r) + abs(ua12) != 0) {
                    if (aua12 / (abs(ua11r) + abs(ua12)) <= avb12 / (abs(vb11r) + abs(vb12))) {
                        Dlartg.dlartg(-ua11r, ua12, out, off);
                    } else {
                        Dlartg.dlartg(-vb11r, vb12, out, off);
                    }
                } else {
                    Dlartg.dlartg(-vb11r, vb12, out, off);
                }
                csq = out[off];
                snq = out[off + 1];

                csu = csl;
                snu = -snl;
                csv = csr;
                snv = -snr;
            } else {
                double ua21 = -snl * a1;
                double ua22 = -snl * a2 + csl * a3;

                double vb21 = -snr * b1;
                double vb22 = -snr * b2 + csr * b3;

                double aua22 = abs(snl) * abs(a2) + abs(csl) * abs(a3);
                double avb22 = abs(snr) * abs(b2) + abs(csr) * abs(b3);

                if (abs(ua21) + abs(ua22) != 0) {
                    if (aua22 / (abs(ua21) + abs(ua22)) <= avb22 / (abs(vb21) + abs(vb22))) {
                        Dlartg.dlartg(-ua21, ua22, out, off);
                    } else {
                        Dlartg.dlartg(-vb21, vb22, out, off);
                    }
                } else {
                    Dlartg.dlartg(-vb21, vb22, out, off);
                }
                csq = out[off];
                snq = out[off + 1];

                csu = snl;
                snu = csl;
                csv = snr;
                snv = csr;
            }
        } else {
            double a = a1 * b3;
            double d = a3 * b1;
            double c = a2 * b3 - a3 * b2;

            Dlas2.dlasv2(a, c, d, out, off);
            double snr = out[off + 2];
            double csr = out[off + 3];
            double snl = out[off + 4];
            double csl = out[off + 5];

            if (abs(csr) >= abs(snr) || abs(csl) >= abs(snl)) {
                double ua21 = -snr * a1 + csr * a2;
                double ua22r = csr * a3;

                double vb21 = -snl * b1 + csl * b2;
                double vb22r = csl * b3;

                double aua21 = abs(snr) * abs(a1) + abs(csr) * abs(a2);
                double avb21 = abs(snl) * abs(b1) + abs(csl) * abs(b2);

                if (abs(ua21) + abs(ua22r) != 0) {
                    if (aua21 / (abs(ua21) + abs(ua22r)) <= avb21 / (abs(vb21) + abs(vb22r))) {
                        Dlartg.dlartg(ua22r, ua21, out, off);
                    } else {
                        Dlartg.dlartg(vb22r, vb21, out, off);
                    }
                } else {
                    Dlartg.dlartg(vb22r, vb21, out, off);
                }
                csq = out[off];
                snq = out[off + 1];

                csu = csr;
                snu = -snr;
                csv = csl;
                snv = -snl;
            } else {
                double ua11 = csr * a1 + snr * a2;
                double ua12 = snr * a3;

                double vb11 = csl * b1 + snl * b2;
                double vb12 = snl * b3;

                double aua11 = abs(csr) * abs(a1) + abs(snr) * abs(a2);
                double avb11 = abs(csl) * abs(b1) + abs(snl) * abs(b2);

                if (abs(ua11) + abs(ua12) != 0) {
                    if (aua11 / (abs(ua11) + abs(ua12)) <= avb11 / (abs(vb11) + abs(vb12))) {
                        Dlartg.dlartg(ua12, ua11, out, off);
                    } else {
                        Dlartg.dlartg(vb12, vb11, out, off);
                    }
                } else {
                    Dlartg.dlartg(vb12, vb11, out, off);
                }
                csq = out[off];
                snq = out[off + 1];

                csu = snr;
                snu = csr;
                csv = snl;
                snv = csl;
            }
        }

        out[off] = csu;
        out[off + 1] = snu;
        out[off + 2] = csv;
        out[off + 3] = snv;
        out[off + 4] = csq;
        out[off + 5] = snq;
    }

    static double dlapll(int n, double[] x, int xOff, int incx,
                         double[] y, int yOff, int incy, double[] work, int wOff) {
        if (n < 1) return 0;
        if (n == 1) return 0;

        double a00 = Dlarfg.dlarfg(n, x[xOff], x, xOff + incx, incx, work, wOff);
        double tau = work[wOff];
        x[xOff] = 1;

        double c = -tau * BLAS.ddot(n, x, xOff, incx, y, yOff, incy);
        BLAS.daxpy(n, c, x, xOff, incx, y, yOff, incy);
        double a11 = Dlarfg.dlarfg(n - 1, y[yOff + incy], y, yOff + 2 * incy, incy, work, wOff);

        Dlas2.dlas2(a00, y[yOff], a11, work, wOff);
        return work[wOff];
    }

    static int[] dggsvp3(BLAS.GsvdJob jobU, BLAS.GsvdJob jobV, BLAS.GsvdJob jobQ, int m, int p, int n,
                         double[] A, int aOff, int lda, double[] B, int bOff, int ldb,
                         double tola, double tolb,
                         double[] U, int uOff, int ldu, double[] V, int vOff, int ldv,
                         double[] Q, int qOff, int ldq,
                         int[] iwork, double[] tau, double[] work, int workOff, int lwork) {
        boolean wantu = (jobU == BLAS.GsvdJob.Compute);
        boolean wantv = (jobV == BLAS.GsvdJob.Compute);
        boolean wantq = (jobQ == BLAS.GsvdJob.Compute);

        if (m < 0 || p < 0 || n < 0 || lda < max(1, n) || ldb < max(1, n) ||
            ldu < 1 || (wantu && ldu < m) ||
            ldv < 1 || (wantv && ldv < p) ||
            ldq < 1 || (wantq && ldq < n)) {
            return new int[]{0, 0};
        }

        if (lwork == -1) {
            work[workOff] = max(max(n, m), p);
            return new int[]{0, 0};
        }

        for (int i = 0; i < n; i++) {
            iwork[i] = 0;
        }

        Dgeqp.dgeqp3(p, n, B, bOff, ldb, iwork, tau, work, workOff, lwork);

        Dgeqp.dlapmt(true, m, n, A, aOff, lda, iwork, 0);

        int l = 0;
        for (int i = 0; i < min(p, n); i++) {
            if (abs(B[bOff + i * ldb + i]) > tolb) {
                l++;
            }
        }

        if (wantv) {
            BLAS.dlaset('A', p, p, 0, 0, V, vOff, ldv);
            if (p > 1) {
                BLAS.dlacpy('L', p - 1, min(p, n), B, bOff + ldb, ldb, V, vOff + ldv, ldv);
            }
            Dgeqr.dorg2r(p, p, min(p, n), V, vOff, ldv, tau, 0, work, workOff);
        }

        for (int i = 1; i < l; i++) {
            for (int j = 0; j < i; j++) {
                B[bOff + i * ldb + j] = 0;
            }
        }
        if (p > l) {
            BLAS.dlaset('A', p - l, n, 0, 0, B, bOff + l * ldb, ldb);
        }

        if (wantq) {
            BLAS.dlaset('A', n, n, 0, 1, Q, qOff, ldq);
            Dgeqp.dlapmt(true, n, n, Q, qOff, ldq, iwork, 0);
        }

        if (p >= l && n != l) {
            Dgerq.dgerq2(l, n, B, bOff, ldb, tau, 0, work, workOff);

            Dgerq.dormr2(BLAS.Side.Right, BLAS.Transpose.Trans, m, n, l, B, bOff, ldb, tau, 0, A, aOff, lda, work, workOff);

            if (wantq) {
                Dgerq.dormr2(BLAS.Side.Right, BLAS.Transpose.Trans, n, n, l, B, bOff, ldb, tau, 0, Q, qOff, ldq, work, workOff);
            }

            BLAS.dlaset('A', l, n - l, 0, 0, B, bOff, ldb);
            for (int i = 1; i < l; i++) {
                for (int j = 0; j < i; j++) {
                    B[bOff + i * ldb + (n - l + j)] = 0;
                }
            }
        }

        for (int i = 0; i < n - l; i++) {
            iwork[i] = 0;
        }

        Dgeqp.dgeqp3(m, n - l, A, aOff, lda, iwork, tau, work, workOff, lwork);

        int k = 0;
        for (int i = 0; i < min(m, n - l); i++) {
            if (abs(A[aOff + i * lda + i]) > tola) {
                k++;
            }
        }

        Dormqr.dorm2r(BLAS.Side.Left, BLAS.Transpose.Trans, m, l, min(m, n - l), A, aOff, lda, tau, 0, A, aOff + (n - l), lda, work, workOff);

        if (wantu) {
            BLAS.dlaset('A', m, m, 0, 0, U, uOff, ldu);
            if (m > 1) {
                BLAS.dlacpy('L', m - 1, min(m, n - l), A, aOff + lda, lda, U, uOff + ldu, ldu);
            }
            int kMin = min(m, n - l);
            Dgeqr.dorg2r(m, m, kMin, U, uOff, ldu, tau, 0, work, workOff);
        }

        if (wantq) {
            Dgeqp.dlapmt(true, n, n - l, Q, qOff, ldq, iwork, 0);
        }

        for (int i = 1; i < k; i++) {
            for (int j = 0; j < i; j++) {
                A[aOff + i * lda + j] = 0;
            }
        }
        if (m > k) {
            BLAS.dlaset('A', m - k, n - l, 0, 0, A, aOff + k * lda, lda);
        }

        if (n - l > k) {
            Dgerq.dgerq2(k, n - l, A, aOff, lda, tau, 0, work, workOff);

            if (wantq) {
                Dgerq.dormr2(BLAS.Side.Right, BLAS.Transpose.Trans, n, n - l, k, A, aOff, lda, tau, 0, Q, qOff, ldq, work, workOff);
            }

            BLAS.dlaset('A', k, n - l - k, 0, 0, A, aOff, lda);
            for (int i = 1; i < k; i++) {
                for (int j = 0; j < i; j++) {
                    A[aOff + i * lda + (n - k - l + j)] = 0;
                }
            }
        }

        if (m > k) {
            Dgeqr.dgeqr2(m - k, l, A, aOff + k * lda + (n - l), lda, tau, 0, work, workOff);
            if (wantu) {
                Dormqr.dorm2r(BLAS.Side.Right, BLAS.Transpose.NoTrans, m, m - k, min(m - k, l),
                              A, aOff + k * lda + (n - l), lda, tau, 0,
                              U, uOff + k, ldu, work, workOff);
            }

            for (int i = k + 1; i < m; i++) {
                for (int j = n - l; j < min(n - l + i - k, n); j++) {
                    A[aOff + i * lda + j] = 0;
                }
            }
        }

        return new int[]{k, l};
    }

    static boolean dtgsja(BLAS.GsvdJob jobU, BLAS.GsvdJob jobV, BLAS.GsvdJob jobQ, int m, int p, int n, int k, int l,
                           double[] A, int aOff, int lda, double[] B, int bOff, int ldb,
                           double tola, double tolb,
                           double[] alpha, int alphaOff, double[] beta, int betaOff,
                           double[] U, int uOff, int ldu, double[] V, int vOff, int ldv,
                           double[] Q, int qOff, int ldq, double[] work, int workOff) {
        boolean initu = (jobU == BLAS.GsvdJob.Initialize);
        boolean wantu = initu || (jobU == BLAS.GsvdJob.Compute);
        boolean initv = (jobV == BLAS.GsvdJob.Initialize);
        boolean wantv = initv || (jobV == BLAS.GsvdJob.Compute);
        boolean initq = (jobQ == BLAS.GsvdJob.Initialize);
        boolean wantq = initq || (jobQ == BLAS.GsvdJob.Compute);

        if (initu) {
            BLAS.dlaset('A', m, m, 0, 1, U, uOff, ldu);
        }
        if (initv) {
            BLAS.dlaset('A', p, p, 0, 1, V, vOff, ldv);
        }
        if (initq) {
            BLAS.dlaset('A', n, n, 0, 1, Q, qOff, ldq);
        }

        double minTol = min(tola, tolb);

        boolean upper = false;
        int cycles;
        boolean ok = false;

        for (cycles = 1; cycles <= MAXIT; cycles++) {
            upper = !upper;

            for (int i = 0; i < l - 1; i++) {
                for (int j = i + 1; j < l; j++) {
                    double a1 = 0, a2 = 0, a3 = 0;
                    if (k + i < m) {
                        a1 = A[aOff + (k + i) * lda + (n - l + i)];
                    }
                    if (k + j < m) {
                        a3 = A[aOff + (k + j) * lda + (n - l + j)];
                    }

                    double b1 = B[bOff + i * ldb + (n - l + i)];
                    double b3 = B[bOff + j * ldb + (n - l + j)];

                    double b2 = 0;
                    if (upper) {
                        if (k + i < m) {
                            a2 = A[aOff + (k + i) * lda + (n - l + j)];
                        }
                        b2 = B[bOff + i * ldb + (n - l + j)];
                    } else {
                        if (k + j < m) {
                            a2 = A[aOff + (k + j) * lda + (n - l + i)];
                        }
                        b2 = B[bOff + j * ldb + (n - l + i)];
                    }

                    dlags2(upper, a1, a2, a3, b1, b2, b3, work, workOff);
                    double csu = work[workOff];
                    double snu = work[workOff + 1];
                    double csv = work[workOff + 2];
                    double snv = work[workOff + 3];
                    double csq = work[workOff + 4];
                    double snq = work[workOff + 5];

                    if (k + j < m) {
                        BLAS.drot(l, A, aOff + (k + j) * lda + (n - l), 1, A, aOff + (k + i) * lda + (n - l), 1, csu, snu);
                    }

                    BLAS.drot(l, B, bOff + j * ldb + (n - l), 1, B, bOff + i * ldb + (n - l), 1, csv, snv);

                    BLAS.drot(min(k + l, m), A, aOff + (n - l + j), lda, A, aOff + (n - l + i), lda, csq, snq);
                    BLAS.drot(l, B, bOff + (n - l + j), ldb, B, bOff + (n - l + i), ldb, csq, snq);

                    if (upper) {
                        if (k + i < m) {
                            A[aOff + (k + i) * lda + (n - l + j)] = 0;
                        }
                        B[bOff + i * ldb + (n - l + j)] = 0;
                    } else {
                        if (k + j < m) {
                            A[aOff + (k + j) * lda + (n - l + i)] = 0;
                        }
                        B[bOff + j * ldb + (n - l + i)] = 0;
                    }

                    if (wantu && k + j < m) {
                        BLAS.drot(m, U, uOff + (k + j), ldu, U, uOff + (k + i), ldu, csu, snu);
                    }
                    if (wantv) {
                        BLAS.drot(p, V, vOff + j, ldv, V, vOff + i, ldv, csv, snv);
                    }
                    if (wantq) {
                        BLAS.drot(n, Q, qOff + (n - l + j), ldq, Q, qOff + (n - l + i), ldq, csq, snq);
                    }
                }
            }

            if (!upper) {
                double error = 0;
                for (int i = 0; i < min(l, m - k); i++) {
                    Dlamv.dcopy(l - i, A, aOff + (k + i) * lda + (n - l + i), 1, work, workOff, 1);
                    Dlamv.dcopy(l - i, B, bOff + i * ldb + (n - l + i), 1, work, workOff + l, 1);
                    double ssmin = dlapll(l - i, work, workOff, 1, work, workOff + l, 1, work, workOff + 2 * l);
                    error = max(error, ssmin);
                }

                if (abs(error) <= minTol) {
                    for (int i = 0; i < k; i++) {
                        alpha[alphaOff + i] = 1;
                        beta[betaOff + i] = 0;
                    }

                    for (int i = 0; i < min(l, m - k); i++) {
                        double a1 = A[aOff + (k + i) * lda + (n - l + i)];
                        double b1 = B[bOff + i * ldb + (n - l + i)];
                        double gamma = b1 / a1;

                        if (!Double.isInfinite(gamma)) {
                            if (gamma < 0) {
                                BLAS.dscal(l - i, -1, B, bOff + i * ldb + (n - l + i), 1);
                                if (wantv) {
                                    BLAS.dscal(p, -1, V, vOff + i, ldv);
                                }
                            }

                            Dlartg.dlartg(abs(gamma), 1, work, workOff);
                            double cs = work[workOff];
                            double sn = work[workOff + 1];
                            beta[betaOff + k + i] = cs;
                            alpha[alphaOff + k + i] = sn;

                            if (alpha[alphaOff + k + i] >= beta[betaOff + k + i]) {
                                BLAS.dscal(l - i, 1 / alpha[alphaOff + k + i], A, aOff + (k + i) * lda + (n - l + i), 1);
                            } else {
                                BLAS.dscal(l - i, 1 / beta[betaOff + k + i], B, bOff + i * ldb + (n - l + i), 1);
                                Dlamv.dcopy(l - i, B, bOff + i * ldb + (n - l + i), 1, A, aOff + (k + i) * lda + (n - l + i), 1);
                            }
                        } else {
                            alpha[alphaOff + k + i] = 0;
                            beta[betaOff + k + i] = 1;
                            Dlamv.dcopy(l - i, B, bOff + i * ldb + (n - l + i), 1, A, aOff + (k + i) * lda + (n - l + i), 1);
                        }
                    }

                    for (int i = m; i < k + l; i++) {
                        alpha[alphaOff + i] = 0;
                        beta[betaOff + i] = 1;
                    }
                    if (k + l < n) {
                        for (int i = k + l; i < n; i++) {
                            alpha[alphaOff + i] = 0;
                            beta[betaOff + i] = 0;
                        }
                    }

                    ok = true;
                    work[workOff] = cycles;
                    return ok;
                }
            }
        }

        work[workOff] = cycles;
        return ok;
    }
}
