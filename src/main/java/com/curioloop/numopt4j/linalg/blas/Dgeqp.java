/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * Computes a QR factorization with column pivoting of a general m×n matrix A.
 *
 * <p>Computes A * P = Q * R where P is a permutation matrix, Q is orthogonal,
 * and R is upper trapezoidal. Uses a blocked algorithm (dlaqps) for efficiency
 * and falls back to unblocked (dlaqp2) for the remainder.
 * Also provides dlapmt for applying a permutation to a matrix.
 *
 * @see #dgeqp3(int, int, double[], int, int, int[], double[], double[], int, int)
 * @see #dlaqps(int, int, int, int, double[], int, int, int[], int, double[], int, double[], int, double[], int, double[], int, double[], int, int)
 * @see #dlaqp2(int, int, int, double[], int, int, int[], int, double[], int, double[], int, double[], int, double[], int)
 * @see #dlapmt(boolean, int, int, double[], int, int, int[], int)
 */
interface Dgeqp {

    static void dlapmt(boolean forward, int m, int n, double[] A, int aOff, int lda, int[] k, int kOff) {
        if (m == 0 || n == 0) return;
        if (n == 1) return;

        for (int i = 0; i < n; i++) {
            k[kOff + i]++;
            k[kOff + i] = -k[kOff + i];
        }

        if (forward) {
            for (int j = 0; j < n; j++) {
                if (k[kOff + j] >= 0) continue;

                k[kOff + j] = -k[kOff + j];
                int i = k[kOff + j] - 1;

                while (k[kOff + i] < 0) {
                    BLAS.dswap(m, A, aOff + j, lda, A, aOff + i, lda);

                    k[kOff + i] = -k[kOff + i];
                    j = i;
                    i = k[kOff + i] - 1;
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                if (k[kOff + i] >= 0) continue;

                k[kOff + i] = -k[kOff + i];
                int j = k[kOff + i] - 1;

                while (j != i) {
                    BLAS.dswap(m, A, aOff + j, lda, A, aOff + i, lda);

                    k[kOff + j] = -k[kOff + j];
                    j = k[kOff + j] - 1;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            k[kOff + i]--;
        }
    }

    static int dgeqp3(int m, int n, double[] A, int aOff, int lda,
                      int[] jpvt, double[] tau, double[] work, int workOff, int lwork) {
        if (m < 0 || n < 0 || lda < max(1, n)) {
            return -1;
        }

        int minmn = min(m, n);
        int iws = 3 * n + 1;
        if (minmn == 0) {
            if (lwork >= 1) {
                work[workOff] = 1;
            }
            return 0;
        }

        if (lwork < iws && lwork != -1) {
            return -1;
        }

        int nb = Ilaenv.ilaenv(1, "DGEQRF", " ", m, n, -1, -1);
        if (lwork == -1) {
            work[workOff] = 2 * n + (n + 1) * nb;
            return 0;
        }

        int nfxd = 0;
        for (int j = 0; j < n; j++) {
            if (jpvt[j] == -1) {
                jpvt[j] = j;
                continue;
            }
            if (j != nfxd) {
                BLAS.dswap(m, A, aOff + j, lda, A, aOff + nfxd, lda);
                jpvt[j] = jpvt[nfxd];
                jpvt[nfxd] = j;
            } else {
                jpvt[j] = j;
            }
            nfxd++;
        }

        if (nfxd > 0) {
            int na = min(m, nfxd);
            Dgeqr.dgeqrf(m, na, A, aOff, lda, tau, 0, work, workOff, lwork);
            iws = max(iws, (int) work[workOff]);
            if (na < n) {
                Dormqr.dormqr(BLAS.Side.Left, BLAS.Trans.Trans, m, n - na, na, A, aOff, lda, tau, 0,
                        A, aOff + na, lda, work, workOff, lwork);
                iws = max(iws, (int) work[workOff]);
            }
        }

        if (nfxd >= minmn) {
            work[workOff] = iws;
            return 0;
        }

        int sm = m - nfxd;
        int sn = n - nfxd;
        int sminmn = minmn - nfxd;

        nb = Ilaenv.ilaenv(1, "DGEQRF", " ", sm, sn, -1, -1);
        int nbmin = 2;
        int nx = 0;

        if (1 < nb && nb < sminmn) {
            nx = max(0, Ilaenv.ilaenv(3, "DGEQRF", " ", sm, sn, -1, -1));
            if (nx < sminmn) {
                int minws = 2 * sn + (sn + 1) * nb;
                iws = max(iws, minws);
                if (lwork < minws) {
                    nb = (lwork - 2 * sn) / (sn + 1);
                    nbmin = max(2, Ilaenv.ilaenv(2, "DGEQRF", " ", sm, sn, -1, -1));
                }
            }
        }

        for (int j = nfxd; j < n; j++) {
            work[workOff + j] = BLAS.dnrm2(sm, A, aOff + nfxd * lda + j, lda);
            work[workOff + n + j] = work[workOff + j];
        }

        int j = nfxd;
        if (nbmin <= nb && nb < sminmn && nx < sminmn) {
            int topbmn = minmn - nx;
            int fjb;
            while (j < topbmn) {
                int jb = min(nb, topbmn - j);
                fjb = dlaqps(m, n - j, j, jb, A, aOff + j, lda, jpvt, j, tau, j,
                        work, workOff + j, work, workOff + n + j,
                        work, workOff + 2 * n, work, workOff + 2 * n + jb, jb);
                j += fjb;
                if (fjb < jb) break;
            }
        }

        if (j < minmn) {
            dlaqp2(m, n - j, j, A, aOff + j, lda, jpvt, j, tau, j,
                    work, workOff + j, work, workOff + n + j, work, workOff + 2 * n);
        }

        work[workOff] = iws;
        return 0;
    }

    static int dlaqps(int m, int n, int offset, int nb,
                      double[] A, int aOff, int lda,
                      int[] jpvt, int jpvtOff, double[] tau, int tauOff,
                      double[] vn1, int vn1Off, double[] vn2, int vn2Off,
                      double[] auxv, int auxvOff, double[] f, int fOff, int ldf) {

        int lastrk = min(m, n + offset);
        int lsticc = -1;
        double tol3z = sqrt(2.220446049250313E-16);

        int k;
        for (k = 0; k < nb && lsticc == -1; k++) {
            int rk = offset + k;

            int p = k + BLAS.idamax(n - k, vn1, vn1Off + k, 1);
            if (p != k) {
                BLAS.dswap(m, A, aOff + p, lda, A, aOff + k, lda);
                BLAS.dswap(k, f, fOff + p * ldf, 1, f, fOff + k * ldf, 1);
                int tmp = jpvt[jpvtOff + p];
                jpvt[jpvtOff + p] = jpvt[jpvtOff + k];
                jpvt[jpvtOff + k] = tmp;
                vn1[vn1Off + p] = vn1[vn1Off + k];
                vn2[vn2Off + p] = vn2[vn2Off + k];
            }

            if (k > 0) {
                BLAS.dgemv(BLAS.Trans.NoTrans, m - rk, k, -1,
                        A, aOff + rk * lda, lda,
                        f, fOff + k * ldf, 1,
                        1,
                        A, aOff + rk * lda + k, lda);
            }

            if (rk < m - 1) {
                int aii = aOff + rk * lda + k;
                double alpha = A[aii];
                A[aii] = BLAS.dlarfg(m - rk, alpha, A, aii + lda, lda, tau, tauOff + k);
            } else {
                tau[tauOff + k] = 0;
            }

            double akk = A[aOff + rk * lda + k];
            A[aOff + rk * lda + k] = 1;

            if (k < n - 1) {
                BLAS.dgemv(BLAS.Trans.Trans, m - rk, n - k - 1, tau[tauOff + k],
                        A, aOff + rk * lda + k + 1, lda,
                        A, aOff + rk * lda + k, lda,
                        0,
                        f, fOff + (k + 1) * ldf + k, ldf);
            }

            for (int j = 0; j < k; j++) {
                f[fOff + j * ldf + k] = 0;
            }

            if (k > 0) {
                BLAS.dgemv(BLAS.Trans.Trans, m - rk, k, -tau[tauOff + k],
                        A, aOff + rk * lda, lda,
                        A, aOff + rk * lda + k, lda,
                        0,
                        auxv, auxvOff, 1);
                BLAS.dgemv(BLAS.Trans.NoTrans, n, k, 1,
                        f, fOff, ldf,
                        auxv, auxvOff, 1,
                        1,
                        f, fOff + k, ldf);
            }

            if (k < n - 1) {
                BLAS.dgemv(BLAS.Trans.NoTrans, n - k - 1, k + 1, -1,
                        f, fOff + (k + 1) * ldf, ldf,
                        A, aOff + rk * lda, 1,
                        1,
                        A, aOff + rk * lda + k + 1, 1);
            }

            if (rk < lastrk - 1) {
                for (int j = k + 1; j < n; j++) {
                    if (vn1[vn1Off + j] == 0) continue;

                    double r = abs(A[aOff + rk * lda + j]) / vn1[vn1Off + j];
                    double temp = max(0, 1 - r * r);
                    r = vn1[vn1Off + j] / vn2[vn2Off + j];
                    double temp2 = temp * r * r;
                    if (temp2 < tol3z) {
                        vn2[vn2Off + j] = lsticc;
                        lsticc = j;
                    } else {
                        vn1[vn1Off + j] *= sqrt(temp);
                    }
                }
            }

            A[aOff + rk * lda + k] = akk;
        }

        int kb = k;
        int rk = offset + kb;

        if (kb < min(n, m - offset)) {
            BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.Trans, m - rk, n - kb, kb, -1,
                    A, aOff + rk * lda, lda,
                    f, fOff + kb * ldf, ldf,
                    1,
                    A, aOff + rk * lda + kb, lda);
        }

        while (lsticc >= 0) {
            int itemp = (int) vn2[vn2Off + lsticc];
            double v = BLAS.dnrm2(m - rk, A, aOff + rk * lda + lsticc, lda);
            vn1[vn1Off + lsticc] = v;
            vn2[vn2Off + lsticc] = v;
            lsticc = itemp;
        }

        return kb;
    }

    static void dlaqp2(int m, int n, int offset,
                       double[] A, int aOff, int lda,
                       int[] jpvt, int jpvtOff, double[] tau, int tauOff,
                       double[] vn1, int vn1Off, double[] vn2, int vn2Off,
                       double[] work, int workOff) {

        int mn = min(m - offset, n);
        double tol3z = sqrt(2.220446049250313E-16);

        for (int i = 0; i < mn; i++) {
            int offpi = offset + i;

            int p = i + BLAS.idamax(n - i, vn1, vn1Off + i, 1);
            if (p != i) {
                BLAS.dswap(m, A, aOff + p, lda, A, aOff + i, lda);
                int tmp = jpvt[jpvtOff + p];
                jpvt[jpvtOff + p] = jpvt[jpvtOff + i];
                jpvt[jpvtOff + i] = tmp;
                vn1[vn1Off + p] = vn1[vn1Off + i];
                vn2[vn2Off + p] = vn2[vn2Off + i];
            }

            if (offpi < m - 1) {
                int aii = aOff + offpi * lda + i;
                double alpha = A[aii];
                A[aii] = BLAS.dlarfg(m - offpi, alpha, A, aii + lda, lda, tau, tauOff + i);
            } else {
                tau[tauOff + i] = 0;
            }

            if (i < n - 1) {
                double aii = A[aOff + offpi * lda + i];
                A[aOff + offpi * lda + i] = 1;
                BLAS.dlarf(BLAS.Side.Left, m - offpi, n - i - 1, A, aOff + offpi * lda + i, lda,
                        tau[tauOff + i], A, aOff + offpi * lda + i + 1, lda, work, workOff);
                A[aOff + offpi * lda + i] = aii;
            }

            for (int j = i + 1; j < n; j++) {
                if (vn1[vn1Off + j] == 0) continue;

                double r = abs(A[aOff + offpi * lda + j]) / vn1[vn1Off + j];
                double temp = max(0, 1 - r * r);
                r = vn1[vn1Off + j] / vn2[vn2Off + j];
                double temp2 = temp * r * r;
                if (temp2 < tol3z) {
                    double v = 0;
                    if (offpi < m - 1) {
                        v = BLAS.dnrm2(m - offpi - 1, A, aOff + (offpi + 1) * lda + j, lda);
                    }
                    vn1[vn1Off + j] = v;
                    vn2[vn2Off + j] = v;
                } else {
                    vn1[vn1Off + j] *= sqrt(temp);
                }
            }
        }
    }
}
