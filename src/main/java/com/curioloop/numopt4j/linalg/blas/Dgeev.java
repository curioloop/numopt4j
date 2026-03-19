/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

interface Dgeev {

    /**
     * Computes eigenvalues and optionally left/right eigenvectors of a general real matrix.
     *
     * <p>Computes the eigenvalues and, optionally, the left and/or right eigenvectors
     * for an n×n real nonsymmetric matrix A. The right eigenvector v_j satisfies
     * A*v_j = λ_j*v_j, and the left eigenvector u_j satisfies u_j^H*A = λ_j*u_j^H.
     * Computed eigenvectors are normalized to have Euclidean norm 1 and largest component real.
     *
     * @param wantvl  whether to compute left eigenvectors
     * @param wantvr  whether to compute right eigenvectors
     * @param n       order of the matrix A
     * @param A       n×n matrix (row-major, overwritten on exit)
     * @param lda     leading dimension of A
     * @param wr      real parts of eigenvalues (length n)
     * @param wi      imaginary parts of eigenvalues (length n)
     * @param vl      left eigenvector matrix (n×n, row-major); unused if !wantvl
     * @param ldvl    leading dimension of vl
     * @param vr      right eigenvector matrix (n×n, row-major); unused if !wantvr
     * @param ldvr    leading dimension of vr
     * @param work    workspace array
     * @param workOff offset into work
     * @param lwork   workspace size; use -1 for workspace query
     * @return 0 on success; positive value = index of first unconverged eigenvalue
     */
    static int dgeev(boolean wantvl, boolean wantvr, int n, double[] A, int lda,
                     double[] wr, double[] wi, double[] vl, int ldvl,
                     double[] vr, int ldvr, double[] work, int workOff, int lwork) {
        int minwrk = (wantvl || wantvr) ? Math.max(1, 4 * n) : Math.max(1, 3 * n);

        if (n < 0) {
            return -3;
        }
        if (lda < Math.max(1, n)) {
            return -5;
        }
        if ((wantvl && ldvl < n) || (!wantvl && ldvl < 1)) {
            return -9;
        }
        if ((wantvr && ldvr < n) || (!wantvr && ldvr < 1)) {
            return -11;
        }
        if (lwork < minwrk && lwork != -1) {
            return -13;
        }

        if (n == 0) {
            work[workOff] = 1;
            return 0;
        }

        int maxwrk = 2 * n + n * Ilaenv.ilaenv(1, "DGEHRD", " ", n, 1, n, 0);
        if (wantvl || wantvr) {
            maxwrk = Math.max(maxwrk, 2 * n + (n - 1) * Ilaenv.ilaenv(1, "DORGHR", " ", n, 1, n, -1));
            int hseqrWork = Dhseqr.dhseqrQuery(n, 0, n - 1);
            maxwrk = Math.max(maxwrk, Math.max(n + 1, n + hseqrWork));
            char side = wantvr ? 'R' : 'L';
            int trevcWork = Dtrevc3.dtrevc3Query(side, 'B', n);
            maxwrk = Math.max(maxwrk, n + trevcWork);
            maxwrk = Math.max(maxwrk, 4 * n);
        } else {
            int hseqrWork = Dhseqr.dhseqrQuery(n, 0, n - 1);
            maxwrk = Math.max(maxwrk, Math.max(n + 1, n + hseqrWork));
        }
        maxwrk = Math.max(maxwrk, minwrk);

        if (lwork == -1) {
            work[workOff] = maxwrk;
            return 0;
        }

        double safmin = BLAS.safmin();
        double eps = BLAS.eps();
        double smlnum = Math.sqrt(safmin) / eps;
        double bignum = 1.0 / smlnum;

        double anrm = BLAS.dlange(BLAS.Norm.MaxAbs, n, n, A, 0, lda);
        boolean scalea = false;
        double cscale = 1.0;

        if (anrm > 0 && anrm < smlnum) {
            scalea = true;
            cscale = smlnum;
        } else if (anrm > bignum) {
            scalea = true;
            cscale = bignum;
        }
        if (scalea) {
            BLAS.dlascl('G', 0, 0, anrm, cscale, n, n, A, 0, lda);
        }

        BLAS.dgebal('B', n, A, lda, work, 0, work, lwork - 2);
        int ilo = (int) work[lwork - 2];
        int ihi = (int) work[lwork - 1];

        int iwrk = 2 * n;
        int tauOff = n;
        BLAS.dgehrd(n, ilo, ihi, A, lda, work, tauOff, work, iwrk, lwork - iwrk);

        int info = 0;
        char side = 'N';

        if (wantvl) {
            side = 'L';
            BLAS.dlacpy(BLAS.Uplo.Lower, n, n, A, 0, lda, vl, 0, ldvl);
            BLAS.dorghr(n, ilo, ihi, vl, 0, ldvl, work, tauOff, work, iwrk, lwork - iwrk);
            iwrk = n;
            info = BLAS.dhseqr('S', 'V', n, ilo, ihi, A, lda, wr, wi, vl, ldvl, work, iwrk, lwork - iwrk);
            if (wantvr) {
                side = 'B';
                BLAS.dlacpy(BLAS.Uplo.All, n, n, vl, 0, ldvl, vr, 0, ldvr);
            }
        } else if (wantvr) {
            side = 'R';
            BLAS.dlacpy(BLAS.Uplo.Lower, n, n, A, 0, lda, vr, 0, ldvr);
            BLAS.dorghr(n, ilo, ihi, vr, 0, ldvr, work, tauOff, work, iwrk, lwork - iwrk);
            iwrk = n;
            info = BLAS.dhseqr('S', 'V', n, ilo, ihi, A, lda, wr, wi, vr, ldvr, work, iwrk, lwork - iwrk);
        } else {
            iwrk = n;
            info = BLAS.dhseqr('E', 'N', n, ilo, ihi, A, lda, wr, wi, null, 1, work, iwrk, lwork - iwrk);
        }

        if (info > 0) {
            if (scalea) {
                BLAS.dlascl('G', 0, 0, cscale, anrm, n - info, 1, wr, info, 1);
                BLAS.dlascl('G', 0, 0, cscale, anrm, n - info, 1, wi, info, 1);
                BLAS.dlascl('G', 0, 0, cscale, anrm, ilo, 1, wr, 0, 1);
                BLAS.dlascl('G', 0, 0, cscale, anrm, ilo, 1, wi, 0, 1);
            }
            return info;
        }

        if (wantvl || wantvr) {
            BLAS.dtrevc3(side, 'B', null, n, A, 0, lda,
                    vl, 0, ldvl, vr, 0, ldvr, n, work, iwrk, lwork - iwrk);
        }

        if (wantvl && vl != null) {
            BLAS.dgebak('B', BLAS.Side.Left, n, ilo, ihi, work, 0, n, vl, ldvl);
            normalizeEigenvectorsLeft(n, vl, ldvl, wr, wi, work, iwrk);
        }
        if (wantvr && vr != null) {
            BLAS.dgebak('B', BLAS.Side.Right, n, ilo, ihi, work, 0, n, vr, ldvr);
            normalizeEigenvectorsRight(n, vr, ldvr, wr, wi, work, iwrk);
        }

        if (scalea) {
            BLAS.dlascl('G', 0, 0, cscale, anrm, n, 1, wr, 0, 1);
            BLAS.dlascl('G', 0, 0, cscale, anrm, n, 1, wi, 0, 1);
        }

        work[workOff] = maxwrk;
        return info;
    }

    static void normalizeEigenvectorsRight(int n, double[] vr, int ldvr,
                                           double[] wr, double[] wi, double[] work, int workOff) {
        double eps = BLAS.eps();
        int rotOff = workOff + n;
        for (int j = 0; j < n; j++) {
            if (Math.abs(wi[j]) < eps) {
                double norm = BLAS.dnrm2(n, vr, j, ldvr);
                if (norm > 0) {
                    BLAS.dscal(n, 1.0 / norm, vr, j, ldvr);
                }
            } else if (wi[j] > 0 && j + 1 < n) {
                double norm1 = BLAS.dnrm2(n, vr, j, ldvr);
                double norm2 = BLAS.dnrm2(n, vr, j + 1, ldvr);
                double norm = Math.hypot(norm1, norm2);
                if (norm > 0) {
                    BLAS.dscal(n, 1.0 / norm, vr, j, ldvr);
                    BLAS.dscal(n, 1.0 / norm, vr, j + 1, ldvr);
                }

                for (int k = 0; k < n; k++) {
                    double vi = vr[k * ldvr + j];
                    double vi1 = vr[k * ldvr + j + 1];
                    work[workOff + k] = vi * vi + vi1 * vi1;
                }
                int maxIdx = BLAS.idamax(n, work, workOff, 1);
                double vi = vr[maxIdx * ldvr + j];
                double vi1 = vr[maxIdx * ldvr + j + 1];

                Dlartg.dlartg(vi, vi1, work, rotOff);
                double cs = work[rotOff];
                double sn = work[rotOff + 1];

                BLAS.drot(n, vr, j, ldvr, vr, j + 1, ldvr, cs, sn);
                vr[maxIdx * ldvr + j + 1] = 0.0;

                j++;
            }
        }
    }

    static void normalizeEigenvectorsLeft(int n, double[] vl, int ldvl,
                                          double[] wr, double[] wi, double[] work, int workOff) {
        double eps = BLAS.eps();
        int rotOff = workOff + n;
        for (int j = 0; j < n; j++) {
            if (Math.abs(wi[j]) < eps) {
                double norm = BLAS.dnrm2(n, vl, j, ldvl);
                if (norm > 0) {
                    BLAS.dscal(n, 1.0 / norm, vl, j, ldvl);
                }
            } else if (wi[j] > 0 && j + 1 < n) {
                double norm1 = BLAS.dnrm2(n, vl, j, ldvl);
                double norm2 = BLAS.dnrm2(n, vl, j + 1, ldvl);
                double norm = Math.hypot(norm1, norm2);
                if (norm > 0) {
                    BLAS.dscal(n, 1.0 / norm, vl, j, ldvl);
                    BLAS.dscal(n, 1.0 / norm, vl, j + 1, ldvl);
                }

                for (int k = 0; k < n; k++) {
                    double vi = vl[k * ldvl + j];
                    double vi1 = vl[k * ldvl + j + 1];
                    work[workOff + k] = vi * vi + vi1 * vi1;
                }
                int maxIdx = BLAS.idamax(n, work, workOff, 1);
                double vi = vl[maxIdx * ldvl + j];
                double vi1 = vl[maxIdx * ldvl + j + 1];

                Dlartg.dlartg(vi, vi1, work, rotOff);
                double cs = work[rotOff];
                double sn = work[rotOff + 1];

                BLAS.drot(n, vl, j, ldvl, vl, j + 1, ldvl, cs, sn);
                vl[maxIdx * ldvl + j + 1] = 0.0;

                j++;
            }
        }
    }
}
