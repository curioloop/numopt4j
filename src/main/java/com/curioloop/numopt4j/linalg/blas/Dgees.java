/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.max;

/**
 * DGEES computes for an N-by-N real nonsymmetric matrix A, the
 * eigenvalues, the real Schur form T, and, optionally, the matrix of
 * Schur vectors Z. This gives the Schur factorization A = Z*T*Z^T.
 *
 * <p>Optionally, it also orders the eigenvalues on the diagonal of the
 * real Schur form so that selected eigenvalues are at the top left.
 * The leading columns of Z then form an orthonormal basis for the
 * invariant subspace corresponding to the selected eigenvalues.</p>
 *
 * <p>A matrix is in real Schur form if it is upper quasi-triangular with
 * 1-by-1 and 2-by-2 blocks. 2-by-2 blocks will be standardized in the form:</p>
 * <pre>
 *   [ a  b ]
 *   [ c  a ]
 * </pre>
 * <p>where b*c &lt; 0. The eigenvalues of such a block are a ± sqrt(bc).</p>
 *
 * <p>Reference: LAPACK DGEES</p>
 */
interface Dgees {

    /**
     * Computes the Schur factorization of a general matrix: A = Z*T*Z^T (LAPACK DGEES).
     * Optionally reorders eigenvalues so that selected ones appear first.
     * Reference: gonum/lapack/gonum/dgees.go
     *
     * @param jobvs  'V' to compute Schur vectors Z, 'N' otherwise
     * @param sort   'S' to reorder eigenvalues, 'N' otherwise
     * @param select eigenvalue selection function (used if sort='S')
     * @param n      order of matrix A
     * @param A      general matrix (n × n, row-major), overwritten with quasi-triangular T
     * @param lda    leading dimension of A
     * @param wr     real parts of eigenvalues (output, length n)
     * @param wi     imaginary parts of eigenvalues (output, length n)
     * @param vs     Schur vectors (n × n, row-major); used if jobvs='V'
     * @param ldvs   leading dimension of vs
     * @param work   workspace
     * @param workOff offset into work; if lwork=-1, workspace query writes optimal size to work[workOff]
     * @param lwork  size of work; use -1 for workspace query
     * @param bwork  boolean workspace (length n); used if sort='S'
     * @return 0 on success; positive value if QR algorithm failed or reordering failed
     */
    static int dgees(char jobvs, char sort, Select select, int n,
                     double[] A, int lda,
                     double[] wr, double[] wi,
                     double[] vs, int ldvs,
                     double[] work, int workOff, int lwork,
                     boolean[] bwork) {

        boolean wantvs = jobvs == 'V' || jobvs == 'v';
        boolean wantst = sort == 'S' || sort == 's';

        if (!wantvs && jobvs != 'N' && jobvs != 'n') {
            return -1;
        }
        if (!wantst && sort != 'N' && sort != 'n') {
            return -2;
        }
        if (n < 0) {
            return -3;
        }
        if (lda < max(1, n)) {
            return -5;
        }
        if (ldvs < 1 || (wantvs && ldvs < n)) {
            return -10;
        }
        if (work == null || work.length < 1) {
            return -14;
        }
        if (wantst && (bwork == null || bwork.length < n)) {
            return -15;
        }

        int minwrk = 1;
        int maxwrk = 1;
        if (n == 0) {
            if (lwork != -1) work[workOff] = 1;
            return 0;
        }

        minwrk = max(10, 3 * n);
        minwrk = max(minwrk, n + 10);
        maxwrk = 2 * n + n * Ilaenv.ilaenv(1, "DGEHRD", " ", n, 1, n, 0);

        Dhseqr.dhseqr('S', wantvs ? 'V' : 'N', n, 0, n - 1, A, lda, wr, wi, vs, ldvs, work, workOff, -1);
        int hswork = work[workOff] > 0 ? (int) work[workOff] : 1;

        if (!wantvs) {
            maxwrk = max(maxwrk, n + hswork);
        } else {
            maxwrk = max(maxwrk, 2 * n + (n - 1) * Ilaenv.ilaenv(1, "DORGHR", " ", n, 1, n, -1));
            maxwrk = max(maxwrk, n + hswork);
        }
        maxwrk = max(maxwrk, minwrk);

        if (lwork == -1) {
            work[workOff] = maxwrk;
            return 0;
        } else if (lwork < minwrk) {
            return -14;
        }

        if (A == null || A.length < (n - 1) * lda + n) {
            return -4;
        }
        if (wr == null || wr.length < n) {
            return -7;
        }
        if (wi == null || wi.length < n) {
            return -8;
        }
        if (wantvs && (vs == null || vs.length < (n - 1) * ldvs + n)) {
            return -9;
        }

        int sdim = 0;

        double smlnum = Math.sqrt(Dlamch.dlamch('S')) / Dlamch.dlamch('P');
        double bignum = 1.0 / smlnum;

        double anrm = Dlange.dlange('M', n, n, A, 0, lda);
        boolean scalea = false;
        double cscale = 0;

        if (anrm > 0 && anrm < smlnum) {
            scalea = true;
            cscale = smlnum;
        } else if (anrm > bignum) {
            scalea = true;
            cscale = bignum;
        }

        if (scalea) {
            Dlamv.dlascl('G', 0, 0, anrm, cscale, n, n, A, 0, lda);
        }

        Dgebal.dgebal('B', n, A, lda, work, 0, work, lwork - 2);
        int ilo = (int) work[lwork - 2];
        int ihi = (int) work[lwork - 1];

        int iwrk = 2 * n;
        Dgehrd.dgehrd(n, ilo, ihi, A, lda, work, n, work, iwrk, lwork - iwrk);

        if (wantvs) {
            Dlamv.dlacpy('L', n, n, A, 0, lda, vs, 0, ldvs);
            Dorghr.dorghr(n, ilo, ihi, vs, ldvs, work, n, work, iwrk, lwork - iwrk);
        }

        int info = 0;
        iwrk = n;
        int ieval = Dhseqr.dhseqr('S', wantvs ? 'V' : 'N', n, ilo, ihi,
                A, lda, wr, wi, vs, ldvs, work, iwrk, lwork - iwrk);

        if (ieval > 0) {
            info = ieval;
        }

        if (wantst && info == 0) {
            if (scalea) {
                Dlamv.dlascl('G', 0, 0, cscale, anrm, n, 1, wr, 0, 1);
                Dlamv.dlascl('G', 0, 0, cscale, anrm, n, 1, wi, 0, 1);
            }

            for (int i = 0; i < n; i++) {
                bwork[i] = select.select(wr[i], wi[i]);
            }

            int[] trsenIwork = new int[2];
            boolean trsenOk = Dtrsen.dtrsen(Dtrsen.NO_COND, wantvs, bwork, n,
                    A, 0, lda, vs, 0, ldvs, wr, wi, work, lwork - iwrk, trsenIwork, 1);

            if (!trsenOk) {
                info = n + 1;
            }

            sdim = trsenIwork[1];
        }

        if (wantvs) {
            Dgebak.dgebak('P', BLAS.Side.Right, n, ilo, ihi, work, 0, n, vs, ldvs);
        }

        if (scalea) {
            Dlamv.dlascl('H', 0, 0, cscale, anrm, n, n, A, 0, lda);
            BLAS.dcopy(n, A, 0, lda + 1, wr, 0, 1);

            if (cscale == smlnum) {
                int i1, i2;
                if (ieval > 0) {
                    i1 = ieval + 1;
                    i2 = ihi - 1;
                    Dlamv.dlascl('G', 0, 0, cscale, anrm, ilo - 1, 1, wi, 0, 1);
                } else if (wantst) {
                    i1 = 1;
                    i2 = n - 1;
                } else {
                    i1 = ilo;
                    i2 = ihi - 1;
                }

                int inxt = i1 - 2;
                for (int i = i1 - 1; i <= i2 - 1; i++) {
                    if (i < inxt) {
                        continue;
                    }
                    if (wi[i] == 0) {
                        inxt = i + 1;
                    } else {
                        if (A[(i + 1) * lda + i] == 0) {
                            wi[i] = 0;
                            wi[i + 1] = 0;
                        } else if (A[(i + 1) * lda + i] != 0 && A[i * lda + i + 1] == 0) {
                            wi[i] = 0;
                            wi[i + 1] = 0;
                            if (i > 0) {
                                BLAS.dswap(i, A, i, 1, A, i + 1, 1);
                            }
                            if (i < n - 2) {
                                BLAS.dswap(n - i - 2, A, i * lda + i + 2, lda, A, (i + 1) * lda + i + 2, lda);
                            }
                            if (wantvs) {
                                BLAS.dswap(n, vs, i, 1, vs, i + 1, 1);
                            }
                            A[i * lda + i + 1] = A[(i + 1) * lda + i];
                            A[(i + 1) * lda + i] = 0;
                        }
                        inxt = i + 2;
                    }
                }
            }

            Dlamv.dlascl('G', 0, 0, cscale, anrm, n - ieval, 1, wi, ieval, 1);
        }

        if (wantst && info == 0) {
            int sdimCheck = 0;
            boolean lastsl = true;
            boolean lst2sl = true;
            int ip = 0;
            for (int i = 0; i < n; i++) {
                boolean cursl = select.select(wr[i], wi[i]);
                if (wi[i] == 0) {
                    if (cursl) {
                        sdimCheck += 1;
                    }
                    ip = 0;
                    if (cursl && !lastsl) {
                        info = n + 2;
                    }
                } else {
                    if (ip == 1) {
                        cursl = cursl || lastsl;
                        lastsl = cursl;
                        if (cursl) {
                            sdimCheck += 2;
                        }
                        ip = -1;
                        if (cursl && !lst2sl) {
                            info = n + 2;
                        }
                    } else {
                        ip = 1;
                    }
                }
                lst2sl = lastsl;
                lastsl = cursl;
            }
            sdim = sdimCheck;
        }

        work[workOff] = sdim;
        return info;
    }
}
