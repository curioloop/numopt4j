/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

interface Dgesvd {

    static int dgesvd(char jobU, char jobVT, int m, int n,
                      double[] A, int aOff, int lda,
                      double[] s, int sOff,
                      double[] u, int uOff, int ldu,
                      double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork) {
        boolean wantua = jobU == 'A' || jobU == 'a';
        boolean wantus = jobU == 'S' || jobU == 's';
        boolean wantuas = wantua || wantus;
        boolean wantuo = jobU == 'O' || jobU == 'o';
        boolean wantun = jobU == 'N' || jobU == 'n';

        boolean wantva = jobVT == 'A' || jobVT == 'a';
        boolean wantvs = jobVT == 'S' || jobVT == 's';
        boolean wantvas = wantva || wantvs;
        boolean wantvo = jobVT == 'O' || jobVT == 'o';
        boolean wantvn = jobVT == 'N' || jobVT == 'n';

        if (wantuo && wantvo) {
            return -1;
        }

        int minmn = min(m, n);
        int minwork = 1;
        if (minmn > 0) {
            minwork = max(3 * minmn + max(m, n), 5 * minmn);
        }

        if (m < 0) return -3;
        if (n < 0) return -4;
        if (lda < max(1, n)) return -6;
        if (ldu < 1 || (wantua && ldu < m) || (wantus && ldu < minmn)) return -9;
        if (ldvt < 1 || (wantvas && ldvt < n)) return -11;
        if (lwork < minwork && lwork != -1) return -13;

        if (minmn == 0) {
            if (lwork >= 1) work[workOff] = 1;
            return 0;
        }

        int mnthr = Ilaenv.ilaenv(6, "DGESVD", String.valueOf(jobU) + jobVT, m, n, 0, 0);

        int maxwrk = 1;
        int wrkbl = 0;
        int bdspac = 0;

        if (m >= n) {
            bdspac = 4 * n + 11;
            int lworkDgeqrf = queryDgeqrf(m, n, work, workOff);
            int lworkDorgqrN = queryDorgqr(m, n, n, work, workOff);
            int lworkDorgqrM = queryDorgqr(m, m, n, work, workOff);
            int lworkDgebrd = queryDgebrd(n, n, work, workOff);
            int lworkDorgbrP = queryDorgbr('P', n, n, n, work, workOff);
            int lworkDorgbrQ = queryDorgbr('Q', n, n, n, work, workOff);

            if (m >= mnthr) {
                if (wantun) {
                    maxwrk = n + lworkDgeqrf;
                    maxwrk = max(maxwrk, 3 * n + lworkDgebrd);
                    if (wantvo || wantvas) {
                        maxwrk = max(maxwrk, 3 * n + lworkDorgbrP);
                    }
                    maxwrk = max(maxwrk, bdspac);
                } else if (wantuo && wantvn) {
                    wrkbl = n + lworkDgeqrf;
                    wrkbl = max(wrkbl, n + lworkDorgqrN);
                    wrkbl = max(wrkbl, 3 * n + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = max(n * n + wrkbl, n * n + m * n + n);
                } else if (wantuo && wantvas) {
                    wrkbl = n + lworkDgeqrf;
                    wrkbl = max(wrkbl, n + lworkDorgqrN);
                    wrkbl = max(wrkbl, 3 * n + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrQ);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = max(n * n + wrkbl, n * n + m * n + n);
                } else if (wantus && wantvn) {
                    wrkbl = n + lworkDgeqrf;
                    wrkbl = max(wrkbl, n + lworkDorgqrN);
                    wrkbl = max(wrkbl, 3 * n + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = n * n + wrkbl;
                } else if (wantus && wantvo) {
                    wrkbl = n + lworkDgeqrf;
                    wrkbl = max(wrkbl, n + lworkDorgqrN);
                    wrkbl = max(wrkbl, 3 * n + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrQ);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = 2 * n * n + wrkbl;
                } else if (wantus && wantvas) {
                    wrkbl = n + lworkDgeqrf;
                    wrkbl = max(wrkbl, n + lworkDorgqrN);
                    wrkbl = max(wrkbl, 3 * n + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrQ);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = n * n + wrkbl;
                } else if (wantua && wantvn) {
                    wrkbl = n + lworkDgeqrf;
                    wrkbl = max(wrkbl, n + lworkDorgqrM);
                    wrkbl = max(wrkbl, 3 * n + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = n * n + wrkbl;
                } else if (wantua && wantvo) {
                    wrkbl = n + lworkDgeqrf;
                    wrkbl = max(wrkbl, n + lworkDorgqrM);
                    wrkbl = max(wrkbl, 3 * n + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrQ);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = 2 * n * n + wrkbl;
                } else if (wantua && wantvas) {
                    wrkbl = n + lworkDgeqrf;
                    wrkbl = max(wrkbl, n + lworkDorgqrM);
                    wrkbl = max(wrkbl, 3 * n + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrQ);
                    wrkbl = max(wrkbl, 3 * n + lworkDorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = n * n + wrkbl;
                }
            } else {
                maxwrk = 3 * n + lworkDgebrd;
                if (wantus || wantuo) {
                    maxwrk = max(maxwrk, 3 * n + lworkDorgbrQ);
                }
                if (wantua) {
                    maxwrk = max(maxwrk, 3 * n + lworkDorgbrQ);
                }
                if (!wantvn) {
                    maxwrk = max(maxwrk, 3 * n + lworkDorgbrP);
                }
                maxwrk = max(maxwrk, bdspac);
            }
        } else {
            bdspac = 4 * m + 11;
            int lworkDgelqf = queryDgelqf(m, n, work, workOff);
            int lworkDorglqN = queryDorglq(n, n, m, work, workOff);
            int lworkDorglqM = queryDorglq(m, n, m, work, workOff);
            int lworkDgebrd = queryDgebrd(m, m, work, workOff);
            int lworkDorgbrP = queryDorgbr('P', m, m, m, work, workOff);
            int lworkDorgbrQ = queryDorgbr('Q', m, m, m, work, workOff);

            if (n >= mnthr) {
                if (wantvn) {
                    maxwrk = m + lworkDgelqf;
                    maxwrk = max(maxwrk, 3 * m + lworkDgebrd);
                    if (wantuo || wantuas) {
                        maxwrk = max(maxwrk, 3 * m + lworkDorgbrQ);
                    }
                    maxwrk = max(maxwrk, bdspac);
                } else if (wantvo && wantun) {
                    wrkbl = m + lworkDgelqf;
                    wrkbl = max(wrkbl, m + lworkDorglqM);
                    wrkbl = max(wrkbl, 3 * m + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = max(m * m + wrkbl, m * m + m * n + m);
                } else if (wantvo && wantuas) {
                    wrkbl = m + lworkDgelqf;
                    wrkbl = max(wrkbl, m + lworkDorglqM);
                    wrkbl = max(wrkbl, 3 * m + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrP);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = max(m * m + wrkbl, m * m + m * n + m);
                } else if (wantvs && wantun) {
                    wrkbl = m + lworkDgelqf;
                    wrkbl = max(wrkbl, m + lworkDorglqM);
                    wrkbl = max(wrkbl, 3 * m + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = m * m + wrkbl;
                } else if (wantvs && wantuo) {
                    wrkbl = m + lworkDgelqf;
                    wrkbl = max(wrkbl, m + lworkDorglqM);
                    wrkbl = max(wrkbl, 3 * m + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrP);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = 2 * m * m + wrkbl;
                } else if (wantvs && wantuas) {
                    wrkbl = m + lworkDgelqf;
                    wrkbl = max(wrkbl, m + lworkDorglqM);
                    wrkbl = max(wrkbl, 3 * m + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrP);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = m * m + wrkbl;
                } else if (wantva && wantun) {
                    wrkbl = m + lworkDgelqf;
                    wrkbl = max(wrkbl, m + lworkDorglqN);
                    wrkbl = max(wrkbl, 3 * m + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrP);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = m * m + wrkbl;
                } else if (wantva && wantuo) {
                    wrkbl = m + lworkDgelqf;
                    wrkbl = max(wrkbl, m + lworkDorglqN);
                    wrkbl = max(wrkbl, 3 * m + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrP);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = 2 * m * m + wrkbl;
                } else if (wantva && wantuas) {
                    wrkbl = m + lworkDgelqf;
                    wrkbl = max(wrkbl, m + lworkDorglqN);
                    wrkbl = max(wrkbl, 3 * m + lworkDgebrd);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrP);
                    wrkbl = max(wrkbl, 3 * m + lworkDorgbrQ);
                    wrkbl = max(wrkbl, bdspac);
                    maxwrk = m * m + wrkbl;
                }
            } else {
                maxwrk = 3 * m + lworkDgebrd;
                if (wantvs || wantvo) {
                    maxwrk = max(maxwrk, 3 * m + lworkDorgbrP);
                }
                if (wantva) {
                    maxwrk = max(maxwrk, 3 * m + lworkDorgbrP);
                }
                if (!wantun) {
                    maxwrk = max(maxwrk, 3 * m + lworkDorgbrQ);
                }
                maxwrk = max(maxwrk, bdspac);
            }
        }

        maxwrk = max(maxwrk, minwork);
        if (lwork == -1) {
            work[workOff] = maxwrk;
            return 0;
        }

        double eps = Dlamch.EPSILON;
        double smlnum = sqrt(Dlamch.SAFE_MIN) / eps;
        double bignum = 1.0 / smlnum;

        double anrm = Dlange.dlange('M', m, n, A, aOff, lda);
        boolean iscl = false;
        if (anrm > 0 && anrm < smlnum) {
            iscl = true;
            Dlamv.dlascl('G', 0, 0, anrm, smlnum, m, n, A, aOff, lda);
        } else if (anrm > bignum) {
            iscl = true;
            Dlamv.dlascl('G', 0, 0, anrm, bignum, m, n, A, aOff, lda);
        }

        boolean ok = true;
        int ie = 0;

        if (m >= n) {
            if (m >= mnthr) {
                if (wantun) {
                    ie = path1(m, n, A, aOff, lda, s, sOff, vt, vtOff, ldvt,
                               work, workOff, lwork, wantvo, wantvas);
                    ok = ie >= 0;
                    if (ie < 0) ie = 0;
                } else if (wantuo && wantvn) {
                    return -1;
                } else if (wantuo && wantvas) {
                    return -1;
                } else if (wantus) {
                    if (wantvn) {
                        ie = path4(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                   work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    } else if (wantvo) {
                        return -1;
                    } else if (wantvas) {
                        ie = path6(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                   work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    }
                } else if (wantua) {
                    if (wantvn) {
                        ie = path7(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                   work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    } else if (wantvo) {
                        return -1;
                    } else if (wantvas) {
                        ie = path9(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                   work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    }
                }
            } else {
                ie = path10(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                            work, workOff, lwork, wantuas, wantua, wantvas, wantva);
                ok = ie >= 0;
                if (ie < 0) ie = 0;
            }
        } else {
            if (n >= mnthr) {
                if (wantvn) {
                    ie = path1t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu,
                                work, workOff, lwork, wantuo, wantuas);
                    ok = ie >= 0;
                    if (ie < 0) ie = 0;
                } else if (wantvo && wantun) {
                    return -1;
                } else if (wantvo && wantuas) {
                    return -1;
                } else if (wantvs) {
                    if (wantun) {
                        ie = path4t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                    work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    } else if (wantuo) {
                        return -1;
                    } else if (wantuas) {
                        ie = path6t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                    work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    }
                } else if (wantva) {
                    if (wantun) {
                        ie = path7t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                    work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    } else if (wantuo) {
                        return -1;
                    } else if (wantuas) {
                        ie = path9t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                                    work, workOff, lwork, wrkbl, bdspac, lda);
                        ok = ie >= 0;
                        if (ie < 0) ie = 0;
                    }
                }
            } else {
                ie = path10t(m, n, A, aOff, lda, s, sOff, u, uOff, ldu, vt, vtOff, ldvt,
                             work, workOff, lwork, wantuas, wantva, wantvas, wantva);
                ok = ie >= 0;
                if (ie < 0) ie = 0;
            }
        }

        if (!ok) {
            if (ie > 1) {
                for (int i = 0; i < minmn - 1; i++) {
                    work[workOff + i + 1] = work[workOff + i + ie];
                }
            }
            if (ie < 1) {
                for (int i = minmn - 2; i >= 0; i--) {
                    work[workOff + i + 1] = work[workOff + i + ie];
                }
            }
        }

        if (iscl) {
            if (anrm > bignum) {
                Dlamv.dlascl('G', 0, 0, bignum, anrm, 1, minmn, s, sOff, 1);
            }
            if (!ok && anrm > bignum) {
                Dlamv.dlascl('G', 0, 0, bignum, anrm, 1, minmn - 1, work, workOff + 1, 1);
            }
            if (anrm < smlnum) {
                Dlamv.dlascl('G', 0, 0, smlnum, anrm, 1, minmn, s, sOff, 1);
            }
            if (!ok && anrm < smlnum) {
                Dlamv.dlascl('G', 0, 0, smlnum, anrm, 1, minmn - 1, work, workOff + 1, 1);
            }
        }

        work[workOff] = maxwrk;
        return ok ? 0 : 1;
    }

    static int path1(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                     double[] vt, int vtOff, int ldvt,
                     double[] work, int workOff, int lwork,
                     boolean wantvo, boolean wantvas) {
        int itau = 0;
        int iwork = itau + n;

        Dgeqr.dgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

        if (n > 1) {
            Dlamv.dlaset('L', n - 1, n - 1, 0, 0, A, aOff + lda, lda);
        }

        int ie = 0;
        int itauq = ie + n;
        int itaup = itauq + n;
        iwork = itaup + n;

        Dgebrd.dgebrd(n, n, A, aOff, lda, s, sOff, work, workOff + ie, work, workOff + itauq,
                      work, workOff + itaup, work, workOff + iwork, lwork - iwork);

        int ncvt = 0;
        if (wantvo || wantvas) {
            Dorgbr.dorgbr('P', n, n, n, A, aOff, lda, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            ncvt = n;
        }
        iwork = ie + n;

        boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, ncvt, 0, 0, s, sOff, work, workOff + ie, A, aOff, lda,
                                   work, workOff, 1, work, workOff, 1, work, workOff + iwork);

        if (wantvas) {
            Dlamv.dlacpy('A', n, n, A, aOff, lda, vt, vtOff, ldvt);
        }

        return ok ? ie : -ie;
    }

    static int path4(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                     double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                     double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworkr) {
        if (lwork >= n * n + max(max(4 * n, bdspac), wrkbl)) {
            int ir = 0;
            if (lwork >= wrkbl + ldworkr * n) {
                ldworkr = lda;
            } else {
                ldworkr = n;
            }
            int itau = ir + ldworkr * n;
            int iwork = itau + n;

            Dgeqr.dgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            Dlamv.dlacpy('U', n, n, A, aOff, lda, work, workOff + ir, ldworkr);
            Dlamv.dlaset('L', n - 1, n - 1, 0, 0, work, workOff + ir + ldworkr, ldworkr);

            Dgeqr.dorgqr(m, n, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n;
            iwork = itaup + n;

            Dgebrd.dgebrd(n, n, work, workOff + ir, ldworkr, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Dorgbr.dorgbr('Q', n, n, n, work, workOff + ir, ldworkr, work, workOff + itauq,
                          work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, 0, n, 0, s, sOff, work, workOff + ie, work, workOff, 1,
                                       work, workOff + ir, ldworkr, work, workOff, 1, work, workOff + iwork);

            Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1, A, aOff, lda, work, workOff + ir, ldworkr, 0, u, uOff, ldu);

            return ok ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + n;

            Dgeqr.dgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);

            Dgeqr.dorgqr(m, n, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n;
            iwork = itaup + n;

            Dlamv.dlaset('L', n - 1, n - 1, 0, 0, A, aOff + lda, lda);

            Dgebrd.dgebrd(n, n, A, aOff, lda, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Dormbr.dormbr('Q', BLAS.Side.Right, BLAS.Trans.NoTrans, m, n, n, A, aOff, lda, work, workOff + itauq,
                          u, uOff, ldu, work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, 0, m, 0, s, sOff, work, workOff + ie, work, workOff, 1,
                                       u, uOff, ldu, work, workOff, 1, work, workOff + iwork);

            return ok ? ie : -ie;
        }
    }

    static int path6(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                     double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                     double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworku) {
        if (lwork >= n * n + max(max(4 * n, bdspac), wrkbl)) {
            int iu = 0;
            if (lwork >= wrkbl + ldworku * n) {
                ldworku = lda;
            } else {
                ldworku = n;
            }
            int itau = iu + ldworku * n;
            int iwork = itau + n;

            Dgeqr.dgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            Dlamv.dlacpy('U', n, n, A, aOff, lda, work, workOff + iu, ldworku);
            Dlamv.dlaset('L', n - 1, n - 1, 0, 0, work, workOff + iu + ldworku, ldworku);

            Dgeqr.dorgqr(m, n, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n;
            iwork = itaup + n;

            Dgebrd.dgebrd(n, n, work, workOff + iu, ldworku, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('U', n, n, work, workOff + iu, ldworku, vt, vtOff, ldvt);

            Dorgbr.dorgbr('Q', n, n, n, work, workOff + iu, ldworku, work, workOff + itauq,
                          work, workOff + iwork, lwork - iwork);

            Dorgbr.dorgbr('P', n, n, n, vt, vtOff, ldvt, work, workOff + itaup,
                          work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, n, n, 0, s, sOff, work, workOff + ie, vt, vtOff, ldvt,
                                       work, workOff + iu, ldworku, work, workOff, 1, work, workOff + iwork);

            Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1, A, aOff, lda, work, workOff + iu, ldworku, 0, u, uOff, ldu);

            return ok ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + n;

            Dgeqr.dgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);

            Dgeqr.dorgqr(m, n, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            Dlamv.dlacpy('U', n, n, A, aOff, lda, vt, vtOff, ldvt);
            Dlamv.dlaset('L', n - 1, n - 1, 0, 0, vt, vtOff + ldvt, ldvt);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n;
            iwork = itaup + n;

            Dgebrd.dgebrd(n, n, vt, vtOff, ldvt, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Dormbr.dormbr('Q', BLAS.Side.Right, BLAS.Trans.NoTrans, m, n, n, vt, vtOff, ldvt, work, workOff + itauq,
                          u, uOff, ldu, work, workOff + iwork, lwork - iwork);

            Dorgbr.dorgbr('P', n, n, n, vt, vtOff, ldvt, work, workOff + itaup,
                          work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, n, m, 0, s, sOff, work, workOff + ie, vt, vtOff, ldvt,
                                       u, uOff, ldu, work, workOff, 1, work, workOff + iwork);

            return ok ? ie : -ie;
        }
    }

    static int path7(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                     double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                     double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworkr) {
        if (lwork >= n * n + max(max(n + m, 4 * n), max(bdspac, wrkbl))) {
            int ir = 0;
            if (lwork >= wrkbl + ldworkr * n) {
                ldworkr = lda;
            } else {
                ldworkr = n;
            }
            int itau = ir + ldworkr * n;
            int iwork = itau + n;

            Dgeqr.dgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);

            Dlamv.dlacpy('U', n, n, A, aOff, lda, work, workOff + ir, ldworkr);
            Dlamv.dlaset('L', n - 1, n - 1, 0, 0, work, workOff + ir + ldworkr, ldworkr);

            Dgeqr.dorgqr(m, m, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n;
            iwork = itaup + n;

            Dgebrd.dgebrd(n, n, work, workOff + ir, ldworkr, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Dorgbr.dorgbr('Q', n, n, n, work, workOff + ir, ldworkr, work, workOff + itauq,
                          work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, 0, n, 0, s, sOff, work, workOff + ie, work, workOff, 1,
                                       work, workOff + ir, ldworkr, work, workOff, 1, work, workOff + iwork);

            Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1, u, uOff, ldu, work, workOff + ir, ldworkr, 0, A, aOff, lda);

            Dlamv.dlacpy('A', m, n, A, aOff, lda, u, uOff, ldu);

            return ok ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + n;

            Dgeqr.dgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);

            Dgeqr.dorgqr(m, m, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n;
            iwork = itaup + n;

            Dlamv.dlaset('L', n - 1, n - 1, 0, 0, A, aOff + lda, lda);

            Dgebrd.dgebrd(n, n, A, aOff, lda, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Dormbr.dormbr('Q', BLAS.Side.Right, BLAS.Trans.NoTrans, m, n, n, A, aOff, lda, work, workOff + itauq,
                          u, uOff, ldu, work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, 0, m, 0, s, sOff, work, workOff + ie, work, workOff, 1,
                                       u, uOff, ldu, work, workOff, 1, work, workOff + iwork);

            return ok ? ie : -ie;
        }
    }

    static int path9(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                     double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                     double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworku) {
        if (lwork >= n * n + max(max(n + m, 4 * n), max(bdspac, wrkbl))) {
            int iu = 0;
            if (lwork >= wrkbl + ldworku * n) {
                ldworku = lda;
            } else {
                ldworku = n;
            }
            int itau = iu + ldworku * n;
            int iwork = itau + n;

            Dgeqr.dgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);

            Dgeqr.dorgqr(m, m, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            Dlamv.dlacpy('U', n, n, A, aOff, lda, work, workOff + iu, ldworku);
            Dlamv.dlaset('L', n - 1, n - 1, 0, 0, work, workOff + iu + ldworku, ldworku);

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n;
            iwork = itaup + n;

            Dgebrd.dgebrd(n, n, work, workOff + iu, ldworku, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('U', n, n, work, workOff + iu, ldworku, vt, vtOff, ldvt);

            Dorgbr.dorgbr('Q', n, n, n, work, workOff + iu, ldworku, work, workOff + itauq,
                          work, workOff + iwork, lwork - iwork);

            Dorgbr.dorgbr('P', n, n, n, vt, vtOff, ldvt, work, workOff + itaup,
                          work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, n, n, 0, s, sOff, work, workOff + ie, vt, vtOff, ldvt,
                                       work, workOff + iu, ldworku, work, workOff, 1, work, workOff + iwork);

            Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, n, 1, u, uOff, ldu, work, workOff + iu, ldworku, 0, A, aOff, lda);

            Dlamv.dlacpy('A', m, n, A, aOff, lda, u, uOff, ldu);

            return ok ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + n;

            Dgeqr.dgeqrf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);

            Dgeqr.dorgqr(m, m, n, u, uOff, ldu, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            Dlamv.dlacpy('U', n, n, A, aOff, lda, vt, vtOff, ldvt);
            if (n > 1) {
                Dlamv.dlaset('L', n - 1, n - 1, 0, 0, vt, vtOff + ldvt, ldvt);
            }

            int ie = itau;
            int itauq = ie + n;
            int itaup = itauq + n;
            iwork = itaup + n;

            Dgebrd.dgebrd(n, n, vt, vtOff, ldvt, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Dormbr.dormbr('Q', BLAS.Side.Right, BLAS.Trans.NoTrans, m, n, n, vt, vtOff, ldvt, work, workOff + itauq,
                          u, uOff, ldu, work, workOff + iwork, lwork - iwork);

            Dorgbr.dorgbr('P', n, n, n, vt, vtOff, ldvt, work, workOff + itaup,
                          work, workOff + iwork, lwork - iwork);
            iwork = ie + n;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, n, m, 0, s, sOff, work, workOff + ie, vt, vtOff, ldvt,
                                       u, uOff, ldu, work, workOff, 1, work, workOff + iwork);

            return ok ? ie : -ie;
        }
    }

    static int path10(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                      double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork,
                      boolean wantuas, boolean wantua, boolean wantvas, boolean wantva) {
        int ie = 0;
        int itauq = ie + n;
        int itaup = itauq + n;
        int iwork = itaup + n;

        Dgebrd.dgebrd(m, n, A, aOff, lda, s, sOff, work, workOff + ie, work, workOff + itauq,
                      work, workOff + itaup, work, workOff + iwork, lwork - iwork);

        if (wantuas) {
            Dlamv.dlacpy('L', m, n, A, aOff, lda, u, uOff, ldu);
            int ncu = wantua ? m : n;
            Dorgbr.dorgbr('Q', m, ncu, n, u, uOff, ldu, work, workOff + itauq,
                          work, workOff + iwork, lwork - iwork);
        }

        if (wantvas) {
            Dlamv.dlacpy('U', n, n, A, aOff, lda, vt, vtOff, ldvt);
            Dorgbr.dorgbr('P', n, n, n, vt, vtOff, ldvt, work, workOff + itaup,
                          work, workOff + iwork, lwork - iwork);
        }

        iwork = ie + n;
        int nru = wantuas ? m : 0;
        int ncvt = wantvas ? n : 0;

        boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, n, ncvt, nru, 0, s, sOff, work, workOff + ie, vt, vtOff, ldvt,
                                   u, uOff, ldu, work, workOff, 1, work, workOff + iwork);

        return ok ? ie : -ie;
    }

    static int path1t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                      double[] u, int uOff, int ldu,
                      double[] work, int workOff, int lwork,
                      boolean wantuo, boolean wantuas) {
        int itau = 0;
        int iwork = itau + m;

        Dgelq.dgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

        if (m > 1) {
            Dlamv.dlaset('U', m - 1, m - 1, 0, 0, A, aOff + 1, lda);
        }

        int ie = 0;
        int itauq = ie + m;
        int itaup = itauq + m;
        iwork = itaup + m;

        Dgebrd.dgebrd(m, m, A, aOff, lda, s, sOff, work, workOff + ie, work, workOff + itauq,
                      work, workOff + itaup, work, workOff + iwork, lwork - iwork);

        if (wantuo || wantuas) {
            Dorgbr.dorgbr('Q', m, m, m, A, aOff, lda, work, workOff + itauq,
                          work, workOff + iwork, lwork - iwork);
        }
        iwork = ie + m;
        int nru = (wantuo || wantuas) ? m : 0;

        boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, m, 0, nru, 0, s, sOff, work, workOff + ie, A, aOff, lda,
                                   work, workOff, 1, work, workOff, 1, work, workOff + iwork);

        if (wantuas) {
            Dlamv.dlacpy('A', m, m, A, aOff, lda, u, uOff, ldu);
        }

        return ok ? ie : -ie;
    }

    static int path4t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                      double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworkr) {
        if (lwork >= m * m + max(max(4 * m, bdspac), wrkbl)) {
            int ir = 0;
            if (lwork >= wrkbl + ldworkr * m) {
                ldworkr = lda;
            } else {
                ldworkr = m;
            }
            int itau = ir + ldworkr * m;
            int iwork = itau + m;

            Dgelq.dgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            Dlamv.dlacpy('L', m, m, A, aOff, lda, work, workOff + ir, ldworkr);
            Dlamv.dlaset('U', m - 1, m - 1, 0, 0, work, workOff + ir + 1, ldworkr);

            Dgelq.dorglq(m, n, m, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m;
            iwork = itaup + m;

            Dgebrd.dgebrd(m, m, work, workOff + ir, ldworkr, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Dorgbr.dorgbr('P', m, m, m, work, workOff + ir, ldworkr, work, workOff + itaup,
                          work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, m, m, 0, 0, s, sOff, work, workOff + ie, work, workOff + ir, ldworkr,
                                       work, workOff, 1, work, workOff, 1, work, workOff + iwork);

            Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, m, 1, work, workOff + ir, ldworkr, A, aOff, lda, 0, vt, vtOff, ldvt);

            return ok ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + m;

            Dgelq.dgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            Dlamv.dlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);

            Dgelq.dorglq(m, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m;
            iwork = itaup + m;

            Dlamv.dlaset('U', m - 1, m - 1, 0, 0, A, aOff + 1, lda);

            Dgebrd.dgebrd(m, m, A, aOff, lda, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Dormbr.dormbr('P', BLAS.Side.Left, BLAS.Trans.Trans, m, n, m, A, aOff, lda, work, workOff + itaup,
                          vt, vtOff, ldvt, work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, m, n, 0, 0, s, sOff, work, workOff + ie, vt, vtOff, ldvt,
                                       work, workOff, 1, work, workOff, 1, work, workOff + iwork);

            return ok ? ie : -ie;
        }
    }

    static int path6t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                      double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworku) {
        if (lwork >= m * m + max(max(4 * m, bdspac), wrkbl)) {
            int iu = 0;
            if (lwork >= wrkbl + ldworku * m) {
                ldworku = lda;
            } else {
                ldworku = m;
            }
            int itau = iu + ldworku * m;
            int iwork = itau + m;

            Dgelq.dgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            Dlamv.dlacpy('L', m, m, A, aOff, lda, work, workOff + iu, ldworku);
            Dlamv.dlaset('U', m - 1, m - 1, 0, 0, work, workOff + iu + 1, ldworku);

            Dgelq.dorglq(m, n, m, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m;
            iwork = itaup + m;

            Dgebrd.dgebrd(m, m, work, workOff + iu, ldworku, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('L', m, m, work, workOff + iu, ldworku, u, uOff, ldu);

            Dorgbr.dorgbr('P', m, m, m, work, workOff + iu, ldworku, work, workOff + itaup,
                          work, workOff + iwork, lwork - iwork);

            Dorgbr.dorgbr('Q', m, m, m, u, uOff, ldu, work, workOff + itauq,
                          work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, m, m, m, 0, s, sOff, work, workOff + ie, work, workOff + iu, ldworku,
                                       u, uOff, ldu, work, workOff, 1, work, workOff + iwork);

            Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, m, 1, work, workOff + iu, ldworku, A, aOff, lda, 0, vt, vtOff, ldvt);

            return ok ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + m;

            Dgelq.dgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            Dlamv.dlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);

            Dgelq.dorglq(m, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            Dlamv.dlacpy('L', m, m, A, aOff, lda, u, uOff, ldu);
            Dlamv.dlaset('U', m - 1, m - 1, 0, 0, u, uOff + 1, ldu);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m;
            iwork = itaup + m;

            Dgebrd.dgebrd(m, m, u, uOff, ldu, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Dormbr.dormbr('P', BLAS.Side.Left, BLAS.Trans.Trans, m, n, m, u, uOff, ldu, work, workOff + itaup,
                          vt, vtOff, ldvt, work, workOff + iwork, lwork - iwork);

            Dorgbr.dorgbr('Q', m, m, m, u, uOff, ldu, work, workOff + itauq,
                          work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, m, n, m, 0, s, sOff, work, workOff + ie, vt, vtOff, ldvt,
                                       u, uOff, ldu, work, workOff, 1, work, workOff + iwork);

            return ok ? ie : -ie;
        }
    }

    static int path7t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                      double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworkr) {
        if (lwork >= m * m + max(max(n + m, 4 * m), max(bdspac, wrkbl))) {
            int ir = 0;
            if (lwork >= wrkbl + ldworkr * m) {
                ldworkr = lda;
            } else {
                ldworkr = m;
            }
            int itau = ir + ldworkr * m;
            int iwork = itau + m;

            Dgelq.dgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);

            Dlamv.dlacpy('L', m, m, A, aOff, lda, work, workOff + ir, ldworkr);
            Dlamv.dlaset('U', m - 1, m - 1, 0, 0, work, workOff + ir + 1, ldworkr);

            Dgelq.dorglq(n, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m;
            iwork = itaup + m;

            Dgebrd.dgebrd(m, m, work, workOff + ir, ldworkr, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Dorgbr.dorgbr('P', m, m, m, work, workOff + ir, ldworkr, work, workOff + itaup,
                          work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, m, m, 0, 0, s, sOff, work, workOff + ie, work, workOff + ir, ldworkr,
                                       work, workOff, 1, work, workOff, 1, work, workOff + iwork);

            Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, m, 1, work, workOff + ir, ldworkr, vt, vtOff, ldvt, 0, A, aOff, lda);

            Dlamv.dlacpy('A', m, n, A, aOff, lda, vt, vtOff, ldvt);

            return ok ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + m;

            Dgelq.dgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);

            Dgelq.dorglq(n, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m;
            iwork = itaup + m;

            Dlamv.dlaset('U', m - 1, m - 1, 0, 0, A, aOff + 1, lda);

            Dgebrd.dgebrd(m, m, A, aOff, lda, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Dormbr.dormbr('P', BLAS.Side.Left, BLAS.Trans.Trans, m, n, m, A, aOff, lda, work, workOff + itaup,
                          vt, vtOff, ldvt, work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, m, n, 0, 0, s, sOff, work, workOff + ie, vt, vtOff, ldvt,
                                       work, workOff, 1, work, workOff, 1, work, workOff + iwork);

            return ok ? ie : -ie;
        }
    }

    static int path9t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                      double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                      double[] work, int workOff, int lwork, int wrkbl, int bdspac, int ldworku) {
        if (lwork >= m * m + max(max(m + n, 4 * m), max(bdspac, wrkbl))) {
            int iu = 0;
            if (lwork >= wrkbl + ldworku * m) {
                ldworku = lda;
            } else {
                ldworku = m;
            }
            int itau = iu + ldworku * m;
            int iwork = itau + m;

            Dgelq.dgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);

            Dgelq.dorglq(n, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            Dlamv.dlacpy('L', m, m, A, aOff, lda, work, workOff + iu, ldworku);
            Dlamv.dlaset('U', m - 1, m - 1, 0, 0, work, workOff + iu + 1, ldworku);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m;
            iwork = itaup + m;

            Dgebrd.dgebrd(m, m, work, workOff + iu, ldworku, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('L', m, m, work, workOff + iu, ldworku, u, uOff, ldu);

            Dorgbr.dorgbr('P', m, m, m, work, workOff + iu, ldworku, work, workOff + itaup,
                          work, workOff + iwork, lwork - iwork);

            Dorgbr.dorgbr('Q', m, m, m, u, uOff, ldu, work, workOff + itauq,
                          work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, m, m, m, 0, s, sOff, work, workOff + ie, work, workOff + iu, ldworku,
                                       u, uOff, ldu, work, workOff, 1, work, workOff + iwork);

            Dgemm.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, m, n, m, 1, work, workOff + iu, ldworku, vt, vtOff, ldvt, 0, A, aOff, lda);

            Dlamv.dlacpy('A', m, n, A, aOff, lda, vt, vtOff, ldvt);

            return ok ? ie : -ie;
        } else {
            int itau = 0;
            int iwork = itau + m;

            Dgelq.dgelqf(m, n, A, aOff, lda, work, workOff + itau, work, workOff + iwork, lwork - iwork);
            Dlamv.dlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);

            Dgelq.dorglq(n, n, m, vt, vtOff, ldvt, work, workOff + itau, work, workOff + iwork, lwork - iwork);

            Dlamv.dlacpy('L', m, m, A, aOff, lda, u, uOff, ldu);
            Dlamv.dlaset('U', m - 1, m - 1, 0, 0, u, uOff + 1, ldu);

            int ie = itau;
            int itauq = ie + m;
            int itaup = itauq + m;
            iwork = itaup + m;

            Dgebrd.dgebrd(m, m, u, uOff, ldu, s, sOff, work, workOff + ie,
                          work, workOff + itauq, work, workOff + itaup, work, workOff + iwork, lwork - iwork);

            Dormbr.dormbr('P', BLAS.Side.Left, BLAS.Trans.Trans, m, n, m, u, uOff, ldu, work, workOff + itaup,
                          vt, vtOff, ldvt, work, workOff + iwork, lwork - iwork);

            Dorgbr.dorgbr('Q', m, m, m, u, uOff, ldu, work, workOff + itauq,
                          work, workOff + iwork, lwork - iwork);
            iwork = ie + m;

            boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Upper, m, n, m, 0, s, sOff, work, workOff + ie, vt, vtOff, ldvt,
                                       u, uOff, ldu, work, workOff, 1, work, workOff + iwork);

            return ok ? ie : -ie;
        }
    }

    static int path10t(int m, int n, double[] A, int aOff, int lda, double[] s, int sOff,
                       double[] u, int uOff, int ldu, double[] vt, int vtOff, int ldvt,
                       double[] work, int workOff, int lwork,
                       boolean wantuas, boolean wantva, boolean wantvas, boolean wantvs) {
        int ie = 0;
        int itauq = ie + m;
        int itaup = itauq + m;
        int iwork = itaup + m;

        Dgebrd.dgebrd(m, n, A, aOff, lda, s, sOff, work, workOff + ie, work, workOff + itauq,
                      work, workOff + itaup, work, workOff + iwork, lwork - iwork);

        if (wantuas) {
            Dlamv.dlacpy('L', m, m, A, aOff, lda, u, uOff, ldu);
            Dorgbr.dorgbr('Q', m, m, n, u, uOff, ldu, work, workOff + itauq,
                          work, workOff + iwork, lwork - iwork);
        }

        if (wantvas) {
            Dlamv.dlacpy('U', m, n, A, aOff, lda, vt, vtOff, ldvt);
            int nrvt = wantva ? n : m;
            Dorgbr.dorgbr('P', nrvt, n, m, vt, vtOff, ldvt, work, workOff + itaup,
                          work, workOff + iwork, lwork - iwork);
        }

        iwork = ie + m;
        int nru = wantuas ? m : 0;
        int ncvt = wantvas ? n : 0;

        boolean ok = Dbdsqr.dbdsqr(BLAS.Uplo.Lower, m, ncvt, nru, 0, s, sOff, work, workOff + ie, vt, vtOff, ldvt,
                                   u, uOff, ldu, work, workOff, 1, work, workOff + iwork);

        return ok ? ie : -ie;
    }

    static int queryDgeqrf(int m, int n, double[] work, int workOff) {
        Dgeqr.dgeqrf(m, n, null, 0, m, work, workOff, work, workOff, -1);
        return (int) work[workOff];
    }

    static int queryDorgqr(int m, int n, int k, double[] work, int workOff) {
        Dgeqr.dorgqr(m, n, k, null, 0, m, null, 0, work, workOff, -1);
        return (int) work[workOff];
    }

    static int queryDgebrd(int m, int n, double[] work, int workOff) {
        Dgebrd.dgebrd(m, n, null, 0, m, null, 0, null, 0, null, 0, null, 0, work, workOff, -1);
        return (int) work[workOff];
    }

    static int queryDorgbr(char vect, int m, int n, int k, double[] work, int workOff) {
        Dorgbr.dorgbr(vect, m, n, k, null, 0, m, null, 0, work, workOff, -1);
        return (int) work[workOff];
    }

    static int queryDgelqf(int m, int n, double[] work, int workOff) {
        Dgelq.dgelqf(m, n, null, 0, m, null, 0, work, workOff, -1);
        return (int) work[workOff];
    }

    static int queryDorglq(int m, int n, int k, double[] work, int workOff) {
        Dgelq.dorglq(m, n, k, null, 0, m, null, 0, work, workOff, -1);
        return (int) work[workOff];
    }
}
