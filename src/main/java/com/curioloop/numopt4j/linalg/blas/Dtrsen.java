/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.sqrt;

/**
 * Dtrsen reorders the real Schur factorization of a real matrix
 * A = Q*T*Q^T, so that a selected cluster of eigenvalues appears in
 * the leading diagonal blocks of the upper quasi-triangular matrix T,
 * and the leading columns of Q form an orthonormal basis of the
 * corresponding right invariant subspace.
 *
 * <p>Optionally the routine computes the reciprocal condition numbers of
 * the cluster of eigenvalues and/or the invariant subspace.</p>
 *
 * <p>T must be in Schur canonical form (as returned by Dhseqr), that is,
 * block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
 * 2-by-2 diagonal block has its diagonal elements equal and its
 * off-diagonal elements of opposite sign.</p>
 *
 * <p>Output values are written to arrays before returning:</p>
 * <ul>
 *   <li>{@code iwork[1]} = m (number of selected eigenvalues)</li>
 *   <li>{@code work[1]} = s (condition number estimate, only when job requires it)</li>
 *   <li>{@code work[2]} = sep (subspace separation, only when job requires it)</li>
 * </ul>
 *
 */
interface Dtrsen {

    char NO_COND = 'N';
    char EIGENVAL_COND = 'E';
    char SUBSPACE_COND = 'V';
    char BOTH_COND = 'B';

    static boolean dtrsen(char job, boolean wantq, boolean[] selects, int n,
                          double[] t, int tOff, int ldt,
                          double[] q, int qOff, int ldq,
                          double[] wr, double[] wi,
                          double[] work, int lwork,
                          int[] iwork, int liwork) {

        boolean wants = job == SUBSPACE_COND || job == BOTH_COND;
        boolean wantsp = job == EIGENVAL_COND || job == BOTH_COND;

        if (n == 0) {
            iwork[1] = 0;
            if (wants) {
                work[1] = 1.0;
            }
            if (wantsp) {
                work[2] = 0.0;
            }
            return true;
        }

        int m = 0;
        boolean ok = true;
        boolean pair = false;
        for (int k = 0; k < n; k++) {
            if (pair) {
                pair = false;
            } else {
                if (k < n - 1) {
                    if (t[tOff + (k + 1) * ldt + k] == 0) {
                        if (selects[k]) {
                            m += 1;
                        }
                    } else {
                        pair = true;
                        if (selects[k] || selects[k + 1]) {
                            m += 2;
                        }
                    }
                } else {
                    if (selects[n - 1]) {
                        m += 1;
                    }
                }
            }
        }

        int n1 = m;
        int n2 = n - m;
        int nn = n1 * n2;

        int lwmin = 1;
        int liwmin = 1;
        if (wantsp) {
            lwmin = max(n, 2 * nn);
            liwmin = max(1, nn);
        } else if (job == NO_COND) {
            lwmin = max(1, n);
        } else if (job == SUBSPACE_COND) {
            lwmin = max(1, nn);
        }

        if (lwork == -1) {
            work[0] = lwmin;
            iwork[0] = liwmin;
            iwork[1] = m;
            return true;
        }

        if (m == n || m == 0) {
            double s = 0.0;
            double sep = 0.0;
            if (wants) {
                s = 1.0;
            }
            if (wantsp) {
                sep = Dlange.dlange('1', n, n, t, tOff, ldt);
            }
            storeEigen(t, tOff, ldt, wr, wi, n);
            iwork[1] = m;
            if (wants) {
                work[1] = s;
            }
            if (wantsp) {
                work[2] = sep;
            }
            return true;
        }

        int ks = -1;
        pair = false;
        for (int k = 0; k < n; k++) {
            if (pair) {
                pair = false;
            } else {
                boolean swap = selects[k];
                if (k < n - 1) {
                    if (t[tOff + (k + 1) * ldt + k] != 0) {
                        pair = true;
                        swap = swap || selects[k + 1];
                    }
                }
                if (swap) {
                    ks += 1;
                    if (k != ks) {
                        if (!Dtrexc.dtrexc(wantq, n, t, tOff, ldt, q, qOff, ldq, k, ks, work, 0)) {
                            storeEigen(t, tOff, ldt, wr, wi, n);
                            iwork[1] = m;
                            if (wants) {
                                work[1] = 0.0;
                            }
                            if (wantsp) {
                                work[2] = 0.0;
                            }
                            return false;
                        }
                        ks = (int) work[1];
                    }
                    if (pair) {
                        ks += 1;
                    }
                }
            }
        }

        double s = 0.0;
        double sep = 0.0;
        double scale = 1.0;

        if (wants) {
            Dlamv.dlacpy('A', n1, n2, t, tOff + n1, ldt, work, 0, n2);
            scale = Dtrsyl.dtrsyl(false, false, -1, n1, n2,
                    t, tOff, ldt, t, tOff + n1 * ldt + n1, ldt, work, 0, n2, null);

            double rnorm = Dlange.dlange('F', n1, n2, work, 0, n2);
            if (rnorm == 0) {
                s = 1;
            } else {
                s = scale / (sqrt(scale * scale / rnorm + rnorm) * sqrt(rnorm));
            }
        }

        if (wantsp) {
            int[] isave = new int[4];
            double est = 0;

            while (true) {
                est = Dlacn2.dlacn2(nn, work, nn, work, 0, iwork, 0, est, isave, 3, isave, 0);

                if (isave[3] == 0) {
                    break;
                }

                if (isave[3] == 1) {
                    scale = Dtrsyl.dtrsyl(false, false, -1, n1, n2,
                            t, tOff, ldt, t, tOff + n1 * ldt + n1, ldt, work, 0, n2, null);
                } else {
                    scale = Dtrsyl.dtrsyl(true, true, -1, n1, n2,
                            t, tOff, ldt, t, tOff + n1 * ldt + n1, ldt, work, 0, n2, null);
                }
            }
            sep = scale / est;
        }

        storeEigen(t, tOff, ldt, wr, wi, n);
        work[0] = lwmin;
        iwork[0] = liwmin;
        iwork[1] = m;
        if (wants) {
            work[1] = s;
        }
        if (wantsp) {
            work[2] = sep;
        }
        return ok;
    }

    static void storeEigen(double[] t, int tOff, int ldt, double[] wr, double[] wi, int n) {
        for (int k = 0; k < n; k++) {
            wr[k] = t[tOff + k * ldt + k];
            wi[k] = 0;
        }
        for (int k = 0; k < n - 1; k++) {
            double v = t[tOff + (k + 1) * ldt + k];
            if (v != 0) {
                wi[k] = sqrt(abs(t[tOff + k * ldt + k + 1])) * sqrt(abs(v));
                wi[k + 1] = -wi[k];
            }
        }
    }

}
