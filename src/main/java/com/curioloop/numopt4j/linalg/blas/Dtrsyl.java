/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;

/**
 * Dtrsyl solves the real Sylvester matrix equation:
 *
 * <pre>
 *   op(A)*X + X*op(B) = scale*C
 *   A*X + sgn*X*B = scale*C
 *   Aᵀ*X + sgn*X*B = scale*C
 *   A*X + sgn*X*Bᵀ = scale*C
 *   Aᵀ*X + sgn*X*Bᵀ = scale*C
 * </pre>
 *
 * <p>A and B are both upper quasi-triangular (Schur canonical form).
 * A is m×m and B is n×n, C and X are m×n.</p>
 *
 */
public interface Dtrsyl {

    public static double dtrsyl(boolean trana, boolean tranb, int isgn, int m, int n,
                                double[] a, int aOff, int lda,
                                double[] b, int bOff, int ldb,
                                double[] c, int cOff, int ldc,
                                boolean[] okOut) {
        double scale = 1.0;
        boolean ok = true;

        if (m == 0 || n == 0) {
            if (okOut != null) {
                okOut[0] = ok;
            }
            return scale;
        }

        double eps = Dlamch.dlamch('P');
        double smlnum = Dlamch.dlamch('S') * (m * n) / eps;
        double bignum = 1.0 / smlnum;
        double sgn = isgn;

        double smin = max(smlnum, eps * Dlange.dlange('M', m, m, a, aOff, lda));
        smin = max(smin, eps * Dlange.dlange('M', n, n, b, bOff, ldb));

        double[] vec = new double[4];
        double[] x = new double[6];

        if (!trana && !tranb) {
            int lnext = 0;
            for (int l = lnext; l <= n - 1; l++) {
                if (l < lnext) continue;

                int l1, l2;
                if (l == n - 1) {
                    l1 = l;
                    l2 = l;
                } else {
                    if (b[(l + 1) * ldb + bOff + l] != 0) {
                        l1 = l;
                        l2 = l + 1;
                        lnext = l + 2;
                    } else {
                        l1 = l;
                        l2 = l;
                        lnext = l + 1;
                    }
                }

                int knext = m - 1;
                for (int k = knext; k >= 0; k--) {
                    if (k > knext) continue;

                    int k1, k2;
                    if (k == 0) {
                        k1 = k;
                        k2 = k;
                    } else {
                        if (a[k * lda + aOff + k - 1] != 0) {
                            k1 = k - 1;
                            k2 = k;
                            knext = k - 2;
                        } else {
                            k1 = k;
                            k2 = k;
                            knext = k - 1;
                        }
                    }

                    if (l1 == l2 && k1 == k2) {
                        int nextK = min(k1 + 1, m - 1);
                        double suml = BLAS.ddot(m - k1 - 1, a, aOff + k1 * lda + nextK, 1,
                                                c, cOff + nextK * ldc + l1, ldc);
                        double sumr = BLAS.ddot(l1, c, cOff + k1 * ldc, 1,
                                                b, bOff + l1, ldb);
                        double vec0 = c[cOff + k1 * ldc + l1] - (suml + sgn * sumr);

                        double scaloc = 1.0;
                        double a11 = a[aOff + k1 * lda + k1] + sgn * b[bOff + l1 * ldb + l1];
                        double da11 = abs(a11);
                        if (da11 <= smin) {
                            a11 = smin;
                            da11 = smin;
                            ok = false;
                        }
                        double db = abs(vec0);
                        if (da11 < 1 && db > 1 && db > bignum * da11) {
                            scaloc = 1.0 / db;
                        }
                        double x0 = (vec0 * scaloc) / a11;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x0;
                    } else if (l1 == l2 && k1 != k2) {
                        int nextK = min(k2 + 1, m - 1);
                        double suml = BLAS.ddot(m - k2 - 1, a, aOff + k1 * lda + nextK, 1,
                                                c, cOff + nextK * ldc + l1, ldc);
                        double sumr = BLAS.ddot(l1, c, cOff + k1 * ldc, 1,
                                                b, bOff + l1, ldb);
                        vec[0] = c[cOff + k1 * ldc + l1] - (suml + sgn * sumr);

                        suml = BLAS.ddot(m - k2 - 1, a, aOff + k2 * lda + nextK, 1,
                                         c, cOff + nextK * ldc + l1, ldc);
                        sumr = BLAS.ddot(l1, c, cOff + k2 * ldc, 1,
                                         b, bOff + l1, ldb);
                        vec[2] = c[cOff + k2 * ldc + l1] - (suml + sgn * sumr);

                        Dlaln2.dlaln2(false, 2, 1, smin, 1, a, aOff + k1 * lda + k1, lda, 1, 1,
                                      vec, 0, 2, -sgn * b[bOff + l1 * ldb + l1], 0, x, 0, 2);
                        double scaloc = x[4];
                        ok = ok && x[5] >= 0;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x[0];
                        c[cOff + k2 * ldc + l1] = x[2];
                    } else if (l1 != l2 && k1 == k2) {
                        int nextK = min(k1 + 1, m - 1);
                        double suml = BLAS.ddot(m - k1 - 1, a, aOff + k1 * lda + nextK, 1,
                                                c, cOff + nextK * ldc + l1, ldc);
                        double sumr = BLAS.ddot(l1, c, cOff + k1 * ldc, 1,
                                                b, bOff + l1, ldb);
                        vec[0] = sgn * (c[cOff + k1 * ldc + l1] - (suml + sgn * sumr));

                        suml = BLAS.ddot(m - k1 - 1, a, aOff + k1 * lda + nextK, 1,
                                         c, cOff + nextK * ldc + l2, ldc);
                        sumr = BLAS.ddot(l1, c, cOff + k1 * ldc, 1,
                                         b, bOff + l2, ldb);
                        vec[2] = sgn * (c[cOff + k1 * ldc + l2] - (suml + sgn * sumr));

                        Dlaln2.dlaln2(true, 2, 1, smin, 1, b, bOff + l1 * ldb + l1, ldb, 1, 1,
                                      vec, 0, 2, -sgn * a[aOff + k1 * lda + k1], 0, x, 0, 2);
                        double scaloc = x[4];
                        ok = ok && x[5] >= 0;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x[0];
                        c[cOff + k1 * ldc + l2] = x[2];
                    } else {
                        int nextK = min(k2 + 1, m - 1);
                        double suml = BLAS.ddot(m - k2 - 1, a, aOff + k1 * lda + nextK, 1,
                                                c, cOff + nextK * ldc + l1, ldc);
                        double sumr = BLAS.ddot(l1, c, cOff + k1 * ldc, 1,
                                                b, bOff + l1, ldb);
                        vec[0] = c[cOff + k1 * ldc + l1] - (suml + sgn * sumr);

                        suml = BLAS.ddot(m - k2 - 1, a, aOff + k1 * lda + nextK, 1,
                                         c, cOff + nextK * ldc + l2, ldc);
                        sumr = BLAS.ddot(l1, c, cOff + k1 * ldc, 1,
                                         b, bOff + l2, ldb);
                        vec[1] = c[cOff + k1 * ldc + l2] - (suml + sgn * sumr);

                        suml = BLAS.ddot(m - k2 - 1, a, aOff + k2 * lda + nextK, 1,
                                         c, cOff + nextK * ldc + l1, ldc);
                        sumr = BLAS.ddot(l1, c, cOff + k2 * ldc, 1,
                                         b, bOff + l1, ldb);
                        vec[2] = c[cOff + k2 * ldc + l1] - (suml + sgn * sumr);

                        suml = BLAS.ddot(m - k2 - 1, a, aOff + k2 * lda + nextK, 1,
                                         c, cOff + nextK * ldc + l2, ldc);
                        sumr = BLAS.ddot(l1, c, cOff + k2 * ldc, 1,
                                         b, bOff + l2, ldb);
                        vec[3] = c[cOff + k2 * ldc + l2] - (suml + sgn * sumr);

                        Dlasy2.dlasy2(false, false, isgn, 2, 2,
                                      a, aOff + k1 * lda + k1, lda,
                                      b, bOff + l1 * ldb + l1, ldb,
                                      vec, 0, 2, x, 0, 2);
                        double scaloc = x[4];
                        ok = ok && x[5] >= 0;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x[0];
                        c[cOff + k1 * ldc + l2] = x[1];
                        c[cOff + k2 * ldc + l1] = x[2];
                        c[cOff + k2 * ldc + l2] = x[3];
                    }
                }
            }
        } else if (trana && !tranb) {
            int lnext = 0;
            for (int l = lnext; l <= n - 1; l++) {
                if (l < lnext) continue;

                int l1, l2;
                if (l == n - 1) {
                    l1 = l;
                    l2 = l;
                } else {
                    if (b[(l + 1) * ldb + bOff + l] != 0) {
                        l1 = l;
                        l2 = l + 1;
                        lnext = l + 2;
                    } else {
                        l1 = l;
                        l2 = l;
                        lnext = l + 1;
                    }
                }

                int knext = 0;
                for (int k = knext; k <= m - 1; k++) {
                    if (k < knext) continue;

                    int k1, k2;
                    if (k == m - 1) {
                        k1 = k;
                        k2 = k;
                    } else {
                        if (a[(k + 1) * lda + aOff + k] != 0) {
                            k1 = k;
                            k2 = k + 1;
                            knext = k + 2;
                        } else {
                            k1 = k;
                            k2 = k;
                            knext = k + 1;
                        }
                    }

                    if (l1 == l2 && k1 == k2) {
                        double suml = BLAS.ddot(k1, a, aOff + k1, lda, c, cOff + l1, ldc);
                        double sumr = BLAS.ddot(l1, c, cOff + k1 * ldc, 1, b, bOff + l1, ldb);
                        double vec0 = c[cOff + k1 * ldc + l1] - (suml + sgn * sumr);

                        double scaloc = 1.0;
                        double a11 = a[aOff + k1 * lda + k1] + sgn * b[bOff + l1 * ldb + l1];
                        double da11 = abs(a11);
                        if (da11 <= smin) {
                            a11 = smin;
                            da11 = smin;
                            ok = false;
                        }
                        double db = abs(vec0);
                        if (da11 < 1 && db > 1 && db > bignum * da11) {
                            scaloc = 1.0 / db;
                        }
                        double x0 = (vec0 * scaloc) / a11;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x0;
                    } else if (l1 == l2 && k1 != k2) {
                        double suml = BLAS.ddot(k1, a, aOff + k1, lda, c, cOff + l1, ldc);
                        double sumr = BLAS.ddot(l1, c, cOff + k1 * ldc, 1, b, bOff + l1, ldb);
                        vec[0] = c[cOff + k1 * ldc + l1] - (suml + sgn * sumr);

                        suml = BLAS.ddot(k1, a, aOff + k2, lda, c, cOff + l1, ldc);
                        sumr = BLAS.ddot(l1, c, cOff + k2 * ldc, 1, b, bOff + l1, ldb);
                        vec[2] = c[cOff + k2 * ldc + l1] - (suml + sgn * sumr);

                        Dlaln2.dlaln2(true, 2, 1, smin, 1, a, aOff + k1 * lda + k1, lda, 1, 1,
                                      vec, 0, 2, -sgn * b[bOff + l1 * ldb + l1], 0, x, 0, 2);
                        double scaloc = x[4];
                        ok = ok && x[5] >= 0;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x[0];
                        c[cOff + k2 * ldc + l1] = x[2];
                    } else if (l1 != l2 && k1 == k2) {
                        double suml = BLAS.ddot(k1, a, aOff + k1, lda, c, cOff + l1, ldc);
                        double sumr = BLAS.ddot(l1, c, cOff + k1 * ldc, 1, b, bOff + l1, ldb);
                        vec[0] = sgn * (c[cOff + k1 * ldc + l1] - (suml + sgn * sumr));

                        suml = BLAS.ddot(k1, a, aOff + k2, lda, c, cOff + l2, ldc);
                        sumr = BLAS.ddot(l1, c, cOff + k1 * ldc, 1, b, bOff + l2, ldb);
                        vec[2] = sgn * (c[cOff + k1 * ldc + l2] - (suml + sgn * sumr));

                        Dlaln2.dlaln2(true, 2, 1, smin, 1, b, bOff + l1 * ldb + l1, ldb, 1, 1,
                                      vec, 0, 2, -sgn * a[aOff + k1 * lda + k1], 0, x, 0, 2);
                        double scaloc = x[4];
                        ok = ok && x[5] >= 0;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x[0];
                        c[cOff + k1 * ldc + l2] = x[2];
                    } else {
                        double suml = BLAS.ddot(k1, a, aOff + k1, lda, c, cOff + l1, ldc);
                        double sumr = BLAS.ddot(l1, c, cOff + k1 * ldc, 1, b, bOff + l1, ldb);
                        vec[0] = c[cOff + k1 * ldc + l1] - (suml + sgn * sumr);

                        suml = BLAS.ddot(k1, a, aOff + k1, lda, c, cOff + l2, ldc);
                        sumr = BLAS.ddot(l1, c, cOff + k1 * ldc, 1, b, bOff + l2, ldb);
                        vec[1] = c[cOff + k1 * ldc + l2] - (suml + sgn * sumr);

                        suml = BLAS.ddot(k1, a, aOff + k2, lda, c, cOff + l1, ldc);
                        sumr = BLAS.ddot(l1, c, cOff + k2 * ldc, 1, b, bOff + l1, ldb);
                        vec[2] = c[cOff + k2 * ldc + l1] - (suml + sgn * sumr);

                        suml = BLAS.ddot(k1, a, aOff + k2, lda, c, cOff + l2, ldc);
                        sumr = BLAS.ddot(l1, c, cOff + k2 * ldc, 1, b, bOff + l2, ldb);
                        vec[3] = c[cOff + k2 * ldc + l2] - (suml + sgn * sumr);

                        Dlasy2.dlasy2(true, false, isgn, 2, 2,
                                      a, aOff + k1 * lda + k1, lda,
                                      b, bOff + l1 * ldb + l1, ldb,
                                      vec, 0, 2, x, 0, 2);
                        double scaloc = x[4];
                        ok = ok && x[5] >= 0;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x[0];
                        c[cOff + k1 * ldc + l2] = x[1];
                        c[cOff + k2 * ldc + l1] = x[2];
                        c[cOff + k2 * ldc + l2] = x[3];
                    }
                }
            }
        } else if (trana && tranb) {
            int lnext = n - 1;
            for (int l = lnext; l >= 0; l--) {
                if (l > lnext) continue;

                int l1, l2;
                if (l == 0) {
                    l1 = l;
                    l2 = l;
                } else {
                    if (b[l * ldb + bOff + l - 1] != 0) {
                        l1 = l - 1;
                        l2 = l;
                        lnext = l - 2;
                    } else {
                        l1 = l;
                        l2 = l;
                        lnext = l - 1;
                    }
                }

                int knext = 0;
                for (int k = knext; k <= m - 1; k++) {
                    if (k < knext) continue;

                    int k1, k2;
                    if (k == m - 1) {
                        k1 = k;
                        k2 = k;
                    } else {
                        if (a[(k + 1) * lda + aOff + k] != 0) {
                            k1 = k;
                            k2 = k + 1;
                            knext = k + 2;
                        } else {
                            k1 = k;
                            k2 = k;
                            knext = k + 1;
                        }
                    }

                    int nextL = min(l1 + 1, n - 1);
                    if (l1 == l2 && k1 == k2) {
                        double suml = BLAS.ddot(k1, a, aOff + k1, lda, c, cOff + l1, ldc);
                        double sumr = BLAS.ddot(n - l1 - 1, c, cOff + k1 * ldc + nextL, 1,
                                                b, bOff + l1 * ldb + nextL, 1);
                        double vec0 = c[cOff + k1 * ldc + l1] - (suml + sgn * sumr);

                        double scaloc = 1.0;
                        double a11 = a[aOff + k1 * lda + k1] + sgn * b[bOff + l1 * ldb + l1];
                        double da11 = abs(a11);
                        if (da11 <= smin) {
                            a11 = smin;
                            da11 = smin;
                            ok = false;
                        }
                        double db = abs(vec0);
                        if (da11 < 1 && db > 1 && db > bignum * da11) {
                            scaloc = 1.0 / db;
                        }
                        double x0 = (vec0 * scaloc) / a11;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x0;
                    } else if (l1 == l2 && k1 != k2) {
                        int nextL2 = min(l2 + 1, n - 1);
                        double suml = BLAS.ddot(k1, a, aOff + k1, lda, c, cOff + l1, ldc);
                        double sumr = BLAS.ddot(n - l2 - 1, c, cOff + k1 * ldc + nextL2, 1,
                                                b, bOff + l1 * ldb + nextL2, 1);
                        vec[0] = c[cOff + k1 * ldc + l1] - (suml + sgn * sumr);

                        suml = BLAS.ddot(k1, a, aOff + k2, lda, c, cOff + l1, ldc);
                        sumr = BLAS.ddot(n - l2 - 1, c, cOff + k2 * ldc + nextL2, 1,
                                         b, bOff + l1 * ldb + nextL2, 1);
                        vec[2] = c[cOff + k2 * ldc + l1] - (suml + sgn * sumr);

                        Dlaln2.dlaln2(true, 2, 1, smin, 1, a, aOff + k1 * lda + k1, lda, 1, 1,
                                      vec, 0, 2, -sgn * b[bOff + l1 * ldb + l1], 0, x, 0, 2);
                        double scaloc = x[4];
                        ok = ok && x[5] >= 0;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x[0];
                        c[cOff + k2 * ldc + l1] = x[2];
                    } else if (l1 != l2 && k1 == k2) {
                        int nextL2 = min(l2 + 1, n - 1);
                        double suml = BLAS.ddot(k1, a, aOff + k1, lda, c, cOff + l1, ldc);
                        double sumr = BLAS.ddot(n - l2 - 1, c, cOff + k1 * ldc + nextL2, 1,
                                                b, bOff + l1 * ldb + nextL2, 1);
                        vec[0] = sgn * (c[cOff + k1 * ldc + l1] - (suml + sgn * sumr));

                        suml = BLAS.ddot(k1, a, aOff + k2, lda, c, cOff + l2, ldc);
                        sumr = BLAS.ddot(n - l2 - 1, c, cOff + k1 * ldc + nextL2, 1,
                                         b, bOff + l2 * ldb + nextL2, 1);
                        vec[2] = sgn * (c[cOff + k1 * ldc + l2] - (suml + sgn * sumr));

                        Dlaln2.dlaln2(false, 2, 1, smin, 1, b, bOff + l1 * ldb + l1, ldb, 1, 1,
                                      vec, 0, 2, -sgn * a[aOff + k1 * lda + k1], 0, x, 0, 2);
                        double scaloc = x[4];
                        ok = ok && x[5] >= 0;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x[0];
                        c[cOff + k1 * ldc + l2] = x[2];
                    } else {
                        int nextL2 = min(l2 + 1, n - 1);
                        double suml = BLAS.ddot(k1, a, aOff + k1, lda, c, cOff + l1, ldc);
                        double sumr = BLAS.ddot(n - l2 - 1, c, cOff + k1 * ldc + nextL2, 1,
                                                b, bOff + l1 * ldb + nextL2, 1);
                        vec[0] = c[cOff + k1 * ldc + l1] - (suml + sgn * sumr);

                        suml = BLAS.ddot(k1, a, aOff + k1, lda, c, cOff + l2, ldc);
                        sumr = BLAS.ddot(n - l2 - 1, c, cOff + k1 * ldc + nextL2, 1,
                                         b, bOff + l2 * ldb + nextL2, 1);
                        vec[1] = c[cOff + k1 * ldc + l2] - (suml + sgn * sumr);

                        suml = BLAS.ddot(k1, a, aOff + k2, lda, c, cOff + l1, ldc);
                        sumr = BLAS.ddot(n - l2 - 1, c, cOff + k2 * ldc + nextL2, 1,
                                         b, bOff + l1 * ldb + nextL2, 1);
                        vec[2] = c[cOff + k2 * ldc + l1] - (suml + sgn * sumr);

                        suml = BLAS.ddot(k1, a, aOff + k2, lda, c, cOff + l2, ldc);
                        sumr = BLAS.ddot(n - l2 - 1, c, cOff + k2 * ldc + nextL2, 1,
                                         b, bOff + l2 * ldb + nextL2, 1);
                        vec[3] = c[cOff + k2 * ldc + l2] - (suml + sgn * sumr);

                        Dlasy2.dlasy2(true, true, isgn, 2, 2,
                                      a, aOff + k1 * lda + k1, lda,
                                      b, bOff + l1 * ldb + l1, ldb,
                                      vec, 0, 2, x, 0, 2);
                        double scaloc = x[4];
                        ok = ok && x[5] >= 0;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x[0];
                        c[cOff + k1 * ldc + l2] = x[1];
                        c[cOff + k2 * ldc + l1] = x[2];
                        c[cOff + k2 * ldc + l2] = x[3];
                    }
                }
            }
        } else {
            int lnext = n - 1;
            for (int l = lnext; l >= 0; l--) {
                if (l > lnext) continue;

                int l1, l2;
                if (l == 0) {
                    l1 = l;
                    l2 = l;
                } else {
                    if (b[l * ldb + bOff + l - 1] != 0) {
                        l1 = l - 1;
                        l2 = l;
                        lnext = l - 2;
                    } else {
                        l1 = l;
                        l2 = l;
                        lnext = l - 1;
                    }
                }

                int knext = m - 1;
                for (int k = knext; k >= 0; k--) {
                    if (k > knext) continue;

                    int k1, k2;
                    if (k == 0) {
                        k1 = k;
                        k2 = k;
                    } else {
                        if (a[k * lda + aOff + k - 1] != 0) {
                            k1 = k - 1;
                            k2 = k;
                            knext = k - 2;
                        } else {
                            k1 = k;
                            k2 = k;
                            knext = k - 1;
                        }
                    }

                    int nextK = min(k1 + 1, m - 1);
                    int nextL = min(l1 + 1, n - 1);
                    if (l1 == l2 && k1 == k2) {
                        double suml = BLAS.ddot(m - k1 - 1, a, aOff + k1 * lda + nextK, 1,
                                                c, cOff + nextK * ldc + l1, ldc);
                        double sumr = BLAS.ddot(n - l1 - 1, c, cOff + k1 * ldc + nextL, 1,
                                                b, bOff + l1 * ldb + nextL, 1);
                        double vec0 = c[cOff + k1 * ldc + l1] - (suml + sgn * sumr);

                        double scaloc = 1.0;
                        double a11 = a[aOff + k1 * lda + k1] + sgn * b[bOff + l1 * ldb + l1];
                        double da11 = abs(a11);
                        if (da11 <= smin) {
                            a11 = smin;
                            da11 = smin;
                            ok = false;
                        }
                        double db = abs(vec0);
                        if (da11 < 1 && db > 1 && db > bignum * da11) {
                            scaloc = 1.0 / db;
                        }
                        double x0 = (vec0 * scaloc) / a11;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x0;
                    } else if (l1 == l2 && k1 != k2) {
                        int nextK2 = min(k2 + 1, m - 1);
                        int nextL2 = min(l2 + 1, n - 1);
                        double suml = BLAS.ddot(m - k2 - 1, a, aOff + k1 * lda + nextK2, 1,
                                                c, cOff + nextK2 * ldc + l1, ldc);
                        double sumr = BLAS.ddot(n - l2 - 1, c, cOff + k1 * ldc + nextL2, 1,
                                                b, bOff + l1 * ldb + nextL2, 1);
                        vec[0] = c[cOff + k1 * ldc + l1] - (suml + sgn * sumr);

                        suml = BLAS.ddot(m - k2 - 1, a, aOff + k2 * lda + nextK2, 1,
                                         c, cOff + nextK2 * ldc + l1, ldc);
                        sumr = BLAS.ddot(n - l2 - 1, c, cOff + k2 * ldc + nextL2, 1,
                                         b, bOff + l1 * ldb + nextL2, 1);
                        vec[2] = c[cOff + k2 * ldc + l1] - (suml + sgn * sumr);

                        Dlaln2.dlaln2(false, 2, 1, smin, 1, a, aOff + k1 * lda + k1, lda, 1, 1,
                                      vec, 0, 2, -sgn * b[bOff + l1 * ldb + l1], 0, x, 0, 2);
                        double scaloc = x[4];
                        ok = ok && x[5] >= 0;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x[0];
                        c[cOff + k2 * ldc + l1] = x[2];
                    } else if (l1 != l2 && k1 == k2) {
                        int nextL2 = min(l2 + 1, n - 1);
                        double suml = BLAS.ddot(m - k1 - 1, a, aOff + k1 * lda + nextK, 1,
                                                c, cOff + nextK * ldc + l1, ldc);
                        double sumr = BLAS.ddot(n - l2 - 1, c, cOff + k1 * ldc + nextL2, 1,
                                                b, bOff + l1 * ldb + nextL2, 1);
                        vec[0] = sgn * (c[cOff + k1 * ldc + l1] - (suml + sgn * sumr));

                        suml = BLAS.ddot(m - k1 - 1, a, aOff + k1 * lda + nextK, 1,
                                         c, cOff + nextK * ldc + l2, ldc);
                        sumr = BLAS.ddot(n - l2 - 1, c, cOff + k1 * ldc + nextL2, 1,
                                         b, bOff + l2 * ldb + nextL2, 1);
                        vec[2] = sgn * (c[cOff + k1 * ldc + l2] - (suml + sgn * sumr));

                        Dlaln2.dlaln2(false, 2, 1, smin, 1, b, bOff + l1 * ldb + l1, ldb, 1, 1,
                                      vec, 0, 2, -sgn * a[aOff + k1 * lda + k1], 0, x, 0, 2);
                        double scaloc = x[4];
                        ok = ok && x[5] >= 0;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x[0];
                        c[cOff + k1 * ldc + l2] = x[2];
                    } else {
                        int nextK2 = min(k2 + 1, m - 1);
                        int nextL2 = min(l2 + 1, n - 1);
                        double suml = BLAS.ddot(m - k2 - 1, a, aOff + k1 * lda + nextK2, 1,
                                                c, cOff + nextK2 * ldc + l1, ldc);
                        double sumr = BLAS.ddot(n - l2 - 1, c, cOff + k1 * ldc + nextL2, 1,
                                                b, bOff + l1 * ldb + nextL2, 1);
                        vec[0] = c[cOff + k1 * ldc + l1] - (suml + sgn * sumr);

                        suml = BLAS.ddot(m - k2 - 1, a, aOff + k1 * lda + nextK2, 1,
                                         c, cOff + nextK2 * ldc + l2, ldc);
                        sumr = BLAS.ddot(n - l2 - 1, c, cOff + k1 * ldc + nextL2, 1,
                                         b, bOff + l2 * ldb + nextL2, 1);
                        vec[1] = c[cOff + k1 * ldc + l2] - (suml + sgn * sumr);

                        suml = BLAS.ddot(m - k2 - 1, a, aOff + k2 * lda + nextK2, 1,
                                         c, cOff + nextK2 * ldc + l1, ldc);
                        sumr = BLAS.ddot(n - l2 - 1, c, cOff + k2 * ldc + nextL2, 1,
                                         b, bOff + l1 * ldb + nextL2, 1);
                        vec[2] = c[cOff + k2 * ldc + l1] - (suml + sgn * sumr);

                        suml = BLAS.ddot(m - k2 - 1, a, aOff + k2 * lda + nextK2, 1,
                                         c, cOff + nextK2 * ldc + l2, ldc);
                        sumr = BLAS.ddot(n - l2 - 1, c, cOff + k2 * ldc + nextL2, 1,
                                         b, bOff + l2 * ldb + nextL2, 1);
                        vec[3] = c[cOff + k2 * ldc + l2] - (suml + sgn * sumr);

                        Dlasy2.dlasy2(false, true, isgn, 2, 2,
                                      a, aOff + k1 * lda + k1, lda,
                                      b, bOff + l1 * ldb + l1, ldb,
                                      vec, 0, 2, x, 0, 2);
                        double scaloc = x[4];
                        ok = ok && x[5] >= 0;
                        if (scaloc != 1.0) {
                            for (int j = 0; j < n; j++) {
                                BLAS.dscal(m, scaloc, c, cOff + j, ldc);
                            }
                            scale *= scaloc;
                        }
                        c[cOff + k1 * ldc + l1] = x[0];
                        c[cOff + k1 * ldc + l2] = x[1];
                        c[cOff + k2 * ldc + l1] = x[2];
                        c[cOff + k2 * ldc + l2] = x[3];
                    }
                }
            }
        }

        if (okOut != null) {
            okOut[0] = ok;
        }
        return scale;
    }
}
