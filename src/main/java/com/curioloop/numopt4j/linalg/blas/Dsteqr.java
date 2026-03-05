/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * LAPACK DSTEQR: Computes all eigenvalues and optionally eigenvectors of a
 * symmetric tridiagonal matrix using implicit QL/QR method.
 * 
 * Based on gonum implementation.
 */
interface Dsteqr {

    /** Machine epsilon: 2^-53 for IEEE */
    static final double DLAMCH_E = 0x1p-53;
    
    /** Machine safe minimum: 2^-1022 for IEEE */
    static final double DLAMCH_S = 0x1p-1022;
    
    /** Maximum iterations per eigenvalue */
    static final int MAXIT = 30;

    /**
     * DSTEQR computes eigenvalues and eigenvectors of symmetric tridiagonal matrix.
     * 
     * @param compz 'N': eigenvalues only, 'V': eigenvectors of original matrix,
     *              'I': eigenvectors of tridiagonal matrix
     * @param n matrix order
     * @param d diagonal elements (modified in place)
     * @param e off-diagonal elements (modified in place)
     * @param z eigenvector matrix (row-major, n×n)
     * @param ldz leading dimension of z
     * @param work workspace (size max(1, 2*n-2) if computing eigenvectors)
     * @return 0 for success, >0 if convergence failure
     */
    static int dsteqr(char compz, int n, double[] d, double[] e,
                      double[] z, int ldz, double[] work) {
        return dsteqr(compz, n, d, 0, e, 0, z, 0, ldz, work, 0);
    }

    /**
     * DSTEQR computes eigenvalues and eigenvectors of symmetric tridiagonal matrix.
     * This version supports array offsets for workspace reuse.
     * 
     * @param compz 'N': eigenvalues only, 'V': eigenvectors of original matrix,
     *              'I': eigenvectors of tridiagonal matrix
     * @param n matrix order
     * @param d diagonal elements (modified in place)
     * @param dOff starting offset in d
     * @param e off-diagonal elements (modified in place)
     * @param eOff starting offset in e
     * @param z eigenvector matrix (row-major, n×n)
     * @param zOff starting offset in z
     * @param ldz leading dimension of z
     * @param work workspace (size max(1, 2*n-2) if computing eigenvectors)
     * @param workOff starting offset in work
     * @return 0 for success, >0 if convergence failure
     */
    static int dsteqr(char compz, int n, double[] d, int dOff, double[] e, int eOff,
                      double[] z, int zOff, int ldz, double[] work, int workOff) {
        if (n < 0) {
            return -2;
        }
        if (n == 0) {
            return 0;
        }

        int icompz;
        if (compz == 'N' || compz == 'n') {
            icompz = 0;
        } else if (compz == 'V' || compz == 'v') {
            icompz = 1;
        } else if (compz == 'I' || compz == 'i') {
            icompz = 2;
        } else {
            icompz = -1;
        }
        if (icompz < 0) {
            return -1;
        }
        if (ldz < 1 || (icompz > 0 && ldz < n)) {
            return -6;
        }

        if (n == 1) {
            if (icompz == 2 && z != null) {
                z[zOff] = 1.0;
            }
            return 0;
        }

        double eps = DLAMCH_E;
        double eps2 = eps * eps;
        double safmin = DLAMCH_S;
        double safmax = 1.0 / safmin;
        double ssfmax = Math.sqrt(safmax) / 3.0;
        double ssfmin = Math.sqrt(safmin) / eps2;

        if (icompz == 2) {
            Dlamv.dlaset('A', n, n, 0.0, 1.0, z, zOff, ldz);
        }

        int nmaxit = n * MAXIT;
        int jtot = 0;
        double[] temp = {0 ,0, 0, 0};

        int l1 = 0;
        int nm1 = n - 1;

        while (true) {
            if (l1 > n - 1) {
                if (icompz == 0) {
                    Dlasrt.dlasrt('I', n, d, dOff);
                } else {
                    for (int ii = 1; ii < n; ii++) {
                        int i = ii - 1;
                        int k = i;
                        double p = d[dOff + i];
                        for (int j = ii; j < n; j++) {
                            if (d[dOff + j] < p) {
                                k = j;
                                p = d[dOff + j];
                            }
                        }
                        if (k != i) {
                            d[dOff + k] = d[dOff + i];
                            d[dOff + i] = p;
                            if (z != null) {
                                for (int row = 0; row < n; row++) {
                                    double tmp = z[zOff + row * ldz + i];
                                    z[zOff + row * ldz + i] = z[zOff + row * ldz + k];
                                    z[zOff + row * ldz + k] = tmp;
                                }
                            }
                        }
                    }
                }
                return 0;
            }
            if (l1 > 0) {
                e[eOff + l1 - 1] = 0;
            }

            int m = n - 1;
            if (l1 <= nm1) {
                for (m = l1; m < nm1; m++) {
                    double test = Math.abs(e[eOff + m]);
                    if (test == 0.0) {
                        break;
                    }
                    if (test <= (Math.sqrt(Math.abs(d[dOff + m])) * Math.sqrt(Math.abs(d[dOff + m + 1]))) * eps) {
                        e[eOff + m] = 0.0;
                        break;
                    }
                }
            }

            int l = l1;
            int lsv = l;
            int lend = m;
            int lendsv = lend;
            l1 = m + 1;
            if (lend == l) {
                continue;
            }

            int len = lend - l + 1;
            double anorm = Dlanst.dlanst('M', len, d, dOff + l, e, eOff + l);
            int iscale = 0;
            if (anorm == 0.0) {
                continue;
            }
            if (anorm > ssfmax) {
                iscale = 1;
                Dlamv.dlascl('G', anorm, ssfmax, len, d, dOff + l, 1);
                Dlamv.dlascl('G', anorm, ssfmax, len - 1, e, eOff + l, 1);
            } else if (anorm < ssfmin) {
                iscale = 2;
                Dlamv.dlascl('G', anorm, ssfmin, len, d, dOff + l, 1);
                Dlamv.dlascl('G', anorm, ssfmin, len - 1, e, eOff + l, 1);
            }

            if (Math.abs(d[dOff + lend]) < Math.abs(d[dOff + l])) {
                lend = lsv;
                l = lendsv;
            }

            if (lend > l) {
                while (true) {
                    if (l != lend) {
                        for (m = l; m < lend; m++) {
                            double v = Math.abs(e[eOff + m]);
                            if (v * v <= (eps2 * Math.abs(d[dOff + m])) * Math.abs(d[dOff + m + 1]) + safmin) {
                                break;
                            }
                        }
                    } else {
                        m = lend;
                    }

                    if (m < lend) {
                        e[eOff + m] = 0.0;
                    }

                    double p = d[dOff + l];
                    if (m == l) {
                        l++;
                        if (l > lend) {
                            break;
                        }
                        continue;
                    }

                    if (m == l + 1) {
                        if (icompz > 0) {
                            Dlae2.dlaev2(d[dOff + l], e[eOff + l], d[dOff + l + 1], temp, 0);
                            d[dOff + l] = temp[0];
                            d[dOff + l + 1] = temp[1];
                            work[workOff + l] = temp[2];
                            work[workOff + n - 1 + l] = temp[3];
                            Dlasr.dlasr(BLAS.Side.Right, 'V', 'B', n, 2, work, workOff + l, work, workOff + n - 1 + l, z, zOff + l, ldz);
                        } else {
                            Dlae2.dlae2(d[dOff + l], e[eOff + l], d[dOff + l + 1], d, dOff + l);
                        }
                        e[eOff + l] = 0.0;
                        l += 2;
                        if (l > lend) {
                            break;
                        }
                        continue;
                    }

                    if (jtot >= nmaxit) {
                        break;
                    }
                    jtot++;

                    double g = (d[dOff + l + 1] - p) / (2.0 * e[eOff + l]);
                    double r = Math.hypot(g, 1.0);
                    g = d[dOff + m] - p + e[eOff + l] / (g + Math.copySign(r, g));
                    double s = 1.0;
                    double c = 1.0;
                    p = 0.0;

                    for (int i = m - 1; i >= l; i--) {
                        double f = s * e[eOff + i];
                        double b = c * e[eOff + i];
                        Dlartg.dlartg(g, f, temp, 0);
                        c = temp[0];
                        s = temp[1];
                        r = temp[2];
                        if (i != m - 1) {
                            e[eOff + i + 1] = r;
                        }
                        g = d[dOff + i + 1] - p;
                        r = (d[dOff + i] - g) * s + 2.0 * c * b;
                        p = s * r;
                        d[dOff + i + 1] = g + p;
                        g = c * r - b;

                        if (icompz > 0) {
                            work[workOff + i] = c;
                            work[workOff + n - 1 + i] = -s;
                        }
                    }

                    if (icompz > 0) {
                        int mm = m - l + 1;
                        Dlasr.dlasr(BLAS.Side.Right, 'V', 'B', n, mm, work, workOff + l, work, workOff + n - 1 + l, z, zOff + l, ldz);
                    }
                    d[dOff + l] -= p;
                    e[eOff + l] = g;
                }
            } else {
                while (true) {
                    if (l != lend) {
                        for (m = l; m > lend; m--) {
                            double v = Math.abs(e[eOff + m - 1]);
                            if (v * v <= (eps2 * Math.abs(d[dOff + m]) * Math.abs(d[dOff + m - 1]) + safmin)) {
                                break;
                            }
                        }
                    } else {
                        m = lend;
                    }

                    if (m > lend) {
                        e[eOff + m - 1] = 0.0;
                    }

                    double p = d[dOff + l];
                    if (m == l) {
                        l--;
                        if (l < lend) {
                            break;
                        }
                        continue;
                    }

                    if (m == l - 1) {
                        if (icompz > 0) {
                            Dlae2.dlaev2(d[dOff + l - 1], e[eOff + l - 1], d[dOff + l], temp, 0);
                            d[dOff + l - 1] = temp[0];
                            d[dOff + l] = temp[1];
                            work[workOff + m] = temp[2];
                            work[workOff + n - 1 + m] = temp[3];
                            Dlasr.dlasr(BLAS.Side.Right, 'V', 'F', n, 2, work, workOff + m, work, workOff + n - 1 + m, z, zOff + l - 1, ldz);
                        } else {
                            Dlae2.dlae2(d[dOff + l - 1], e[eOff + l - 1], d[dOff + l], d, dOff + l - 1);
                        }
                        e[eOff + l - 1] = 0.0;
                        l -= 2;
                        if (l < lend) {
                            break;
                        }
                        continue;
                    }

                    if (jtot >= nmaxit) {
                        break;
                    }
                    jtot++;

                    double g = (d[dOff + l - 1] - p) / (2.0 * e[eOff + l - 1]);
                    double r = Math.hypot(g, 1.0);
                    g = d[dOff + m] - p + e[eOff + l - 1] / (g + Math.copySign(r, g));
                    double s = 1.0;
                    double c = 1.0;
                    p = 0.0;

                    for (int i = m; i < l; i++) {
                        double f = s * e[eOff + i];
                        double b = c * e[eOff + i];
                        Dlartg.dlartg(g, f, temp, 0);
                        c = temp[0];
                        s = temp[1];
                        r = temp[2];
                        if (i != m) {
                            e[eOff + i - 1] = r;
                        }
                        g = d[dOff + i] - p;
                        r = (d[dOff + i + 1] - g) * s + 2.0 * c * b;
                        p = s * r;
                        d[dOff + i] = g + p;
                        g = c * r - b;

                        if (icompz > 0) {
                            work[workOff + i] = c;
                            work[workOff + n - 1 + i] = s;
                        }
                    }

                    if (icompz > 0) {
                        int mm = l - m + 1;
                        Dlasr.dlasr(BLAS.Side.Right, 'V', 'F', n, mm, work, workOff + m, work, workOff + n - 1 + m, z, zOff + m, ldz);
                    }
                    d[dOff + l] -= p;
                    e[eOff + l - 1] = g;
                }
            }

            switch (iscale) {
                case 1:
                    Dlamv.dlascl('G', ssfmax, anorm, lendsv - lsv + 1, d, dOff + lsv, 1);
                    Dlamv.dlascl('G', ssfmax, anorm, lendsv - lsv, e, eOff + lsv, 1);
                    break;
                case 2:
                    Dlamv.dlascl('G', ssfmin, anorm, lendsv - lsv + 1, d, dOff + lsv, 1);
                    Dlamv.dlascl('G', ssfmin, anorm, lendsv - lsv, e, eOff + lsv, 1);
                    break;
            }

            if (jtot >= nmaxit) {
                break;
            }
        }

        for (int i = 0; i < n - 1; i++) {
            if (e[eOff + i] != 0.0) {
                return i + 1;
            }
        }
        return 0;
    }

    static int dsteqr(int n, double[] d, double[] e, double[] work) {
        return dsteqr('N', n, d, e, null, n, work);
    }
}
