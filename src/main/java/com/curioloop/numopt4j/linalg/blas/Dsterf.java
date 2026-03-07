/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * LAPACK DSTERF: Computes all eigenvalues of a symmetric tridiagonal matrix
 * using the Pal-Walker-Kahan variant of the QL or QR algorithm.
 * 
 */
interface Dsterf {

    /** Machine epsilon: 2^-53 for IEEE */
    static final double DLAMCH_E = 0x1p-53;
    
    /** Machine safe minimum: 2^-1022 for IEEE */
    static final double DLAMCH_S = 0x1p-1022;
    
    /** Maximum iterations per eigenvalue */
    static final int MAXIT = 30;

    /**
     * DSTERF computes all eigenvalues of a symmetric tridiagonal matrix
     * using the QL/QR algorithm.
     *
     * @param n order of the matrix
     * @param d diagonal elements (modified in place, contains eigenvalues on exit)
     * @param e off-diagonal elements (modified in place)
     * @return true if successful, false if convergence failure
     */
    static boolean dsterf(int n, double[] d, double[] e) {
        return dsterf(n, d, 0, e, 0);
    }

    /**
     * DSTERF computes all eigenvalues of a symmetric tridiagonal matrix
     * using the QL/QR algorithm. This version supports array offsets for
     * workspace reuse.
     *
     * @param n order of the matrix
     * @param d diagonal elements (modified in place, contains eigenvalues on exit)
     * @param dOff starting offset in d
     * @param e off-diagonal elements (modified in place)
     * @param eOff starting offset in e
     * @return true if successful, false if convergence failure
     */
    static boolean dsterf(int n, double[] d, int dOff, double[] e, int eOff) {
        if (n < 0) {
            return false;
        }
        if (n == 0) {
            return true;
        }
        if (d == null || d.length < dOff + n) {
            return false;
        }
        if (e == null || e.length < eOff + n - 1) {
            return false;
        }

        if (n == 1) {
            return true;
        }

        // Determine the unit roundoff for this environment
        double eps = DLAMCH_E;
        double eps2 = eps * eps;
        double safmin = DLAMCH_S;
        double safmax = 1.0 / safmin;
        double ssfmax = Math.sqrt(safmax) / 3.0;
        double ssfmin = Math.sqrt(safmin) / eps2;

        // Compute the eigenvalues of the tridiagonal matrix
        int nmaxit = n * MAXIT;
        int jtot = 0;

        int l1 = 0;

        while (true) {
            if (l1 > n - 1) {
                // Sort eigenvalues
                Dlasrt.dlasrt('I', n, d, dOff);
                return true;
            }
            if (l1 > 0) {
                e[eOff + l1 - 1] = 0;
            }

            int m = n - 1;
            for (m = l1; m < n - 1; m++) {
                if (Math.abs(e[eOff + m]) <= Math.sqrt(Math.abs(d[dOff + m])) * Math.sqrt(Math.abs(d[dOff + m + 1])) * eps) {
                    e[eOff + m] = 0;
                    break;
                }
            }

            int l = l1;
            int lsv = l;
            int lend = m;
            int lendsv = lend;
            l1 = m + 1;
            if (lend == 0) {
                continue;
            }

            // Scale submatrix in rows and columns l to lend
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

            // Square the off-diagonal elements
            for (int i = l; i < lend; i++) {
                e[eOff + i] = e[eOff + i] * e[eOff + i];
            }

            // Choose between QL and QR iteration
            if (Math.abs(d[dOff + lend]) < Math.abs(d[dOff + l])) {
                lend = lsv;
                l = lendsv;
            }

            if (lend >= l) {
                // QL Iteration
                while (true) {
                    // Look for small sub-diagonal element
                    if (l != lend) {
                        for (m = l; m < lend; m++) {
                            if (Math.abs(e[eOff + m]) <= eps2 * (Math.abs(d[dOff + m]) * Math.abs(d[dOff + m + 1]))) {
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
                        // Eigenvalue found
                        l++;
                        if (l > lend) {
                            break;
                        }
                        continue;
                    }
                    // If remaining matrix is 2 by 2, use Dlae2 to compute its eigenvalues
                    if (m == l + 1) {
                        double sqrtEl = Math.sqrt(e[eOff + l]);
                        Dlae2.dlae2(d[dOff + l], sqrtEl, d[dOff + l + 1], d, dOff + l);
                        e[eOff + l] = 0.0;
                        l += 2;
                        if (l > lend) {
                            break;
                        }
                        continue;
                    }
                    if (jtot == nmaxit) {
                        break;
                    }
                    jtot++;

                    // Form shift
                    double rte = Math.sqrt(e[eOff + l]);
                    double sigma = (d[dOff + l + 1] - p) / (2.0 * rte);
                    double r = Math.hypot(sigma, 1.0);
                    sigma = p - (rte / (sigma + (sigma >= 0 ? r : -r)));

                    double c = 1.0;
                    double s = 0.0;
                    double gamma = d[dOff + m] - sigma;
                    p = gamma * gamma;

                    // Inner loop
                    for (int i = m - 1; i >= l; i--) {
                        double bb = e[eOff + i];
                        double rp = p + bb;
                        if (i != m - 1) {
                            e[eOff + i + 1] = s * rp;
                        }
                        double oldc = c;
                        c = p / rp;
                        s = bb / rp;
                        double oldgam = gamma;
                        double alpha = d[dOff + i];
                        gamma = c * (alpha - sigma) - s * oldgam;
                        d[dOff + i + 1] = oldgam + (alpha - gamma);
                        if (c != 0.0) {
                            p = (gamma * gamma) / c;
                        } else {
                            p = oldc * bb;
                        }
                    }
                    e[eOff + l] = s * p;
                    d[dOff + l] = sigma + gamma;
                }
            } else {
                // QR Iteration
                while (true) {
                    // Look for small super-diagonal element
                    for (m = l; m > lend; m--) {
                        if (Math.abs(e[eOff + m - 1]) <= eps2 * Math.abs(d[dOff + m] * d[dOff + m - 1])) {
                            break;
                        }
                    }
                    if (m > lend) {
                        e[eOff + m - 1] = 0.0;
                    }
                    double p = d[dOff + l];
                    if (m == l) {
                        // Eigenvalue found
                        l--;
                        if (l < lend) {
                            break;
                        }
                        continue;
                    }

                    // If remaining matrix is 2 by 2, use Dlae2
                    if (m == l - 1) {
                        double sqrtEl = Math.sqrt(e[eOff + l - 1]);
                        Dlae2.dlae2(d[dOff + l], sqrtEl, d[dOff + l - 1], d, dOff + l - 1);
                        e[eOff + l - 1] = 0.0;
                        l -= 2;
                        if (l < lend) {
                            break;
                        }
                        continue;
                    }
                    if (jtot == nmaxit) {
                        break;
                    }
                    jtot++;

                    // Form shift
                    double rte = Math.sqrt(e[eOff + l - 1]);
                    double sigma = (d[dOff + l - 1] - p) / (2.0 * rte);
                    double r = Math.hypot(sigma, 1.0);
                    sigma = p - (rte / (sigma + (sigma >= 0 ? r : -r)));

                    double c = 1.0;
                    double s = 0.0;
                    double gamma = d[dOff + m] - sigma;
                    p = gamma * gamma;

                    // Inner loop
                    for (int i = m; i < l; i++) {
                        double bb = e[eOff + i];
                        double rp = p + bb;
                        if (i != m) {
                            e[eOff + i - 1] = s * rp;
                        }
                        double oldc = c;
                        c = p / rp;
                        s = bb / rp;
                        double oldgam = gamma;
                        double alpha = d[dOff + i + 1];
                        gamma = c * (alpha - sigma) - s * oldgam;
                        d[dOff + i] = oldgam + alpha - gamma;
                        if (c != 0.0) {
                            p = (gamma * gamma) / c;
                        } else {
                            p = oldc * bb;
                        }
                    }
                    e[eOff + l - 1] = s * p;
                    d[dOff + l] = sigma + gamma;
                }
            }

            // Undo scaling if necessary
            if (iscale == 1) {
                Dlamv.dlascl('G', ssfmax, anorm, lendsv - lsv + 1, d, dOff + lsv, 1);
            } else if (iscale == 2) {
                Dlamv.dlascl('G', ssfmin, anorm, lendsv - lsv + 1, d, dOff + lsv, 1);
            }

            // Check for no convergence
            if (jtot >= nmaxit) {
                break;
            }
        }

        // Check for convergence failure
        for (int i = 0; i < n - 1; i++) {
            if (e[eOff + i] != 0.0) {
                return false;
            }
        }

        Dlasrt.dlasrt('I', n, d, dOff);
        return true;
    }
}
