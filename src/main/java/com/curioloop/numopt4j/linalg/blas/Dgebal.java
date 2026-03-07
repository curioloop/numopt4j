/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Balances a general matrix to improve eigenvalue computation accuracy.
 * LAPACK DGEBAL routine.
 *
 * <p>Balancing consists of two stages, permuting and scaling:</p>
 * <ul>
 *   <li>Permuting: Apply permutation P such that P^T*A*P takes upper block triangular form</li>
 *   <li>Scaling: Apply diagonal similarity D such that D^{-1}*B*D has row and column norms nearly equal</li>
 * </ul>
 *
 */
interface Dgebal {

    static void dgebal(char job, int n, double[] A, int lda, double[] scale, int scaleOff, double[] out, int outOff) {
        if (n == 0) {
            out[outOff] = 0;
            out[outOff + 1] = 0;
            return;
        }

        char jobUpper = Character.toUpperCase(job);

        int ilo = 0;
        int ihi = n - 1;

        if (jobUpper == 'N') {
            for (int i = 0; i < n; i++) {
                scale[scaleOff + i] = 1.0;
            }
            out[outOff] = ilo;
            out[outOff + 1] = ihi;
            return;
        }

        for (int i = 0; i < n; i++) {
            scale[scaleOff + i] = 1.0;
        }

        if (jobUpper == 'S') {
            gotoScaling(A, lda, n, ilo, ihi, scale, scaleOff);
            out[outOff] = ilo;
            out[outOff + 1] = ihi;
            return;
        }

        boolean swapped = true;

        // Permutation to isolate eigenvalues if possible.
        // Search for rows isolating an eigenvalue and push them down.
        while (swapped) {
            swapped = false;
            rowLoop:
            for (int i = ihi; i >= 0; i--) {
                for (int j = 0; j <= ihi; j++) {
                    if (i == j) {
                        continue;
                    }
                    if (A[i * lda + j] != 0) {
                        continue rowLoop;
                    }
                }
                // Row i has only zero off-diagonal elements
                scale[scaleOff + ihi] = i;
                if (i != ihi) {
                    // Swap rows i and ihi
                    for (int k = 0; k <= ihi; k++) {
                        double tmp = A[i * lda + k];
                        A[i * lda + k] = A[ihi * lda + k];
                        A[ihi * lda + k] = tmp;
                    }
                    // Swap columns i and ihi
                    for (int k = 0; k < n; k++) {
                        double tmp = A[k * lda + i];
                        A[k * lda + i] = A[k * lda + ihi];
                        A[k * lda + ihi] = tmp;
                    }
                }
                if (ihi == 0) {
                    scale[scaleOff] = 1;
                    out[outOff] = 0;
                    out[outOff + 1] = 0;
                    return;
                }
                ihi--;
                swapped = true;
                break;
            }
        }

        // Search for columns isolating an eigenvalue and push them left.
        swapped = true;
        while (swapped) {
            swapped = false;
            colLoop:
            for (int j = ilo; j <= ihi; j++) {
                for (int i = ilo; i <= ihi; i++) {
                    if (i == j) {
                        continue;
                    }
                    if (A[i * lda + j] != 0) {
                        continue colLoop;
                    }
                }
                // Column j has only zero off-diagonal elements
                scale[scaleOff + ilo] = j;
                if (j != ilo) {
                    // Swap columns j and ilo
                    for (int k = 0; k <= ihi; k++) {
                        double tmp = A[k * lda + j];
                        A[k * lda + j] = A[k * lda + ilo];
                        A[k * lda + ilo] = tmp;
                    }
                    // Swap rows j and ilo
                    for (int k = ilo; k < n; k++) {
                        double tmp = A[j * lda + k];
                        A[j * lda + k] = A[ilo * lda + k];
                        A[ilo * lda + k] = tmp;
                    }
                }
                swapped = true;
                ilo++;
                break;
            }
        }

        if (jobUpper == 'P') {
            out[outOff] = ilo;
            out[outOff + 1] = ihi;
            return;
        }

        gotoScaling(A, lda, n, ilo, ihi, scale, scaleOff);
        out[outOff] = ilo;
        out[outOff + 1] = ihi;
    }

    static void gotoScaling(double[] A, int lda, int n, int ilo, int ihi, double[] scale, int scaleOff) {
        for (int i = ilo; i <= ihi; i++) {
            scale[scaleOff + i] = 1.0;
        }

        final double sclfac = 2.0;
        final double factor = 0.95;
        double sfmin1 = Dlamch.safmin() / Dlamch.eps();
        double sfmax1 = 1.0 / sfmin1;
        double sfmin2 = sfmin1 * sclfac;
        double sfmax2 = 1.0 / sfmin2;

        // WARNING: NaN inputs cause an infinite loop here.
        // When A contains NaN, c and r become NaN. NaN comparisons always return false,
        // so the "c == 0 || r == 0" guard is bypassed, and the convergence check
        // "c + r >= factor * s" also evaluates to false, causing conv to be set false
        // on every iteration and the outer while(!conv) loop to never terminate.
        // Callers must validate input before invoking this method.
        // See: SchurTest — passing NaN to Schur.decompose() caused a multi-hour hang.
        boolean conv = false;
        while (!conv) {
            conv = true;
            for (int i = ilo; i <= ihi; i++) {
                // Compute column norm (excluding diagonal)
                double c = 0.0;
                for (int k = ilo; k <= ihi; k++) {
                    if (k != i) {
                        c += Math.abs(A[k * lda + i]);
                    }
                }
                // Compute row norm (excluding diagonal)
                double r = 0.0;
                for (int k = ilo; k <= ihi; k++) {
                    if (k != i) {
                        r += Math.abs(A[i * lda + k]);
                    }
                }

                // Find max element in column i
                double ca = 0.0;
                for (int k = 0; k <= ihi; k++) {
                    double val = Math.abs(A[k * lda + i]);
                    if (val > ca) {
                        ca = val;
                    }
                }
                // Find max element in row i
                double ra = 0.0;
                for (int k = ilo; k < n; k++) {
                    double val = Math.abs(A[i * lda + k]);
                    if (val > ra) {
                        ra = val;
                    }
                }

                if (c == 0 || r == 0) {
                    continue;
                }

                double g = r / sclfac;
                double f = 1.0;
                double s = c + r;

                while (c < g && Math.max(f, Math.max(c, ca)) < sfmax2 && Math.min(r, Math.min(g, ra)) > sfmin2) {
                    if (Double.isNaN(c + f + ca + r + g + ra)) {
                        throw new RuntimeException("lapack: NaN");
                    }
                    f *= sclfac;
                    c *= sclfac;
                    ca *= sclfac;
                    g /= sclfac;
                    r /= sclfac;
                    ra /= sclfac;
                }

                g = c / sclfac;
                while (r <= g && Math.max(r, ra) < sfmax2 && Math.min(Math.min(f, c), Math.min(g, ca)) > sfmin2) {
                    f /= sclfac;
                    c /= sclfac;
                    ca /= sclfac;
                    g /= sclfac;
                    r *= sclfac;
                    ra *= sclfac;
                }

                if (c + r >= factor * s) {
                    continue;
                }
                if (f < 1 && scale[scaleOff + i] < 1 && f * scale[scaleOff + i] <= sfmin1) {
                    continue;
                }
                if (f > 1 && scale[scaleOff + i] > 1 && scale[scaleOff + i] >= sfmax1 / f) {
                    continue;
                }

                // Now balance
                scale[scaleOff + i] *= f;
                for (int k = ilo; k < n; k++) {
                    A[i * lda + k] /= f;
                }
                for (int k = 0; k <= ihi; k++) {
                    A[k * lda + i] *= f;
                }
                conv = false;
            }
        }
    }
}
