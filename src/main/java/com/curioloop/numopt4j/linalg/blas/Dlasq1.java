/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

interface Dlasq1 {

    /**
     * Computes the singular values of an n×n bidiagonal matrix with diagonal {@code d}
     * and off-diagonal {@code e} using the dqds algorithm. On exit, {@code d} contains
     * the singular values in decreasing order.
     *
     * <p>The algorithm squares the input data and calls {@link Dlasq2} to compute
     * eigenvalues of the associated symmetric positive definite tridiagonal matrix,
     * then takes square roots to recover singular values.
     *
     *
     * @param n       order of the bidiagonal matrix (n ≥ 0)
     * @param d       diagonal elements (length ≥ dOff+n); overwritten with singular values on exit
     * @param dOff    offset into d
     * @param e       off-diagonal elements (length ≥ eOff+n-1); overwritten on exit
     * @param eOff    offset into e
     * @param work    workspace array (length ≥ workOff+4*n+16)
     * @param workOff offset into work
     * @return 0 on success; 2 if max iterations exceeded (d and e contain partial results)
     */
    static int dlasq1(int n, double[] d, int dOff, double[] e, int eOff, double[] work, int workOff) {
        if (n == 0) {
            return 0;
        }

        if (n == 1) {
            d[dOff] = abs(d[dOff]);
            return 0;
        }

        if (n == 2) {
            Dlas2.dlas2(d[dOff], e[eOff], d[dOff + 1], work, workOff);
            d[dOff] = max(work[workOff], work[workOff + 1]);
            d[dOff + 1] = min(work[workOff], work[workOff + 1]);
            return 0;
        }

        double sigmx = 0;
        for (int i = 0; i < n - 1; i++) {
            d[dOff + i] = abs(d[dOff + i]);
            sigmx = max(sigmx, abs(e[eOff + i]));
        }
        d[dOff + n - 1] = abs(d[dOff + n - 1]);

        if (sigmx == 0) {
            Dlasrt.dlasrt('D', n, d, dOff);
            return 0;
        }

        for (int i = 0; i < n; i++) {
            sigmx = max(sigmx, d[dOff + i]);
        }

        double eps = Dlamch.EPSILON;
        double safmin = Dlamch.SAFE_MIN;
        double scale = sqrt(eps / safmin);

        for (int i = 0; i < n; i++) {
            work[workOff + 2 * i] = d[dOff + i] / sigmx * scale;
        }
        for (int i = 0; i < n - 1; i++) {
            work[workOff + 2 * i + 1] = e[eOff + i] / sigmx * scale;
        }

        for (int i = 0; i < 2 * n - 1; i++) {
            work[workOff + i] *= work[workOff + i];
        }
        work[workOff + 2 * n - 1] = 0;

        int dstateOff = workOff + 4 * n;
        int[] istate = new int[7];

        int info = Dlasq2.dlasq2(n, work, workOff, istate, 0, work, dstateOff);

        if (info == 0) {
            for (int i = 0; i < n; i++) {
                d[dOff + i] = sqrt(work[workOff + i]) * sigmx / scale;
            }
        } else if (info == 2) {
            for (int i = 0; i < n; i++) {
                d[dOff + i] = sqrt(work[workOff + 2 * i]) * sigmx / scale;
                e[eOff + i] = sqrt(work[workOff + 2 * i + 1]) * sigmx / scale;
            }
        }

        return info;
    }
}
