/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Forms the eigenvectors of a balanced matrix by backward transformation.
 * LAPACK DGEBAK routine.
 *
 * <p>Performs the inverse of DGEBAL to transform eigenvectors back
 * to the original coordinate system.</p>
 *
 * <p>Updates V as:</p>
 * <ul>
 *   <li>V = P D V       if side == 'R' (right eigenvectors)</li>
 *   <li>V = P D^{-1} V  if side == 'L' (left eigenvectors)</li>
 * </ul>
 *
 * <p>Based on gonum/lapack/gonum/dgebak.go</p>
 */
interface Dgebak {

    static void dgebak(char job, BLAS.Side side, int n, int ilo, int ihi,
                        double[] scale, int scaleOff, int m, double[] V, int ldv) {
        if (n == 0 || m == 0) {
            return;
        }

        char jobUpper = Character.toUpperCase(job);

        if (jobUpper == 'N') {
            return;
        }

        if (ilo != ihi && jobUpper != 'P') {
            if (side == BLAS.Side.Right) {
                for (int i = ilo; i <= ihi; i++) {
                    double s = scale[scaleOff + i];
                    if (s != 1.0) {
                        for (int j = 0; j < m; j++) {
                            V[i * ldv + j] *= s;
                        }
                    }
                }
            } else {
                for (int i = ilo; i <= ihi; i++) {
                    double s = scale[scaleOff + i];
                    if (s != 1.0) {
                        for (int j = 0; j < m; j++) {
                            V[i * ldv + j] /= s;
                        }
                    }
                }
            }
        }

        if (jobUpper == 'S') {
            return;
        }

        for (int i = ilo - 1; i >= 0; i--) {
            int k = (int) scale[scaleOff + i];
            if (k == i) {
                continue;
            }
            for (int j = 0; j < m; j++) {
                double tmp = V[i * ldv + j];
                V[i * ldv + j] = V[k * ldv + j];
                V[k * ldv + j] = tmp;
            }
        }

        for (int i = ihi + 1; i < n; i++) {
            int k = (int) scale[scaleOff + i];
            if (k == i) {
                continue;
            }
            for (int j = 0; j < m; j++) {
                double tmp = V[i * ldv + j];
                V[i * ldv + j] = V[k * ldv + j];
                V[k * ldv + j] = tmp;
            }
        }
    }

    static void dgebak(char job, BLAS.Side side, int n, int ilo, int ihi,
                        double[] scale, int m, double[] V, int ldv) {
        dgebak(job, side, n, ilo, ihi, scale, 0, m, V, ldv);
    }
}
