/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Generates the orthogonal matrix Q from Hessenberg reduction.
 * LAPACK DORGHR algorithm.
 *
 * <p>Generates the orthogonal matrix Q from the Householder
 * transformations in the Hessenberg reduction.</p>
 * 
 * <p>Algorithm:</p>
 * <ul>
 *   <li>Initialize Q to identity</li>
 *   <li>For i from ihi-1 down to ilo:</li>
 *   <li>Extract v from A[i+1:i+ihi+1, i] where v[0]=1 implicitly</li>
 *   <li>Apply H = I - tau * v * v^T from the right</li>
 * </ul>
 */
interface Dorghr {

    static void dorghr(int n, int ilo, int ihi, double[] A, int lda,
                       double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        if (n <= 1) {
            return;
        }

        int minWork = 2 * n;
        if (lwork < minWork && lwork != -1) {
            return;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * lda + j] = (i == j) ? 1.0 : 0.0;
            }
        }

        for (int i = ihi - 1; i >= ilo; i--) {
            int m = ihi - i + 1;
            
            if (tau[tauOff + i] != 0.0 && m > 2) {
                int nrow = n - i;
                int ncol = m - 1;
                
                work[workOff] = 1.0;
                for (int j = 0; j < ncol - 1; j++) {
                    work[workOff + j + 1] = A[(i + 2 + j) * lda + i];
                }
                
                int wOff = workOff + n;
                for (int ii = 0; ii < nrow; ii++) {
                    double sum = 0.0;
                    for (int jj = 0; jj < ncol; jj++) {
                        sum += A[(i + ii) * lda + (i + 1 + jj)] * work[workOff + jj];
                    }
                    work[wOff + ii] = sum;
                }
                
                for (int ii = 0; ii < nrow; ii++) {
                    double coeff = -tau[tauOff + i] * work[wOff + ii];
                    if (coeff != 0.0) {
                        for (int jj = 0; jj < ncol; jj++) {
                            A[(i + ii) * lda + (i + 1 + jj)] += coeff * work[workOff + jj];
                        }
                    }
                }
            }
        }
    }

}
