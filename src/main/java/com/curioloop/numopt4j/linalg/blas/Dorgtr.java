/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Generates the orthogonal matrix Q from symmetric tridiagonalization.
 * LAPACK DORGTR algorithm.
 */
interface Dorgtr {

    static void dorgtr(BLAS.Uplo uplo, int n, double[] A, int lda,
                       double[] tau, int tauOff, double[] work, int workOff, int lwork) {
        if (n <= 0) {
            return;
        }

        if (n == 1) {
            A[0] = 1.0;
            return;
        }

        boolean upper = uplo == BLAS.Uplo.Upper;

        if (upper) {
            // Q was determined by a call to Dsytrd with uplo == 'U'.
            // Shift the vectors which define the elementary reflectors one column
            // to the left, and set the last row and column of Q to those of the unit
            // matrix.
            
            // Preprocessing: shift vectors left
            for (int j = 0; j < n - 1; j++) {
                for (int i = 0; i < j; i++) {
                    A[i * lda + j] = A[i * lda + j + 1];
                }
                A[(n - 1) * lda + j] = 0;
            }
            for (int i = 0; i < n - 1; i++) {
                A[i * lda + n - 1] = 0;
            }
            A[(n - 1) * lda + n - 1] = 1;

            // Generate Q[0:n-1, 0:n-1] using Dorgql
            if (n > 1) {
                Dorgql.dorgql(n - 1, n - 1, n - 1, A, lda, 0, tau, tauOff, work, workOff, lwork);
            }
        } else {
            // Lower triangular case
            // Q = H_0 * H_1 * ... * H_{n-2}
            // Preprocessing: shift vectors one column to the right
            for (int j = n - 1; j > 0; j--) {
                A[j] = 0.0;
                for (int i = j + 1; i < n; i++) {
                    A[i * lda + j] = A[i * lda + j - 1];
                }
            }
            A[0] = 1.0;
            for (int i = 1; i < n; i++) {
                A[i * lda] = 0.0;
            }
            
            // Generate Q[1:n, 1:n]
            if (n > 1) {
                int aOff = lda + 1;
                Dgeqr.dorgqr(n - 1, n - 1, n - 1, A, aOff, lda, tau, tauOff, work, workOff, lwork);
            }
        }
    }
}
