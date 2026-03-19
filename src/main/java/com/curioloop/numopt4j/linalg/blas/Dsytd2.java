/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * LAPACK DSYTD2: Reduces a symmetric matrix to symmetric tridiagonal form.
 * Unblocked algorithm.
 * 
 * Based on LAPACK, adapted for row-major storage.
 */
interface Dsytd2 {

    /**
     * Reduces a symmetric matrix to tridiagonal form (unblocked).
     * Version with external work array for zero-allocation.
     *
     * @param uplo  uplo enum: Upper or Lower
     * @param n     order of matrix A
     * @param A     symmetric matrix (n x n, row-major), overwritten
     * @param aOff  offset into A
     * @param lda   leading dimension of A
     * @param d     diagonal elements (output, length n)
     * @param dOff  offset into d
     * @param e     off-diagonal elements (output, length n-1)
     * @param eOff  offset into e
     * @param tau   Householder scalars (output, length n-1), also used as temp workspace
     * @param tauOff offset into tau
     * @param work  external work array (unused, kept for API compatibility)
     * @param workOff offset into work (unused)
     */
    static void dsytd2(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda,
                       double[] d, int dOff, double[] e, int eOff, double[] tau, int tauOff,
                       double[] work, int workOff) {
        if (n == 0) {
            return;
        }

        boolean upper = uplo == BLAS.Uplo.Upper;

        if (upper) {
            // Reduce the upper triangle of A.
            // Following Gonum: use tau[tauOff:tauOff+i] as temporary workspace
            for (int i = n - 2; i >= 0; i--) {
                // Generate elementary reflector H_i = I - tau * v * v^T to
                // annihilate A[0:i, i+1].
                // In Row-Major: alpha = A[i, i+1] = A[i*lda + i+1]
                // x = A[0:i+1, i+1] with stride lda starting at offset i+1
                double beta = Dlarfg.dlarfg(i + 1, A[aOff + i * lda + i + 1], A, aOff + i + 1, lda, tau, tauOff + i);
                
                double taui = tau[tauOff + i];
                e[eOff + i] = beta;
                
                if (taui != 0.0) {
                    // Apply H_i from both sides to A[0:i, 0:i].
                    A[aOff + i * lda + i + 1] = 1.0;
                    
                    // Compute x := tau * A * v, storing in tau[tauOff:tauOff+i+1]
                    // Gonum: bi.Dsymv(uplo, i+1, taui, a, lda, a[i+1:], lda, 0, tau, 1)
                    // v is stored at A[0:i+1, i+1] with stride lda starting at offset i+1
                    Dsymv.dsymv(BLAS.Uplo.Upper, i + 1, taui, A, aOff, lda, A, aOff + i + 1, lda, 0.0, tau, tauOff, 1);
                    
                    // Compute w := x - 1/2 * tau * (x^T * v) * v
                    // Gonum: alpha := -0.5 * taui * bi.Ddot(i+1, tau, 1, a[i+1:], lda)
                    double alpha = -0.5 * taui * Ddot.ddot(i + 1, tau, tauOff, 1, A, aOff + i + 1, lda);
                    // Gonum: bi.Daxpy(i+1, alpha, a[i+1:], lda, tau, 1)
                    Daxpy.daxpy(i + 1, alpha, A, aOff + i + 1, lda, tau, tauOff, 1);
                    
                    // Apply the transformation as a rank-2 update
                    // A = A - v * w^T - w * v^T
                    // Gonum: bi.Dsyr2(uplo, i+1, -1, a[i+1:], lda, tau, 1, a, lda)
                    Dsyr2.dsyr2(BLAS.Uplo.Upper, i + 1, -1.0, A, aOff + i + 1, lda, tau, tauOff, 1, A, aOff, lda);
                    A[aOff + i * lda + i + 1] = e[eOff + i];
                }
                
                d[dOff + i + 1] = A[aOff + (i + 1) * lda + i + 1];
                tau[tauOff + i] = taui;
            }
            d[dOff] = A[aOff];
        } else {
            // Reduce the lower triangle of A.
            // Following Gonum: use tau[tauOff+i:tauOff+n-1] as temporary workspace
            for (int i = 0; i < n - 1; i++) {
                int m = n - i - 1;
                
                // Generate elementary reflector H_i
                // For row-major: alpha at A[i+1, i] = A[(i+1)*lda + i]
                // x starts at A[i+2, i] = A[(i+2)*lda + i] for the remaining m-1 elements
                double beta = Dlarfg.dlarfg(m, A[aOff + (i + 1) * lda + i], A, aOff + Math.min(i + 2, n - 1) * lda + i, lda, tau, tauOff + i);
                
                double taui = tau[tauOff + i];
                e[eOff + i] = beta;
                
                if (taui != 0.0) {
                    // Apply H_i from both sides to A[i+1:n, i+1:n].
                    A[aOff + (i + 1) * lda + i] = 1.0;
                    
                    // Compute x := tau * A * v, storing in tau[tauOff+i:tauOff+n-1]
                    // Gonum: bi.Dsymv(uplo, n-i-1, taui, a[(i+1)*lda+i+1:], lda, a[(i+1)*lda+i:], lda, 0, tau[i:], 1)
                    Dsymv.dsymv(BLAS.Uplo.Lower, m, taui, A, aOff + (i + 1) * lda + i + 1, lda, 
                               A, aOff + (i + 1) * lda + i, lda, 0.0, tau, tauOff + i, 1);
                    
                    // Compute w := x - 1/2 * tau * (x^T * v) * v
                    // Gonum: alpha := -0.5 * taui * bi.Ddot(n-i-1, tau[i:], 1, a[(i+1)*lda+i:], lda)
                    double alpha = -0.5 * taui * Ddot.ddot(m, tau, tauOff + i, 1, A, aOff + (i + 1) * lda + i, lda);
                    // Gonum: bi.Daxpy(n-i-1, alpha, a[(i+1)*lda+i:], lda, tau[i:], 1)
                    Daxpy.daxpy(m, alpha, A, aOff + (i + 1) * lda + i, lda, tau, tauOff + i, 1);
                    
                    // Apply the transformation as a rank-2 update
                    // Gonum: bi.Dsyr2(uplo, n-i-1, -1, a[(i+1)*lda+i:], lda, tau[i:], 1, a[(i+1)*lda+i+1:], lda)
                    Dsyr2.dsyr2(BLAS.Uplo.Lower, m, -1.0, A, aOff + (i + 1) * lda + i, lda, tau, tauOff + i, 1, 
                               A, aOff + (i + 1) * lda + i + 1, lda);
                    A[aOff + (i + 1) * lda + i] = e[eOff + i];
                }
                
                d[dOff + i] = A[aOff + i * lda + i];
                tau[tauOff + i] = taui;
            }
            d[dOff + n - 1] = A[aOff + (n - 1) * lda + n - 1];
        }
    }
}
