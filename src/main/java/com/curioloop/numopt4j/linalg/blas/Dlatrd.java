/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * LAPACK DLATRD: Reduces a symmetric matrix to banded form.
 * Blocked algorithm - computes the first nb columns of the tridiagonal 
 * matrix T and the matrix W needed to update the unreduced part of A.
 * 
 * Based on gonum implementation, adapted for row-major storage.
 */
interface Dlatrd {

    /**
     * DLATRD reduces the first nb columns of a symmetric matrix A
     * to tridiagonal form.
     *
     * @param uplo  'U' for upper triangular, 'L' for lower triangular
     * @param n     order of matrix A
     * @param nb    number of columns to reduce
     * @param A     symmetric matrix (n x n, row-major), overwritten
     * @param lda   leading dimension of A
     * @param e     off-diagonal elements (output, length n-1)
     * @param tau   Householder scalars (output, length n-1)
     * @param w     workspace matrix (n x nb)
     * @param ldw   leading dimension of w
     */
    static void dlatrd(BLAS.Uplo uplo, int n, int nb, double[] A, int lda,
                       double[] e, double[] tau, double[] w, int ldw) {
        dlatrd(uplo, n, nb, A, 0, lda, e, 0, tau, 0, w, 0, ldw);
    }

    static void dlatrd(BLAS.Uplo uplo, int n, int nb, double[] A, int aOff, int lda,
                       double[] e, int eOff, double[] tau, int tauOff,
                       double[] w, int wOff, int ldw) {
        if (n == 0) {
            return;
        }

        boolean upper = uplo == BLAS.Uplo.Upper;

        if (upper) {
            // Reduce last nb columns of upper triangle
            for (int i = n - 1; i >= n - nb; i--) {
                int iw = i - n + nb;
                
                if (i < n - 1) {
                    // Update A(0:i, i)
                    // A[0:i, i] := A[0:i, i] - A[0:i, i+1:n] * w[i, iw+1:n]
                    // A[0:i, i] := A[0:i, i] - w[0:i, iw+1:n] * A[i, i+1:n]
                    // dgemv(trans, m, n, alpha, A, aOff, lda, x, xOff, beta, y, yOff)
                    Dgemv.dgemv(BLAS.Transpose.NoTrans, i + 1, n - i - 1, -1.0,
                               A, aOff + i + 1, lda,
                               w, wOff + i * ldw + iw + 1, 1.0,
                               A, aOff + i * lda);
                    
                    Dgemv.dgemv(BLAS.Transpose.NoTrans, i + 1, n - i - 1, -1.0,
                               w, wOff + iw + 1, ldw,
                               A, aOff + i * lda + i + 1, 1.0,
                               A, aOff + i * lda);
                }
                
                if (i > 0) {
                    // Generate elementary reflector H_i to annihilate A(0:i-2, i)
                    // Use in-place vector from A[0:i, i] - already contiguous in row-major
                    // Call Dlarfg with in-place vector: x at column i, stride lda
                    double beta = Dlarfg.dlarfg(i, A[aOff + (i - 1) * lda + i], A, aOff + i, lda, tau, tauOff + i - 1);
                    
                    e[eOff + i - 1] = beta;
                    A[aOff + (i - 1) * lda + i] = 1.0;

                    // Compute W(0:i-1, i)
                    // dsymv(uplo, n, alpha, A, lda, x, xOff, incx, beta, y, yOff, incy)
                    // Use in-place v from A: x at A[aOff + i], stride lda
                    BLAS.dsymv(BLAS.Uplo.Upper, i, 1.0, A, aOff, lda,
                               A, aOff + i, lda, 0.0,
                               w, wOff + iw, ldw);
                    
                    if (i < n - 1) {
                        // dgemv(trans, m, n, alpha, A, aOff, lda, x, xOff, beta, y, yOff)
                        // w[0:i-1, iw] += A[0:i-1, i+1:n]^T * w[0:i-1, iw+1:n]
                        Dgemv.dgemv(BLAS.Transpose.Trans, i, n - i - 1, 1.0,
                                   w, wOff + iw + 1, ldw,
                                   A, aOff + i * lda + i + 1, 0.0,
                                   w, wOff + (i + 1) * ldw + iw);
                        
                        // w[0:i-1, iw] -= A[0:i-1, i+1:n] * w[0:i-1, iw+1:n]
                        Dgemv.dgemv(BLAS.Transpose.NoTrans, i, n - i - 1, -1.0,
                                   A, aOff + i * lda + i + 1, lda,
                                   w, wOff + (i + 1) * ldw + iw, 1, 0.0,
                                   w, wOff + iw);
                        
                        // w[0:i-1, iw+1:n] := A[0:i-1, i+1:n]^T * v
                        // v is in A at column i with stride lda
                        Dgemv.dgemv(BLAS.Transpose.Trans, i, n - i - 1, 1.0,
                                   A, aOff + i * lda + i + 1, lda,
                                   A, aOff + i, lda, 0.0,
                                   w, wOff + (i + 1) * ldw + iw);
                        
                        // w[0:i-1, iw+1:n] -= w[0:i-1, iw+1:n] * v
                        Dgemv.dgemv(BLAS.Transpose.NoTrans, i, n - i - 1, -1.0,
                                   w, wOff + iw + 1, ldw,
                                   A, aOff + i, lda, 1.0,
                                   w, wOff + (i + 1) * ldw + iw);
                    }
                    
                    // dscal(n, alpha, x, xOff, xInc)
                    BLAS.dscal(i, tau[tauOff + i - 1], w, wOff + iw, ldw);
                    
                    double dot = BLAS.ddot(i, w, wOff + iw, ldw, A, aOff + i, lda);
                    double alpha = -0.5 * tau[tauOff + i - 1] * dot;
                    BLAS.daxpy(i, alpha, A, aOff + i, lda, w, wOff + iw, ldw);
                }
            }
        } else {
            // Reduce first nb columns of lower triangle
            for (int i = 0; i < nb; i++) {
                // Update A(i:n, i)
                Dgemv.dgemv(BLAS.Transpose.NoTrans, n - i, i, -1.0,
                           A, aOff + i * lda, lda,
                           w, wOff + i * ldw, 1.0,
                           A, aOff + i * lda + i);
                
                Dgemv.dgemv(BLAS.Transpose.NoTrans, n - i, i, -1.0,
                           w, wOff + i * ldw, ldw,
                           A, aOff + i * lda, 1.0,
                           A, aOff + i * lda + i);
                
                if (i < n - 1) {
                    // Generate elementary reflector H_i to annihilate A(i+2:n, i)
                    // Use in-place vector from A[i+1:n, i] - already contiguous in row-major
                    int m = n - i - 1;
                    double beta = Dlarfg.dlarfg(m, A[aOff + (i + 1) * lda + i], A, aOff + (i + 1) * lda + i, lda, tau, tauOff + i);
                    
                    e[eOff + i] = beta;
                    A[aOff + (i + 1) * lda + i] = 1.0;

                    // Compute W(i+1:n, i)
                    BLAS.dsymv(BLAS.Uplo.Lower, m, 1.0, 
                               A, aOff + (i + 1) * lda + i + 1, lda,
                               A, aOff + (i + 1) * lda + i, lda, 0.0,
                               w, wOff + (i + 1) * ldw + i, ldw);
                    
                    // w[0:i, i] := A[i+1:n, 0:i]^T * v
                    // v is in A at column i starting from row i+1
                    Dgemv.dgemv(BLAS.Transpose.Trans, m, i, 1.0,
                               w, wOff + (i + 1) * ldw, ldw,
                               A, aOff + (i + 1) * lda + i, lda, 0.0,
                               w, wOff + i);
                    
                    // w[0:i, i] -= A[i+1:n, 0:i] * w[i+1:n, i]
                    Dgemv.dgemv(BLAS.Transpose.NoTrans, m, i, -1.0,
                               A, aOff + (i + 1) * lda, lda,
                               w, wOff + (i + 1) * ldw + i, 1, 0.0,
                               w, wOff + i);
                    
                    // w[i+1:n, i+1] := A[i+1:n, 0:i]^T * v
                    Dgemv.dgemv(BLAS.Transpose.Trans, m, i, 1.0,
                               A, aOff + (i + 1) * lda, lda,
                               A, aOff + (i + 1) * lda + i, lda, 0.0,
                               w, wOff + (i + 1) * ldw + i + 1);
                    
                    // w[i+1:n, i+1] -= w[i+1:n, 0:i] * v
                    Dgemv.dgemv(BLAS.Transpose.NoTrans, m, i, -1.0,
                               w, wOff + (i + 1) * ldw, ldw,
                               A, aOff + (i + 1) * lda + i, lda, 1.0,
                               w, wOff + (i + 1) * ldw + i + 1);
                    
                    // w[i+1:n, i] := tau[i] * w[i+1:n, i]
                    BLAS.dscal(m, tau[tauOff + i], w, wOff + (i + 1) * ldw + i, ldw);
                    
                    // w[i+1:n, i] -= 0.5 * tau[i] * (w[i+1:n, i]^T * v) * v
                    double dot = BLAS.ddot(m, w, wOff + (i + 1) * ldw + i, ldw, A, aOff + (i + 1) * lda + i, lda);
                    double alpha = -0.5 * tau[tauOff + i] * dot;
                    BLAS.daxpy(m, alpha, A, aOff + (i + 1) * lda + i, lda, w, wOff + (i + 1) * ldw + i, ldw);
                }
            }
        }
    }
}
