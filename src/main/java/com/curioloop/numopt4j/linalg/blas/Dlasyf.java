/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * DLASYF computes a partial factorization of a real symmetric matrix A
 * using the Bunch-Kaufman diagonal pivoting method. The partial
 * factorization has the form:
 * <pre>
 *   A = ( I  U12 ) ( A11  0  ) (  I    0   )  if UPLO = 'U'
 *       ( 0   U22 ) (  0   D  ) ( U12**T U22**T )
 *
 *   A = ( L11   0  ) ( D    0  ) ( L11**T L21**T )  if UPLO = 'L'
 *       ( L21  I  ) (  0  A22 ) (  0      I    )
 * </pre>
 * where the order of D is at most {@code nb}. The actual order is returned
 * in the argument {@code kb}, and is either {@code nb} or {@code nb-1},
 * or is equal to {@code n} if the matrix is small.
 *
 * <p>DLASYF is an auxiliary routine called by DSYTRF. It uses blocked
 * code (based on BLAS Level 3) to factorize {@code nb} columns at a time.
 *
 * <p>Reference: Fortran netlib LAPACK DLASYF.
 *
 * @see Dsytrf
 */
interface Dlasyf {

    double ALPHA = (1.0 + sqrt(17.0)) / 8.0;

    static int dlasyf(BLAS.Uplo uplo, int n, int nb, double[] A, int aOff, int lda,
                       int[] ipiv, int ipivOff, double[] W, int wOff, int ldw) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        
        int kb = 0;
        
        if (upper) {
            int k = n - 1;
            while (true) {
                int kw = k + nb - n;
                
                if ((k <= n - nb && nb < n) || k < 0) {
                    break;
                }
                
                for (int i = 0; i <= k; i++) {
                    W[wOff + kw * ldw + i] = A[aOff + i * lda + k];
                }
                
                if (k < n - 1) {
                    for (int i = 0; i <= k; i++) {
                        double sum = 0;
                        for (int j = k + 1; j < n; j++) {
                            int jw = j + nb - n;
                            sum += A[aOff + i * lda + j] * W[wOff + jw * ldw + k];
                        }
                        W[wOff + kw * ldw + i] -= sum;
                    }
                }
                
                int kstep = 1;
                double absakk = abs(W[wOff + kw * ldw + k]);
                
                double colmax = 0;
                int imax = k;
                if (k > 0) {
                    for (int i = 0; i < k; i++) {
                        double absAik = abs(W[wOff + kw * ldw + i]);
                        if (absAik > colmax) {
                            colmax = absAik;
                            imax = i;
                        }
                    }
                }
                
                int kp;
                if (max(absakk, colmax) == 0) {
                    kp = k;
                } else if (absakk >= ALPHA * colmax) {
                    kp = k;
                } else {
                    int kw1 = kw - 1;
                    
                    for (int i = 0; i < imax; i++) {
                        W[wOff + kw1 * ldw + i] = A[aOff + i * lda + imax];
                    }
                    for (int i = imax; i <= k; i++) {
                        W[wOff + kw1 * ldw + i] = A[aOff + imax * lda + i];
                    }
                    
                    if (k < n - 1) {
                        for (int i = 0; i <= k; i++) {
                            double sum = 0;
                            for (int j = k + 1; j < n; j++) {
                                int jw = j + nb - n;
                                sum += A[aOff + i * lda + j] * W[wOff + jw * ldw + imax];
                            }
                            W[wOff + kw1 * ldw + i] -= sum;
                        }
                    }
                    
                    double rowmax = 0;
                    int jmax = imax;
                    for (int j = imax + 1; j <= k; j++) {
                        double absAij = abs(W[wOff + kw1 * ldw + j]);
                        if (absAij > rowmax) {
                            rowmax = absAij;
                            jmax = j;
                        }
                    }
                    if (imax > 0) {
                        for (int j = 0; j < imax; j++) {
                            double absAij = abs(W[wOff + kw1 * ldw + j]);
                            if (absAij > rowmax) {
                                rowmax = absAij;
                                jmax = j;
                            }
                        }
                    }
                    
                    if (absakk >= ALPHA * colmax * (colmax / rowmax)) {
                        kp = k;
                    } else if (abs(W[wOff + kw1 * ldw + imax]) >= ALPHA * rowmax) {
                        kp = imax;
                        for (int i = 0; i <= k; i++) {
                            W[wOff + kw * ldw + i] = W[wOff + kw1 * ldw + i];
                        }
                    } else {
                        kp = imax;
                        kstep = 2;
                    }
                }
                
                int kk = k - kstep + 1;
                int kkw = kk + nb - n;
                
                if (kp != kk) {
                    A[aOff + kp * lda + kp] = A[aOff + kk * lda + kk];
                    
                    for (int j = kp + 1; j < kk; j++) {
                        A[aOff + kp * lda + j] = A[aOff + j * lda + kk];
                    }
                    
                    if (kp > 0) {
                        for (int i = 0; i < kp; i++) {
                            A[aOff + i * lda + kp] = A[aOff + i * lda + kk];
                        }
                    }
                    
                    if (k < n - 1) {
                        for (int j = k + 1; j < n; j++) {
                            double temp = A[aOff + kk * lda + j];
                            A[aOff + kk * lda + j] = A[aOff + kp * lda + j];
                            A[aOff + kp * lda + j] = temp;
                        }
                    }
                    
                    for (int j = kkw; j < nb; j++) {
                        double temp = W[wOff + j * ldw + kk];
                        W[wOff + j * ldw + kk] = W[wOff + j * ldw + kp];
                        W[wOff + j * ldw + kp] = temp;
                    }
                }
                
                if (kstep == 1) {
                    for (int i = 0; i <= k; i++) {
                        A[aOff + i * lda + k] = W[wOff + kw * ldw + i];
                    }
                    
                    if (k > 0) {
                        double r1 = 1.0 / A[aOff + k * lda + k];
                        for (int i = 0; i < k; i++) {
                            A[aOff + i * lda + k] *= r1;
                        }
                    }
                    
                    ipiv[ipivOff + k] = kp + 1;
                } else {
                    if (k > 1) {
                        double d21 = W[wOff + kw * ldw + (k - 1)];
                        double d11 = W[wOff + kw * ldw + k] / d21;
                        double d22 = W[wOff + (kw - 1) * ldw + (k - 1)] / d21;
                        double t = 1.0 / (d11 * d22 - 1.0);
                        d21 = t / d21;
                        
                        for (int j = 0; j < k - 1; j++) {
                            A[aOff + j * lda + (k - 1)] = d21 * (d11 * W[wOff + (kw - 1) * ldw + j] - W[wOff + kw * ldw + j]);
                            A[aOff + j * lda + k] = d21 * (d22 * W[wOff + kw * ldw + j] - W[wOff + (kw - 1) * ldw + j]);
                        }
                    }
                    
                    A[aOff + (k - 1) * lda + (k - 1)] = W[wOff + (kw - 1) * ldw + (k - 1)];
                    A[aOff + (k - 1) * lda + k] = W[wOff + kw * ldw + (k - 1)];
                    A[aOff + k * lda + k] = W[wOff + kw * ldw + k];
                    
                    ipiv[ipivOff + k] = -(kp + 1);
                    ipiv[ipivOff + (k - 1)] = -(kp + 1);
                }
                
                k -= kstep;
            }
            
            kb = n - 1 - k;
            
            if (kb > 0 && k >= 0) {
                for (int j = 0; j <= k; j++) {
                    for (int i = 0; i <= j; i++) {
                        double sum = 0;
                        for (int l = k + 1; l < n; l++) {
                            int lw = l + nb - n;
                            sum += A[aOff + i * lda + l] * W[wOff + lw * ldw + j];
                        }
                        A[aOff + i * lda + j] -= sum;
                    }
                }
                
                int j = k + 1;
                while (j < n) {
                    int jj = j;
                    int jp = ipiv[ipivOff + j];
                    if (jp < 0) {
                        jp = -jp;
                        j++;
                    }
                    j++;
                    if (jp != jj + 1 && j < n) {
                        jp--;
                        for (int i = j; i < n; i++) {
                            double temp = A[aOff + jp * lda + i];
                            A[aOff + jp * lda + i] = A[aOff + jj * lda + i];
                            A[aOff + jj * lda + i] = temp;
                        }
                    }
                }
            }
            
        } else {
            int k = 0;
            while (true) {
                if ((k >= nb && nb < n) || k >= n) {
                    break;
                }
                
                for (int i = k; i < n; i++) {
                    W[wOff + k * ldw + i] = A[aOff + i * lda + k];
                }
                
                if (k > 0) {
                    for (int i = k; i < n; i++) {
                        double sum = 0;
                        for (int j = 0; j < k; j++) {
                            sum += A[aOff + i * lda + j] * W[wOff + j * ldw + k];
                        }
                        W[wOff + k * ldw + i] -= sum;
                    }
                }
                
                int kstep = 1;
                double absakk = abs(W[wOff + k * ldw + k]);
                
                double colmax = 0;
                int imax = k;
                if (k < n - 1) {
                    for (int i = k + 1; i < n; i++) {
                        double absAik = abs(W[wOff + k * ldw + i]);
                        if (absAik > colmax) {
                            colmax = absAik;
                            imax = i;
                        }
                    }
                }
                
                int kp;
                if (max(absakk, colmax) == 0) {
                    kp = k;
                } else if (absakk >= ALPHA * colmax) {
                    kp = k;
                } else {
                    for (int i = k; i < imax; i++) {
                        W[wOff + (k + 1) * ldw + i] = A[aOff + imax * lda + i];
                    }
                    for (int i = imax; i < n; i++) {
                        W[wOff + (k + 1) * ldw + i] = A[aOff + i * lda + imax];
                    }
                    
                    if (k > 0) {
                        for (int i = k; i < n; i++) {
                            double sum = 0;
                            for (int j = 0; j < k; j++) {
                                sum += A[aOff + i * lda + j] * W[wOff + j * ldw + imax];
                            }
                            W[wOff + (k + 1) * ldw + i] -= sum;
                        }
                    }
                    
                    double rowmax = 0;
                    int jmax = k;
                    for (int j = k + 1; j < imax; j++) {
                        double absAij = abs(W[wOff + (k + 1) * ldw + j]);
                        if (absAij > rowmax) {
                            rowmax = absAij;
                            jmax = j;
                        }
                    }
                    if (imax < n - 1) {
                        for (int j = imax + 1; j < n; j++) {
                            double absAij = abs(W[wOff + (k + 1) * ldw + j]);
                            if (absAij > rowmax) {
                                rowmax = absAij;
                                jmax = j;
                            }
                        }
                    }
                    
                    if (absakk >= ALPHA * colmax * (colmax / rowmax)) {
                        kp = k;
                    } else if (abs(W[wOff + (k + 1) * ldw + imax]) >= ALPHA * rowmax) {
                        kp = imax;
                        for (int i = k; i < n; i++) {
                            W[wOff + k * ldw + i] = W[wOff + (k + 1) * ldw + i];
                        }
                    } else {
                        kp = imax;
                        kstep = 2;
                    }
                }
                
                int kk = k + kstep - 1;
                
                if (kp != kk) {
                    if (kp < n - 1) {
                        for (int i = kp + 1; i < n; i++) {
                            double temp = A[aOff + i * lda + kk];
                            A[aOff + i * lda + kk] = A[aOff + i * lda + kp];
                            A[aOff + i * lda + kp] = temp;
                        }
                    }
                    
                    for (int j = kk + 1; j < kp; j++) {
                        double temp = A[aOff + kp * lda + j];
                        A[aOff + kp * lda + j] = A[aOff + j * lda + kk];
                        A[aOff + j * lda + kk] = temp;
                    }
                    
                    double temp = A[aOff + kk * lda + kk];
                    A[aOff + kk * lda + kk] = A[aOff + kp * lda + kp];
                    A[aOff + kp * lda + kp] = temp;
                    
                    if (kstep == 2) {
                        temp = A[aOff + (k + 1) * lda + k];
                        A[aOff + (k + 1) * lda + k] = A[aOff + kp * lda + k];
                        A[aOff + kp * lda + k] = temp;
                    }
                    
                    if (k > 0) {
                        for (int j = 0; j < k; j++) {
                            temp = A[aOff + kk * lda + j];
                            A[aOff + kk * lda + j] = A[aOff + kp * lda + j];
                            A[aOff + kp * lda + j] = temp;
                        }
                    }
                    
                    for (int j = 0; j <= kk; j++) {
                        temp = W[wOff + j * ldw + kk];
                        W[wOff + j * ldw + kk] = W[wOff + j * ldw + kp];
                        W[wOff + j * ldw + kp] = temp;
                    }
                }
                
                if (kstep == 1) {
                    for (int i = k; i < n; i++) {
                        A[aOff + i * lda + k] = W[wOff + k * ldw + i];
                    }
                    
                    if (k < n - 1) {
                        double r1 = 1.0 / A[aOff + k * lda + k];
                        for (int i = k + 1; i < n; i++) {
                            A[aOff + i * lda + k] *= r1;
                        }
                    }
                    
                    ipiv[ipivOff + k] = kp + 1;
                } else {
                    if (k < n - 2) {
                        double d21 = W[wOff + (k + 1) * ldw + k];
                        double d11 = W[wOff + (k + 1) * ldw + (k + 1)] / d21;
                        double d22 = W[wOff + k * ldw + k] / d21;
                        double t = 1.0 / (d11 * d22 - 1.0);
                        d21 = t / d21;
                        
                        for (int j = k + 2; j < n; j++) {
                            A[aOff + j * lda + k] = d21 * (d11 * W[wOff + k * ldw + j] - W[wOff + (k + 1) * ldw + j]);
                            A[aOff + j * lda + (k + 1)] = d21 * (d22 * W[wOff + (k + 1) * ldw + j] - W[wOff + k * ldw + j]);
                        }
                    }
                    
                    A[aOff + k * lda + k] = W[wOff + k * ldw + k];
                    A[aOff + (k + 1) * lda + k] = W[wOff + (k + 1) * ldw + k];
                    A[aOff + (k + 1) * lda + (k + 1)] = W[wOff + (k + 1) * ldw + (k + 1)];
                    
                    ipiv[ipivOff + k] = -(kp + 1);
                    ipiv[ipivOff + (k + 1)] = -(kp + 1);
                }
                
                k += kstep;
            }
            
            kb = k;
            
            if (kb > 0 && k < n) {
                for (int j = k; j < n; j++) {
                    for (int i = j; i < n; i++) {
                        double sum = 0;
                        for (int l = 0; l < kb; l++) {
                            sum += A[aOff + i * lda + l] * W[wOff + l * ldw + j];
                        }
                        A[aOff + i * lda + j] -= sum;
                    }
                }
                
                int j = k - 1;
                while (j >= 0) {
                    int jj = j;
                    int jp = ipiv[ipivOff + j];
                    if (jp < 0) {
                        jp = -jp;
                        j--;
                    }
                    j--;
                    if (jp != jj + 1 && j >= 0) {
                        jp--;
                        for (int l = 0; l <= j; l++) {
                            double temp = A[aOff + jp * lda + l];
                            A[aOff + jp * lda + l] = A[aOff + jj * lda + l];
                            A[aOff + jj * lda + l] = temp;
                        }
                    }
                }
            }
        }
        
        return kb;
    }
}
