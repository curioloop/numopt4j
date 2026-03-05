/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * LAPACK DSYTRF: Computes the factorization of a real symmetric matrix A
 * using the Bunch-Kaufman diagonal pivoting method.
 *
 * <p>The factorization has the form:</p>
 * <pre>
 *   A = U * D * U^T  (if uplo = Upper)
 *   A = L * D * L^T  (if uplo = Lower)
 * </pre>
 * <p>where U (or L) is a product of permutation and unit upper (lower) triangular
 * matrices, and D is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal
 * blocks.</p>
 *
 * <p>Uses a blocked algorithm that calls {@code Dlasyf} for large blocks and
 * falls back to the unblocked {@code dsytf2} for small blocks or when workspace
 * is insufficient.</p>
 *
 * <p>Reference: Fortran netlib LAPACK DSYTRF / DSYTF2</p>
 */
interface Dsytrf {

    double ALPHA = (1.0 + sqrt(17.0)) / 8.0;
    double SAFMIN = Dlamch.dlamch('S');

    static int dsytrf(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda, 
                      int[] ipiv, int ipivOff, double[] work, int lwork) {
        boolean upper = uplo == BLAS.Uplo.Upper;
        
        if (n < 0) {
            return -2;
        }
        if (lda < max(1, n)) {
            return -4;
        }

        int nb = Ilaenv.ilaenv(1, "DSYTRF", String.valueOf(uplo.code), n, -1, -1, -1);
        int lwkopt = max(1, n * nb);
        
        if (n == 0) {
            work[0] = lwkopt;
            return 0;
        }

        boolean lquery = lwork == -1;
        if (lquery) {
            work[0] = lwkopt;
            return 0;
        }

        if (lwork < 1) {
            return -7;
        }
        
        work[0] = lwkopt;

        int nbmin = 2;
        int iws = 1;
        
        if (nb > 1 && nb < n) {
            iws = n * nb;
            if (lwork < iws) {
                nb = max(lwork / n, 1);
                nbmin = max(2, Ilaenv.ilaenv(2, "DSYTRF", String.valueOf(uplo.code), n, -1, -1, -1));
            }
        } else {
            iws = 1;
        }
        
        if (nb < nbmin) {
            nb = n;
        }

        int info = 0;
        
        if (upper) {
            int k = n;
            while (k >= 1) {
                int kb;
                
                if (k > nb) {
                    kb = Dlasyf.dlasyf(uplo, k, nb, A, aOff, lda, ipiv, ipivOff, work, 0, n);
                    if (kb < 1) {
                        kb = k;
                        if (!dsytf2(uplo, k, A, aOff, lda, ipiv, ipivOff, work)) {
                            if (info == 0) {
                                info = k;
                            }
                        }
                    }
                } else {
                    kb = k;
                    if (!dsytf2(uplo, k, A, aOff, lda, ipiv, ipivOff, work)) {
                        if (info == 0) {
                            info = k;
                        }
                    }
                }
                
                k -= kb;
            }
        } else {
            int k = 1;
            while (k <= n) {
                int kb;
                
                if (k <= n - nb) {
                    kb = Dlasyf.dlasyf(uplo, n - k + 1, nb, A, aOff + (k - 1) * lda + (k - 1), lda, 
                                       ipiv, ipivOff + k - 1, work, 0, n);
                    if (kb < 1) {
                        kb = n - k + 1;
                        if (!dsytf2(uplo, n - k + 1, A, aOff + (k - 1) * lda + (k - 1), lda, 
                                    ipiv, ipivOff + k - 1, work)) {
                            if (info == 0) {
                                info = k;
                            }
                        }
                    }
                } else {
                    kb = n - k + 1;
                    if (!dsytf2(uplo, n - k + 1, A, aOff + (k - 1) * lda + (k - 1), lda, 
                                ipiv, ipivOff + k - 1, work)) {
                        if (info == 0) {
                            info = k;
                        }
                    }
                }
                
                for (int j = k; j < k + kb; j++) {
                    int piv = ipiv[ipivOff + j - 1];
                    if (piv > 0) {
                        ipiv[ipivOff + j - 1] = piv + k - 1;
                    } else {
                        ipiv[ipivOff + j - 1] = piv - k + 1;
                    }
                }
                
                k += kb;
            }
        }
        
        work[0] = lwkopt;
        return info;
    }

    static boolean dsytf2(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda, int[] ipiv, int ipivOff, double[] work) {
        if (n < 0 || lda < max(1, n)) {
            return false;
        }
        if (n == 0) {
            return true;
        }

        boolean lower = uplo == BLAS.Uplo.Lower;
        return lower ? dsytf2Lower(n, A, aOff, lda, ipiv, ipivOff, work) 
                     : dsytf2Upper(n, A, aOff, lda, ipiv, ipivOff, work);
    }

    static boolean dsytf2Lower(int n, double[] A, int aOff, int lda, int[] ipiv, int ipivOff, double[] work) {
        int k = 0;
        while (k < n) {
            int kk = aOff + k * lda + k;
            double absakk = abs(A[kk]);

            double colmax = 0;
            int imax = k;
            for (int i = k + 1; i < n; i++) {
                double absAik = abs(A[aOff + i * lda + k]);
                if (absAik > colmax) {
                    colmax = absAik;
                    imax = i;
                }
            }

            if (colmax == 0 && absakk == 0) {
                ipiv[ipivOff + k] = k + 1;
                k++;
                continue;
            }

            int kp;
            int kstep = 1;
            
            if (absakk >= ALPHA * colmax) {
                kp = k;
            } else {
                double rowmax = 0;
                int jmax = k;
                for (int j = k; j < imax; j++) {
                    double absAij = abs(A[aOff + imax * lda + j]);
                    if (absAij > rowmax) {
                        rowmax = absAij;
                        jmax = j;
                    }
                }
                for (int i = imax + 1; i < n; i++) {
                    double absAij = abs(A[aOff + i * lda + imax]);
                    if (absAij > rowmax) {
                        rowmax = absAij;
                        jmax = i;
                    }
                }

                if (absakk >= ALPHA * colmax * (colmax / rowmax)) {
                    kp = k;
                } else if (abs(A[aOff + imax * lda + imax]) >= ALPHA * rowmax) {
                    kp = imax;
                } else {
                    kp = imax;
                    kstep = 2;
                }
            }

            int kkStep = k + kstep - 1;
            if (kp != kkStep) {
                if (kp < n - 1) {
                    BLAS.dswap(n - kp - 1, A, aOff + (kp + 1) * lda + kkStep, lda, A, aOff + (kp + 1) * lda + kp, lda);
                }
                if (kp - kkStep > 1) {
                    BLAS.dswap(kp - kkStep - 1, A, aOff + (kkStep + 1) * lda + kkStep, lda, A, aOff + kp * lda + kkStep + 1, 1);
                }
                
                double temp = A[aOff + kkStep * lda + kkStep];
                A[aOff + kkStep * lda + kkStep] = A[aOff + kp * lda + kp];
                A[aOff + kp * lda + kp] = temp;
                
                if (kstep == 2) {
                    temp = A[aOff + (k + 1) * lda + k];
                    A[aOff + (k + 1) * lda + k] = A[aOff + kp * lda + k];
                    A[aOff + kp * lda + k] = temp;
                }
            }

            if (kstep == 1) {
                if (k < n - 1 && abs(A[kk]) >= SAFMIN) {
                    double d11 = 1.0 / A[kk];
                    
                    for (int j = k + 1; j < n; j++) {
                        double temp = A[aOff + j * lda + k] * d11;
                        for (int i = j; i < n; i++) {
                            A[aOff + i * lda + j] -= temp * A[aOff + i * lda + k];
                        }
                    }
                    
                    for (int i = k + 1; i < n; i++) {
                        A[aOff + i * lda + k] *= d11;
                    }
                }
                ipiv[ipivOff + k] = kp + 1;
                k++;
            } else {
                if (k < n - 1) {
                    double d21 = A[aOff + (k + 1) * lda + k];
                    double d11 = A[aOff + (k + 1) * lda + k + 1] / d21;
                    double d22 = A[kk] / d21;
                    double t = 1.0 / (d11 * d22 - 1.0);
                    d21 = t / d21;

                    for (int j = k + 2; j < n; j++) {
                        double wk = d21 * (d11 * A[aOff + j * lda + k] - A[aOff + j * lda + k + 1]);
                        double wkp1 = d21 * (d22 * A[aOff + j * lda + k + 1] - A[aOff + j * lda + k]);
                        
                        for (int i = j; i < n; i++) {
                            A[aOff + i * lda + j] -= A[aOff + i * lda + k] * wk + A[aOff + i * lda + k + 1] * wkp1;
                        }
                        
                        A[aOff + j * lda + k] = wk;
                        A[aOff + j * lda + k + 1] = wkp1;
                    }
                }
                ipiv[ipivOff + k] = -(kp + 1);
                ipiv[ipivOff + k + 1] = -(kp + 1);
                k += 2;
            }
        }
        return true;
    }

    static boolean dsytf2Upper(int n, double[] A, int aOff, int lda, int[] ipiv, int ipivOff, double[] work) {
        int k = n - 1;
        while (k >= 0) {
            int kk = aOff + k * lda + k;
            double absakk = abs(A[kk]);

            double colmax = 0;
            int imax = k;
            if (k > 0) {
                for (int i = 0; i < k; i++) {
                    double absAik = abs(A[aOff + i * lda + k]);
                    if (absAik > colmax) {
                        colmax = absAik;
                        imax = i;
                    }
                }
            }

            if (colmax == 0 && absakk == 0) {
                ipiv[ipivOff + k] = k + 1;
                k--;
                continue;
            }

            int kp;
            int kstep = 1;
            
            if (absakk >= ALPHA * colmax) {
                kp = k;
            } else {
                double rowmax = 0;
                int jmax = imax;
                for (int j = imax + 1; j <= k; j++) {
                    double absAij = abs(A[aOff + imax * lda + j]);
                    if (absAij > rowmax) {
                        rowmax = absAij;
                        jmax = j;
                    }
                }
                if (imax > 0) {
                    for (int j = 0; j < imax; j++) {
                        double absAij = abs(A[aOff + j * lda + imax]);
                        if (absAij > rowmax) {
                            rowmax = absAij;
                            jmax = j;
                        }
                    }
                }

                if (absakk >= ALPHA * colmax * (colmax / rowmax)) {
                    kp = k;
                } else if (abs(A[aOff + imax * lda + imax]) >= ALPHA * rowmax) {
                    kp = imax;
                } else {
                    kp = imax;
                    kstep = 2;
                }
            }

            int kkStep = k - kstep + 1;
            if (kp != kkStep) {
                for (int i = 0; i < kp; i++) {
                    double temp = A[aOff + i * lda + kkStep];
                    A[aOff + i * lda + kkStep] = A[aOff + i * lda + kp];
                    A[aOff + i * lda + kp] = temp;
                }
                if (kkStep - kp > 1) {
                    for (int j = kp + 1; j < kkStep; j++) {
                        double temp = A[aOff + kp * lda + j];
                        A[aOff + kp * lda + j] = A[aOff + j * lda + kkStep];
                        A[aOff + j * lda + kkStep] = temp;
                    }
                }
                
                double temp = A[aOff + kkStep * lda + kkStep];
                A[aOff + kkStep * lda + kkStep] = A[aOff + kp * lda + kp];
                A[aOff + kp * lda + kp] = temp;
                
                if (kstep == 2) {
                    temp = A[aOff + (k - 1) * lda + k];
                    A[aOff + (k - 1) * lda + k] = A[aOff + kp * lda + k];
                    A[aOff + kp * lda + k] = temp;
                }
            }

            if (kstep == 1) {
                if (k > 0 && abs(A[kk]) >= SAFMIN) {
                    double r1 = 1.0 / A[kk];
                    
                    for (int i = 0; i < k; i++) {
                        work[i] = A[aOff + i * lda + k];
                    }
                    
                    for (int j = 0; j < k; j++) {
                        double temp = r1 * work[j];
                        for (int i = 0; i <= j; i++) {
                            A[aOff + i * lda + j] -= temp * work[i];
                        }
                    }
                    
                    for (int i = 0; i < k; i++) {
                        A[aOff + i * lda + k] = r1 * work[i];
                    }
                }
                ipiv[ipivOff + k] = kp + 1;
                k--;
            } else {
                if (k > 1) {
                    double d12 = A[aOff + (k - 1) * lda + k];
                    double d22 = A[aOff + (k - 1) * lda + k - 1] / d12;
                    double d11 = A[kk] / d12;
                    double t = 1.0 / (d11 * d22 - 1.0);
                    d12 = t / d12;

                    for (int j = k - 2; j >= 0; j--) {
                        double wkm1 = d12 * (d11 * A[aOff + j * lda + k - 1] - A[aOff + j * lda + k]);
                        double wk = d12 * (d22 * A[aOff + j * lda + k] - A[aOff + j * lda + k - 1]);
                        
                        for (int i = j; i >= 0; i--) {
                            A[aOff + i * lda + j] -= A[aOff + i * lda + k] * wk + A[aOff + i * lda + k - 1] * wkm1;
                        }
                        
                        A[aOff + j * lda + k] = wk;
                        A[aOff + j * lda + k - 1] = wkm1;
                    }
                }
                ipiv[ipivOff + k] = -(kp + 1);
                ipiv[ipivOff + k - 1] = -(kp + 1);
                k -= 2;
            }
        }
        return true;
    }
}
