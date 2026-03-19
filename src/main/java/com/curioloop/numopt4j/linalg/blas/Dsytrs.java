/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * LAPACK DSYTRS: Solves a system of linear equations A * X = B with a real
 * symmetric matrix A using the factorization A = U * D * U^T or A = L * D * L^T
 * computed by {@link Dsytrf}.
 *
 * <p>The factorization uses Bunch-Kaufman diagonal pivoting. The solution is
 * computed by applying the inverse of the factored form to B.</p>
 *
 * <p>Reference: Fortran netlib LAPACK DSYTRS</p>
 */
interface Dsytrs {

    static void dsytrs(BLAS.Uplo uplo, int n, int nrhs, double[] A, int aOff, int lda,
                       int[] ipiv, int ipivOff, double[] B, int bOff, int ldb) {
        if (n == 0 || nrhs == 0) return;

        boolean upper = uplo == BLAS.Uplo.Upper;

        if (upper) {
            dsytrsUpper(n, nrhs, A, aOff, lda, ipiv, ipivOff, B, bOff, ldb);
        } else {
            dsytrsLower(n, nrhs, A, aOff, lda, ipiv, ipivOff, B, bOff, ldb);
        }
    }

    static void dsytrsLower(int n, int nrhs, double[] A, int aOff, int lda,
                            int[] ipiv, int ipivOff, double[] B, int bOff, int ldb) {
        for (int j = 0; j < nrhs; j++) {
            int bjOff = bOff + j;

            int k = 0;
            while (k < n) {
                if (ipiv[ipivOff + k] > 0) {
                    int kp = ipiv[ipivOff + k] - 1;
                    if (kp != k) {
                        double temp = B[bjOff + k * ldb];
                        B[bjOff + k * ldb] = B[bjOff + kp * ldb];
                        B[bjOff + kp * ldb] = temp;
                    }

                    if (k < n - 1) {
                        Dgemv.dgemv(BLAS.Trans.NoTrans, n - k - 1, 1, -1.0,
                                A, aOff + (k + 1) * lda + k, lda,
                                B, bjOff + k * ldb, ldb, 1.0,
                                B, bjOff + (k + 1) * ldb, ldb);
                    }

                    double ak = A[aOff + k * lda + k];
                    B[bjOff + k * ldb] /= ak;
                    k++;
                } else {
                    int kp = -ipiv[ipivOff + k] - 1;
                    if (kp != k + 1) {
                        double temp = B[bjOff + (k + 1) * ldb];
                        B[bjOff + (k + 1) * ldb] = B[bjOff + kp * ldb];
                        B[bjOff + kp * ldb] = temp;
                    }

                    if (k < n - 2) {
                        Dgemv.dgemv(BLAS.Trans.NoTrans, n - k - 2, 2, -1.0,
                                A, aOff + (k + 2) * lda + k, lda,
                                B, bjOff + k * ldb, ldb, 1.0,
                                B, bjOff + (k + 2) * ldb, ldb);
                    }

                    double akm1k = A[aOff + (k + 1) * lda + k];
                    double akm1 = A[aOff + k * lda + k] / akm1k;
                    double ak = A[aOff + (k + 1) * lda + k + 1] / akm1k;
                    double denom = akm1 * ak - 1.0;

                    double bkm1 = B[bjOff + k * ldb] / akm1k;
                    double bk = B[bjOff + (k + 1) * ldb] / akm1k;
                    B[bjOff + k * ldb] = (ak * bkm1 - bk) / denom;
                    B[bjOff + (k + 1) * ldb] = (akm1 * bk - bkm1) / denom;
                    k += 2;
                }
            }

            k = n - 1;
            while (k >= 0) {
                if (ipiv[ipivOff + k] > 0) {
                    if (k < n - 1) {
                        double temp = B[bjOff + k * ldb];
                        Dgemv.dgemv(BLAS.Trans.Trans, n - k - 1, 1, -1.0,
                                A, aOff + (k + 1) * lda + k, lda,
                                B, bjOff + (k + 1) * ldb, ldb, 1.0,
                                B, bjOff + k * ldb, ldb);
                    }

                    int kp = ipiv[ipivOff + k] - 1;
                    if (kp != k) {
                        double temp = B[bjOff + k * ldb];
                        B[bjOff + k * ldb] = B[bjOff + kp * ldb];
                        B[bjOff + kp * ldb] = temp;
                    }
                    k--;
                } else {
                    if (k < n - 1) {
                        Dgemv.dgemv(BLAS.Trans.Trans, n - k - 1, 1, -1.0,
                                A, aOff + (k + 1) * lda + k, lda,
                                B, bjOff + (k + 1) * ldb, ldb, 1.0,
                                B, bjOff + k * ldb, ldb);
                        Dgemv.dgemv(BLAS.Trans.Trans, n - k - 1, 1, -1.0,
                                A, aOff + (k + 1) * lda + k - 1, lda,
                                B, bjOff + (k + 1) * ldb, ldb, 1.0,
                                B, bjOff + (k - 1) * ldb, ldb);
                    }

                    int kp = -ipiv[ipivOff + k] - 1;
                    if (kp != k) {
                        double temp = B[bjOff + k * ldb];
                        B[bjOff + k * ldb] = B[bjOff + kp * ldb];
                        B[bjOff + kp * ldb] = temp;
                    }
                    k -= 2;
                }
            }
        }
    }

    static void dsytrsUpper(int n, int nrhs, double[] A, int aOff, int lda,
                            int[] ipiv, int ipivOff, double[] B, int bOff, int ldb) {
        for (int j = 0; j < nrhs; j++) {
            int bjOff = bOff + j;

            int k = n - 1;
            while (k >= 0) {
                if (ipiv[ipivOff + k] > 0) {
                    int kp = ipiv[ipivOff + k] - 1;
                    if (kp != k) {
                        double temp = B[bjOff + k * ldb];
                        B[bjOff + k * ldb] = B[bjOff + kp * ldb];
                        B[bjOff + kp * ldb] = temp;
                    }

                    if (k > 0) {
                        Dgemv.dgemv(BLAS.Trans.NoTrans, k, 1, -1.0,
                                A, aOff + k, lda,
                                B, bjOff + k * ldb, ldb, 1.0,
                                B, bjOff, ldb);
                    }

                    double ak = A[aOff + k * lda + k];
                    B[bjOff + k * ldb] /= ak;
                    k--;
                } else {
                    int kp = -ipiv[ipivOff + k] - 1;
                    if (kp != k - 1) {
                        double temp = B[bjOff + (k - 1) * ldb];
                        B[bjOff + (k - 1) * ldb] = B[bjOff + kp * ldb];
                        B[bjOff + kp * ldb] = temp;
                    }

                    if (k > 1) {
                        Dgemv.dgemv(BLAS.Trans.NoTrans, k - 1, 2, -1.0,
                                A, aOff + k - 1, lda,
                                B, bjOff + (k - 1) * ldb, ldb, 1.0,
                                B, bjOff, ldb);
                    }

                    double akm1k = A[aOff + (k - 1) * lda + k];
                    double akm1 = A[aOff + (k - 1) * lda + k - 1] / akm1k;
                    double ak = A[aOff + k * lda + k] / akm1k;
                    double denom = akm1 * ak - 1.0;

                    double bkm1 = B[bjOff + (k - 1) * ldb] / akm1k;
                    double bk = B[bjOff + k * ldb] / akm1k;
                    B[bjOff + (k - 1) * ldb] = (ak * bkm1 - bk) / denom;
                    B[bjOff + k * ldb] = (akm1 * bk - bkm1) / denom;
                    k -= 2;
                }
            }

            k = 0;
            while (k < n) {
                if (ipiv[ipivOff + k] > 0) {
                    if (k > 0) {
                        Dgemv.dgemv(BLAS.Trans.Trans, k, 1, -1.0,
                                A, aOff + k, lda,
                                B, bjOff, ldb, 1.0,
                                B, bjOff + k * ldb, ldb);
                    }

                    int kp = ipiv[ipivOff + k] - 1;
                    if (kp != k) {
                        double temp = B[bjOff + k * ldb];
                        B[bjOff + k * ldb] = B[bjOff + kp * ldb];
                        B[bjOff + kp * ldb] = temp;
                    }
                    k++;
                } else {
                    if (k > 0) {
                        Dgemv.dgemv(BLAS.Trans.Trans, k, 1, -1.0,
                                A, aOff + k, lda,
                                B, bjOff, ldb, 1.0,
                                B, bjOff + k * ldb, ldb);
                        Dgemv.dgemv(BLAS.Trans.Trans, k, 1, -1.0,
                                A, aOff + k + 1, lda,
                                B, bjOff, ldb, 1.0,
                                B, bjOff + (k + 1) * ldb, ldb);
                    }

                    int kp = -ipiv[ipivOff + k] - 1;
                    if (kp != k) {
                        double temp = B[bjOff + k * ldb];
                        B[bjOff + k * ldb] = B[bjOff + kp * ldb];
                        B[bjOff + kp * ldb] = temp;
                    }
                    k += 2;
                }
            }
        }
    }
}
