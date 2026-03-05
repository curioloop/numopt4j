/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DLASR: Applies a sequence of plane rotations to a matrix.
 * Based on gonum/lapack/gonum/dlasr.go
 * 
 * <p>Note: This implementation uses row-major storage convention:
 * a[row*lda+col] = A[row][col]</p>
 */
interface Dlasr {
    
    static void dlasr(BLAS.Side side, char pivot, char direct, int m, int n,
                      double[] c, double[] s, double[] a, int lda) {
        dlasr(side, pivot, direct, m, n, c, 0, s, 0, a, 0, lda);
    }
    
    static void dlasr(BLAS.Side side, char pivot, char direct, int m, int n,
                      double[] c, int cOff, double[] s, int sOff, 
                      double[] a, int aOff, int lda) {
        if (m == 0 || n == 0) {
            return;
        }
        
        char pivotU = Character.toUpperCase(pivot);
        char directU = Character.toUpperCase(direct);
        
        if (side == BLAS.Side.Left) {
            // Left side transformation: A = P * A
            // In row-major, this means rotating rows
            if (pivotU == 'V') {
                // Variable pivot - rotation in (j, j+1) rows
                if (directU == 'F') {
                    for (int j = 0; j < m - 1; j++) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                double tmp2 = a[aOff + j * lda + i];
                                double tmp = a[aOff + (j + 1) * lda + i];
                                a[aOff + (j + 1) * lda + i] = ctmp * tmp - stmp * tmp2;
                                a[aOff + j * lda + i] = stmp * tmp + ctmp * tmp2;
                            }
                        }
                    }
                } else {
                    for (int j = m - 2; j >= 0; j--) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                double tmp2 = a[aOff + j * lda + i];
                                double tmp = a[aOff + (j + 1) * lda + i];
                                a[aOff + (j + 1) * lda + i] = ctmp * tmp - stmp * tmp2;
                                a[aOff + j * lda + i] = stmp * tmp + ctmp * tmp2;
                            }
                        }
                    }
                }
            } else if (pivotU == 'T') {
                // Top pivot - rotation in (0, j) rows
                if (directU == 'F') {
                    for (int j = 1; j < m; j++) {
                        double ctmp = c[cOff + j - 1];
                        double stmp = s[sOff + j - 1];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                double tmp = a[aOff + j * lda + i];
                                double tmp2 = a[aOff + i];
                                a[aOff + j * lda + i] = ctmp * tmp - stmp * tmp2;
                                a[aOff + i] = stmp * tmp + ctmp * tmp2;
                            }
                        }
                    }
                } else {
                    for (int j = m - 1; j >= 1; j--) {
                        double ctmp = c[cOff + j - 1];
                        double stmp = s[sOff + j - 1];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                double tmp = a[aOff + j * lda + i];
                                double tmp2 = a[aOff + i];
                                a[aOff + j * lda + i] = ctmp * tmp - stmp * tmp2;
                                a[aOff + i] = stmp * tmp + ctmp * tmp2;
                            }
                        }
                    }
                }
            } else if (pivotU == 'B') {
                // Bottom pivot - rotation in (j, m-1) rows
                if (directU == 'F') {
                    for (int j = 0; j < m - 1; j++) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                double tmp = a[aOff + j * lda + i];
                                double tmp2 = a[aOff + (m - 1) * lda + i];
                                a[aOff + j * lda + i] = stmp * tmp2 + ctmp * tmp;
                                a[aOff + (m - 1) * lda + i] = ctmp * tmp2 - stmp * tmp;
                            }
                        }
                    }
                } else {
                    for (int j = m - 2; j >= 0; j--) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < n; i++) {
                                double tmp = a[aOff + j * lda + i];
                                double tmp2 = a[aOff + (m - 1) * lda + i];
                                a[aOff + j * lda + i] = stmp * tmp2 + ctmp * tmp;
                                a[aOff + (m - 1) * lda + i] = ctmp * tmp2 - stmp * tmp;
                            }
                        }
                    }
                }
            }
        } else {
            // Right side transformation: A = A * P^T
            // In row-major, this means rotating columns
            if (pivotU == 'V') {
                // Variable pivot - rotation in (j, j+1) columns
                if (directU == 'F') {
                    for (int j = 0; j < n - 1; j++) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                double tmp = a[aOff + i * lda + j + 1];
                                double tmp2 = a[aOff + i * lda + j];
                                a[aOff + i * lda + j + 1] = ctmp * tmp - stmp * tmp2;
                                a[aOff + i * lda + j] = stmp * tmp + ctmp * tmp2;
                            }
                        }
                    }
                } else {
                    for (int j = n - 2; j >= 0; j--) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                double tmp = a[aOff + i * lda + j + 1];
                                double tmp2 = a[aOff + i * lda + j];
                                a[aOff + i * lda + j + 1] = ctmp * tmp - stmp * tmp2;
                                a[aOff + i * lda + j] = stmp * tmp + ctmp * tmp2;
                            }
                        }
                    }
                }
            } else if (pivotU == 'T') {
                // Top pivot - rotation in (0, j) columns
                if (directU == 'F') {
                    for (int j = 1; j < n; j++) {
                        double ctmp = c[cOff + j - 1];
                        double stmp = s[sOff + j - 1];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                double tmp = a[aOff + i * lda + j];
                                double tmp2 = a[aOff + i * lda];
                                a[aOff + i * lda + j] = ctmp * tmp - stmp * tmp2;
                                a[aOff + i * lda] = stmp * tmp + ctmp * tmp2;
                            }
                        }
                    }
                } else {
                    for (int j = n - 1; j >= 1; j--) {
                        double ctmp = c[cOff + j - 1];
                        double stmp = s[sOff + j - 1];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                double tmp = a[aOff + i * lda + j];
                                double tmp2 = a[aOff + i * lda];
                                a[aOff + i * lda + j] = ctmp * tmp - stmp * tmp2;
                                a[aOff + i * lda] = stmp * tmp + ctmp * tmp2;
                            }
                        }
                    }
                }
            } else if (pivotU == 'B') {
                // Bottom pivot - rotation in (j, n-1) columns
                if (directU == 'F') {
                    for (int j = 0; j < n - 1; j++) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                double tmp = a[aOff + i * lda + j];
                                double tmp2 = a[aOff + i * lda + n - 1];
                                a[aOff + i * lda + j] = stmp * tmp2 + ctmp * tmp;
                                a[aOff + i * lda + n - 1] = ctmp * tmp2 - stmp * tmp;
                            }
                        }
                    }
                } else {
                    for (int j = n - 2; j >= 0; j--) {
                        double ctmp = c[cOff + j];
                        double stmp = s[sOff + j];
                        if (ctmp != 1.0 || stmp != 0.0) {
                            for (int i = 0; i < m; i++) {
                                double tmp = a[aOff + i * lda + j];
                                double tmp2 = a[aOff + i * lda + n - 1];
                                a[aOff + i * lda + j] = stmp * tmp2 + ctmp * tmp;
                                a[aOff + i * lda + n - 1] = ctmp * tmp2 - stmp * tmp;
                            }
                        }
                    }
                }
            }
        }
    }
}
