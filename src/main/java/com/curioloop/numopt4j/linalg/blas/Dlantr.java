/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.abs;
import static java.lang.Math.sqrt;

public interface Dlantr {

    static double dlantr(char norm, BLAS.Uplo uplo, BLAS.Diag diag, int m, int n,
                         double[] A, int lda, double[] work) {
        if (m == 0 || n == 0) return 0.0;

        boolean upper = uplo == BLAS.Uplo.Upper;
        boolean nounit = diag == BLAS.Diag.NonUnit;
        int mn = m < n ? m : n;

        switch (Character.toUpperCase(norm)) {
            case '1':
            case 'O':
                return norm1(upper, nounit, m, n, mn, A, lda, work);
            case 'I':
                return normInf(upper, nounit, m, n, mn, A, lda, work);
            case 'F':
            case 'E':
                return normFro(upper, nounit, m, n, mn, A, lda);
            case 'M':
                return normMax(upper, nounit, m, n, mn, A, lda);
            default:
                throw new IllegalArgumentException("Unknown norm type: " + norm);
        }
    }

    static double norm1(boolean upper, boolean nounit, int m, int n, int mn, double[] A, int lda, double[] work) {
        // Use work array to accumulate column sums (matches Go's MaxColumnSum approach)
        if (work == null || work.length < n) work = new double[n];

        if (!nounit) {
            // Unit diagonal: initialize diagonal columns to 1, rest to 0
            for (int i = 0; i < mn; i++) work[i] = 1.0;
            for (int i = mn; i < n; i++) work[i] = 0.0;
            if (upper) {
                for (int i = 0; i < m; i++) {
                    for (int j = i + 1; j < n; j++) work[j] += abs(A[i * lda + j]);
                }
            } else {
                for (int i = 1; i < m; i++) {
                    for (int j = 0; j < (i < n ? i : n); j++) work[j] += abs(A[i * lda + j]);
                }
            }
        } else {
            for (int i = 0; i < n; i++) work[i] = 0.0;
            if (upper) {
                for (int i = 0; i < m; i++) {
                    for (int j = i; j < n; j++) work[j] += abs(A[i * lda + j]);
                }
            } else {
                for (int i = 0; i < m; i++) {
                    int jEnd = (i < n - 1) ? i + 1 : n;
                    for (int j = 0; j < jEnd; j++) work[j] += abs(A[i * lda + j]);
                }
            }
        }

        double maxColSum = 0.0;
        for (int j = 0; j < n; j++) {
            if (work[j] > maxColSum) maxColSum = work[j];
        }
        return maxColSum;
    }

    static double normInf(boolean upper, boolean nounit, int m, int n, int mn,
                          double[] A, int lda, double[] work) {
        double maxRowSum = 0.0;
        if (!nounit) {
            // Unit diagonal: each row with a diagonal element starts at 1
            if (upper) {
                for (int i = 0; i < m; i++) {
                    double sum = (i < mn) ? 1.0 : 0.0;
                    for (int j = i + 1; j < n; j++) sum += abs(A[i * lda + j]);
                    if (sum > maxRowSum) maxRowSum = sum;
                }
            } else {
                for (int i = 0; i < m; i++) {
                    double sum = (i < mn) ? 1.0 : 0.0;
                    for (int j = 0; j < (i < n ? i : n); j++) sum += abs(A[i * lda + j]);
                    if (sum > maxRowSum) maxRowSum = sum;
                }
            }
        } else {
            if (upper) {
                for (int i = 0; i < m; i++) {
                    double sum = 0.0;
                    for (int j = i; j < n; j++) sum += abs(A[i * lda + j]);
                    if (sum > maxRowSum) maxRowSum = sum;
                }
            } else {
                for (int i = 0; i < m; i++) {
                    double sum = 0.0;
                    int jEnd = (i < n - 1) ? i + 1 : n;
                    for (int j = 0; j < jEnd; j++) sum += abs(A[i * lda + j]);
                    if (sum > maxRowSum) maxRowSum = sum;
                }
            }
        }
        return maxRowSum;
    }

    static double normFro(boolean upper, boolean nounit, int m, int n, int mn, double[] A, int lda) {
        double scale, ssq;
        if (!nounit) {
            // Unit diagonal: start with scale=1, sum=min(m,n) to account for diagonal 1s
            scale = 1.0;
            ssq = mn;
            if (upper) {
                for (int i = 0; i < mn; i++) {
                    for (int j = i + 1; j < n; j++) {
                        double absVal = abs(A[i * lda + j]);
                        if (absVal != 0.0) {
                            if (scale < absVal) { double r = scale / absVal; ssq = 1.0 + ssq * r * r; scale = absVal; }
                            else { double r = absVal / scale; ssq += r * r; }
                        }
                    }
                }
            } else {
                for (int i = 1; i < m; i++) {
                    for (int j = 0; j < (i < n ? i : n); j++) {
                        double absVal = abs(A[i * lda + j]);
                        if (absVal != 0.0) {
                            if (scale < absVal) { double r = scale / absVal; ssq = 1.0 + ssq * r * r; scale = absVal; }
                            else { double r = absVal / scale; ssq += r * r; }
                        }
                    }
                }
            }
        } else {
            scale = 0.0;
            ssq = 1.0;
            if (upper) {
                for (int j = 0; j < mn; j++) {
                    for (int i = 0; i <= j; i++) {
                        double absVal = abs(A[i * lda + j]);
                        if (absVal != 0.0) {
                            if (scale < absVal) { double r = scale / absVal; ssq = 1.0 + ssq * r * r; scale = absVal; }
                            else { double r = absVal / scale; ssq += r * r; }
                        }
                    }
                }
                for (int j = mn; j < n; j++) {
                    for (int i = 0; i < mn; i++) {
                        double absVal = abs(A[i * lda + j]);
                        if (absVal != 0.0) {
                            if (scale < absVal) { double r = scale / absVal; ssq = 1.0 + ssq * r * r; scale = absVal; }
                            else { double r = absVal / scale; ssq += r * r; }
                        }
                    }
                }
            } else {
                for (int j = 0; j < mn; j++) {
                    for (int i = j; i < m; i++) {
                        double absVal = abs(A[i * lda + j]);
                        if (absVal != 0.0) {
                            if (scale < absVal) { double r = scale / absVal; ssq = 1.0 + ssq * r * r; scale = absVal; }
                            else { double r = absVal / scale; ssq += r * r; }
                        }
                    }
                }
            }
        }
        return scale * sqrt(ssq);
    }

    static double normMax(boolean upper, boolean nounit, int m, int n, int mn, double[] A, int lda) {
        if (!nounit) {
            // Unit diagonal: max is at least 1 (if any diagonal exists)
            double maxVal = (mn > 0) ? 1.0 : 0.0;
            if (upper) {
                for (int i = 0; i < m; i++) {
                    for (int j = i + 1; j < n; j++) {
                        double absVal = abs(A[i * lda + j]);
                        if (absVal > maxVal) maxVal = absVal;
                    }
                }
            } else {
                for (int i = 1; i < m; i++) {
                    for (int j = 0; j < (i < n ? i : n); j++) {
                        double absVal = abs(A[i * lda + j]);
                        if (absVal > maxVal) maxVal = absVal;
                    }
                }
            }
            return maxVal;
        }
        double maxVal = 0.0;
        if (upper) {
            for (int j = 0; j < mn; j++) {
                for (int i = 0; i <= j; i++) {
                    double absVal = abs(A[i * lda + j]);
                    if (absVal > maxVal) maxVal = absVal;
                }
            }
            for (int j = mn; j < n; j++) {
                for (int i = 0; i < mn; i++) {
                    double absVal = abs(A[i * lda + j]);
                    if (absVal > maxVal) maxVal = absVal;
                }
            }
        } else {
            for (int j = 0; j < mn; j++) {
                for (int i = j; i < m; i++) {
                    double absVal = abs(A[i * lda + j]);
                    if (absVal > maxVal) maxVal = absVal;
                }
            }
        }
        return maxVal;
    }
}
