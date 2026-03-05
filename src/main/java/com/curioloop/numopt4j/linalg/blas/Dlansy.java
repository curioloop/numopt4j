/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * LAPACK DLANSY: Returns the value of the specified norm of a symmetric matrix.
 * 
 * Based on LAPACK, adapted for row-major storage.
 */
public interface Dlansy {

    /**
     * DLANSY returns the value of the specified norm of an n×n symmetric matrix.
     *
     * @param norm  Norm type: 'M' (max abs), '1'/'O' (1-norm/col sum), 
     *              'I' (inf norm/row sum), 'F'/'E' (Frobenius)
     * @param uplo  'U' for upper triangular, 'L' for lower triangular
     * @param n     order of matrix A
     * @param A     symmetric matrix (n x n, row-major)
     * @param lda   leading dimension of A
     * @param work  workspace (required for 1-norm or inf-norm)
     * @return the computed norm value
     */
    static double dlansy(char norm, BLAS.Uplo uplo, int n, double[] A, int lda, double[] work) {
        if (n == 0) {
            return 0.0;
        }

        char uploChar = uplo.code;
        switch (Character.toUpperCase(norm)) {
            case 'M':
                return normMax(uploChar, n, A, lda);
            case '1':
            case 'O':
                return norm1(uploChar, n, A, lda, work);
            case 'I':
                return normInf(uploChar, n, A, lda, work);
            case 'F':
            case 'E':
                return normFro(uploChar, n, A, lda);
            default:
                throw new IllegalArgumentException("Unknown norm type: " + norm);
        }
    }

    /**
     * Max norm: largest absolute value.
     * ||A||_max = max_ij |A[i,j]|
     */
    static double normMax(char uplo, int n, double[] A, int lda) {
        boolean upper = Character.toUpperCase(uplo) == 'U';
        double max = 0.0;

        if (upper) {
            for (int i = 0; i < n; i++) {
                for (int j = i; j < n; j++) {
                    double v = Math.abs(A[i * lda + j]);
                    if (Double.isNaN(v)) {
                        return Double.NaN;
                    }
                    if (v > max) {
                        max = v;
                    }
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j <= i; j++) {
                    double v = Math.abs(A[i * lda + j]);
                    if (Double.isNaN(v)) {
                        return Double.NaN;
                    }
                    if (v > max) {
                        max = v;
                    }
                }
            }
        }
        return max;
    }

    /**
     * 1-norm: maximum absolute column sum.
     * For symmetric matrix, 1-norm = infinity-norm.
     * ||A||₁ = max_j Σᵢ |A[i,j]|
     */
    static double norm1(char uplo, int n, double[] A, int lda, double[] work) {
        boolean upper = Character.toUpperCase(uplo) == 'U';
        
        // Initialize work array to zero
        for (int i = 0; i < n; i++) {
            work[i] = 0.0;
        }

        if (upper) {
            for (int i = 0; i < n; i++) {
                work[i] += Math.abs(A[i * lda + i]); // diagonal
                for (int j = i + 1; j < n; j++) {
                    double v = Math.abs(A[i * lda + j]);
                    work[i] += v;
                    work[j] += v;
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < i; j++) {
                    double v = Math.abs(A[i * lda + j]);
                    work[i] += v;
                    work[j] += v;
                }
                work[i] += Math.abs(A[i * lda + i]); // diagonal
            }
        }

        double max = 0.0;
        for (int i = 0; i < n; i++) {
            double v = work[i];
            if (Double.isNaN(v)) {
                return Double.NaN;
            }
            if (v > max) {
                max = v;
            }
        }
        return max;
    }

    /**
     * Infinity-norm: maximum absolute row sum.
     * For symmetric matrix, inf-norm = 1-norm.
     * ||A||∞ = max_i Σⱼ |A[i,j]|
     */
    static double normInf(char uplo, int n, double[] A, int lda, double[] work) {
        // For symmetric matrix, 1-norm = infinity-norm
        return norm1(uplo, n, A, lda, work);
    }

    /**
     * Frobenius norm: sqrt of sum of squares.
     * ||A||_F = √(Σᵢⱼ |A[i,j]|²)
     */
    static double normFro(char uplo, int n, double[] A, int lda) {
        double[] ssq = {0.0, 1.0};

        boolean upper = Character.toUpperCase(uplo) == 'U';

        if (upper) {
            for (int i = 0; i < n - 1; i++) {
                dlassq(n - i - 1, A, i * lda + i + 1, 1, ssq);
            }
        } else {
            for (int i = 1; i < n; i++) {
                dlassq(i, A, i * lda, 1, ssq);
            }
        }
        ssq[1] *= 2;

        dlassq(n, A, 0, lda + 1, ssq);

        return ssq[0] * Math.sqrt(ssq[1]);
    }

    /**
     * DLASSQ updates a sum of squares represented in scaled form.
     * ssq[0] = scale, ssq[1] = sumsq on input and output.
     * After call: scale^2 * sumsq = sum(x_i^2) + original_scale^2 * original_sumsq
     */
    static void dlassq(int n, double[] x, int off, int incx, double[] ssq) {
        if (n <= 0) {
            return;
        }

        double scale = ssq[0];
        double sumsq = ssq[1];

        if (Double.isNaN(scale) || Double.isNaN(sumsq)) {
            ssq[0] = Double.NaN;
            ssq[1] = Double.NaN;
            return;
        }

        if (sumsq == 0.0) {
            scale = 1.0;
        }
        if (scale == 0.0) {
            scale = 1.0;
            sumsq = 0.0;
        }

        for (int i = 0; i < n; i++) {
            double xi = x[off + i * incx];
            if (xi != 0.0) {
                double absxi = Math.abs(xi);
                if (scale < absxi) {
                    double ratio = scale / absxi;
                    sumsq = 1.0 + sumsq * ratio * ratio;
                    scale = absxi;
                } else {
                    double ratio = absxi / scale;
                    sumsq += ratio * ratio;
                }
            }
        }

        ssq[0] = scale;
        ssq[1] = sumsq;
    }
}
