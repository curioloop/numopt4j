/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DLANGE computes matrix norms.
 * 
 * <p>Supported norms:</p>
 * <ul>
 *   <li>'1' or 'O': 1-norm (maximum column sum)</li>
 *   <li>'I': Infinity-norm (maximum row sum)</li>
 *   <li>'F' or 'E': Frobenius norm</li>
 *   <li>'M': Max norm (largest absolute value)</li>
 * </ul>
 * 
 * <h2>Optimization Techniques</h2>
 * <ul>
 *   <li>4-way loop unrolling with multiple accumulators for normInf</li>
 * </ul>
 * 
 * <p>Reference: LAPACK DLANGE</p>
 */
public interface Dlange {

    /**
     * Computes the specified matrix norm.
     * 
     * @param norm Norm type: '1'/'O' (1-norm), 'I' (inf-norm), 'F'/'E' (Frobenius), 'M' (max)
     * @param m number of rows
     * @param n number of columns
     * @param A matrix (row-major, m × n)
     * @param aOff offset into A
     * @param lda leading dimension of A
     * @return the computed norm value
     */
    static double dlange(char norm, int m, int n, double[] A, int aOff, int lda) {
        if (m == 0 || n == 0) return 0.0;

        switch (Character.toUpperCase(norm)) {
            case '1':
            case 'O':
                return norm1(m, n, A, aOff, lda);
            case 'I':
                return normInf(m, n, A, aOff, lda);
            case 'F':
            case 'E':
                return normFro(m, n, A, aOff, lda);
            case 'M':
                return normMax(m, n, A, aOff, lda);
            default:
                throw new IllegalArgumentException("Unknown norm type: " + norm);
        }
    }

    /**
     * 1-norm: maximum absolute column sum.
     * ||A||₁ = max_j Σᵢ |A[i,j]|
     */
    static double norm1(int m, int n, double[] A, int aOff, int lda) {
        double maxColSum = 0.0;
        for (int j = 0; j < n; j++) {
            double colSum = 0.0;
            for (int i = 0; i < m; i++) {
                colSum += Math.abs(A[aOff + i * lda + j]);
            }
            if (colSum > maxColSum) {
                maxColSum = colSum;
            }
        }
        return maxColSum;
    }

    /**
     * Infinity-norm: maximum absolute row sum.
     * ||A||∞ = max_i Σⱼ |A[i,j]|
     */
    static double normInf(int m, int n, double[] A, int aOff, int lda) {
        double maxRowSum = 0.0;
        for (int i = 0; i < m; i++) {
            int rowOff = aOff + i * lda;
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            int j = 0;
            int j4 = (n / 4) * 4;
            for (; j < j4; j += 4) {
                sum0 += Math.abs(A[rowOff + j]);
                sum1 += Math.abs(A[rowOff + j + 1]);
                sum2 += Math.abs(A[rowOff + j + 2]);
                sum3 += Math.abs(A[rowOff + j + 3]);
            }
            double rowSum = sum0 + sum1 + sum2 + sum3;
            for (; j < n; j++) {
                rowSum += Math.abs(A[rowOff + j]);
            }
            if (rowSum > maxRowSum) {
                maxRowSum = rowSum;
            }
        }
        return maxRowSum;
    }

    /**
     * Frobenius norm: sqrt of sum of squares.
     * ||A||_F = √(Σᵢⱼ |A[i,j]|²)
     * 
     * Uses scaled summation to avoid overflow/underflow.
     */
    static double normFro(int m, int n, double[] A, int aOff, int lda) {
        double scale = 0.0;
        double ssq = 1.0;
        
        for (int i = 0; i < m; i++) {
            int rowOff = aOff + i * lda;
            for (int j = 0; j < n; j++) {
                double absVal = Math.abs(A[rowOff + j]);
                if (absVal != 0.0) {
                    if (scale < absVal) {
                        double ratio = scale / absVal;
                        ssq = 1.0 + ssq * ratio * ratio;
                        scale = absVal;
                    } else {
                        double ratio = absVal / scale;
                        ssq += ratio * ratio;
                    }
                }
            }
        }
        return scale * Math.sqrt(ssq);
    }

    /**
     * Max norm: largest absolute value.
     * ||A||_max = max_ij |A[i,j]|
     */
    static double normMax(int m, int n, double[] A, int aOff, int lda) {
        double maxVal = 0.0;
        for (int i = 0; i < m; i++) {
            int rowOff = aOff + i * lda;
            for (int j = 0; j < n; j++) {
                double absVal = Math.abs(A[rowOff + j]);
                if (absVal > maxVal) {
                    maxVal = absVal;
                }
            }
        }
        return maxVal;
    }
}
