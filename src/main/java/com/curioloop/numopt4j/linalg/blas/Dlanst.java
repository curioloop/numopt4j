/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DLANST: Computes matrix norm of symmetric tridiagonal matrix.
 * Based on LAPACK.
 */
interface Dlanst {
    
    /**
     * Compute matrix norm with offset support.
     * 
     * @param norm 'M' or 'm' - max abs, '1' or 'O' or 'o' - one norm, 'F' or 'f' or 'E' or 'e' - Frobenius norm
     * @param n matrix order
     * @param d diagonal elements
     * @param doff offset into d array
     * @param e off-diagonal elements
     * @param eoff offset into e array
     * @return matrix norm
     */
    static double dlanst(char norm, int n, double[] d, int doff, double[] e, int eoff) {
        if (n == 0) {
            return 0.0;
        }
        
        char nchar = Character.toUpperCase(norm);
        
        switch (nchar) {
            case 'M':
                // Maximum absolute value
                double res = Math.abs(d[doff + n - 1]);
                for (int i = 0; i < n - 1; i++) {
                    double absD = Math.abs(d[doff + i]);
                    if (res < absD || Double.isNaN(absD)) {
                        res = absD;
                    }
                    double absE = Math.abs(e[eoff + i]);
                    if (res < absE || Double.isNaN(absE)) {
                        res = absE;
                    }
                }
                return res;
                
            case '1':
            case 'O':
                // One norm (max column sum for symmetric tridiagonal)
                if (n == 1) {
                    return Math.abs(d[doff]);
                }
                double anorm = Math.abs(d[doff]) + Math.abs(e[eoff]);
                double sum = Math.abs(e[eoff + n - 2]) + Math.abs(d[doff + n - 1]);
                if (anorm < sum || Double.isNaN(sum)) {
                    anorm = sum;
                }
                for (int i = 1; i < n - 1; i++) {
                    sum = Math.abs(d[doff + i]) + Math.abs(e[eoff + i]) + Math.abs(e[eoff + i - 1]);
                    if (anorm < sum || Double.isNaN(sum)) {
                        anorm = sum;
                    }
                }
                return anorm;
                
            case 'F':
            case 'E':
                // Frobenius norm: sqrt(sum of squares)
                double scale = 0.0;
                double ssq = 1.0;
                // Process off-diagonal elements (e)
                for (int i = 0; i < n - 1; i++) {
                    double absE = Math.abs(e[eoff + i]);
                    if (absE != 0.0) {
                        if (scale < absE) {
                            ssq = 1 + ssq * (scale / absE) * (scale / absE);
                            scale = absE;
                        } else {
                            ssq = ssq + (absE / scale) * (absE / scale);
                        }
                    }
                }
                // Off-diagonal appears twice in symmetric matrix
                ssq *= 2;
                // Process diagonal elements (d)
                for (int i = 0; i < n; i++) {
                    double absD = Math.abs(d[doff + i]);
                    if (absD != 0.0) {
                        if (scale < absD) {
                            ssq = 1 + ssq * (scale / absD) * (scale / absD);
                            scale = absD;
                        } else {
                            ssq = ssq + (absD / scale) * (absD / scale);
                        }
                    }
                }
                return scale * Math.sqrt(ssq);
                
            default:
                // Default to max abs
                return dlanst('M', n, d, doff, e, eoff);
        }
    }
}
