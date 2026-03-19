/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DLASRT: Sorts eigenvalues in increasing or decreasing order.
 * Based on LAPACK.
 */
interface Dlasrt {
    
    /**
     * Sort eigenvalues with offset support.
     * 
     * @param id 'I' for increasing, 'D' for decreasing
     * @param n number of elements
     * @param d array to sort (modified in place)
     * @param dOff starting offset in d
     */
    static void dlasrt(char id, int n, double[] d, int dOff) {
        if (n <= 1) {
            return;
        }
        
        if (id == 'I' || id == 'i') {
            // Sort in increasing order
            java.util.Arrays.sort(d, dOff, dOff + n);
        } else if (id == 'D' || id == 'd') {
            // Sort in decreasing order
            java.util.Arrays.sort(d, dOff, dOff + n);
            for (int i = 0; i < n / 2; i++) {
                double tmp = d[dOff + i];
                d[dOff + i] = d[dOff + n - 1 - i];
                d[dOff + n - 1 - i] = tmp;
            }
        }
    }
}
