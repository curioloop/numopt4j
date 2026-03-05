/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

interface Ilaux {

    static int iladlr(int m, int n, double[] a, int aOff, int lda) {
        if (n == 0 || m == 0) {
            return -1;
        }

        if (a[aOff + (m - 1) * lda] != 0.0 || a[aOff + (m - 1) * lda + n - 1] != 0.0) {
            return m - 1;
        }

        for (int i = m - 1; i >= 0; i--) {
            for (int j = 0; j < n; j++) {
                if (a[aOff + i * lda + j] != 0.0) {
                    return i;
                }
            }
        }
        return -1;
    }

    static int iladlc(int m, int n, double[] a, int aOff, int lda) {
        if (n == 0 || m == 0) {
            return -1;
        }

        if (a[aOff + n - 1] != 0.0 || a[aOff + (m - 1) * lda + (n - 1)] != 0.0) {
            return n - 1;
        }

        int highest = -1;
        for (int i = 0; i < m; i++) {
            for (int j = n - 1; j >= 0; j--) {
                if (a[aOff + i * lda + j] != 0.0) {
                    highest = Math.max(highest, j);
                    break;
                }
            }
        }
        return highest;
    }
}
