/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class DlarfTest {

    @Test
    void testDlarfLeft() {
        double[] C = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        int m = 3, n = 3, ldc = 3;

        double[] v = {1.0, 2.0, 3.0};
        double tau = 0.5;
        double[] work = new double[n];

        System.out.println("Before Dlarf:");
        printMatrix(C, m, n);

        Dlarf.dlarf(BLAS.Side.Left, m, n, v, 0, 1, tau, C, 0, ldc, work);

        System.out.println("After Dlarf:");
        printMatrix(C, m, n);

        double[] expected = {
            -14.0, -16.0, -18.0,
            -26.0, -31.0, -36.0,
            -38.0, -46.0, -54.0
        };

        for (int i = 0; i < m * n; i++) {
            assertThat(C[i]).isCloseTo(expected[i], org.assertj.core.data.Offset.offset(1e-10));
        }
    }

    @Test
    void testDlarfRight() {
        double[] C = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        int m = 3, n = 3, ldc = 3;

        double[] v = {1.0, 2.0, 3.0};
        double tau = 0.5;
        double[] work = new double[m];

        System.out.println("Before Dlarf:");
        printMatrix(C, m, n);

        Dlarf.dlarf(BLAS.Side.Right, m, n, v, 0, 1, tau, C, 0, ldc, work);

        System.out.println("After Dlarf:");
        printMatrix(C, m, n);

        double[] expected = {
            -6.0, -12.0, -18.0,
            -12.0, -27.0, -42.0,
            -18.0, -42.0, -66.0
        };

        for (int i = 0; i < m * n; i++) {
            assertThat(C[i]).isCloseTo(expected[i], org.assertj.core.data.Offset.offset(1e-10));
        }
    }

    private void printMatrix(double[] A, int m, int n) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                System.out.printf("%10.6f ", A[i * n + j]);
            }
            System.out.println();
        }
    }
}
