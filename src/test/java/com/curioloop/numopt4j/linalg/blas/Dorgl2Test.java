package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

class Dorgl2Test {

    @Test
    void testDorgl2WithDgelq2() {
        int m = 3, n = 3, k = 3;
        
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        double[] tau = new double[k];
        double[] work = new double[m];
        
        System.out.println("Original A:");
        printMatrix(A, m, n);
        
        Dgelq.dgelq2(m, n, A, 0, n, tau, 0, work, 0);
        
        System.out.println("\nAfter dgelq2:");
        printMatrix(A, m, n);
        System.out.println("tau: " + java.util.Arrays.toString(tau));
        
        Dgelq.dorgl2(m, n, k, A, 0, n, tau, 0, work, 0);
        
        System.out.println("\nAfter dorgl2 (should be Q):");
        printMatrix(A, m, n);
        
        double[] QQt = new double[m * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int l = 0; l < n; l++) {
                    sum += A[i * n + l] * A[j * n + l];
                }
                QQt[i * m + j] = sum;
            }
        }
        
        System.out.println("\nQ * Q^T (should be identity):");
        printMatrix(QQt, m, m);
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(QQt[i * m + j]).isCloseTo(expected, org.assertj.core.data.Offset.offset(1e-10));
            }
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
