package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

class Dorg2rTest {

    @Test
    void testDorg2rWithDgeqr2() {
        int m = 4, n = 3, k = 3;
        
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        };
        double[] tau = new double[k];
        double[] work = new double[n];
        
        System.out.println("Original A:");
        printMatrix(A, m, n);
        
        Dgeqr.dgeqr2(m, n, A, 0, n, tau, 0, work, 0);
        
        System.out.println("\nAfter dgeqr2:");
        printMatrix(A, m, n);
        System.out.println("tau: " + java.util.Arrays.toString(tau));
        
        Dgeqr.dorg2r(m, n, k, A, 0, n, tau, 0, work, 0);
        
        System.out.println("\nAfter dorg2r (should be Q):");
        printMatrix(A, m, n);
        
        // Check orthogonality: Q^T * Q should be identity
        double[] QtQ = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int l = 0; l < m; l++) {
                    sum += A[l * n + i] * A[l * n + j];
                }
                QtQ[i * n + j] = sum;
            }
        }
        
        System.out.println("\nQ^T * Q (should be identity):");
        printMatrix(QtQ, n, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(QtQ[i * n + j]).isCloseTo(expected, org.assertj.core.data.Offset.offset(1e-10));
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
