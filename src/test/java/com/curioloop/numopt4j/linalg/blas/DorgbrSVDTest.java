package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

class DorgbrSVDTest {

    @Test
    void testDorgbrPInSVD() {
        int m = 4, n = 3, minMN = 3;
        
        double[] A = {
            0.8147, 0.9134, 0.9,
            0.9058, 0.6324, 0.9,
            0.1270, 0.0975, 0.1,
            1.6, 2.8, -3.5
        };
        
        double[] S = new double[minMN];
        double[] e = new double[minMN];
        double[] tauQ = new double[minMN];
        double[] tauP = new double[minMN];
        double[] work = new double[4 * minMN];
        
        System.out.println("Original A:");
        printMatrix(A, m, n);
        
        BLAS.dgebd2(m, n, A, 0, n, S, 0, e, 0, tauQ, 0, tauP, 0, work, 0);
        
        System.out.println("\nAfter dgebd2:");
        printMatrix(A, m, n);
        System.out.println("S: " + java.util.Arrays.toString(S));
        System.out.println("e: " + java.util.Arrays.toString(e));
        System.out.println("tauQ: " + java.util.Arrays.toString(tauQ));
        System.out.println("tauP: " + java.util.Arrays.toString(tauP));
        
        // Generate VT
        double[] VT = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                VT[i * n + j] = A[i * n + j];
            }
        }
        
        System.out.println("\nVT before dorgbr:");
        printMatrix(VT, n, n);
        
        BLAS.dorgbr('P', n, n, n, VT, 0, n, tauP, 0, work, 0, work.length);
        
        System.out.println("\nVT after dorgbr('P', n, n, n):");
        printMatrix(VT, n, n);
        
        // Check orthogonality: VT * VT^T should be identity
        double[] VTVTt = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int l = 0; l < n; l++) {
                    sum += VT[i * n + l] * VT[j * n + l];
                }
                VTVTt[i * n + j] = sum;
            }
        }
        
        System.out.println("\nVT * VT^T (should be identity):");
        printMatrix(VTVTt, n, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(VTVTt[i * n + j]).isCloseTo(expected, org.assertj.core.data.Offset.offset(1e-10));
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
