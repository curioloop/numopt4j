/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class SVDReconstructionTest {

    private static final double EPSILON = 1e-10;

    @Test
    void testSVDReconstruction2x2() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int m = 2, n = 2;
        int lda = n;
        
        System.out.println("Original A:");
        printMatrix(A, m, n);
        
        double[] d = new double[Math.min(m, n)];
        double[] e = new double[Math.min(m, n)];
        double[] tauQ = new double[Math.min(m, n)];
        double[] tauP = new double[Math.min(m, n)];
        double[] work = new double[Math.max(m, n) * 10];
        
        double[] AWork = A.clone();
        com.curioloop.numopt4j.linalg.blas.BLAS.dgebd2(m, n, AWork, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);
        
        System.out.println("\nAfter Dgebd2:");
        System.out.println("AWork = ");
        printMatrix(AWork, m, n);
        System.out.println("d = " + java.util.Arrays.toString(d));
        System.out.println("e = " + java.util.Arrays.toString(e));
        System.out.println("tauQ = " + java.util.Arrays.toString(tauQ));
        System.out.println("tauP = " + java.util.Arrays.toString(tauP));
        
        double[] Q = new double[m * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Q[i * m + j] = AWork[i * lda + j];
            }
        }
        
        System.out.println("\nBefore Dorgbr Q:");
        printMatrix(Q, m, m);
        
        com.curioloop.numopt4j.linalg.blas.BLAS.dorgbr('Q', m, m, n, Q, 0, m, tauQ, 0, work, 0, work.length);
        
        System.out.println("\nAfter Dorgbr Q:");
        printMatrix(Q, m, m);
        
        double[] PT = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                PT[i * n + j] = AWork[i * lda + j];
            }
        }
        
        System.out.println("\nBefore Dorgbr PT:");
        printMatrix(PT, n, n);
        
        com.curioloop.numopt4j.linalg.blas.BLAS.dorgbr('P', n, n, n, PT, 0, n, tauP, 0, work, 0, work.length);
        
        System.out.println("\nAfter Dorgbr PT:");
        printMatrix(PT, n, n);
        
        double[] S = d.clone();
        double[] E = e.clone();
        
        System.out.println("\nBefore Dbdsqr:");
        System.out.println("S = " + java.util.Arrays.toString(S));
        System.out.println("E = " + java.util.Arrays.toString(E));
        System.out.println("Q = ");
        printMatrix(Q, m, m);
        System.out.println("PT = ");
        printMatrix(PT, n, n);
        
        int ncvt = n;
        int nru = m;
        int ldvt = n;
        int ldu = m;
        
        System.out.println("\nCalling Dbdsqr with ncvt=" + ncvt + ", nru=" + nru + ", ldvt=" + ldvt + ", ldu=" + ldu);
        
        boolean ok = com.curioloop.numopt4j.linalg.blas.BLAS.dbdsqr(com.curioloop.numopt4j.linalg.blas.BLAS.Uplo.Upper, Math.min(m, n), ncvt, nru, 0, S, 0, E, 0, PT, 0, ldvt, Q, 0, ldu, null, 0, 0, work, 0);
        
        System.out.println("\nAfter Dbdsqr:");
        System.out.println("S = " + java.util.Arrays.toString(S));
        System.out.println("Q (U) = ");
        printMatrix(Q, m, m);
        System.out.println("PT (VT) = ");
        printMatrix(PT, n, n);
        
        System.out.println("\nReconstruction A = U * S * VT:");
        double[] reconstructed = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < Math.min(m, n); k++) {
                    sum += Q[i * m + k] * S[k] * PT[k * n + j];
                }
                reconstructed[i * n + j] = sum;
            }
        }
        printMatrix(reconstructed, m, n);
        
        for (int i = 0; i < m * n; i++) {
            assertThat(reconstructed[i]).isCloseTo(A[i], org.assertj.core.data.Offset.offset(EPSILON));
        }
    }

    @Test
    void testSVDReconstruction3x2() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        };
        int m = 3, n = 2;
        int lda = n;
        
        System.out.println("Original A:");
        printMatrix(A, m, n);
        
        double[] d = new double[Math.min(m, n)];
        double[] e = new double[Math.min(m, n)];
        double[] tauQ = new double[Math.min(m, n)];
        double[] tauP = new double[Math.min(m, n)];
        double[] work = new double[Math.max(m, n) * 10];
        
        double[] AWork = A.clone();
        com.curioloop.numopt4j.linalg.blas.BLAS.dgebd2(m, n, AWork, 0, lda, d, 0, e, 0, tauQ, 0, tauP, 0, work, 0);
        
        System.out.println("\nAfter Dgebd2:");
        System.out.println("AWork = ");
        printMatrix(AWork, m, n);
        System.out.println("d = " + java.util.Arrays.toString(d));
        System.out.println("e = " + java.util.Arrays.toString(e));
        System.out.println("tauQ = " + java.util.Arrays.toString(tauQ));
        System.out.println("tauP = " + java.util.Arrays.toString(tauP));
        
        double[] Q = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                Q[i * n + j] = AWork[i * lda + j];
            }
        }
        
        System.out.println("\nBefore Dorgbr Q:");
        printMatrix(Q, m, n);
        
        com.curioloop.numopt4j.linalg.blas.BLAS.dorgbr('Q', m, n, n, Q, 0, n, tauQ, 0, work, 0, work.length);
        
        System.out.println("\nAfter Dorgbr Q:");
        printMatrix(Q, m, n);
        
        double[] PT = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                PT[i * n + j] = AWork[i * lda + j];
            }
        }
        
        System.out.println("\nBefore Dorgbr PT:");
        printMatrix(PT, n, n);
        
        com.curioloop.numopt4j.linalg.blas.BLAS.dorgbr('P', n, n, n, PT, 0, n, tauP, 0, work, 0, work.length);
        
        System.out.println("\nAfter Dorgbr PT:");
        printMatrix(PT, n, n);
        
        double[] S = d.clone();
        double[] E = e.clone();
        
        System.out.println("\nBefore Dbdsqr:");
        System.out.println("S = " + java.util.Arrays.toString(S));
        System.out.println("E = " + java.util.Arrays.toString(E));
        System.out.println("Q = ");
        printMatrix(Q, m, n);
        System.out.println("PT = ");
        printMatrix(PT, n, n);
        
        int ncvt = n;
        int nru = m;
        int ldvt = n;
        int ldu = n;
        
        System.out.println("\nCalling Dbdsqr with ncvt=" + ncvt + ", nru=" + nru + ", ldvt=" + ldvt + ", ldu=" + ldu);
        
        boolean ok = com.curioloop.numopt4j.linalg.blas.BLAS.dbdsqr(com.curioloop.numopt4j.linalg.blas.BLAS.Uplo.Upper, Math.min(m, n), ncvt, nru, 0, S, 0, E, 0, PT, 0, ldvt, Q, 0, ldu, null, 0, 0, work, 0);
        
        System.out.println("\nAfter Dbdsqr:");
        System.out.println("S = " + java.util.Arrays.toString(S));
        System.out.println("Q (U) = ");
        printMatrix(Q, m, n);
        System.out.println("PT (VT) = ");
        printMatrix(PT, n, n);
        
        System.out.println("\nReconstruction A = U * S * VT:");
        double[] reconstructed = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < Math.min(m, n); k++) {
                    sum += Q[i * n + k] * S[k] * PT[k * n + j];
                }
                reconstructed[i * n + j] = sum;
            }
        }
        printMatrix(reconstructed, m, n);
        
        for (int i = 0; i < m * n; i++) {
            assertThat(reconstructed[i]).isCloseTo(A[i], org.assertj.core.data.Offset.offset(EPSILON));
        }
    }

    private void printMatrix(double[] M, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.printf("%12.6f ", M[i * cols + j]);
            }
            System.out.println();
        }
    }
}
