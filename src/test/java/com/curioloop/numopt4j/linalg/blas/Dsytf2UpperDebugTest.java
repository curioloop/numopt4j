/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.*;

class Dsytf2UpperDebugTest {

    @Test
    void testIndefiniteUpper() {
        double[] A = {
            1, 2,
            0, 1
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        System.out.println("=== Indefinite Upper Test ===");
        System.out.println("Original symmetric matrix:");
        System.out.println("  [1, 2]");
        System.out.println("  [2, 1]");
        System.out.println("\nUpper triangular storage (row-major):");
        printMatrix(A, n);

        boolean success = Dsytrf.dsytf2(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work);

        System.out.println("\nAfter decomposition:");
        printMatrix(A, n);
        System.out.println("\nipiv:");
        for (int i = 0; i < n; i++) {
            System.out.println("ipiv[" + i + "] = " + ipiv[i]);
        }
        
        assertThat(success).isTrue();
        
        double[] b = {3, 4};
        double[] bOrig = b.clone();
        
        Dsytrs.dsytrs(BLAS.Uplo.Upper, n, 1, A, 0, n, ipiv, 0, b, 0, 1);
        
        System.out.println("\nSolution:");
        for (int i = 0; i < n; i++) {
            System.out.println("x[" + i + "] = " + b[i]);
        }
        
        double[] Ax = new double[n];
        Ax[0] = 1 * b[0] + 2 * b[1];
        Ax[1] = 2 * b[0] + 1 * b[1];
        
        System.out.println("\nA*x:");
        for (int i = 0; i < n; i++) {
            System.out.println("Ax[" + i + "] = " + Ax[i] + " (expected: " + bOrig[i] + ")");
        }
        
        for (int i = 0; i < n; i++) {
            assertThat(Ax[i]).isCloseTo(bOrig[i], offset(1e-10));
        }
    }
    
    @Test
    void testPositiveDefiniteUpper() {
        double[] A = {
            4, 2, 2,
            0, 5, 3,
            0, 0, 6
        };
        int n = 3;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        System.out.println("\n=== Positive Definite Upper Test ===");
        System.out.println("Original symmetric matrix:");
        System.out.println("  [4, 2, 2]");
        System.out.println("  [2, 5, 3]");
        System.out.println("  [2, 3, 6]");
        System.out.println("\nUpper triangular storage (row-major):");
        printMatrix(A, n);

        boolean success = Dsytrf.dsytf2(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work);

        System.out.println("\nAfter decomposition:");
        printMatrix(A, n);
        System.out.println("\nipiv:");
        for (int i = 0; i < n; i++) {
            System.out.println("ipiv[" + i + "] = " + ipiv[i]);
        }
        
        assertThat(success).isTrue();
        
        double[] b = {8, 7, 11};
        double[] bOrig = b.clone();
        
        Dsytrs.dsytrs(BLAS.Uplo.Upper, n, 1, A, 0, n, ipiv, 0, b, 0, 1);
        
        System.out.println("\nSolution:");
        for (int i = 0; i < n; i++) {
            System.out.println("x[" + i + "] = " + b[i]);
        }
        
        double[] Ax = new double[n];
        double[] AOrig = {
            4, 2, 2,
            2, 5, 3,
            2, 3, 6
        };
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Ax[i] += AOrig[i * n + j] * b[j];
            }
        }
        
        System.out.println("\nA*x:");
        for (int i = 0; i < n; i++) {
            System.out.println("Ax[" + i + "] = " + Ax[i] + " (expected: " + bOrig[i] + ")");
        }
        
        for (int i = 0; i < n; i++) {
            assertThat(Ax[i]).isCloseTo(bOrig[i], offset(1e-10));
        }
    }
    
    @Test
    void testManualSolveUpper() {
        double[] A = {
            4, 2, 2,
            0, 5, 3,
            0, 0, 6
        };
        int n = 3;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        System.out.println("\n=== Manual Solve Upper Test ===");
        System.out.println("Original symmetric matrix:");
        System.out.println("  [4, 2, 2]");
        System.out.println("  [2, 5, 3]");
        System.out.println("  [2, 3, 6]");

        boolean success = Dsytrf.dsytf2(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work);

        System.out.println("\nAfter decomposition:");
        printMatrix(A, n);
        System.out.println("\nipiv: " + java.util.Arrays.toString(ipiv));
        
        assertThat(success).isTrue();
        
        double[] b = {8, 7, 11};
        System.out.println("\nRight-hand side b: " + java.util.Arrays.toString(b));
        
        double[] x = b.clone();
        
        System.out.println("\n--- Step 1: Solve U*D*X = B ---");
        System.out.println("Processing from k = n-1 down to 0");
        
        int k = n - 1;
        while (k >= 0) {
            System.out.println("\nk = " + k);
            if (ipiv[k] > 0) {
                System.out.println("  1x1 block, ipiv[" + k + "] = " + ipiv[k]);
                double ak = A[k * n + k];
                System.out.println("  A[" + k + "," + k + "] = " + ak);
                System.out.println("  Before division: x[" + k + "] = " + x[k]);
                x[k] /= ak;
                System.out.println("  After division: x[" + k + "] = " + x[k]);
                
                for (int i = k - 1; i >= 0; i--) {
                    double aik = A[k * n + i];
                    System.out.println("  A[" + k + "," + i + "] = " + aik);
                    System.out.println("  x[" + i + "] -= " + aik + " * " + x[k] + " = " + (x[i] - aik * x[k]));
                    x[i] -= aik * x[k];
                }
                k--;
            } else {
                System.out.println("  2x2 block, ipiv[" + k + "] = " + ipiv[k]);
                k -= 2;
            }
        }
        
        System.out.println("\nAfter U*D*X = B: x = " + java.util.Arrays.toString(x));
        
        System.out.println("\n--- Step 2: Solve U^T*X = B ---");
        System.out.println("Processing from k = 0 up to n-1");
        
        k = 0;
        while (k < n) {
            System.out.println("\nk = " + k);
            if (ipiv[k] > 0) {
                System.out.println("  1x1 block");
                for (int i = k + 1; i < n; i++) {
                    double aki = A[k * n + i];
                    System.out.println("  A[" + k + "," + i + "] = " + aki);
                    System.out.println("  x[" + i + "] -= " + aki + " * " + x[k] + " = " + (x[i] - aki * x[k]));
                    x[i] -= aki * x[k];
                }
                k++;
            } else {
                System.out.println("  2x2 block");
                k += 2;
            }
        }
        
        System.out.println("\nFinal solution: x = " + java.util.Arrays.toString(x));
        
        double[] Ax = new double[n];
        double[] AOrig = {4, 2, 2, 2, 5, 3, 2, 3, 6};
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Ax[i] += AOrig[i * n + j] * x[j];
            }
        }
        
        System.out.println("\nVerification A*x = " + java.util.Arrays.toString(Ax));
        System.out.println("Expected b = " + java.util.Arrays.toString(b));
    }
    
    private void printMatrix(double[] A, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.printf("%10.6f ", A[i * n + j]);
            }
            System.out.println();
        }
    }
}
