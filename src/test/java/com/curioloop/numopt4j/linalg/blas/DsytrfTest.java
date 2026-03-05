/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DsytrfTest {

    private static final double TOL = 1e-10;

    @Test
    void testEmpty() {
        int[] ipiv = new int[0];
        double[] work = new double[1];
        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, 0, new double[0], 0, 1, ipiv, 0, work, 1);
        assertEquals(0, info);
    }

    @Test
    void testSingleElement() {
        double[] A = {4.0};
        int[] ipiv = new int[1];
        double[] work = new double[1];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, 1, A, 0, 1, ipiv, 0, work, 1);

        assertEquals(0, info);
        assertEquals(1, ipiv[0]);
    }

    @Test
    void testLowerBasic() {
        double[] A = {
            4, 0,
            2, 3
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n);

        assertEquals(0, info);
    }

    @Test
    void testUpperBasic() {
        double[] A = {
            4, 2,
            0, 3
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work, n);

        assertEquals(0, info);
    }

    @Test
    void testIndefiniteMatrix() {
        double[] A = {
            1, 0,
            0, -1
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n);

        assertEquals(0, info);
    }

    @Test
    void testSingularMatrix() {
        double[] A = {
            0, 0,
            0, 0
        };
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n);

        assertTrue(info >= 0);
        assertEquals(1, ipiv[0]);
        assertEquals(2, ipiv[1]);
    }

    @Test
    void testPositiveDefiniteLower() {
        double[] A = {
            4, 0, 0,
            2, 5, 0,
            2, 3, 6
        };
        double[] Aorig = A.clone();
        int n = 3;
        int[] ipiv = new int[n];
        double[] work = new double[n * 32];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 32);
        assertEquals(0, info);

        verifySolveLower(Aorig, A, ipiv, n);
    }

    @Test
    void testPositiveDefiniteUpper() {
        double[] A = {
            4, 2, 2,
            0, 5, 3,
            0, 0, 6
        };
        double[] Aorig = A.clone();
        int n = 3;
        int[] ipiv = new int[n];
        double[] work = new double[n * 32];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work, n * 32);
        assertEquals(0, info);

        verifySolveUpper(Aorig, A, ipiv, n);
    }

    @Test
    void test2x2BlockPivot() {
        double[] A = {
            1, 2,
            2, 1
        };
        double[] Aorig = A.clone();
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[n];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n);
        assertEquals(0, info);

        assertTrue(ipiv[0] < 0);

        verifySolveLower(Aorig, A, ipiv, n);
    }

    @Test
    void testLargeMatrixLower() {
        int n = 100;
        java.util.Random rand = new java.util.Random(42);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                A[i * n + j] = rand.nextDouble();
                A[j * n + i] = A[i * n + j];
            }
        }
        for (int i = 0; i < n; i++) {
            A[i * n + i] += n;
        }
        double[] Aorig = A.clone();
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, n * 64);
        assertEquals(0, info);

        verifySolveLower(Aorig, A, ipiv, n);
    }

    @Test
    void testLargeMatrixUpper() {
        int n = 100;
        java.util.Random rand = new java.util.Random(42);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                A[i * n + j] = rand.nextDouble();
                A[j * n + i] = A[i * n + j];
            }
        }
        for (int i = 0; i < n; i++) {
            A[i * n + i] += n;
        }
        double[] Aorig = A.clone();
        int[] ipiv = new int[n];
        double[] work = new double[n * 64];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, work, n * 64);
        assertEquals(0, info);

        verifySolveUpper(Aorig, A, ipiv, n);
    }

    @Test
    void testConsistencyWithDsytf2Lower() {
        int n = 50;
        java.util.Random rand = new java.util.Random(123);
        double[] Aorig = new double[n * n];
        double[] A1 = new double[n * n];
        double[] A2 = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double val = rand.nextDouble();
                Aorig[i * n + j] = val;
                Aorig[j * n + i] = val;
                A1[i * n + j] = val;
                A1[j * n + i] = val;
                A2[i * n + j] = val;
                A2[j * n + i] = val;
            }
        }

        int[] ipiv1 = new int[n];
        int[] ipiv2 = new int[n];
        double[] work1 = new double[n];
        double[] work2 = new double[n * 64];

        boolean ok1 = Dsytrf.dsytf2(BLAS.Uplo.Lower, n, A1, 0, n, ipiv1, 0, work1);
        int info2 = Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A2, 0, n, ipiv2, 0, work2, n * 64);

        assertTrue(ok1);
        assertEquals(0, info2);

        verifySolveLower(Aorig, A1, ipiv1, n);
        verifySolveLower(Aorig, A2, ipiv2, n);
    }

    @Test
    void testConsistencyWithDsytf2Upper() {
        int n = 50;
        java.util.Random rand = new java.util.Random(456);
        double[] Aorig = new double[n * n];
        double[] A1 = new double[n * n];
        double[] A2 = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double val = rand.nextDouble();
                Aorig[i * n + j] = val;
                Aorig[j * n + i] = val;
                A1[i * n + j] = val;
                A1[j * n + i] = val;
                A2[i * n + j] = val;
                A2[j * n + i] = val;
            }
        }

        int[] ipiv1 = new int[n];
        int[] ipiv2 = new int[n];
        double[] work1 = new double[n];
        double[] work2 = new double[n * 64];

        boolean ok1 = Dsytrf.dsytf2(BLAS.Uplo.Upper, n, A1, 0, n, ipiv1, 0, work1);
        int info2 = Dsytrf.dsytrf(BLAS.Uplo.Upper, n, A2, 0, n, ipiv2, 0, work2, n * 64);

        assertTrue(ok1);
        assertEquals(0, info2);

        verifySolveUpper(Aorig, A1, ipiv1, n);
        verifySolveUpper(Aorig, A2, ipiv2, n);
    }
    
    @Test
    void testDebugLowerSmall() {
        int n = 50;
        double[] Aorig = new double[n * n];
        double[] A1 = new double[n * n];
        double[] A2 = new double[n * n];
        java.util.Random rand = new java.util.Random(123);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double val = rand.nextDouble() * 10;
                Aorig[i * n + j] = val;
                Aorig[j * n + i] = val;
                A1[i * n + j] = val;
                A1[j * n + i] = val;
                A2[i * n + j] = val;
                A2[j * n + i] = val;
            }
        }
        
        int[] ipiv1 = new int[n];
        int[] ipiv2 = new int[n];
        double[] work1 = new double[n];
        double[] work2 = new double[n * 64];

        boolean ok1 = Dsytrf.dsytf2(BLAS.Uplo.Lower, n, A1, 0, n, ipiv1, 0, work1);
        int info2 = Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A2, 0, n, ipiv2, 0, work2, n * 64);

        System.out.println("ipiv1 (dsytf2): " + java.util.Arrays.toString(ipiv1));
        System.out.println("ipiv2 (dsytrf): " + java.util.Arrays.toString(ipiv2));
        System.out.println("work2[0] (lwkopt): " + work2[0]);
        
        int diffCount = 0;
        for (int i = 0; i < n; i++) {
            if (ipiv1[i] != ipiv2[i]) {
                System.out.printf("ipiv diff at %d: %d vs %d%n", i, ipiv1[i], ipiv2[i]);
                diffCount++;
            }
        }
        System.out.println("Total ipiv differences: " + diffCount);

        assertTrue(ok1);
        assertEquals(0, info2);

        verifySolveLower(Aorig, A1, ipiv1, n);
        verifySolveLower(Aorig, A2, ipiv2, n);
    }
    
    @Test
    void testDebugUpperSmall() {
        int n = 50;
        double[] Aorig = new double[n * n];
        double[] A1 = new double[n * n];
        double[] A2 = new double[n * n];
        java.util.Random rand = new java.util.Random(456);
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double val = rand.nextDouble() * 10;
                Aorig[i * n + j] = val;
                Aorig[j * n + i] = val;
                A1[i * n + j] = val;
                A1[j * n + i] = val;
                A2[i * n + j] = val;
                A2[j * n + i] = val;
            }
        }
        
        int[] ipiv1 = new int[n];
        int[] ipiv2 = new int[n];
        double[] work1 = new double[n];
        double[] work2 = new double[n * 64];

        boolean ok1 = Dsytrf.dsytf2(BLAS.Uplo.Upper, n, A1, 0, n, ipiv1, 0, work1);
        int info2 = Dsytrf.dsytrf(BLAS.Uplo.Upper, n, A2, 0, n, ipiv2, 0, work2, n * 64);

        System.out.println("ipiv1 (dsytf2): " + java.util.Arrays.toString(ipiv1));
        System.out.println("ipiv2 (dsytrf): " + java.util.Arrays.toString(ipiv2));
        System.out.println("work2[0] (lwkopt): " + work2[0]);
        
        int diffCount = 0;
        for (int i = 0; i < n; i++) {
            if (ipiv1[i] != ipiv2[i]) {
                System.out.printf("ipiv diff at %d: %d vs %d%n", i, ipiv1[i], ipiv2[i]);
                diffCount++;
            }
        }
        System.out.println("Total ipiv differences: " + diffCount);

        assertTrue(ok1);
        assertEquals(0, info2);

        verifySolveUpper(Aorig, A1, ipiv1, n);
        verifySolveUpper(Aorig, A2, ipiv2, n);
    }
    
    private void printMatrix(double[] A, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.printf("%10.4f ", A[i * n + j]);
            }
            System.out.println();
        }
    }

    @Test
    void testInvalidUplo() {
        // With enum types, invalid uplo values are prevented at compile time
        // This test verifies that negative n still returns -2
        double[] A = {1.0};
        int[] ipiv = new int[1];
        double[] work = new double[1];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, -1, A, 0, 1, ipiv, 0, work, 1);
        assertEquals(-2, info);
    }

    @Test
    void testNegativeN() {
        double[] A = {1.0};
        int[] ipiv = new int[1];
        double[] work = new double[1];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, -1, A, 0, 1, ipiv, 0, work, 1);
        assertEquals(-2, info);
    }

    @Test
    void testInsufficientWork() {
        double[] A = {1.0, 0, 1.0};
        int n = 2;
        int[] ipiv = new int[n];
        double[] work = new double[0];

        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, work, 0);
        assertEquals(-7, info);
    }

    private void verifySolveLower(double[] Aorig, double[] A, int[] ipiv, int n) {
        double[] b = new double[n];
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            b[i] = i + 1;
            x[i] = b[i];
        }

        Dsytrs.dsytrs(BLAS.Uplo.Lower, n, 1, A, 0, n, ipiv, 0, x, 0, 1);

        double[] Ax = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double aij = (i >= j) ? Aorig[i * n + j] : Aorig[j * n + i];
                Ax[i] += aij * x[j];
            }
        }

        for (int i = 0; i < n; i++) {
            assertEquals(b[i], Ax[i], 1e-8);
        }
    }

    private void verifySolveUpper(double[] Aorig, double[] A, int[] ipiv, int n) {
        double[] b = new double[n];
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            b[i] = i + 1;
            x[i] = b[i];
        }

        Dsytrs.dsytrs(BLAS.Uplo.Upper, n, 1, A, 0, n, ipiv, 0, x, 0, 1);

        double[] Ax = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double aij = (i <= j) ? Aorig[i * n + j] : Aorig[j * n + i];
                Ax[i] += aij * x[j];
            }
        }

        for (int i = 0; i < n; i++) {
            assertEquals(b[i], Ax[i], 1e-8);
        }
    }

    @Test
    void testWorkQuery() {
        int n = 100;
        double[] A = new double[n * n];
        java.util.Random rand = new java.util.Random(789);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double val = rand.nextDouble();
                A[i * n + j] = val;
                A[j * n + i] = val;
            }
        }
        
        int[] ipiv = new int[n];
        double[] workQuery = new double[1];
        
        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A, 0, n, ipiv, 0, workQuery, -1);
        assertEquals(0, info);
        
        int lwkopt = (int) workQuery[0];
        assertTrue(lwkopt > 0);
        assertTrue(lwkopt >= n);
        
        double[] work = new double[lwkopt];
        double[] A2 = A.clone();
        int[] ipiv2 = new int[n];
        
        info = Dsytrf.dsytrf(BLAS.Uplo.Lower, n, A2, 0, n, ipiv2, 0, work, lwkopt);
        assertEquals(0, info);
    }

    @Test
    void testWorkQueryUpper() {
        int n = 100;
        double[] A = new double[n * n];
        java.util.Random rand = new java.util.Random(789);
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double val = rand.nextDouble();
                A[i * n + j] = val;
                A[j * n + i] = val;
            }
        }
        
        int[] ipiv = new int[n];
        double[] workQuery = new double[1];
        
        int info = Dsytrf.dsytrf(BLAS.Uplo.Upper, n, A, 0, n, ipiv, 0, workQuery, -1);
        assertEquals(0, info);
        
        int lwkopt = (int) workQuery[0];
        assertTrue(lwkopt > 0);
        assertTrue(lwkopt >= n);
        
        double[] work = new double[lwkopt];
        double[] A2 = A.clone();
        int[] ipiv2 = new int[n];
        
        info = Dsytrf.dsytrf(BLAS.Uplo.Upper, n, A2, 0, n, ipiv2, 0, work, lwkopt);
        assertEquals(0, info);
    }

    @Test
    void testWorkQueryEmpty() {
        double[] workQuery = new double[1];
        int info = Dsytrf.dsytrf(BLAS.Uplo.Lower, 0, new double[0], 0, 1, new int[0], 0, workQuery, -1);
        assertEquals(0, info);
        assertTrue(workQuery[0] >= 0);
    }
}
