package com.curioloop.numopt4j.linalg.mat;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class LUAccuracyTest {
    
    static final double TOL = 1e-10;
    
    @Test
    void test2x2Inverse() {
        double[] A = {4, 3, 6, 3};
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 2);
        assertTrue(lu.ok(), "Should be nonsingular");
        double[] inv = lu.inverse(null);
        assertNotNull(inv, "Inverse should succeed");
        
        double[] expectedInv = {-0.5, 0.5, 1, -2.0/3.0};
        for (int i = 0; i < 4; i++) {
            assertEquals(expectedInv[i], inv[i], TOL, "Inverse mismatch at index " + i);
        }
        
        double[] product = multiplyMatrices(Aorig, inv, 2);
        assertIdentity(product, 2, TOL);
    }
    
    @Test
    void test3x3Inverse() {
        double[] A = {1, 2, 3, 0, 1, 4, 5, 6, 0};
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, 3);
        assertTrue(lu.ok(), "Should be nonsingular");
        double[] inv = lu.inverse(null);
        assertNotNull(inv, "Inverse should succeed");
        
        double[] product = multiplyMatrices(Aorig, inv, 3);
        assertIdentity(product, 3, 1e-9);
    }
    
    @Test
    void testDiagonalInverse() {
        double[] A = {2, 0, 0, 0, 3, 0, 0, 0, 4};

        LU lu = LU.decompose(A, 3);
        assertTrue(lu.ok(), "Should be nonsingular");
        double[] inv = lu.inverse(null);
        assertNotNull(inv, "Inverse should succeed");
        
        double[] expected = {0.5, 0, 0, 0, 1.0/3.0, 0, 0, 0, 0.25};
        assertArrayEquals(expected, inv, TOL);
    }
    
    @Test
    void testIdentityInverse() {
        double[] A = {1, 0, 0, 0, 1, 0, 0, 0, 1};

        LU lu = LU.decompose(A, 3);
        assertTrue(lu.ok(), "Should be nonsingular");
        double[] inv = lu.inverse(null);
        assertNotNull(inv, "Inverse should succeed");
        
        double[] expected = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        assertArrayEquals(expected, inv, TOL);
    }
    
    @Test
    void testHilbertInverse() {
        int n = 8;
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = 1.0 / (i + j + 1);
            }
        }
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, n);
        assertTrue(lu.ok(), "Should be nonsingular");
        double[] inv = lu.inverse(null);
        assertNotNull(inv, "Inverse should succeed");
        
        double[] product = multiplyMatrices(Aorig, inv, n);
        assertIdentity(product, n, 1e-4);
    }
    
    @Test
    void testRandomMatrixInverse() {
        int n = 50;
        double[] A = generateRandomMatrix(n, 12345);
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, n);
        assertTrue(lu.ok(), "Should be nonsingular");
        double[] inv = lu.inverse(null);
        assertNotNull(inv, "Inverse should succeed");
        
        double[] product = multiplyMatrices(Aorig, inv, n);
        assertIdentity(product, n, 1e-8);
    }
    
    @Test
    void testLargeMatrixInverse() {
        int n = 150;
        double[] A = generateDiagonallyDominantMatrix(n, 99999);
        double[] Aorig = A.clone();

        LU lu = LU.decompose(A, n);
        assertTrue(lu.ok(), "Should be nonsingular");
        double[] inv = lu.inverse(null);
        assertNotNull(inv, "Inverse should succeed");
        
        double[] product = multiplyMatrices(Aorig, inv, n);
        assertIdentity(product, n, 1e-6);
    }
    
    @Test
    void testSingularMatrix() {
        double[] A = {1, 2, 2, 4};

        LU lu = LU.decompose(A, 2);
        assertFalse(lu.ok(), "Should detect singularity");
    }
    
    double[] multiplyMatrices(double[] A, double[] B, int n) {
        double[] C = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
        return C;
    }
    
    void assertIdentity(double[] M, int n, double tol) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, M[i * n + j], tol, 
                    "Not identity at (" + i + "," + j + ")");
            }
        }
    }
    
    void assertArrayEquals(double[] expected, double[] actual, double tol) {
        assertEquals(expected.length, actual.length, "Array length mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tol, "Mismatch at index " + i);
        }
    }
    
    double[] generateRandomMatrix(int n, long seed) {
        java.util.Random rand = new java.util.Random(seed);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = rand.nextDouble() * 2 - 1;
            }
            A[i * n + i] += n;
        }
        return A;
    }
    
    double[] generateDiagonallyDominantMatrix(int n, long seed) {
        java.util.Random rand = new java.util.Random(seed);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            double rowSum = 0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    A[i * n + j] = rand.nextDouble() - 0.5;
                    rowSum += Math.abs(A[i * n + j]);
                }
            }
            A[i * n + i] = rowSum + 1 + rand.nextDouble();
        }
        return A;
    }
}
