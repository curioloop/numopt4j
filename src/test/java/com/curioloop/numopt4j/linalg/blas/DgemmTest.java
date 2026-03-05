/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Comprehensive tests for DGEMM to verify all parameter combinations and edge cases.
 * Tests: transpose combinations (NN, NT, TN, TT), alpha/beta values, matrix sizes,
 * offsets, and leading dimensions.
 */
public class DgemmTest {

    private final Random rand = new Random(12345);
    
    // Test sizes for various scenarios
    private static final int[] SMALL_SIZES = {1, 2, 3, 4, 5, 7, 8, 10};
    private static final int[] MEDIUM_SIZES = {11, 15, 16, 20, 31, 32, 48, 63, 64};
    private static final int[] LARGE_SIZES = {65, 80, 96, 100, 128};
    private static final int[] ALL_SIZES = {1, 2, 3, 4, 5, 7, 8, 10, 15, 16, 20, 31, 32, 48, 64, 65, 80, 100, 128};

    // ========================================
    // Basic correctness tests: C = A * B
    // ========================================

    @Test
    void testNN_Small() {
        for (int m : SMALL_SIZES) {
            for (int n : SMALL_SIZES) {
                for (int k : SMALL_SIZES) {
                    testNN(m, n, k);
                }
            }
        }
    }

    @Test
    void testNN_Medium() {
        for (int m : MEDIUM_SIZES) {
            for (int n : MEDIUM_SIZES) {
                for (int k : MEDIUM_SIZES) {
                    testNN(m, n, k);
                }
            }
        }
    }

    @Test
    void testNN_Large() {
        for (int m : LARGE_SIZES) {
            for (int n : LARGE_SIZES) {
                for (int k : LARGE_SIZES) {
                    testNN(m, n, k);
                }
            }
        }
    }

    private void testNN(int m, int n, int k) {
        double[] A = createRandomMatrix(m, k);
        double[] B = createRandomMatrix(k, n);
        double[] C = new double[m * n];
        
        double[] C_expected = computeExpectedNN(A, m, k, B, n);
        
        Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m, n, k, 1.0, A, 0, k, B, 0, n, 0.0, C, 0, n);
        
        assertThat(maxDiff(C, C_expected)).isLessThan(1e-10)
            .as("NN: m=%d, n=%d, k=%d", m, n, k);
    }

    // ========================================
    // Transpose tests: C = A^T * B
    // ========================================

    @Test
    void testTN() {
        for (int m : ALL_SIZES) {
            for (int n : ALL_SIZES) {
                for (int k : SMALL_SIZES) {
                    testTN(m, n, k);
                }
            }
        }
    }

    private void testTN(int m, int n, int k) {
        // For C = A^T * B:
        // - A is k x m (transpose, so stored as k rows, m cols)
        // - B is k x n (no transpose, stored as k rows, n cols)
        double[] A = createRandomMatrix(k, m);  // stored as k rows, m cols
        double[] B = createRandomMatrix(k, n);  // stored as k rows, n cols
        double[] C = new double[m * n];
        
        // Expected: C[i,j] = sum_l A[l,i] * B[l,j]
        // Since A is stored as A[l*m + i] (k rows, m cols)
        // And B is stored as B[l*n + j] (k rows, n cols)
        double[] C_expected = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[l * m + i] * B[l * n + j];
                }
                C_expected[i * n + j] = sum;
            }
        }
        
        // DGEMM: C = A^T * B
        // - A: k x m (transposed to m x k), lda = m
        // - B: k x n, ldb = n
        Dgemm.dgemm(BLAS.Transpose.Trans, BLAS.Transpose.NoTrans, m, n, k, 1.0, A, 0, m, B, 0, n, 0.0, C, 0, n);
        
        assertThat(maxDiff(C, C_expected)).isLessThan(1e-10)
            .as("TN: m=%d, n=%d, k=%d", m, n, k);
    }

    // ========================================
    // Transpose tests: C = A * B^T
    // ========================================

    @Test
    void testNT() {
        for (int m : ALL_SIZES) {
            for (int n : ALL_SIZES) {
                for (int k : SMALL_SIZES) {
                    testNT(m, n, k);
                }
            }
        }
    }

    private void testNT(int m, int n, int k) {
        // For C = A * B^T:
        // - A is m x k (no transpose, so stored as m rows, k cols)
        // - B^T is k x n, meaning B is stored as n rows, k cols (each row has k elements)
        double[] A = createRandomMatrix(m, k);
        double[] B = createRandomMatrix(n, k);  // stored as n rows, k cols
        double[] C = new double[m * n];
        
        // Expected: C[i,j] = sum_l A[i,l] * B[j,l]
        // Since B is stored as B[row*k + col], and we're treating it as B^T:
        // B[j,l] = B[l*n + j] would be for B as k x n
        // But B is stored as n x k, so B[j,l] = B[j*k + l]
        double[] C_expected = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[i * k + l] * B[j * k + l];
                }
                C_expected[i * n + j] = sum;
            }
        }
        
        // DGEMM: C = A * B^T
        // - A: m x k, lda = k
        // - B: n x k (but treated as k x n for transpose), ldb = k
        Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.Trans, m, n, k, 1.0, A, 0, k, B, 0, k, 0.0, C, 0, n);
        
        assertThat(maxDiff(C, C_expected)).isLessThan(1e-10)
            .as("NT: m=%d, n=%d, k=%d", m, n, k);
    }

    // ========================================
    // Transpose tests: C = A^T * B^T
    // ========================================

    @Test
    void testTT() {
        for (int m : SMALL_SIZES) {
            for (int n : SMALL_SIZES) {
                for (int k : SMALL_SIZES) {
                    testTT(m, n, k);
                }
            }
        }
    }

    private void testTT(int m, int n, int k) {
        // A is k x m in row-major, but we treat it as A^T which is m x k
        double[] A = createRandomMatrix(k, m);
        // B is n x k in row-major, but we treat it as B^T which is k x n
        double[] B = createRandomMatrix(n, k);
        double[] C = new double[m * n];
        
        // Expected: C[i,j] = sum_l A[l,i] * B[j,l]
        double[] C_expected = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[l * m + i] * B[j * k + l];
                }
                C_expected[i * n + j] = sum;
            }
        }
        
        Dgemm.dgemm(BLAS.Transpose.Trans, BLAS.Transpose.Trans, m, n, k, 1.0, A, 0, m, B, 0, k, 0.0, C, 0, n);
        
        assertThat(maxDiff(C, C_expected)).isLessThan(1e-10)
            .as("TT: m=%d, n=%d, k=%d", m, n, k);
    }

    // ========================================
    // Alpha = 0 tests
    // ========================================

    @Test
    void testAlphaZero() {
        for (int m : SMALL_SIZES) {
            for (int n : SMALL_SIZES) {
                double[] A = createRandomMatrix(m, 5);
                double[] B = createRandomMatrix(5, n);
                double[] C = createRandomMatrix(m, n);
                double[] C_original = C.clone();
                
                // With alpha=0, C should remain unchanged regardless of A and B
                Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m, n, 5, 0.0, A, 0, 5, B, 0, n, 1.0, C, 0, n);
                
                assertThat(maxDiff(C, C_original)).isLessThan(1e-10)
                    .as("Alpha=0: m=%d, n=%d", m, n);
            }
        }
    }

    // ========================================
    // Beta = 0 tests (overwrite C)
    // ========================================

    @Test
    void testBetaZero() {
        for (int m : SMALL_SIZES) {
            for (int n : SMALL_SIZES) {
                for (int k : SMALL_SIZES) {
                    double[] A = createRandomMatrix(m, k);
                    double[] B = createRandomMatrix(k, n);
                    double[] C = createRandomMatrix(m, n);  // should be ignored
                    double[] C_expected = computeExpectedNN(A, m, k, B, n);
                    
                    Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m, n, k, 1.0, A, 0, k, B, 0, n, 0.0, C, 0, n);
                    
                    assertThat(maxDiff(C, C_expected)).isLessThan(1e-10)
                        .as("Beta=0: m=%d, n=%d, k=%d", m, n, k);
                }
            }
        }
    }

    // ========================================
    // Beta = 1 tests (accumulate)
    // ========================================

    @Test
    void testBetaOne() {
        for (int m : SMALL_SIZES) {
            for (int n : SMALL_SIZES) {
                for (int k : SMALL_SIZES) {
                    double[] A = createRandomMatrix(m, k);
                    double[] B = createRandomMatrix(k, n);
                    double[] C = createRandomMatrix(m, n);
                    double[] C_original = C.clone();
                    
                    double[] C_expected = computeExpectedNN(A, m, k, B, n);
                    // Add original C
                    for (int i = 0; i < m * n; i++) {
                        C_expected[i] += C_original[i];
                    }
                    
                    Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m, n, k, 1.0, A, 0, k, B, 0, n, 1.0, C, 0, n);
                    
                    assertThat(maxDiff(C, C_expected)).isLessThan(1e-10)
                        .as("Beta=1: m=%d, n=%d, k=%d", m, n, k);
                }
            }
        }
    }

    // ========================================
    // Negative alpha tests
    // ========================================

    @Test
    void testNegativeAlpha() {
        for (int m : SMALL_SIZES) {
            for (int n : SMALL_SIZES) {
                for (int k : SMALL_SIZES) {
                    double[] A = createRandomMatrix(m, k);
                    double[] B = createRandomMatrix(k, n);
                    double[] C = new double[m * n];
                    
                    double[] C_expected = computeExpectedNN(A, m, k, B, n);
                    for (int i = 0; i < m * n; i++) {
                        C_expected[i] *= -2.0;
                    }
                    
                    Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m, n, k, -2.0, A, 0, k, B, 0, n, 0.0, C, 0, n);
                    
                    assertThat(maxDiff(C, C_expected)).isLessThan(1e-10)
                        .as("Negative alpha: m=%d, n=%d, k=%d", m, n, k);
                }
            }
        }
    }

    // ========================================
    // Zero-sized matrix tests
    // ========================================

    @Test
    void testZeroM() {
        double[] A = createRandomMatrix(0, 5);
        double[] B = createRandomMatrix(5, 3);
        double[] C = createRandomMatrix(0, 3);
        
        Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, 0, 3, 5, 1.0, A, 0, 5, B, 0, 3, 0.0, C, 0, 3);
        // Should not throw
    }

    @Test
    void testZeroN() {
        double[] A = createRandomMatrix(4, 5);
        double[] B = createRandomMatrix(5, 0);
        double[] C = createRandomMatrix(4, 0);
        
        Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, 4, 0, 5, 1.0, A, 0, 5, B, 0, 0, 0.0, C, 0, 0);
        // Should not throw
    }

    @Test
    void testZeroK() {
        double[] A = createRandomMatrix(4, 0);
        double[] B = createRandomMatrix(0, 3);
        double[] C = new double[12];
        
        Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, 4, 3, 0, 1.0, A, 0, 0, B, 0, 0, 0.0, C, 0, 3);
        // With k=0, C should remain 0 (since alpha=0 effectively)
    }

    // ========================================
    // Offset tests (non-zero array offsets)
    // ========================================

    @Test
    void testWithOffsets() {
        for (int m : new int[]{3, 5, 8}) {
            for (int n : new int[]{3, 5, 8}) {
                for (int k : new int[]{3, 5, 8}) {
                    // Use leading dimension large enough for offsets
                    int lda = m + 10;
                    int ldb = k + 10;
                    int ldc = n + 10;
                    int offset = 5;
                    
                    // A has offset of (offset,offset) within larger matrix
                    // Need: (offset + m - 1) * lda + (offset + k - 1) + 1 elements
                    double[] A = new double[(offset + m) * lda + offset + k];
                    fillSubMatrix(A, offset, offset, m, k, lda);
                    
                    // B has offset of (offset,offset) within larger matrix  
                    double[] B = new double[(offset + k) * ldb + offset + n];
                    fillSubMatrix(B, offset, offset, k, n, ldb);
                    
                    // C has offset of (offset,offset) within larger matrix
                    double[] C = new double[(offset + m) * ldc + offset + n];
                    
                    // Compute expected
                    double[] C_expected = new double[m * n];
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                            double sum = 0.0;
                            for (int l = 0; l < k; l++) {
                                sum += A[(offset + i) * lda + (offset + l)] * B[(offset + l) * ldb + (offset + j)];
                            }
                            C_expected[i * n + j] = sum;
                        }
                    }
                    
                    Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m, n, k, 1.0, A, offset * lda + offset, lda, B, offset * ldb + offset, ldb, 0.0, C, offset * ldc + offset, ldc);
                    
                    // Verify the submatrix
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                            assertThat(Math.abs(C[(offset + i) * ldc + (offset + j)] - C_expected[i * n + j]))
                                .isLessThan(1e-10)
                                .as("Offset test: m=%d, n=%d, k=%d", m, n, k);
                        }
                    }
                }
            }
        }
    }

    // ========================================
    // Leading dimension tests
    // ========================================

    @Test
    void testNonStandardLeadingDimension() {
        for (int m : new int[]{3, 5, 8}) {
            for (int n : new int[]{3, 5, 8}) {
                for (int k : new int[]{3, 5, 8}) {
                    // Use larger leading dimensions than matrix dimensions
                    int lda = k + 5;
                    int ldb = n + 3;
                    int ldc = n + 2;
                    
                    double[] A = new double[m * lda];
                    fillSubMatrix(A, 0, 0, m, k, lda);
                    
                    double[] B = new double[k * ldb];
                    fillSubMatrix(B, 0, 0, k, n, ldb);
                    
                    double[] C = new double[m * ldc];
                    
                    double[] C_expected = new double[m * n];
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                            double sum = 0.0;
                            for (int l = 0; l < k; l++) {
                                sum += A[i * lda + l] * B[l * ldb + j];
                            }
                            C_expected[i * n + j] = sum;
                        }
                    }
                    
                    Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m, n, k, 1.0, A, 0, lda, B, 0, ldb, 0.0, C, 0, ldc);
                    
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                            assertThat(Math.abs(C[i * ldc + j] - C_expected[i * n + j]))
                                .isLessThan(1e-10)
                                .as("Leading dim test: m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d", 
                                    m, n, k, lda, ldb, ldc);
                        }
                    }
                }
            }
        }
    }

    // ========================================
    // Idempotence tests: C = 1*C + 0*A*B
    // ========================================

    @Test
    void testIdempotence() {
        for (int m : SMALL_SIZES) {
            for (int n : SMALL_SIZES) {
                double[] C = createRandomMatrix(m, n);
                double[] C_original = C.clone();
                
                // alpha=0, beta=1 should not change C
                Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m, n, 5, 0.0, 
                    new double[m*5], 0, 5, 
                    new double[5*n], 0, n, 
                    1.0, C, 0, n);
                
                assertThat(maxDiff(C, C_original)).isLessThan(1e-10)
                    .as("Idempotence: m=%d, n=%d", m, n);
            }
        }
    }

    // ========================================
    // Combined alpha and beta tests
    // ========================================

    @Test
    void testAlphaBetaCombination() {
        double[][] alphas = {{1.0}, {2.0}, {-1.0}, {0.5}, {-0.5}, {3.0}};
        double[][] betas = {{0.0}, {1.0}, {2.0}, {0.5}, {-1.0}};
        
        for (double[] alphaArr : alphas) {
            for (double[] betaArr : betas) {
                double alpha = alphaArr[0];
                double beta = betaArr[0];
                
                for (int m : new int[]{4, 8}) {
                    for (int n : new int[]{4, 8}) {
                        for (int k : new int[]{4, 8}) {
                            double[] A = createRandomMatrix(m, k);
                            double[] B = createRandomMatrix(k, n);
                            double[] C = createRandomMatrix(m, n);
                            double[] C_original = C.clone();
                            
                            // C_expected = alpha * A * B + beta * C_original
                            double[] C_expected = computeExpectedNN(A, m, k, B, n);
                            for (int i = 0; i < m * n; i++) {
                                C_expected[i] = alpha * C_expected[i] + beta * C_original[i];
                            }
                            
                            Dgemm.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, m, n, k, alpha, A, 0, k, B, 0, n, beta, C, 0, n);
                            
                            assertThat(maxDiff(C, C_expected)).isLessThan(1e-10)
                                .as("Alpha=%.1f, beta=%.1f: m=%d, n=%d, k=%d", 
                                    alpha, beta, m, n, k);
                        }
                    }
                }
            }
        }
    }

    // ========================================
    // Large matrix test with blocking
    // ========================================

    @Test
    void testLargeMatrices() {
        // Test matrices that trigger blocked algorithm
        int[] sizes = {100, 128, 200, 256};
        
        for (int m : sizes) {
            for (int n : sizes) {
                for (int k : sizes) {
                    testNN(m, n, k);
                }
            }
        }
    }

    // ========================================
    // Helper methods
    // ========================================

    private double[] createRandomMatrix(int rows, int cols) {
        if (rows <= 0 || cols <= 0) return new double[0];
        double[] M = new double[rows * cols];
        for (int i = 0; i < M.length; i++) {
            M[i] = (rand.nextDouble() - 0.5) * 2;  // [-1, 1]
        }
        return M;
    }

    private void fillSubMatrix(double[] A, int rowOff, int colOff, int rows, int cols, int lda) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                A[(rowOff + i) * lda + (colOff + j)] = (rand.nextDouble() - 0.5) * 2;
            }
        }
    }

    private double[] computeExpectedNN(double[] A, int m, int k, double[] B, int n) {
        double[] C = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[i * k + l] * B[l * n + j];
                }
                C[i * n + j] = sum;
            }
        }
        return C;
    }

    private double maxDiff(double[] A, double[] B) {
        if (A.length != B.length) return Double.MAX_VALUE;
        double max = 0.0;
        for (int i = 0; i < A.length; i++) {
            max = Math.max(max, Math.abs(A[i] - B[i]));
        }
        return max;
    }
}
