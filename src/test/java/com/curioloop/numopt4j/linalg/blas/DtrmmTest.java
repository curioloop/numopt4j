/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Comprehensive tests for DTRMM to verify all parameter combinations.
 * Tests: side (L/R), uplo (U/L), trans (N/T), diag (U/N)
 */
public class DtrmmTest {

    private final Random rand = new Random(12345);
    
    // Test sizes to verify
    private static final int[] SIZES = {2, 3, 4, 5, 7, 8, 10, 15, 16, 20, 31, 32, 48, 64};

    // ========================================
    // LEFT SIDE TESTS: B := alpha * A * B
    // ========================================

    // Left + No Transpose + Upper
    @Test
    void testLeftUpperNN_NonUnit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createUpperTriangular(m, false);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                // Expected: B[i,j] = sum_{k=i}^{m-1} A[i,k] * B[k,j]
                computeExpectedLeftNN(A, m, B_expected, n);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, n, 1.0, A, 0, m, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Left+Upper+NN non-unit: m=%d, n=%d", m, n);
            }
        }
    }

    @Test
    void testLeftUpperNN_Unit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createUpperTriangular(m, true);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedLeftNN_Unit(A, m, B_expected, n);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, n, 1.0, A, 0, m, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Left+Upper+NN unit: m=%d, n=%d", m, n);
            }
        }
    }

    // Left + No Transpose + Lower
    @Test
    void testLeftLowerNN_NonUnit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createLowerTriangular(m, false);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedLeftLowerNN(A, m, B_expected, n);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, n, 1.0, A, 0, m, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Left+Lower+NN non-unit: m=%d, n=%d", m, n);
            }
        }
    }

    @Test
    void testLeftLowerNN_Unit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createLowerTriangular(m, true);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedLeftLowerNN_Unit(A, m, B_expected, n);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, n, 1.0, A, 0, m, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Left+Lower+NN unit: m=%d, n=%d", m, n);
            }
        }
    }

    // Left + Transpose + Upper
    @Test
    void testLeftUpperTN_NonUnit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createUpperTriangular(m, false);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedLeftTN(A, m, B_expected, n);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, n, 1.0, A, 0, m, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Left+Upper+TN non-unit: m=%d, n=%d", m, n);
            }
        }
    }

    @Test
    void testLeftUpperTN_Unit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createUpperTriangular(m, true);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedLeftTN_Unit(A, m, B_expected, n);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit, m, n, 1.0, A, 0, m, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Left+Upper+TN unit: m=%d, n=%d", m, n);
            }
        }
    }

    // Left + Transpose + Lower
    @Test
    void testLeftLowerTN_NonUnit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createLowerTriangular(m, false);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedLeftLowerTN(A, m, B_expected, n);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, n, 1.0, A, 0, m, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Left+Lower+TN non-unit: m=%d, n=%d", m, n);
            }
        }
    }

    @Test
    void testLeftLowerTN_Unit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createLowerTriangular(m, true);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedLeftLowerTN_Unit(A, m, B_expected, n);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, m, n, 1.0, A, 0, m, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Left+Lower+TN unit: m=%d, n=%d", m, n);
            }
        }
    }

    // ========================================
    // RIGHT SIDE TESTS: B := alpha * B * A
    // ========================================

    // Right + No Transpose + Upper
    @Test
    void testRightUpperNN_NonUnit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createUpperTriangular(n, false);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedRightNN(A, n, B_expected, m);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, n, 1.0, A, 0, n, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Right+Upper+NN non-unit: m=%d, n=%d", m, n);
            }
        }
    }

    @Test
    void testRightUpperNN_Unit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createUpperTriangular(n, true);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedRightNN_Unit(A, n, B_expected, m);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, n, 1.0, A, 0, n, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Right+Upper+NN unit: m=%d, n=%d", m, n);
            }
        }
    }

    // Right + No Transpose + Lower
    @Test
    void testRightLowerNN_NonUnit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createLowerTriangular(n, false);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedRightLowerNN(A, n, B_expected, m);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, n, 1.0, A, 0, n, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Right+Lower+NN non-unit: m=%d, n=%d", m, n);
            }
        }
    }

    @Test
    void testRightLowerNN_Unit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createLowerTriangular(n, true);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedRightLowerNN_Unit(A, n, B_expected, m);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.Unit, m, n, 1.0, A, 0, n, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Right+Lower+NN unit: m=%d, n=%d", m, n);
            }
        }
    }

    // Right + Transpose + Upper
    @Test
    void testRightUpperTN_NonUnit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createUpperTriangular(n, false);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedRightTN(A, n, B_expected, m);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, n, 1.0, A, 0, n, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Right+Upper+TN non-unit: m=%d, n=%d", m, n);
            }
        }
    }

    @Test
    void testRightUpperTN_Unit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createUpperTriangular(n, true);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedRightTN_Unit(A, n, B_expected, m);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Right, BLAS.Uplo.Upper, BLAS.Trans.Trans, BLAS.Diag.Unit, m, n, 1.0, A, 0, n, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Right+Upper+TN unit: m=%d, n=%d", m, n);
            }
        }
    }

    // Right + Transpose + Lower
    @Test
    void testRightLowerTN_NonUnit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createLowerTriangular(n, false);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedRightLowerTN(A, n, B_expected, m);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit, m, n, 1.0, A, 0, n, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Right+Lower+TN non-unit: m=%d, n=%d", m, n);
            }
        }
    }

    @Test
    void testRightLowerTN_Unit() {
        for (int m : SIZES) {
            for (int n : SIZES) {
                double[] A = createLowerTriangular(n, true);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                computeExpectedRightLowerTN_Unit(A, n, B_expected, m);
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Right, BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.Unit, m, n, 1.0, A, 0, n, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Right+Lower+TN unit: m=%d, n=%d", m, n);
            }
        }
    }

    // ========================================
    // ALPHA TESTS
    // ========================================

    @Test
    void testAlphaMultiplier() {
        for (int m : new int[]{5, 10, 20}) {
            for (int n : new int[]{5, 10, 20}) {
                double alpha = rand.nextDouble() * 3 - 1.5; // random between -1.5 and 1.5
                double[] A = createUpperTriangular(m, false);
                double[] B = createRandomMatrix(m, n);
                double[] B_expected = B.clone();
                
                // Expected with alpha
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int k = i; k < m; k++) {
                            sum += A[i * m + k] * B_expected[k * n + j];
                        }
                        B_expected[i * n + j] = alpha * sum;
                    }
                }
                
                double[] B_result = B.clone();
                Dtrmm.dtrmm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit, m, n, alpha, A, 0, m, B_result, 0, n);
                
                assertThat(maxDiff(B_result, B_expected)).isLessThan(1e-10)
                    .as("Alpha test: m=%d, n=%d, alpha=%.3f", m, n, alpha);
            }
        }
    }

    // ========================================
    // Helper Methods
    // ========================================

    private double[] createUpperTriangular(int n, boolean unitDiag) {
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            A[i * n + i] = unitDiag ? 1.0 : rand.nextDouble() * 10 + 1;
            for (int j = i + 1; j < n; j++) {
                A[i * n + j] = rand.nextDouble() * 2 - 1;
            }
        }
        return A;
    }

    private double[] createLowerTriangular(int n, boolean unitDiag) {
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            A[i * n + i] = unitDiag ? 1.0 : rand.nextDouble() * 10 + 1;
            for (int j = 0; j < i; j++) {
                A[i * n + j] = rand.nextDouble() * 2 - 1;
            }
        }
        return A;
    }

    private double[] createRandomMatrix(int m, int n) {
        double[] A = new double[m * n];
        for (int i = 0; i < m * n; i++) {
            A[i] = rand.nextDouble() * 2 - 1;
        }
        return A;
    }

    // B := A * B (Left, NoTranspose, Upper)
    private void computeExpectedLeftNN(double[] A, int m, double[] B, int n) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = i; k < m; k++) {
                    sum += A[i * m + k] * Bcopy[k * n + j];
                }
                B[i * n + j] = sum;
            }
        }
    }

    private void computeExpectedLeftNN_Unit(double[] A, int m, double[] B, int n) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = Bcopy[i * n + j]; // diagonal is 1
                for (int k = i + 1; k < m; k++) {
                    sum += A[i * m + k] * Bcopy[k * n + j];
                }
                B[i * n + j] = sum;
            }
        }
    }

    // B := A * B (Left, NoTranspose, Lower)
    private void computeExpectedLeftLowerNN(double[] A, int m, double[] B, int n) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k <= i; k++) {
                    sum += A[i * m + k] * Bcopy[k * n + j];
                }
                B[i * n + j] = sum;
            }
        }
    }

    private void computeExpectedLeftLowerNN_Unit(double[] A, int m, double[] B, int n) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = Bcopy[i * n + j]; // diagonal is 1
                for (int k = 0; k < i; k++) {
                    sum += A[i * m + k] * Bcopy[k * n + j];
                }
                B[i * n + j] = sum;
            }
        }
    }

    // B := A^T * B (Left, Transpose, Upper)
    private void computeExpectedLeftTN(double[] A, int m, double[] B, int n) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = i; k < m; k++) {
                    sum += A[k * m + i] * Bcopy[k * n + j];
                }
                B[i * n + j] = sum;
            }
        }
    }

    private void computeExpectedLeftTN_Unit(double[] A, int m, double[] B, int n) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = Bcopy[i * n + j]; // diagonal is 1
                for (int k = i + 1; k < m; k++) {
                    sum += A[k * m + i] * Bcopy[k * n + j];
                }
                B[i * n + j] = sum;
            }
        }
    }

    // B := A^T * B (Left, Transpose, Lower)
    private void computeExpectedLeftLowerTN(double[] A, int m, double[] B, int n) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k <= i; k++) {
                    sum += A[k * m + i] * Bcopy[k * n + j];
                }
                B[i * n + j] = sum;
            }
        }
    }

    private void computeExpectedLeftLowerTN_Unit(double[] A, int m, double[] B, int n) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = Bcopy[i * n + j]; // diagonal is 1
                for (int k = 0; k < i; k++) {
                    sum += A[k * m + i] * Bcopy[k * n + j];
                }
                B[i * n + j] = sum;
            }
        }
    }

    // B := B * A (Right, NoTranspose, Upper)
    private void computeExpectedRightNN(double[] A, int n, double[] B, int m) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k <= j; k++) {
                    sum += Bcopy[i * n + k] * A[k * n + j];
                }
                B[i * n + j] = sum;
            }
        }
    }

    private void computeExpectedRightNN_Unit(double[] A, int n, double[] B, int m) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        // For Right+Upper+NN+Unit: result[i,j] = B[i,j] + sum_{k=0}^{j-1} B[i,k] * A[k,j]
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = Bcopy[i * n + j]; // diagonal is 1
                for (int k = 0; k < j; k++) {
                    sum += Bcopy[i * n + k] * A[k * n + j];
                }
                B[i * n + j] = sum;
            }
        }
    }

    // B := B * A (Right, NoTranspose, Lower)
    private void computeExpectedRightLowerNN(double[] A, int n, double[] B, int m) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = j; k < n; k++) {
                    sum += Bcopy[i * n + k] * A[k * n + j];
                }
                B[i * n + j] = sum;
            }
        }
    }

    private void computeExpectedRightLowerNN_Unit(double[] A, int n, double[] B, int m) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = Bcopy[i * n + j]; // diagonal is 1
                for (int k = j + 1; k < n; k++) {
                    sum += Bcopy[i * n + k] * A[k * n + j];
                }
                B[i * n + j] = sum;
            }
        }
    }

    // B := B * A^T (Right, Transpose, Upper)
    private void computeExpectedRightTN(double[] A, int n, double[] B, int m) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = j; k < n; k++) {
                    sum += Bcopy[i * n + k] * A[j * n + k];
                }
                B[i * n + j] = sum;
            }
        }
    }

    private void computeExpectedRightTN_Unit(double[] A, int n, double[] B, int m) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = Bcopy[i * n + j]; // diagonal is 1
                for (int k = j + 1; k < n; k++) {
                    sum += Bcopy[i * n + k] * A[j * n + k];
                }
                B[i * n + j] = sum;
            }
        }
    }

    // B := B * A^T (Right, Transpose, Lower)
    private void computeExpectedRightLowerTN(double[] A, int n, double[] B, int m) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k <= j; k++) {
                    sum += Bcopy[i * n + k] * A[j * n + k];
                }
                B[i * n + j] = sum;
            }
        }
    }

    private void computeExpectedRightLowerTN_Unit(double[] A, int n, double[] B, int m) {
        // Use a copy to avoid in-place modification issues
        double[] Bcopy = B.clone();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = Bcopy[i * n + j]; // diagonal is 1
                for (int k = 0; k < j; k++) {
                    sum += Bcopy[i * n + k] * A[j * n + k];
                }
                B[i * n + j] = sum;
            }
        }
    }

    private double maxDiff(double[] A, double[] B) {
        double max = 0.0;
        for (int i = 0; i < A.length; i++) {
            max = Math.max(max, Math.abs(A[i] - B[i]));
        }
        return max;
    }
}
