/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Comprehensive tests for Cholesky decomposition.
 */
public class CholeskyTest {

    private final Random rand = new Random(12345);

    private static final double EPSILON = 1e-10;

    private static double readCondition(Cholesky cholesky) {
        try {
            Field field = Cholesky.class.getDeclaredField("condition");
            field.setAccessible(true);
            return field.getDouble(cholesky);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(e);
        }
    }

    @Test
    void testDecomposeLower() {
        int[] sizes = {1, 2, 3, 4, 5, 8, 10, 16, 32, 64};
        for (int n : sizes) {
            double[] A = createRandomPositiveDefinite(n);
            double[] A_original = A.clone();

            Cholesky cholesky = Cholesky.decompose(A, n, BLAS.Uplo.Lower);
            assertThat(cholesky.ok()).isTrue().as("Decomposition should succeed for n=%d", n);

            double[] L = extractLower(A, n);
            double[] LLT = multiplyLLT(L, n);
            assertThat(maxDiff(LLT, A_original)).isLessThan(EPSILON)
                .as("A should equal L*L^T for n=%d", n);
        }
    }

    @Test
    void testDecomposeUpper() {
        int[] sizes = {1, 2, 3, 4, 5, 8, 10, 16, 32, 64};
        for (int n : sizes) {
            double[] A = createRandomPositiveDefinite(n);
            double[] A_original = A.clone();

            Cholesky cholesky = Cholesky.decompose(A, n, BLAS.Uplo.Upper);
            assertThat(cholesky.ok()).isTrue().as("Decomposition should succeed for n=%d", n);

            double[] U = extractUpper(A, n);
            double[] UTU = multiplyUTU(U, n);
            assertThat(maxDiff(UTU, A_original)).isLessThan(EPSILON)
                .as("A should equal U^T*U for n=%d", n);
        }
    }

    @Test
    void testLowerUpperConsistency() {
        int[] sizes = {3, 5, 8, 10};
        for (int n : sizes) {
            double[] A = createRandomPositiveDefinite(n);

            double[] A_lower = A.clone();
            Cholesky.decompose(A_lower, n, BLAS.Uplo.Lower);

            double[] A_upper = A.clone();
            Cholesky.decompose(A_upper, n, BLAS.Uplo.Upper);

            double[] L = extractLower(A_lower, n);
            double[] LLT = multiplyLLT(L, n);

            double[] U = extractUpper(A_upper, n);
            double[] UTU = multiplyUTU(U, n);

            assertThat(maxDiff(LLT, UTU)).isLessThan(EPSILON)
                .as("Lower and upper should produce same result for n=%d", n);
        }
    }

    @Test
    void testSolveLower() {
        int[] sizes = {1, 2, 3, 5, 8, 10, 16};
        for (int n : sizes) {
            double[] A = createRandomPositiveDefinite(n);
            double[] b = createRandomVector(n);
            double[] b_original = b.clone();
            double[] x_expected = new double[n];
            
            double[] A_inv = computeInverse(A, n);
            for (int i = 0; i < n; i++) {
                x_expected[i] = 0;
                for (int j = 0; j < n; j++) {
                    x_expected[i] += A_inv[i * n + j] * b_original[j];
                }
            }

            double[] x = Cholesky.decompose(A, n, BLAS.Uplo.Lower).solve(b, null);
            assertThat(x).isSameAs(b);

            assertThat(maxDiff(x, x_expected)).isLessThan(EPSILON * 100)
                .as("x should equal A^-1*b for n=%d", n);
        }
    }

    @Test
    void testSolveUpper() {
        int[] sizes = {1, 2, 3, 5, 8, 10, 16};
        for (int n : sizes) {
            double[] A = createRandomPositiveDefinite(n);
            double[] b = createRandomVector(n);
            double[] b_original = b.clone();
            double[] x_expected = new double[n];
            
            double[] A_inv = computeInverse(A, n);
            for (int i = 0; i < n; i++) {
                x_expected[i] = 0;
                for (int j = 0; j < n; j++) {
                    x_expected[i] += A_inv[i * n + j] * b_original[j];
                }
            }

            double[] x = Cholesky.decompose(A, n, BLAS.Uplo.Upper).solve(b, null);
            assertThat(x).isSameAs(b);

            assertThat(maxDiff(x, x_expected)).isLessThan(EPSILON * 1000)
                .as("x should equal A^-1*b for n=%d", n);
        }
    }

    @Test
    void testInverseLower() {
        int[] sizes = {1, 2, 3, 5, 8, 10, 16};
        for (int n : sizes) {
            double[] A = createRandomPositiveDefinite(n);
            double[] A_original = A.clone();

            double[] inv = Cholesky.decompose(A, n, BLAS.Uplo.Lower).inverse(null);
            expandSymmetric(inv, n, false);

            double[] I = multiply(inv, A_original, n);
            assertThat(maxDiff(I, identity(n))).isLessThan(EPSILON * 1000)
                .as("A*A^-1 should equal I for n=%d", n);
        }
    }

    @Test
    void testInverseUpper() {
        int[] sizes = {1, 2, 3, 5, 8, 10, 16};
        for (int n : sizes) {
            double[] A = createRandomPositiveDefinite(n);
            double[] A_original = A.clone();

            double[] inv = Cholesky.decompose(A, n, BLAS.Uplo.Upper).inverse(null);
            expandSymmetric(inv, n, true);

            double[] I = multiply(inv, A_original, n);
            assertThat(maxDiff(I, identity(n))).isLessThan(EPSILON * 1000)
                .as("A*A^-1 should equal I for n=%d", n);
        }
    }

    @Test
    void testDeterminant() {
        int[] sizes = {1, 2, 3, 5, 8, 10};
        for (int n : sizes) {
            double[] A = createRandomPositiveDefinite(n);
            double[] A_original = A.clone();

            Cholesky cholesky = Cholesky.decompose(A, n, BLAS.Uplo.Lower);

            double det = cholesky.determinant();
            double expected = computeDeterminant(A_original, n);

            assertThat(Math.abs(det - expected)).isLessThan(Math.abs(expected) * EPSILON)
                .as("Determinant should match for n=%d", n);
        }
    }

    @Test
    void testLogDet() {
        int[] sizes = {1, 2, 3, 5, 8, 10};
        for (int n : sizes) {
            double[] A = createRandomPositiveDefinite(n);

            Cholesky cholesky = Cholesky.decompose(A, n, BLAS.Uplo.Lower);

            double det = cholesky.determinant();
            double logDet = cholesky.logDet();

            assertThat(Math.abs(logDet - Math.log(det))).isLessThan(Math.abs(Math.log(det)) * EPSILON)
                .as("LogDet should match log(det) for n=%d", n);
        }
    }

    @Test
    void testLogDetNumericalStability() {
        int n = 5;
        double[] A = createRandomPositiveDefinite(n);
        for (int i = 0; i < n; i++) {
            A[i * n + i] *= 1e10;
        }
        double[] A_original = A.clone();

        Cholesky cholesky = Cholesky.decompose(A, n, BLAS.Uplo.Lower);
        assertThat(cholesky.ok()).isTrue();

        double logDet = cholesky.logDet();
        double expectedLogDet = Math.log(Math.abs(computeDeterminant(A_original, n)));

        assertThat(Math.abs(logDet - expectedLogDet)).isLessThan(Math.abs(expectedLogDet) * 1e-6)
            .as("LogDet should be numerically stable");
    }

    @Test
    void testCond() {
        int[] sizes = {3, 5, 10};
        for (int n : sizes) {
            double[] A = createRandomPositiveDefinite(n);

            Cholesky cholesky = Cholesky.decompose(A, n, BLAS.Uplo.Lower);
            assertThat(cholesky.ok()).isTrue();

            double cond = cholesky.cond();
            assertThat(cond).isGreaterThan(0);
            assertThat(cond).isGreaterThanOrEqualTo(1.0);
        }
    }

    @Test
    void testCondIsLazyInitialized() {
        double[] A = createRandomPositiveDefinite(4);

        Cholesky cholesky = Cholesky.decompose(A, 4, BLAS.Uplo.Lower);
        assertThat(cholesky.ok()).isTrue();
        assertThat(readCondition(cholesky)).isNaN();

        double cond = cholesky.cond();
        assertThat(cond).isGreaterThanOrEqualTo(1.0);
        assertThat(readCondition(cholesky)).isEqualTo(cond);
    }

    @Test
    void testCondIllConditioned() {
        int n = 5;
        java.util.Random rand = new java.util.Random(42);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = rand.nextDouble();
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double avg = (A[i * n + j] + A[j * n + i]) / 2;
                A[i * n + j] = avg;
                A[j * n + i] = avg;
            }
        }
        for (int i = 0; i < n; i++) {
            A[i * n + i] += n;
        }
        A[0] *= 1e10;

        Cholesky cholesky = Cholesky.decompose(A, n, BLAS.Uplo.Lower);
        assertThat(cholesky.ok()).isTrue();

        double cond = cholesky.cond();
        assertThat(cond).isGreaterThan(1e3);
    }

    @Test
    void testSingularMatrix() {
        double[] A = {4, 2, 2, 2, 1, 1, 2, 1, 1};

        Cholesky cholesky = Cholesky.decompose(A, 3, BLAS.Uplo.Lower);
        assertThat(cholesky.ok()).isFalse();
        assertThat(cholesky.cond()).isEqualTo(Double.POSITIVE_INFINITY);
        assertThatThrownBy(() -> cholesky.solve(new double[]{1, 1, 1}, null))
            .isInstanceOf(ArithmeticException.class);
        assertThatThrownBy(() -> cholesky.inverse(null))
            .isInstanceOf(ArithmeticException.class);
    }

    @Test
    void testDimensionOne() {
        double[] A = {4.0};
        double[] A_original = A.clone();

        Cholesky cholesky = Cholesky.decompose(A, 1, BLAS.Uplo.Lower);
        assertThat(cholesky.ok()).isTrue();
        assertThat(A[0]).isEqualTo(2.0);
        assertThat(maxDiff(multiplyLLT(A, 1), A_original)).isLessThan(EPSILON);
    }

    @Test
    void testWorkspaceReuse() {
        int n = 10;
        Cholesky.Pool ws = (Cholesky.Pool) Cholesky.workspace();
        
        for (int i = 0; i < 5; i++) {
            double[] A = createRandomPositiveDefinite(n);
            double[] A_original = A.clone();

            Cholesky cholesky = Cholesky.decompose(A, n, BLAS.Uplo.Lower, false, ws);
            assertThat(cholesky.ok()).isTrue();

            double[] L = extractLower(A, n);
            double[] LLT = multiplyLLT(L, n);
            assertThat(maxDiff(LLT, A_original)).isLessThan(EPSILON);
        }
    }

    private double[] createRandomPositiveDefinite(int n) {
        double[] A = new double[n * n];
        for (int i = 0; i < n * n; i++) {
            A[i] = (rand.nextDouble() - 0.5) * 2;
        }

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double avg = (A[i * n + j] + A[j * n + i]) / 2;
                A[i * n + j] = avg;
                A[j * n + i] = avg;
            }
        }

        for (int i = 0; i < n; i++) {
            A[i * n + i] += n;
        }

        return A;
    }

    private double[] createRandomVector(int n) {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            v[i] = (rand.nextDouble() - 0.5) * 2;
        }
        return v;
    }

    private double[] extractLower(double[] A, int n) {
        double[] L = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                L[i * n + j] = A[i * n + j];
            }
        }
        return L;
    }

    private double[] extractUpper(double[] A, int n) {
        double[] U = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                U[i * n + j] = A[i * n + j];
            }
        }
        return U;
    }

    private double[] multiplyLLT(double[] L, int n) {
        double[] result = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k <= Math.max(i, j); k++) {
                    sum += L[i * n + k] * L[j * n + k];
                }
                result[i * n + j] = sum;
            }
        }
        return result;
    }

    private double[] multiplyUTU(double[] U, int n) {
        double[] result = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k <= Math.min(i, j); k++) {
                    sum += U[k * n + i] * U[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }
        return result;
    }

    private double[] multiply(double[] A, double[] B, int n) {
        double[] result = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A[i * n + k] * B[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }
        return result;
    }

    private double[] identity(int n) {
        double[] I = new double[n * n];
        for (int i = 0; i < n; i++) {
            I[i * n + i] = 1.0;
        }
        return I;
    }

    private double computeDeterminant(double[] A, int n) {
        double det = 1.0;
        double[] temp = A.clone();

        for (int i = 0; i < n; i++) {
            int maxRow = i;
            double maxVal = Math.abs(temp[i * n + i]);
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(temp[k * n + i]) > maxVal) {
                    maxVal = Math.abs(temp[k * n + i]);
                    maxRow = k;
                }
            }

            if (maxVal < EPSILON) {
                return 0.0;
            }

            if (maxRow != i) {
                for (int j = i; j < n; j++) {
                    double tmp = temp[i * n + j];
                    temp[i * n + j] = temp[maxRow * n + j];
                    temp[maxRow * n + j] = tmp;
                }
                det = -det;
            }

            det *= temp[i * n + i];

            double pivot = temp[i * n + i];
            for (int k = i + 1; k < n; k++) {
                temp[k * n + i] /= pivot;
                for (int j = i + 1; j < n; j++) {
                    temp[k * n + j] -= temp[k * n + i] * temp[i * n + j];
                }
            }
        }

        return det;
    }

    private double maxDiff(double[] A, double[] B) {
        double max = 0.0;
        for (int i = 0; i < A.length; i++) {
            max = Math.max(max, Math.abs(A[i] - B[i]));
        }
        return max;
    }

    private void expandSymmetric(double[] A, int n, boolean upper) {
        if (upper) {
            for (int i = 1; i < n; i++) {
                for (int j = 0; j < i; j++) {
                    A[i * n + j] = A[j * n + i];
                }
            }
        } else {
            for (int i = 0; i < n - 1; i++) {
                for (int j = i + 1; j < n; j++) {
                    A[i * n + j] = A[j * n + i];
                }
            }
        }
    }
    
    private double[] computeInverse(double[] A, int n) {
        double[] inv = new double[n * n];
        
        double[] aug = new double[n * 2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aug[i * 2 * n + j] = A[i * n + j];
            }
            aug[i * 2 * n + n + i] = 1.0;
        }
        
        for (int i = 0; i < n; i++) {
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(aug[k * 2 * n + i]) > Math.abs(aug[maxRow * 2 * n + i])) {
                    maxRow = k;
                }
            }
            
            for (int j = 0; j < 2 * n; j++) {
                double tmp = aug[i * 2 * n + j];
                aug[i * 2 * n + j] = aug[maxRow * 2 * n + j];
                aug[maxRow * 2 * n + j] = tmp;
            }
            
            double pivot = aug[i * 2 * n + i];
            for (int j = 0; j < 2 * n; j++) {
                aug[i * 2 * n + j] /= pivot;
            }
            
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = aug[k * 2 * n + i];
                    for (int j = 0; j < 2 * n; j++) {
                        aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
                    }
                }
            }
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inv[i * n + j] = aug[i * 2 * n + n + j];
            }
        }
        
        return inv;
    }
}
