/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

public class CholeskyInverseTest {

    @Test
    void testDtrti2LowerSimple() {
        double[] L = {
            2, 0, 0,
            1, 3, 0,
            2, 1, 4
        };
        
        double[] L_expected_inv = {
            0.5, 0, 0,
            -1.0/6, 1.0/3, 0,
            -5.0/24, -1.0/12, 0.25
        };
        
        CholeskyTestAccess.dtrti2('L', 'N', 3, L, 0, 3);
        
        for (int i = 0; i < 9; i++) {
            assertThat(L[i]).isCloseTo(L_expected_inv[i], within(1e-6));
        }
    }

    @Test
    void testDtrti2UpperSimple() {
        double[] U = {
            2, 1, 2,
            0, 3, 1,
            0, 0, 4
        };
        
        double[] U_expected_inv = {
            0.5, -1.0/6, -5.0/24,
            0, 1.0/3, -1.0/12,
            0, 0, 0.25
        };
        
        CholeskyTestAccess.dtrti2('U', 'N', 3, U, 0, 3);
        
        for (int i = 0; i < 9; i++) {
            assertThat(U[i]).isCloseTo(U_expected_inv[i], within(1e-6));
        }
    }

    @Test
    void testDlauu2Lower() {
        double[] L = {
            2, 0, 0,
            1, 3, 0,
            2, 1, 4
        };
        
        CholeskyTestAccess.dlauu2('L', 3, L, 0, 3);
        
        assertThat(L[0]).isCloseTo(9.0, within(1e-6));
        assertThat(L[4]).isCloseTo(10.0, within(1e-6));
        assertThat(L[8]).isCloseTo(16.0, within(1e-6));
        assertThat(L[3]).isCloseTo(5.0, within(1e-6));
        assertThat(L[6]).isCloseTo(8.0, within(1e-6));
        assertThat(L[7]).isCloseTo(4.0, within(1e-6));
    }

    @Test
    void testFullInverseLower() {
        double[] A = {
            4, 1, 1,
            1, 5, 2,
            1, 2, 6
        };
        double[] A_orig = A.clone();
        
        Cholesky cholesky = Cholesky.decompose(A, 3, BLAS.Uplo.Lower);
        assertThat(cholesky.ok()).isTrue();
        
        double[] inv = cholesky.inverse(null);
        assertThat(inv).isNotNull();
        
        expandSymmetric(inv, 3, false);
        
        double[] I = multiply(A_orig, inv, 3);
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(I[i * 3 + j]).isCloseTo(expected, within(1e-6));
            }
        }
    }

    @Test
    void testFullInverseUpper() {
        double[] A = {
            4, 1, 1,
            1, 5, 2,
            1, 2, 6
        };
        double[] A_orig = A.clone();
        
        Cholesky cholesky = Cholesky.decompose(A, 3, BLAS.Uplo.Upper);
        assertThat(cholesky.ok()).isTrue();
        
        double[] inv = cholesky.inverse(null);
        assertThat(inv).isNotNull();
        
        expandSymmetric(inv, 3, true);
        
        double[] I = multiply(A_orig, inv, 3);
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(I[i * 3 + j]).isCloseTo(expected, within(1e-6));
            }
        }
    }

    private static void expandSymmetric(double[] A, int n, boolean upper) {
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

    private static double[] multiply(double[] A, double[] B, int n) {
        double[] C = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    C[i * n + j] += A[i * n + k] * B[k * n + j];
                }
            }
        }
        return C;
    }

    private static org.assertj.core.data.Offset<Double> within(double value) {
        return org.assertj.core.data.Offset.offset(value);
    }
}
