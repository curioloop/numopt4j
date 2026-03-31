/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.offset;

class LQTest {

    private static final double EPSILON = 1e-10;

    private static double readCondition(LQ lq) {
        try {
            Field field = LQ.class.getDeclaredField("condition");
            field.setAccessible(true);
            return field.getDouble(lq);
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(e);
        }
    }

    @Test
    void testBasicDecomposition() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        double[] original = A.clone();

        LQ lq = LQ.decompose(A, 2, 3);

        assertThat(lq.ok()).isTrue();

        Decomposition.Matrix Lm = lq.toL();
        Decomposition.Matrix Qm = lq.toQ();
        double[] L = Lm.data;
        double[] Q = Qm.data;

        assertThat(Lm.m).isEqualTo(2);
        assertThat(Lm.n).isEqualTo(3);
        assertThat(Qm.m).isEqualTo(3);
        assertThat(Qm.n).isEqualTo(3);

        double[] reconstructed = new double[6];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                double sum = 0;
                for (int k = 0; k < 3; k++) {
                    sum += L[i * 3 + k] * Q[k * 3 + j];
                }
                reconstructed[i * 3 + j] = sum;
            }
        }

        for (int i = 0; i < 6; i++) {
            assertThat(reconstructed[i]).isCloseTo(original[i], offset(EPSILON));
        }
    }

    @Test
    void testSolve() {
        double[] A = {
            2.0, 1.0, 1.0,
            1.0, 3.0, 2.0
        };
        double[] b = {4.0, 5.0};
        double[] original = b.clone();

        LQ lq = LQ.decompose(A, 2, 3);
        assertThat(lq.ok()).isTrue();

        double[] x = lq.solve(b, null);

        assertThat(2.0 * x[0] + 1.0 * x[1] + 1.0 * x[2]).isCloseTo(4.0, offset(EPSILON));
        assertThat(1.0 * x[0] + 3.0 * x[1] + 2.0 * x[2]).isCloseTo(5.0, offset(EPSILON));
        assertThat(Arrays.equals(b, original)).isFalse();
    }

    @Test
    void testSolveTranspose() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        double[] b = {7.0, 8.0, 9.0};
        double[] original = b.clone();

        LQ lq = LQ.decompose(A, 2, 3);
        assertThat(lq.ok()).isTrue();

        double[] x = lq.solveTranspose(b, null);

        assertThat(1.0 * x[0] + 4.0 * x[1]).isCloseTo(7.0, offset(EPSILON));
        assertThat(2.0 * x[0] + 5.0 * x[1]).isCloseTo(8.0, offset(EPSILON));
        assertThat(3.0 * x[0] + 6.0 * x[1]).isCloseTo(9.0, offset(EPSILON));
        assertThat(Arrays.equals(b, original)).isFalse();
    }

    @Test
    void testLeastSquares() {
        double[] A = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0
        };
        double[] b = {1.0, 2.0};
        double[] x = new double[3];

        LQ lq = LQ.decompose(A, 2, 3);
        assertThat(lq.ok()).isTrue();

        lq.leastSquares(b, x);

        assertThat(x[0]).isCloseTo(1.0, offset(EPSILON));
        assertThat(x[1]).isCloseTo(2.0, offset(EPSILON));
        assertThat(x[2]).isCloseTo(0.0, offset(EPSILON));
    }

    @Test
    void testOrthogonality() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        };

        LQ lq = LQ.decompose(A, 3, 4);
        assertThat(lq.ok()).isTrue();

        double[] Q = lq.toQ().data;

        double[] QtQ = new double[16];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                double sum = 0;
                for (int k = 0; k < 4; k++) {
                    sum += Q[k * 4 + i] * Q[k * 4 + j];
                }
                QtQ[i * 4 + j] = sum;
            }
        }

        for (int i = 0; i < 4; i++) {
            assertThat(QtQ[i * 4 + i]).isCloseTo(1.0, offset(EPSILON));
            for (int j = i + 1; j < 4; j++) {
                assertThat(QtQ[i * 4 + j]).isCloseTo(0.0, offset(EPSILON));
                assertThat(QtQ[j * 4 + i]).isCloseTo(0.0, offset(EPSILON));
            }
        }
    }

    @Test
    void testCond() {
        double[] A = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0
        };

        LQ lq = LQ.decompose(A, 2, 3);
        assertThat(lq.ok()).isTrue();

        double cond = lq.cond();
        assertThat(cond).isGreaterThanOrEqualTo(1.0);
        assertThat(cond).isLessThan(1.0 / EPSILON);
    }

    @Test
    void testCondIsLazyInitialized() {
        double[] A = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0
        };

        LQ lq = LQ.decompose(A, 2, 3);
        assertThat(lq.ok()).isTrue();
        assertThat(readCondition(lq)).isNaN();

        double cond = lq.cond();
        assertThat(cond).isGreaterThanOrEqualTo(1.0);
        assertThat(readCondition(lq)).isEqualTo(cond);
    }

    @Test
    void testSquareMatrix() {
        double[] A = {
            4.0, 1.0, 2.0,
            3.0, 5.0, 1.0,
            2.0, 3.0, 6.0
        };
        double[] original = A.clone();

        LQ lq = LQ.decompose(A, 3, 3);
        assertThat(lq.ok()).isTrue();

        Decomposition.Matrix Lm = lq.toL();
        Decomposition.Matrix Qm = lq.toQ();
        double[] L = Lm.data;
        double[] Q = Qm.data;

        double[] reconstructed = new double[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double sum = 0;
                for (int k = 0; k < 3; k++) {
                    sum += L[i * 3 + k] * Q[k * 3 + j];
                }
                reconstructed[i * 3 + j] = sum;
            }
        }

        for (int i = 0; i < 9; i++) {
            assertThat(reconstructed[i]).isCloseTo(original[i], offset(EPSILON));
        }
    }

    @Test
    void testSolveSquare() {
        double[] A = {
            2.0, 1.0, 1.0,
            1.0, 3.0, 2.0,
            1.0, 0.0, 2.0
        };
        double[] b = {4.0, 5.0, 6.0};

        LQ lq = LQ.decompose(A, 3, 3);
        assertThat(lq.ok()).isTrue();

        double[] x = lq.solve(b, null);

        assertThat(2.0 * x[0] + 1.0 * x[1] + 1.0 * x[2]).isCloseTo(4.0, offset(EPSILON));
        assertThat(1.0 * x[0] + 3.0 * x[1] + 2.0 * x[2]).isCloseTo(5.0, offset(EPSILON));
        assertThat(1.0 * x[0] + 0.0 * x[1] + 2.0 * x[2]).isCloseTo(6.0, offset(EPSILON));
    }

    @Test
    void testTallMatrixRejected() {
        assertThatThrownBy(() -> LQ.decompose(new double[6], 3, 2))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("m must be <= n");
    }
}
