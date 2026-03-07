/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class QRTest {

    private static final double EPSILON = 1e-10;

    @Test
    void testDecomposeSquareMatrix() {
        double[] A = {
            4.0, 3.0,
            6.0, 3.0
        };
        int m = 2, n = 2;

        QR qr = QR.decompose(A, m, n);

        assertThat(qr.ok()).isTrue();
        assertThat(A[1 * n + 0]).isNotEqualTo(0.0);
        assertThat(A[0 * n + 1]).isNotEqualTo(0.0);
    }

    @Test
    void testLeastSquaresOverdetermined() {
        double[] A = {
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0
        };
        double[] b = {2.0, 3.0, 5.0};
        int m = 3, n = 2;
        double[] x = new double[n];

        QR qr = QR.decompose(A, m, n);
        assertThat(qr.ok()).isTrue();

        assertThat(qr.leastSquares(b, x)).isNotNull();
        assertThat(x[0]).isCloseTo(1.0/3.0, org.assertj.core.data.Offset.offset(1e-10));
        assertThat(x[1]).isCloseTo(1.5, org.assertj.core.data.Offset.offset(1e-10));
    }

    @Test
    void testSolveSquareSystem() {
        double[] A = {
            2.0, 1.0,
            1.0, 3.0
        };
        double[] b = {5.0, 8.0};
        int n = 2;

        QR qr = QR.decompose(A, n, n);
        assertThat(qr.ok()).isTrue();

        double[] x = qr.solve(b, null);
        assertThat(x).isNotNull();
        assertThat(x[0]).isCloseTo(1.4, org.assertj.core.data.Offset.offset(1e-10));
        assertThat(x[1]).isCloseTo(2.2, org.assertj.core.data.Offset.offset(1e-10));
    }

    @Test
    void testExtractR() {
        double[] A = {
            3.0, 2.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 2.0
        };
        int m = 3, n = 3;

        QR qr = QR.decompose(A, m, n);
        assertThat(qr.ok()).isTrue();

        double[] R = qr.toR().data;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                assertThat(R[i * n + j]).isCloseTo(0.0, org.assertj.core.data.Offset.offset(EPSILON));
            }
        }
    }

    @Test
    void testInverseSquareMatrix() {
        double[] A = {
            4.0, 7.0,
            2.0, 6.0
        };
        double[] Aorig = A.clone();
        int n = 2;

        QR qr = QR.decompose(A, n, n);
        assertThat(qr.ok()).isTrue();

        double[] Ainv = qr.inverse(null);
        assertThat(Ainv).isNotNull();

        double[] product = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    product[i * n + j] += Aorig[i * n + k] * Ainv[k * n + j];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(product[i * n + j]).isCloseTo(expected, org.assertj.core.data.Offset.offset(1e-10));
            }
        }
    }

    @Test
    void testInvalidDimensions() {
        double[] A = new double[6];

        try {
            QR qr = QR.decompose(A, 2, 3);
        } catch (IllegalArgumentException ex) {
            assertThat(ex.getMessage()).doesNotContain("m must be >= n");
        }
    }

    @Test
    void testNullArguments() {
        assertThatThrownBy(() -> QR.decompose(null, 2, 2))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Matrix A must have length");

        assertThatThrownBy(() -> QR.decompose(new double[4], 0, 2))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Dimensions must be positive");
    }

    @Test
    void testQRDecompositionReconstruction() {
        int[] sizes = {2, 3, 5, 10, 20};
        java.util.Random rand = new java.util.Random(42);

        for (int n : sizes) {
            double[] A = new double[n * n];
            for (int i = 0; i < n * n; i++) {
                A[i] = rand.nextDouble() * 2 - 1;
            }
            double[] Aorig = A.clone();

            QR qr = QR.decompose(A, n, n);
            assertThat(qr.ok()).isTrue();

            double[] R = qr.toR().data;

            double[] Q = qr.toQ().data;

            double[] QR = new double[n * n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n; k++) {
                        QR[i * n + j] += Q[i * n + k] * R[k * n + j];
                    }
                }
            }

            for (int i = 0; i < n * n; i++) {
                assertThat(QR[i]).isCloseTo(Aorig[i], org.assertj.core.data.Offset.offset(1e-8 * n));
            }
        }
    }

    @Test
    void testQIsOrthogonal() {
        int[] sizes = {2, 3, 5, 10};
        java.util.Random rand = new java.util.Random(42);

        for (int n : sizes) {
            double[] A = new double[n * n];
            for (int i = 0; i < n * n; i++) {
                A[i] = rand.nextDouble() * 2 - 1;
            }

            QR qr = QR.decompose(A, n, n);
            assertThat(qr.ok()).isTrue();

            double[] Q = qr.toQ().data;

            double[] QtQ = new double[n * n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n; k++) {
                        QtQ[i * n + j] += Q[k * n + i] * Q[k * n + j];
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double expected = (i == j) ? 1.0 : 0.0;
                    assertThat(QtQ[i * n + j]).isCloseTo(expected, org.assertj.core.data.Offset.offset(1e-8));
                }
            }
        }
    }

    @Test
    void testRectangularMatrix() {
        java.util.Random rand = new java.util.Random(42);

        int[][] dimensions = {{5, 3}, {3, 5}, {10, 4}, {4, 10}};
        for (int[] dim : dimensions) {
            int m = dim[0], n = dim[1];
            double[] A = new double[m * n];
            for (int i = 0; i < m * n; i++) {
                A[i] = rand.nextDouble() * 2 - 1;
            }

            QR qr = QR.decompose(A, m, n);
            assertThat(qr.ok()).isTrue();
        }
    }

    @Test
    void testSolveMultipleRHS() {
        double[] A = {
            2.0, 1.0,
            1.0, 3.0
        };
        double[] B = {5.0, 7.0, 8.0, 11.0};
        int n = 2, nrhs = 2;

        QR qr = QR.decompose(A, n, n);
        assertThat(qr.ok()).isTrue();

        assertThat(qr.solveMultiple(B, nrhs)).isNotNull();

        assertThat(B[0]).isCloseTo(1.4, org.assertj.core.data.Offset.offset(1e-10));
        assertThat(B[1]).isCloseTo(2.0, org.assertj.core.data.Offset.offset(1e-10));
        assertThat(B[2]).isCloseTo(2.2, org.assertj.core.data.Offset.offset(1e-10));
        assertThat(B[3]).isCloseTo(3.0, org.assertj.core.data.Offset.offset(1e-10));
    }

    @Test
    void testLeastSquaresResidual() {
        double[] A = {
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0
        };
        double[] Aorig = A.clone();
        double[] b = {2.0, 3.0, 5.0};
        int m = 3, n = 2;
        double[] x = new double[n];
        double[] bCopy = b.clone();

        QR qr = QR.decompose(A, m, n);
        assertThat(qr.ok()).isTrue();

        assertThat(qr.leastSquares(b, x)).isNotNull();

        assertThat(x[0]).isCloseTo(1.0/3.0, org.assertj.core.data.Offset.offset(1e-10));
        assertThat(x[1]).isCloseTo(1.5, org.assertj.core.data.Offset.offset(1e-10));

        double[] residual = new double[m];
        for (int i = 0; i < m; i++) {
            residual[i] = bCopy[i];
            for (int j = 0; j < n; j++) {
                residual[i] -= Aorig[i * n + j] * x[j];
            }
        }

        double residualNorm = 0;
        for (int i = 0; i < m; i++) {
            residualNorm += residual[i] * residual[i];
        }
        residualNorm = Math.sqrt(residualNorm);

        assertThat(residualNorm).isLessThan(0.5);
    }
}
