/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class SVDTest {

    private static final double EPSILON = 1e-8;

    @Test
    void testSingularValuesSquareMatrix() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        assertThat(s).hasSize(2);
        assertThat(s[0]).isGreaterThanOrEqualTo(s[1]);
        assertThat(s[0]).isCloseTo(5.464985704219043, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(s[1]).isCloseTo(0.365966190626258, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testSingularValuesRectangularWide() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        int m = 2, n = 3;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        assertThat(s).hasSize(2);
        assertThat(s[0]).isGreaterThanOrEqualTo(s[1]);
        assertThat(s[0]).isCloseTo(9.508032006094326, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(s[1]).isCloseTo(0.772869635673717, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testSingularValuesRectangularTall() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        };
        int m = 3, n = 2;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        assertThat(s).hasSize(2);
        assertThat(s[0]).isGreaterThanOrEqualTo(s[1]);
        assertThat(s[0]).isCloseTo(9.525518091565113, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(s[1]).isCloseTo(0.514300580657645, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testIdentityMatrix() {
        double[] A = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        };
        int n = 3;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        assertThat(s[0]).isCloseTo(1.0, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(s[1]).isCloseTo(1.0, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(s[2]).isCloseTo(1.0, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testDiagonalMatrix() {
        double[] A = {
            3.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 1.0
        };
        int n = 3;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        assertThat(s[0]).isCloseTo(3.0, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(s[1]).isCloseTo(2.0, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(s[2]).isCloseTo(1.0, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testSingularMatrix() {
        double[] A = {
            1.0, 2.0,
            2.0, 4.0
        };
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        assertThat(s[0]).isCloseTo(5.0, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(s[1]).isCloseTo(0.0, org.assertj.core.data.Offset.offset(1e-10));
    }

    @Test
    void testSmallMatrix1x1() {
        double[] A = {5.0};
        int n = 1;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        assertThat(s).hasSize(1);
        assertThat(s[0]).isCloseTo(5.0, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testRankComputation() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        int n = 3;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        double tol = 1e-10 * s[0];
        int rank = svd.rank(tol);

        assertThat(rank).isEqualTo(2);
    }

    @Test
    void testFullRankMatrix() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        double tol = 1e-10 * s[0];
        int rank = svd.rank(tol);

        assertThat(rank).isEqualTo(2);
    }

    @Test
    void testSingularValuesOnly() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        int m = 2, n = 3;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        assertThat(s).hasSize(2);
        assertThat(svd.U()).isNull();
        assertThat(svd.VT()).isNull();
    }

    @Test
    void testWellConditionedMatrix() {
        double[] A = {
            10.0, 0.0, 0.0,
            0.0, 5.0, 0.0,
            0.0, 0.0, 1.0
        };
        int n = 3;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        double cond = s[0] / s[2];
        assertThat(cond).isCloseTo(10.0, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testLargeMatrix() {
        int n = 50;
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            A[i * n + i] = i + 1;
            if (i > 0) {
                A[i * n + i - 1] = 0.5;
            }
            if (i < n - 1) {
                A[i * n + i + 1] = 0.5;
            }
        }

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        assertThat(s).hasSize(n);
        for (int i = 0; i < n - 1; i++) {
            assertThat(s[i]).isGreaterThanOrEqualTo(s[i + 1]);
        }
    }

    @Test
    void testReconstructionSquare() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_ALL, null);

        double[] s = svd.singularValues();
        double[] U = svd.U();
        double[] VT = svd.VT();

        assertThat(U).isNotNull();
        assertThat(VT).isNotNull();
        assertThat(U).hasSize(n * n);
        assertThat(VT).hasSize(n * n);

        double[] reconstructed = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += U[i * n + k] * s[k] * VT[k * n + j];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        for (int i = 0; i < n * n; i++) {
            assertThat(reconstructed[i]).isCloseTo(A[i], org.assertj.core.data.Offset.offset(EPSILON));
        }
    }

    @Test
    void testReconstructionWide() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        int m = 2, n = 3;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_ALL, null);

        double[] s = svd.singularValues();
        double[] U = svd.U();
        double[] VT = svd.VT();

        assertThat(U).hasSizeGreaterThanOrEqualTo(m * m);
        assertThat(VT).hasSizeGreaterThanOrEqualTo(m * n);

        double[] reconstructed = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < m; k++) {
                    sum += U[i * m + k] * s[k] * VT[k * n + j];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        for (int i = 0; i < m * n; i++) {
            assertThat(reconstructed[i]).isCloseTo(A[i], org.assertj.core.data.Offset.offset(EPSILON));
        }
    }

    @Test
    void testReconstructionTall() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        };
        int m = 3, n = 2;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_ALL, null);

        double[] s = svd.singularValues();
        double[] U = svd.U();
        double[] VT = svd.VT();

        assertThat(U).hasSizeGreaterThanOrEqualTo(m * n);
        assertThat(VT).hasSizeGreaterThanOrEqualTo(n * n);

        double[] reconstructed = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += U[i * n + k] * s[k] * VT[k * n + j];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        for (int i = 0; i < m * n; i++) {
            assertThat(reconstructed[i]).isCloseTo(A[i], org.assertj.core.data.Offset.offset(EPSILON));
        }
    }

    @Test
    void testOrthogonalityU() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        int n = 3;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_WANT_U, null);

        double[] U = svd.U();
        assertThat(U).isNotNull();

        double[] UtU = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += U[k * n + i] * U[k * n + j];
                }
                UtU[i * n + j] = sum;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(UtU[i * n + j]).isCloseTo(expected, org.assertj.core.data.Offset.offset(1e-10));
            }
        }
    }

    @Test
    void testOrthogonalityV() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        int n = 3;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_WANT_V, null);

        double[] VT = svd.VT();
        assertThat(VT).isNotNull();

        double[] VVt = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += VT[i * n + k] * VT[j * n + k];
                }
                VVt[i * n + j] = sum;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(VVt[i * n + j]).isCloseTo(expected, org.assertj.core.data.Offset.offset(1e-10));
            }
        }
    }

    @Test
    void testSolveSquareSystem() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        double[] b = {5.0, 6.0};
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_ALL, null);

        double[] x = new double[n];
        svd.solve(b, x);

        assertThat(x[0]).isCloseTo(-4.0, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(x[1]).isCloseTo(4.5, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testSolveLeastSquares() {
        double[] A = {
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0
        };
        double[] b = {1.0, 2.0, 3.0};
        int m = 3, n = 2;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_ALL, null);

        double[] x = new double[n];
        svd.solve(b, x);

        assertThat(x[0]).isCloseTo(0.0, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(x[1]).isCloseTo(1.0, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testSolveWithSingularMatrix() {
        double[] A = {
            1.0, 2.0,
            2.0, 4.0
        };
        double[] b = {3.0, 6.0};
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_ALL, null);

        double[] x = new double[n];
        svd.solve(b, x);

        double[] residual = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                residual[i] += A[i * n + j] * x[j];
            }
            residual[i] -= b[i];
        }

        double norm = 0;
        for (double r : residual) {
            norm += r * r;
        }
        norm = Math.sqrt(norm);
        assertThat(norm).isCloseTo(0.0, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testWorkspaceReuse() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int n = 2;
        SVD.Pool ws = new SVD.Pool();

        SVD svd1 = SVD.decompose(A.clone(), n, n, SVD.SVD_ALL, ws);
        double[] s1 = svd1.singularValues();

        SVD svd2 = SVD.decompose(A.clone(), n, n, SVD.SVD_ALL, ws);
        double[] s2 = svd2.singularValues();

        assertThat(s1[0]).isCloseTo(s2[0], org.assertj.core.data.Offset.offset(1e-15));
        assertThat(s1[1]).isCloseTo(s2[1], org.assertj.core.data.Offset.offset(1e-15));
    }

    @Test
    void testWorkspaceSize() {
        Decomposition.Workspace ws1 = SVD.workspace(3, 3, SVD.SVD_ALL);
        Decomposition.Workspace ws2 = SVD.workspace(2, 5, SVD.SVD_ALL);
        Decomposition.Workspace ws3 = SVD.workspace(5, 2, SVD.SVD_ALL);

        assertThat(ws1).isNotNull();
        assertThat(ws2).isNotNull();
        assertThat(ws3).isNotNull();
    }

    @Test
    void testSingularValuesOrdering() {
        int n = 10;
        double[] A = new double[n * n];
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < n * n; i++) {
            A[i] = rand.nextDouble();
        }

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        for (int i = 0; i < n - 1; i++) {
            assertThat(s[i]).isGreaterThanOrEqualTo(s[i + 1]);
        }
    }

    @Test
    void testZeroMatrix() {
        int n = 3;
        double[] A = new double[n * n];

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        for (int i = 0; i < n; i++) {
            assertThat(s[i]).isCloseTo(0.0, org.assertj.core.data.Offset.offset(1e-10));
        }
    }

    @Test
    void testConditionNumber() {
        double[] A = {
            10.0, 0.0, 0.0,
            0.0, 5.0, 0.0,
            0.0, 0.0, 1.0
        };
        int n = 3;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        double cond = s[0] / s[2];
        assertThat(cond).isCloseTo(10.0, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testConditionNumberIllConditioned() {
        int n = 5;
        java.util.Random rand = new java.util.Random(42);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            A[i * n + i] = Math.pow(10, i);
        }

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        double cond = s[0] / s[n - 1];
        assertThat(cond).isCloseTo(1e4, org.assertj.core.data.Offset.offset(1e-2));
    }

    @Test
    void testNorm2() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        double norm2 = s[0];
        assertThat(norm2).isCloseTo(5.464985704219043, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testFrobeniusNorm() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int n = 2;

        double expected = 0;
        for (double v : A) {
            expected += v * v;
        }
        expected = Math.sqrt(expected);

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        double frobNorm = 0;
        for (double sv : s) {
            frobNorm += sv * sv;
        }
        frobNorm = Math.sqrt(frobNorm);

        assertThat(frobNorm).isCloseTo(expected, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testPseudoInverse() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_ALL, null);

        double[] s = svd.singularValues();
        double[] U = svd.U();
        double[] VT = svd.VT();

        double[] Ainv = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    if (s[k] > 1e-10) {
                        sum += VT[k * n + i] * (1.0 / s[k]) * U[j * n + k];
                    }
                }
                Ainv[i * n + j] = sum;
            }
        }

        double[] I = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    I[i * n + j] += A[i * n + k] * Ainv[k * n + j];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(I[i * n + j]).isCloseTo(expected, org.assertj.core.data.Offset.offset(EPSILON));
            }
        }
    }

    @Test
    void testPseudoInverseSingular() {
        double[] A = {
            1.0, 2.0,
            2.0, 4.0
        };
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_ALL, null);

        double[] s = svd.singularValues();
        double[] U = svd.U();
        double[] VT = svd.VT();

        double[] Ainv = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    if (s[k] > 1e-10) {
                        sum += VT[k * n + i] * (1.0 / s[k]) * U[j * n + k];
                    }
                }
                Ainv[i * n + j] = sum;
            }
        }

        double[] AinvA = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    AinvA[i * n + j] += Ainv[i * n + k] * A[k * n + j];
                }
            }
        }

        assertThat(AinvA[0]).isCloseTo(0.2, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(AinvA[1]).isCloseTo(0.4, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(AinvA[2]).isCloseTo(0.4, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(AinvA[3]).isCloseTo(0.8, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testNullSpace() {
        double[] A = {
            1.0, 2.0,
            2.0, 4.0
        };
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_WANT_V, null);

        double[] s = svd.singularValues();
        double[] VT = svd.VT();

        double[] nullVector = new double[n];
        int nullIdx = -1;
        for (int i = 0; i < n; i++) {
            if (s[i] < 1e-10) {
                nullIdx = i;
                break;
            }
        }

        if (nullIdx >= 0) {
            for (int i = 0; i < n; i++) {
                nullVector[i] = VT[nullIdx * n + i];
            }

            double[] Anull = new double[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    Anull[i] += A[i * n + j] * nullVector[j];
                }
            }

            for (int i = 0; i < n; i++) {
                assertThat(Anull[i]).isCloseTo(0.0, org.assertj.core.data.Offset.offset(1e-10));
            }
        }
    }

    @Test
    void testDecomposeDefault() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n);

        assertThat(svd.U()).isNotNull();
        assertThat(svd.VT()).isNotNull();
        assertThat(svd.singularValues()).hasSize(2);
    }

    @Test
    void testLargeRectangularMatrix() {
        int m = 100, n = 50;
        double[] A = new double[m * n];
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < m * n; i++) {
            A[i] = rand.nextDouble();
        }

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        assertThat(s).hasSize(n);
        for (int i = 0; i < n - 1; i++) {
            assertThat(s[i]).isGreaterThanOrEqualTo(s[i + 1]);
        }
    }

    @Test
    void testEffectiveRank() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        int n = 3;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        double[] s = svd.singularValues();
        double tol = 1e-8 * s[0];
        int rank = svd.rank(tol);

        assertThat(rank).isEqualTo(2);
    }

    @Test
    void testDecomposeWithExternalArrays() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        int m = 2, n = 3;
        int minMN = Math.min(m, n);

        SVD.Pool ws = new SVD.Pool();

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_ALL, ws);

        double[] S = svd.singularValues();
        double[] U = svd.U();
        double[] VT = svd.VT();

        assertThat(S).isNotNull();
        assertThat(U).isNotNull();
        assertThat(VT).isNotNull();

        double[] reconstructed = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < minMN; k++) {
                    sum += U[i * minMN + k] * S[k] * VT[k * n + j];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        double[] original = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        for (int i = 0; i < m * n; i++) {
            assertThat(reconstructed[i]).isCloseTo(original[i], org.assertj.core.data.Offset.offset(EPSILON));
        }
    }

    @Test
    void testSolveWithExternalTmpArray() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        double[] b = {5.0, 6.0};
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_ALL, null);

        double[] x = new double[n];
        double[] tmp = new double[n];
        svd.solve(b, x, tmp);

        assertThat(x[0]).isCloseTo(-4.0, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(x[1]).isCloseTo(4.5, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testSolveRankDeficient() {
        double[] A = {
            -1.7854591879711257, -0.42687285925779594, -0.12730256811265162,
            -0.5728984211439724, -0.10093393134001777, -0.1181901192353067,
            1.2484316018707418, 0.5646683943038734, -0.48229492403243485,
            0.10174927665169475, -0.5805410929482445, 1.3054473231942054,
            -1.134174808195733, -0.4732430202414438, 0.3528489486370508
        };
        double[] b = {-2.3181340317357653, -0.7146777651358073, 1.8361340927945298, -0.35699930593018775, -1.6359508076249094};
        int m = 5, n = 3;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_ALL, null);

        double[] x = new double[n];
        svd.solve(b, x);

        assertThat(x[0]).isCloseTo(1.2120842180372118, org.assertj.core.data.Offset.offset(1e-6));
        assertThat(x[1]).isCloseTo(0.4154150318658529, org.assertj.core.data.Offset.offset(1e-6));
        assertThat(x[2]).isCloseTo(-0.1832034870198265, org.assertj.core.data.Offset.offset(1e-6));
    }

    @Test
    void testSolveZeroMatrix() {
        int m = 2, n = 2;
        double[] A = new double[m * n];
        double[] b = {3.0, 2.0};

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_ALL, null);

        double[] s = svd.singularValues();
        assertThat(s[0]).isCloseTo(0.0, org.assertj.core.data.Offset.offset(1e-10));
        assertThat(s[1]).isCloseTo(0.0, org.assertj.core.data.Offset.offset(1e-10));
    }

    @Test
    void testSolveWideMatrix() {
        double[] A = {
            0.8147, 0.9134, 0.5528,
            0.9058, 0.6324, 0.8723
        };
        double[] b = {0.278, 0.547};
        int m = 2, n = 3;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_ALL, null);

        double[] x = new double[n];
        svd.solve(b, x);

        double[] AtAx = new double[n];
        double[] Atb = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                Atb[i] += A[j * n + i] * b[j];
                for (int k = 0; k < n; k++) {
                    AtAx[i] += A[j * n + i] * A[j * n + k] * x[k];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            assertThat(AtAx[i]).isCloseTo(Atb[i], org.assertj.core.data.Offset.offset(1e-8));
        }
    }

    @Test
    void testSolveTallMatrix() {
        double[] A = {
            0.8147, 0.9134, 0.9,
            0.9058, 0.6324, 0.9,
            0.1270, 0.0975, 0.1,
            1.6, 2.8, -3.5
        };
        double[] b = {0.278, 0.547, -0.958, 1.452};
        int m = 4, n = 3;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_ALL, null);

        double[] x = new double[n];
        svd.solve(b, x);

        assertThat(x[0]).isCloseTo(0.820970340787782, org.assertj.core.data.Offset.offset(1e-6));
        assertThat(x[1]).isCloseTo(-0.218604626527306, org.assertj.core.data.Offset.offset(1e-6));
        assertThat(x[2]).isCloseTo(-0.212938815234215, org.assertj.core.data.Offset.offset(1e-6));
    }

    @Test
    void testRowsColsMethods() {
        int m = 5, n = 3;
        double[] A = new double[m * n];
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < A.length; i++) {
            A[i] = rand.nextDouble();
        }

        SVD svd = SVD.decompose(A.clone(), m, n);
        assertThat(svd.singularValues()).hasSize(Math.min(m, n));
        assertThat(svd.m()).isEqualTo(m);
        assertThat(svd.n()).isEqualTo(n);
        assertThat(svd.uCols()).isEqualTo(Math.min(m, n));
        assertThat(svd.vtRows()).isEqualTo(Math.min(m, n));
    }

    @Test
    void testOkMethod() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n);

        assertThat(svd.ok()).isTrue();
    }

    @Test
    void testDimensionMethods() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        int m = 2, n = 3;

        SVD svd = SVD.decompose(A.clone(), m, n);

        assertThat(svd.m()).isEqualTo(m);
        assertThat(svd.n()).isEqualTo(n);
    }

    @Test
    void testSVDKindNone() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int n = 2;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_NONE, null);

        assertThat(svd.ok()).isTrue();
        assertThat(svd.kind()).isEqualTo(SVD.SVD_NONE);
        assertThat(svd.U()).isNull();
        assertThat(svd.VT()).isNull();
        assertThat(svd.singularValues()).hasSize(2);
    }

    @Test
    void testSVDKindWantU() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        int m = 3, n = 3;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_WANT_U, null);

        assertThat(svd.ok()).isTrue();
        assertThat(svd.kind()).isEqualTo(SVD.SVD_WANT_U);
        assertThat(svd.U()).isNotNull();
        assertThat(svd.uCols()).isEqualTo(Math.min(m, n));
        assertThat(svd.U().length).isGreaterThanOrEqualTo(m * Math.min(m, n));
        assertThat(svd.VT()).isNull();
    }

    @Test
    void testSVDKindWantV() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        int m = 3, n = 3;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_WANT_V, null);

        assertThat(svd.ok()).isTrue();
        assertThat(svd.kind()).isEqualTo(SVD.SVD_WANT_V);
        assertThat(svd.U()).isNull();
        assertThat(svd.VT()).isNotNull();
        assertThat(svd.vtRows()).isEqualTo(Math.min(m, n));
        assertThat(svd.VT().length).isGreaterThanOrEqualTo(Math.min(m, n) * n);
    }

    @Test
    void testSVDKindAll() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        int m = 2, n = 3;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_ALL, null);

        assertThat(svd.ok()).isTrue();
        assertThat(svd.kind()).isEqualTo(SVD.SVD_ALL);
        assertThat(svd.U()).isNotNull();
        assertThat(svd.uCols()).isEqualTo(Math.min(m, n));
        assertThat(svd.U().length).isGreaterThanOrEqualTo(m * Math.min(m, n));
        assertThat(svd.VT()).isNotNull();
        assertThat(svd.vtRows()).isEqualTo(Math.min(m, n));
        assertThat(svd.VT().length).isGreaterThanOrEqualTo(Math.min(m, n) * n);
    }

    @Test
    void testSVDKindFullU() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        int m = 3, n = 3;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_WANT_U | SVD.SVD_FULL_U, null);

        assertThat(svd.ok()).isTrue();
        assertThat(svd.U()).isNotNull();
        assertThat(svd.uCols()).isEqualTo(m);
        assertThat(svd.U().length).isGreaterThanOrEqualTo(m * m);
        assertThat(svd.VT()).isNull();
    }

    @Test
    void testSVDKindFullV() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        int m = 2, n = 3;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_WANT_V | SVD.SVD_FULL_V, null);

        assertThat(svd.ok()).isTrue();
        assertThat(svd.U()).isNull();
        assertThat(svd.VT()).isNotNull();
        assertThat(svd.vtRows()).isEqualTo(n);
        assertThat(svd.VT().length).isGreaterThanOrEqualTo(n * n);
    }

    @Test
    void testSVDKindFullURectangularTall() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        };
        int m = 3, n = 2;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_WANT_U | SVD.SVD_FULL_U, null);

        assertThat(svd.ok()).isTrue();
        assertThat(svd.U()).isNotNull();
        assertThat(svd.uCols()).isEqualTo(m);
        assertThat(svd.U().length).isGreaterThanOrEqualTo(m * m);
    }

    @Test
    void testSVDKindFullVRectangularTall() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        };
        int m = 3, n = 2;

        SVD svd = SVD.decompose(A.clone(), m, n, SVD.SVD_WANT_V | SVD.SVD_FULL_V, null);

        assertThat(svd.ok()).isTrue();
        assertThat(svd.VT()).isNotNull();
        assertThat(svd.vtRows()).isEqualTo(n);
        assertThat(svd.VT().length).isGreaterThanOrEqualTo(n * n);
    }

    @Test
    void testSVDKindUOrthogonality() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        int n = 3;

        SVD svd = SVD.decompose(A.clone(), n, n, SVD.SVD_WANT_U | SVD.SVD_FULL_U, null);

        double[] U = svd.U();
        
        double[] UtU = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += U[k * n + i] * U[k * n + j];
                }
                UtU[i * n + j] = sum;
            }
        }

        for (int i = 0; i < n; i++) {
            assertThat(UtU[i * n + i]).isCloseTo(1.0, org.assertj.core.data.Offset.offset(EPSILON * 10));
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    assertThat(Math.abs(UtU[i * n + j])).isLessThan(EPSILON * 10);
                }
            }
        }
    }

}
