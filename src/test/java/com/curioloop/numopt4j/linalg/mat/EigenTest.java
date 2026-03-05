/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;

public class EigenTest {

    private static final double EPSILON = 1e-10;
    private static final double LOOSE_EPSILON = 1e-8;
    private static final double STABILITY_EPSILON = 1e-6;

    private static boolean isComplexPair(double[] wr, double[] wi, int idx) {
        double eps = 1e-15;
        return idx < wr.length - 1 && Math.abs(wi[idx]) > 0 && Math.abs(wi[idx + 1]) > 0
               && Math.abs(wr[idx] - wr[idx + 1]) < eps * Math.max(Math.abs(wr[idx]), Math.abs(wr[idx + 1]))
               && Math.abs(wi[idx] + wi[idx + 1]) < eps * Math.abs(wi[idx]);
    }

    @Test
    public void testSymmetricIdentityMatrix() {
        int n = 3;
        double[] A = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvalues()).containsExactly(1.0, 1.0, 1.0);
    }

    @Test
    public void testSymmetricDiagonalMatrix() {
        int n = 4;
        double[] A = {
            4, 0, 0, 0,
            0, 3, 0, 0,
            0, 0, 2, 0,
            0, 0, 0, 1
        };

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvalues()[0]).isCloseTo(1.0, assertWithPrecision(EPSILON));
        assertThat(eigen.eigenvalues()[1]).isCloseTo(2.0, assertWithPrecision(EPSILON));
        assertThat(eigen.eigenvalues()[2]).isCloseTo(3.0, assertWithPrecision(EPSILON));
        assertThat(eigen.eigenvalues()[3]).isCloseTo(4.0, assertWithPrecision(EPSILON));
    }

    @Test
    public void testSymmetric2x2() {
        int n = 2;
        double[] A = {2, 1, 1, 2};

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();
        double min = Math.min(eigen.eigenvalues()[0], eigen.eigenvalues()[1]);
        double max = Math.max(eigen.eigenvalues()[0], eigen.eigenvalues()[1]);
        assertThat(min).isCloseTo(1.0, assertWithPrecision(EPSILON));
        assertThat(max).isCloseTo(3.0, assertWithPrecision(EPSILON));
    }

    @Test
    public void testSymmetric1x1() {
        int n = 1;
        double[] A = {5.0};

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvalues()[0]).isCloseTo(5.0, assertWithPrecision(EPSILON));
    }

    @Test
    public void testSymmetricEigenvectorResidual() {
        int n = 3;
        double[] A = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };
        double[] Aorig = A.clone();

        Eigen eigen = Eigen.symmetric(A, n, 'L');
        assertThat(eigen.ok()).isTrue();

        double[] eigenvalues = eigen.eigenvalues();
        double[] eigenvectors = eigen.eigenvectors();

        for (int j = 0; j < n; j++) {
            double lambda = eigenvalues[j];
            double[] v = new double[n];
            for (int i = 0; i < n; i++) {
                v[i] = eigenvectors[i * n + j];
            }

            double[] Av = new double[n];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    Av[i] += Aorig[i * n + k] * v[k];
                }
                Av[i] -= lambda * v[i];
            }

            double residual = norm2(Av, n);
            assertThat(residual).isLessThan(1e-8);
        }
    }

    @Test
    public void testSymmetricEigenvectorOrthogonality() {
        int n = 3;
        double[] A = {
            3, 1, 0,
            1, 3, 1,
            0, 1, 3
        };

        Eigen eigen = Eigen.symmetric(A, n, 'L');
        assertThat(eigen.ok()).isTrue();

        double[] eigenvectors = eigen.eigenvectors();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += eigenvectors[k * n + i] * eigenvectors[k * n + j];
                }
                if (i == j) {
                    assertThat(sum).isCloseTo(1.0, assertWithPrecision(1e-6));
                } else {
                    assertThat(Math.abs(sum)).isLessThan(1e-6);
                }
            }
        }
    }

    @Test
    public void testSymmetricNegativeEigenvalues() {
        int n = 2;
        double[] A = {-1, 2, 2, -1};

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();
        double min = Math.min(eigen.eigenvalues()[0], eigen.eigenvalues()[1]);
        double max = Math.max(eigen.eigenvalues()[0], eigen.eigenvalues()[1]);
        assertThat(min).isCloseTo(-3.0, assertWithPrecision(EPSILON));
        assertThat(max).isCloseTo(1.0, assertWithPrecision(EPSILON));
    }

    @Test
    public void testSymmetricSingularMatrix() {
        int n = 3;
        double[] A = {
            1, 2, 3,
            2, 4, 6,
            3, 6, 9
        };

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();
        int zeroCount = 0;
        for (double ev : eigen.eigenvalues()) {
            if (Math.abs(ev) < EPSILON) {
                zeroCount++;
            }
        }
        assertThat(zeroCount).isGreaterThanOrEqualTo(1);
    }

    @Test
    public void testSymmetricLargeMatrix() {
        int n = 10;
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            A[i * n + i] = i + 1;
        }

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();

        for (int i = 0; i < n - 1; i++) {
            assertThat(eigen.eigenvalues()[i]).isLessThanOrEqualTo(eigen.eigenvalues()[i + 1]);
        }

        for (int i = 0; i < n; i++) {
            assertThat(eigen.eigenvalues()[i]).isCloseTo(i + 1.0, assertWithPrecision(1e-8));
        }
    }

    @Test
    public void testGeneralDiagonalMatrix() {
        int n = 3;
        double[] A = {1, 0, 0, 0, 2, 0, 0, 0, 3};

        Eigen eigen = Eigen.general(A, n);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvalues()).containsExactly(1.0, 2.0, 3.0);
        assertThat(eigen.eigenvaluesImag()).containsExactly(0.0, 0.0, 0.0);
    }

    @Test
    public void testGeneralRotationMatrix() {
        int n = 2;
        double[] A = {0, -1, 1, 0};

        Eigen eigen = Eigen.general(A, n);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvalues()[0]).isCloseTo(0.0, assertWithPrecision(EPSILON));
        assertThat(eigen.eigenvalues()[1]).isCloseTo(0.0, assertWithPrecision(EPSILON));
        double absWi0 = Math.abs(eigen.eigenvaluesImag()[0]);
        double absWi1 = Math.abs(eigen.eigenvaluesImag()[1]);
        assertThat(absWi0).isCloseTo(1.0, assertWithPrecision(EPSILON));
        assertThat(absWi1).isCloseTo(1.0, assertWithPrecision(EPSILON));
    }

    @Test
    public void testGeneral1x1() {
        int n = 1;
        double[] A = {3.5};

        Eigen eigen = Eigen.general(A, n);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvalues()[0]).isCloseTo(3.5, assertWithPrecision(EPSILON));
        assertThat(eigen.eigenvaluesImag()[0]).isCloseTo(0.0, assertWithPrecision(EPSILON));
    }

    @Test
    public void testGeneralRealEigenvalues() {
        int n = 2;
        double[] A = {4, 2, 1, 3};

        Eigen eigen = Eigen.general(A, n);

        assertThat(eigen.ok()).isTrue();
        assertThat(Math.abs(eigen.eigenvaluesImag()[0])).isLessThan(EPSILON);
        assertThat(Math.abs(eigen.eigenvaluesImag()[1])).isLessThan(EPSILON);
    }

    @Test
    public void testGeneralComplexEigenvalues() {
        int n = 2;
        double[] A = {1, -1, 1, 1};

        Eigen eigen = Eigen.general(A, n);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvalues()[0]).isCloseTo(1.0, assertWithPrecision(EPSILON));
        assertThat(eigen.eigenvalues()[1]).isCloseTo(1.0, assertWithPrecision(EPSILON));
        assertThat(Math.abs(eigen.eigenvaluesImag()[0])).isCloseTo(1.0, assertWithPrecision(EPSILON));
        assertThat(Math.abs(eigen.eigenvaluesImag()[1])).isCloseTo(1.0, assertWithPrecision(EPSILON));
        assertThat(eigen.eigenvaluesImag()[0] + eigen.eigenvaluesImag()[1]).isCloseTo(0.0, assertWithPrecision(EPSILON));
    }

    @Test
    public void testGeneralJordanBlock() {
        int n = 2;
        double lambda = 5.0;
        double[] A = {lambda, 1, 0, lambda};

        Eigen eigen = Eigen.general(A, n);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvalues()[0]).isCloseTo(lambda, assertWithPrecision(EPSILON));
        assertThat(eigen.eigenvalues()[1]).isCloseTo(lambda, assertWithPrecision(EPSILON));
    }

    @Test
    public void testGeneralWithEigenvectors() {
        int n = 3;
        double[] A = {
            4, 2, 1,
            1, 3, 1,
            0, 1, 2
        };

        Eigen eigen = Eigen.general(A, n, false, true, null);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvectors()).isNotNull();
    }

    @Test
    public void testConditionNumber() {
        int n = 3;
        double[] A = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();
        double cond = eigen.cond();
        assertThat(cond).isCloseTo(3.0, assertWithPrecision(EPSILON));
    }

    @Test
    public void testSymmetricZeroMatrix() {
        int n = 3;
        double[] A = new double[n * n];

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();
        for (double ev : eigen.eigenvalues()) {
            assertThat(ev).isCloseTo(0.0, assertWithPrecision(EPSILON));
        }
    }

    @Test
    public void testSymmetricRandom() {
        int n = 4;
        double[] A = new double[n * n];
        java.util.Random rand = new java.util.Random(42);

        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double val = rand.nextDouble() * 2 - 1;
                A[i * n + j] = val;
                A[j * n + i] = val;
            }
            A[i * n + i] += n;
        }

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();
        for (double ev : eigen.eigenvalues()) {
            assertThat(ev).isGreaterThan(0);
        }
    }

    @Test
    public void testIsComplexPair() {
        double[] wr = {1.0, 1.0, 2.0};
        double[] wi = {1.0, -1.0, 0.0};

        assertThat(isComplexPair(wr, wi, 0)).isTrue();
        assertThat(isComplexPair(wr, wi, 1)).isFalse();
        assertThat(isComplexPair(wr, wi, 2)).isFalse();
    }

    @Test
    public void testSymmetricUpperStorage() {
        int n = 3;
        double[] A = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };

        Eigen eigen = Eigen.symmetric(A, n, 'U');

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvalues()[0]).isCloseTo(1.0, assertWithPrecision(EPSILON));
        assertThat(eigen.eigenvalues()[1]).isCloseTo(2.0, assertWithPrecision(EPSILON));
        assertThat(eigen.eigenvalues()[2]).isCloseTo(3.0, assertWithPrecision(EPSILON));
    }

    @Test
    public void testGeneralRightEigenvectorsExist() {
        int n = 3;
        double[] A = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };

        Eigen eigen = Eigen.general(A, n, false, true, null);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvectors()).isNotNull();
    }

    @Test
    public void testSingularMatrixConditionNumber() {
        int n = 3;
        double[] A = {
            1, 2, 3,
            2, 4, 6,
            3, 6, 9
        };

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();
        double cond = eigen.cond();
        assertThat(cond).isInfinite();
    }

    @Test
    public void testWorkspaceSizeSymmetric() {
        int n = 3;
        Decomposition.Workspace ws = Eigen.workspace(n, true, Eigen.EIGEN_RIGHT);
        assertThat(ws.work().length).isGreaterThanOrEqualTo(3 * n - 1);
    }

    @Test
    public void testWorkspaceSizeGeneral() {
        int n = 3;
        Decomposition.Workspace ws = Eigen.workspace(n, false, Eigen.EIGEN_NONE);
        assertThat(ws.work().length).isGreaterThanOrEqualTo(3 * n);
    }

    @Test
    public void testGeneralLargeMatrix() {
        int n = 20;
        double[] A = new double[n * n];
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < n * n; i++) {
            A[i] = rand.nextDouble() * 2 - 1;
        }
        for (int i = 0; i < n; i++) {
            A[i * n + i] += n;
        }

        Eigen eigen = Eigen.general(A, n, false, false, null);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvalues()).hasSize(n);
    }

    @Test
    public void testSymmetricEigenvalueSum() {
        int n = 3;
        double[] A = {
            2, 1, 0,
            1, 3, 1,
            0, 1, 2
        };
        double trace = A[0] + A[4] + A[8];

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();
        double sum = 0;
        for (double ev : eigen.eigenvalues()) {
            sum += ev;
        }
        assertThat(sum).isCloseTo(trace, assertWithPrecision(EPSILON));
    }

    @Test
    public void testSymmetricDeterminant() {
        int n = 3;
        double[] A = {
            2, 1, 0,
            1, 3, 1,
            0, 1, 2
        };

        Eigen eigen = Eigen.symmetric(A, n, 'L');

        assertThat(eigen.ok()).isTrue();
        double det = 1;
        for (double ev : eigen.eigenvalues()) {
            det *= ev;
        }
        assertThat(det).isCloseTo(8.0, assertWithPrecision(EPSILON));
    }

    @Test
    public void testGeneralRepeatedEigenvalues() {
        int n = 3;
        double[] A = {
            2, 0, 0,
            0, 2, 0,
            0, 0, 3
        };

        Eigen eigen = Eigen.general(A, n);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.eigenvalues()[0]).isCloseTo(2.0, assertWithPrecision(EPSILON));
        assertThat(eigen.eigenvalues()[1]).isCloseTo(2.0, assertWithPrecision(EPSILON));
        assertThat(eigen.eigenvalues()[2]).isCloseTo(3.0, assertWithPrecision(EPSILON));
    }

    private static org.assertj.core.data.Offset<Double> assertWithPrecision(double precision) {
        return org.assertj.core.data.Offset.offset(precision);
    }

    private double norm2(double[] v, int n) {
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += v[i] * v[i];
        }
        return Math.sqrt(sum);
    }

    @Test
    void testEigenKindNone() {
        int n = 2;
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };

        Eigen eigen = Eigen.general(A.clone(), n, Eigen.EIGEN_NONE, null);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.kind()).isEqualTo(Eigen.EIGEN_NONE);
        assertThat(eigen.eigenvectors()).isNull();
        assertThat(eigen.leftEigenvectors()).isNull();
        assertThat(eigen.eigenvalues()).hasSize(2);
    }

    @Test
    void testEigenKindLeft() {
        int n = 2;
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };

        Eigen eigen = Eigen.general(A.clone(), n, Eigen.EIGEN_LEFT, null);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.kind()).isEqualTo(Eigen.EIGEN_LEFT);
        assertThat(eigen.leftEigenvectors()).isNotNull();
        assertThat(eigen.leftEigenvectors().length).isEqualTo(n * n);
        assertThat(eigen.eigenvectors()).isNull();
    }

    @Test
    void testEigenKindRight() {
        int n = 2;
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };

        Eigen eigen = Eigen.general(A.clone(), n, Eigen.EIGEN_RIGHT, null);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.kind()).isEqualTo(Eigen.EIGEN_RIGHT);
        assertThat(eigen.eigenvectors()).isNotNull();
        assertThat(eigen.eigenvectors().length).isEqualTo(n * n);
        assertThat(eigen.leftEigenvectors()).isNull();
    }

    @Test
    void testEigenKindBoth() {
        int n = 2;
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };

        Eigen eigen = Eigen.general(A.clone(), n, Eigen.EIGEN_BOTH, null);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.kind()).isEqualTo(Eigen.EIGEN_BOTH);
        assertThat(eigen.eigenvectors()).isNotNull();
        assertThat(eigen.eigenvectors().length).isEqualTo(n * n);
        assertThat(eigen.leftEigenvectors()).isNotNull();
        assertThat(eigen.leftEigenvectors().length).isEqualTo(n * n);
    }

    @Test
    void testEigenKindSymmetric() {
        int n = 2;
        double[] A = {
            2.0, 1.0,
            1.0, 2.0
        };

        Eigen eigen = Eigen.symmetric(A.clone(), n, 'U');

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.kind()).isEqualTo(Eigen.EIGEN_RIGHT);
        assertThat(eigen.eigenvectors()).isNotNull();
        assertThat(eigen.isSymmetric()).isTrue();
    }

    @Test
    void testSymmetricValuesOnly() {
        int n = 3;
        double[] A = {
            4.0, 1.0, 1.0,
            1.0, 3.0, 0.0,
            1.0, 0.0, 2.0
        };

        Eigen eigen = Eigen.symmetricValues(A.clone(), n, 'L');

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.kind()).isEqualTo(Eigen.EIGEN_NONE);
        assertThat(eigen.eigenvalues()).hasSize(n);
        assertThat(eigen.eigenvectors()).isNull();

        double trace = 4.0 + 3.0 + 2.0;
        double sumEigenvalues = 0;
        for (int i = 0; i < n; i++) {
            sumEigenvalues += eigen.eigenvalues()[i];
        }
        assertThat(sumEigenvalues).isCloseTo(trace, org.assertj.core.data.Offset.offset(LOOSE_EPSILON));
    }

    @Test
    void testGeneralValuesOnly() {
        int n = 2;
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };

        Eigen eigen = Eigen.generalValues(A.clone(), n);

        assertThat(eigen.ok()).isTrue();
        assertThat(eigen.kind()).isEqualTo(Eigen.EIGEN_NONE);
        assertThat(eigen.eigenvalues()).hasSize(n);
        assertThat(eigen.eigenvectors()).isNull();
        assertThat(eigen.leftEigenvectors()).isNull();

        double trace = 5.0;
        double det = -2.0;
        double sumEigenvalues = eigen.eigenvalues()[0] + eigen.eigenvalues()[1];
        assertThat(sumEigenvalues).isCloseTo(trace, org.assertj.core.data.Offset.offset(LOOSE_EPSILON));
    }
}
