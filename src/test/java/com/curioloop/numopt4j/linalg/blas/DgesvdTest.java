/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class DgesvdTest {

    private static final double EPSILON = 1e-8;

    private static int queryWorkspace(char jobU, char jobVT, int m, int n, int ldu, int ldvt) {
        double[] work = new double[1];
        int info = BLAS.dgesvd(jobU, jobVT, m, n, null, 0, Math.max(1, n), null, 0,
                null, 0, ldu, null, 0, ldvt, work, 0, -1);
        assertThat(info).isEqualTo(0);
        return (int) work[0];
    }

    private static int tallOverwriteMinWork(int m, int n) {
        return Math.max(3 * n + m, 5 * n);
    }

    private static int wideOverwriteMinWork(int m, int n) {
        return Math.max(3 * m + n, 5 * m);
    }

    private static double[] sampleMatrix(int m, int n, long seed) {
        java.util.Random random = new java.util.Random(seed);
        double[] data = new double[m * n];
        for (int i = 0; i < data.length; i++) {
            data[i] = random.nextDouble() * 2.0 - 1.0;
        }
        return data;
    }

    private static void assertThinReconstruction(double[] original, int m, int n, double[] s,
                                                 double[] u, int uOff, int ldu,
                                                 double[] vt, int vtOff, int ldvt) {
        int k = Math.min(m, n);
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                double sum = 0.0;
                for (int i = 0; i < k; i++) {
                    sum += u[uOff + row * ldu + i] * s[i] * vt[vtOff + i * ldvt + col];
                }
                assertThat(sum).isCloseTo(original[row * n + col], org.assertj.core.data.Offset.offset(1e-7));
            }
        }
    }

    @Test
    void testSingularValuesSquareMatrix() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int m = 2, n = 2, lda = n;

        double[] s = new double[Math.min(m, n)];
        double[] u = new double[m * m];
        double[] vt = new double[n * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('A', 'A', m, n, A.clone(), 0, lda, s, 0, u, 0, m, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);
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
        int m = 2, n = 3, lda = n;

        double[] s = new double[Math.min(m, n)];
        double[] u = new double[m * m];
        double[] vt = new double[Math.min(m, n) * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('A', 'S', m, n, A.clone(), 0, lda, s, 0, u, 0, m, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);
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
        int m = 3, n = 2, lda = n;

        double[] s = new double[Math.min(m, n)];
        double[] u = new double[m * Math.min(m, n)];
        double[] vt = new double[n * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('S', 'A', m, n, A.clone(), 0, lda, s, 0, u, 0, Math.min(m, n), vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);
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

        double[] s = new double[n];
        double[] u = new double[n * n];
        double[] vt = new double[n * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('A', 'A', n, n, A.clone(), 0, n, s, 0, u, 0, n, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);
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

        double[] s = new double[n];
        double[] u = new double[n * n];
        double[] vt = new double[n * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('A', 'A', n, n, A.clone(), 0, n, s, 0, u, 0, n, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);
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

        double[] s = new double[n];
        double[] u = new double[n * n];
        double[] vt = new double[n * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('A', 'A', n, n, A.clone(), 0, n, s, 0, u, 0, n, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);
        assertThat(s[0]).isCloseTo(5.0, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(s[1]).isCloseTo(0.0, org.assertj.core.data.Offset.offset(1e-10));
    }

    @Test
    void testSmallMatrix1x1() {
        double[] A = {5.0};
        int n = 1;

        double[] s = new double[n];
        double[] u = new double[n * n];
        double[] vt = new double[n * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('A', 'A', n, n, A.clone(), 0, n, s, 0, u, 0, n, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);
        assertThat(s[0]).isCloseTo(5.0, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testZeroMatrix() {
        double[] A = new double[4];
        int n = 2;

        double[] s = new double[n];
        double[] u = new double[n * n];
        double[] vt = new double[n * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('A', 'A', n, n, A.clone(), 0, n, s, 0, u, 0, n, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);
        assertThat(s[0]).isCloseTo(0.0, org.assertj.core.data.Offset.offset(1e-15));
        assertThat(s[1]).isCloseTo(0.0, org.assertj.core.data.Offset.offset(1e-15));
    }

    @Test
    void testSingularValuesOnly() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        int m = 2, n = 3, lda = n;

        double[] s = new double[Math.min(m, n)];
        double[] work = new double[100];

        int info = BLAS.dgesvd('N', 'N', m, n, A.clone(), 0, lda, s, 0, null, 0, 1, null, 0, 1, work, 0, work.length);

        assertThat(info).isEqualTo(0);
        assertThat(s[0]).isCloseTo(9.508032006094326, org.assertj.core.data.Offset.offset(EPSILON));
        assertThat(s[1]).isCloseTo(0.772869635673717, org.assertj.core.data.Offset.offset(EPSILON));
    }

    @Test
    void testReconstructionSquare() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int n = 2;
        double[] AOrig = A.clone();

        double[] s = new double[n];
        double[] u = new double[n * n];
        double[] vt = new double[n * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('A', 'A', n, n, A, 0, n, s, 0, u, 0, n, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);

        double[] reconstructed = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += u[i * n + k] * s[k] * vt[k * n + j];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        for (int i = 0; i < n * n; i++) {
            assertThat(reconstructed[i]).isCloseTo(AOrig[i], org.assertj.core.data.Offset.offset(EPSILON));
        }
    }

    @Test
    void testReconstructionWide() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        int m = 2, n = 3;
        double[] AOrig = A.clone();

        double[] s = new double[m];
        double[] u = new double[m * m];
        double[] vt = new double[m * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('A', 'S', m, n, A, 0, n, s, 0, u, 0, m, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);

        double[] reconstructed = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < m; k++) {
                    sum += u[i * m + k] * s[k] * vt[k * n + j];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        for (int i = 0; i < m * n; i++) {
            assertThat(reconstructed[i]).isCloseTo(AOrig[i], org.assertj.core.data.Offset.offset(EPSILON));
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
        double[] AOrig = A.clone();

        double[] s = new double[n];
        double[] u = new double[m * n];
        double[] vt = new double[n * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('S', 'A', m, n, A, 0, n, s, 0, u, 0, n, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);

        double[] reconstructed = new double[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += u[i * n + k] * s[k] * vt[k * n + j];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        for (int i = 0; i < m * n; i++) {
            assertThat(reconstructed[i]).isCloseTo(AOrig[i], org.assertj.core.data.Offset.offset(EPSILON));
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

        double[] s = new double[n];
        double[] u = new double[n * n];
        double[] vt = new double[n * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('A', 'N', n, n, A.clone(), 0, n, s, 0, u, 0, n, vt, 0, 1, work, 0, work.length);

        assertThat(info).isEqualTo(0);

        double[] UtU = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += u[k * n + i] * u[k * n + j];
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

        double[] s = new double[n];
        double[] u = new double[n * n];
        double[] vt = new double[n * n];
        double[] work = new double[100];

        int info = BLAS.dgesvd('N', 'A', n, n, A.clone(), 0, n, s, 0, u, 0, 1, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);

        double[] VVt = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += vt[i * n + k] * vt[j * n + k];
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
    void testWorkspaceQuery() {
        double[] A = {
            1.0, 2.0,
            3.0, 4.0
        };
        int n = 2;

        double[] work = new double[1];
        int info = BLAS.dgesvd('A', 'A', n, n, A.clone(), 0, n, new double[n], 0, new double[n * n], 0, n, new double[n * n], 0, n, work, 0, -1);

        assertThat(info).isEqualTo(0);
        assertThat(work[0]).isGreaterThan(0);
    }

    @Test
    void testSingularValuesOrdering() {
        int n = 10;
        double[] A = new double[n * n];
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < n * n; i++) {
            A[i] = rand.nextDouble();
        }

        double[] s = new double[n];
        double[] u = new double[n * n];
        double[] vt = new double[n * n];
        double[] work = new double[500];

        int info = BLAS.dgesvd('A', 'A', n, n, A.clone(), 0, n, s, 0, u, 0, n, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);
        for (int i = 0; i < n - 1; i++) {
            assertThat(s[i]).isGreaterThanOrEqualTo(s[i + 1]);
        }
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

        double[] s = new double[n];
        double[] u = new double[n * n];
        double[] vt = new double[n * n];
        double[] work = new double[2000];

        int info = BLAS.dgesvd('A', 'A', n, n, A.clone(), 0, n, s, 0, u, 0, n, vt, 0, n, work, 0, work.length);

        assertThat(info).isEqualTo(0);
        assertThat(s).hasSize(n);
        for (int i = 0; i < n - 1; i++) {
            assertThat(s[i]).isGreaterThanOrEqualTo(s[i + 1]);
        }
    }

    @Test
    void testTallOverwriteUReducedWorkspaceBranch() {
        int m = 20;
        int n = 5;
        int lda = n;
        int lwork = tallOverwriteMinWork(m, n);
        int fastThreshold = n * n + Math.max(4 * n, 5 * n);

        assertThat(m).isGreaterThanOrEqualTo(Ilaenv.ilaenv(6, "DGESVD", "OS", m, n, 0, 0));
        assertThat(lwork).isLessThan(fastThreshold);
        assertThat(lwork).isLessThan(queryWorkspace('O', 'S', m, n, 1, n));

        double[] original = sampleMatrix(m, n, 11L);
        double[] a = original.clone();
        double[] s = new double[n];
        double[] vt = new double[n * n];
        double[] work = new double[lwork];

        int info = BLAS.dgesvd('O', 'S', m, n, a, 0, lda, s, 0, null, 0, 1, vt, 0, n, work, 0, lwork);

        assertThat(info).isEqualTo(0);
        assertThinReconstruction(original, m, n, s, a, 0, lda, vt, 0, n);
    }

    @Test
    void testTallOverwriteVTReducedWorkspaceBranch() {
        int m = 20;
        int n = 5;
        int lda = n;
        int lwork = tallOverwriteMinWork(m, n);
        int fastThreshold = 2 * n * n + Math.max(Math.max(n + m, 4 * n), 5 * n);

        assertThat(m).isGreaterThanOrEqualTo(Ilaenv.ilaenv(6, "DGESVD", "SO", m, n, 0, 0));
        assertThat(lwork).isLessThan(fastThreshold);
        assertThat(lwork).isLessThan(queryWorkspace('S', 'O', m, n, n, 1));

        double[] original = sampleMatrix(m, n, 23L);
        double[] a = original.clone();
        double[] s = new double[n];
        double[] u = new double[m * n];
        double[] work = new double[lwork];

        int info = BLAS.dgesvd('S', 'O', m, n, a, 0, lda, s, 0, u, 0, n, null, 0, 1, work, 0, lwork);

        assertThat(info).isEqualTo(0);
        assertThinReconstruction(original, m, n, s, u, 0, n, a, 0, lda);
    }

    @Test
    void testWideOverwriteVTReducedWorkspaceBranch() {
        int m = 5;
        int n = 20;
        int lda = n;
        int lwork = wideOverwriteMinWork(m, n);
        int fastThreshold = m * m + Math.max(4 * m, 5 * m);

        assertThat(n).isGreaterThanOrEqualTo(Ilaenv.ilaenv(6, "DGESVD", "SO", m, n, 0, 0));
        assertThat(lwork).isLessThan(fastThreshold);
        assertThat(lwork).isLessThan(queryWorkspace('S', 'O', m, n, m, 1));

        double[] original = sampleMatrix(m, n, 37L);
        double[] a = original.clone();
        double[] s = new double[m];
        double[] u = new double[m * m];
        double[] work = new double[lwork];

        int info = BLAS.dgesvd('S', 'O', m, n, a, 0, lda, s, 0, u, 0, m, null, 0, 1, work, 0, lwork);

        assertThat(info).isEqualTo(0);
        assertThinReconstruction(original, m, n, s, u, 0, m, a, 0, lda);
    }

    @Test
    void testWideOverwriteUReducedWorkspaceBranch() {
        int m = 5;
        int n = 20;
        int lda = n;
        int lwork = wideOverwriteMinWork(m, n);
        int fastThreshold = 2 * m * m + Math.max(Math.max(m + n, 4 * m), 5 * m);

        assertThat(n).isGreaterThanOrEqualTo(Ilaenv.ilaenv(6, "DGESVD", "OS", m, n, 0, 0));
        assertThat(lwork).isLessThan(fastThreshold);
        assertThat(lwork).isLessThan(queryWorkspace('O', 'S', m, n, 1, n));

        double[] original = sampleMatrix(m, n, 41L);
        double[] a = original.clone();
        double[] s = new double[m];
        double[] vt = new double[m * n];
        double[] work = new double[lwork];

        int info = BLAS.dgesvd('O', 'S', m, n, a, 0, lda, s, 0, null, 0, 1, vt, 0, n, work, 0, lwork);

        assertThat(info).isEqualTo(0);
        assertThinReconstruction(original, m, n, s, a, 0, lda, vt, 0, n);
    }
}
