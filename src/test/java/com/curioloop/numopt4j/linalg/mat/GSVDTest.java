/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class GSVDTest {

    private static final double EPSILON = 1e-10;

    @Test
    void testGSVDSimple() {
        double[] A = {
            1.0, 0.0,
            0.0, 1.0
        };
        double[] B = {
            1.0, 0.0,
            0.0, 1.0
        };
        int m = 2, n = 2, p = 2;

        GSVD gsvd = GSVD.decompose(A.clone(), m, n, B.clone(), p, GSVD.GSVD_NONE);

        assertThat(gsvd).isNotNull();
        assertThat(gsvd.alpha()).hasSize(n);
        assertThat(gsvd.beta()).hasSize(n);
    }

    @Test
    void testGSVDWithU() {
        double[] A = {
            1.0, 0.0,
            0.0, 1.0
        };
        double[] B = {
            1.0, 0.0,
            0.0, 1.0
        };
        int m = 2, n = 2, p = 2;

        GSVD gsvd = GSVD.decompose(A.clone(), m, n, B.clone(), p, GSVD.GSVD_U);

        assertThat(gsvd).isNotNull();
        assertThat(gsvd.U()).isNotNull();
        assertThat(gsvd.U().length).isEqualTo(m * m);
    }

    @Test
    void testGSVDWithV() {
        double[] A = {
            1.0, 0.0,
            0.0, 1.0
        };
        double[] B = {
            1.0, 0.0,
            0.0, 1.0
        };
        int m = 2, n = 2, p = 2;

        GSVD gsvd = GSVD.decompose(A.clone(), m, n, B.clone(), p, GSVD.GSVD_V);

        assertThat(gsvd).isNotNull();
        assertThat(gsvd.V()).isNotNull();
        assertThat(gsvd.V().length).isEqualTo(p * p);
    }

    @Test
    void testGSVDWithQ() {
        double[] A = {
            1.0, 0.0,
            0.0, 1.0
        };
        double[] B = {
            1.0, 0.0,
            0.0, 1.0
        };
        int m = 2, n = 2, p = 2;

        GSVD gsvd = GSVD.decompose(A.clone(), m, n, B.clone(), p, GSVD.GSVD_Q);

        assertThat(gsvd).isNotNull();
        assertThat(gsvd.Q()).isNotNull();
        assertThat(gsvd.Q().length).isEqualTo(n * n);
    }

    @Test
    void testDimensions() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        double[] B = {
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0
        };
        int m = 2, n = 3, p = 3;

        GSVD gsvd = GSVD.decompose(A.clone(), m, n, B.clone(), p, GSVD.GSVD_NONE);

        assertThat(gsvd.m()).isEqualTo(m);
        assertThat(gsvd.n()).isEqualTo(n);
        assertThat(gsvd.p()).isEqualTo(p);
    }

    @Test
    void testWorkspaceSize() {
        Decomposition.Workspace ws1 = GSVD.workspace(3, 3, 3);
        Decomposition.Workspace ws2 = GSVD.workspace(2, 5, 3);
        Decomposition.Workspace ws3 = GSVD.workspace(5, 2, 3);

        assertThat(ws1.work().length).isPositive();
        assertThat(ws2.work().length).isPositive();
        assertThat(ws3.work().length).isPositive();
    }

    @Test
    void testOrthogonalityOfU() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0
        };
        double[] B = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        };
        int m = 4, n = 3, p = 3;

        GSVD gsvd = GSVD.decompose(A.clone(), m, n, B.clone(), p, GSVD.GSVD_U);

        assertThat(gsvd.ok()).isTrue();
        assertThat(gsvd.U()).isNotNull();

        double[] U = gsvd.U();
        double[] UtU = new double[m * m];
        BLAS.dgemm(BLAS.Transpose.Trans, BLAS.Transpose.NoTrans, m, m, m, 1.0, U, 0, m, U, 0, m, 0.0, UtU, 0, m);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(Math.abs(UtU[i * m + j] - expected)).isLessThan(EPSILON);
            }
        }
    }

    @Test
    void testOrthogonalityOfV() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        };
        double[] B = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0
        };
        int m = 2, n = 3, p = 4;

        GSVD gsvd = GSVD.decompose(A.clone(), m, n, B.clone(), p, GSVD.GSVD_V);

        assertThat(gsvd.ok()).isTrue();
        assertThat(gsvd.V()).isNotNull();

        double[] V = gsvd.V();
        double[] VtV = new double[p * p];
        BLAS.dgemm(BLAS.Transpose.Trans, BLAS.Transpose.NoTrans, p, p, p, 1.0, V, 0, p, V, 0, p, 0.0, VtV, 0, p);

        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(Math.abs(VtV[i * p + j] - expected)).isLessThan(EPSILON);
            }
        }
    }

    @Test
    void testOrthogonalityOfQ() {
        double[] A = {
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        };
        double[] B = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        };
        int m = 3, n = 3, p = 3;

        GSVD gsvd = GSVD.decompose(A.clone(), m, n, B.clone(), p, GSVD.GSVD_Q);

        assertThat(gsvd.ok()).isTrue();
        assertThat(gsvd.Q()).isNotNull();

        double[] Q = gsvd.Q();
        double[] QtQ = new double[n * n];
        BLAS.dgemm(BLAS.Transpose.Trans, BLAS.Transpose.NoTrans, n, n, n, 1.0, Q, 0, n, Q, 0, n, 0.0, QtQ, 0, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(Math.abs(QtQ[i * n + j] - expected)).isLessThan(EPSILON);
            }
        }
    }

    @Test
    void testOrthogonalityAllMatrices() {
        double[] A = {
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0
        };
        double[] B = {
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        };
        int m = 5, n = 4, p = 3;

        GSVD gsvd = GSVD.decompose(A.clone(), m, n, B.clone(), p, GSVD.GSVD_ALL);

        assertThat(gsvd.ok()).isTrue();

        double[] U = gsvd.U();
        double[] UtU = new double[m * m];
        BLAS.dgemm(BLAS.Transpose.Trans, BLAS.Transpose.NoTrans, m, m, m, 1.0, U, 0, m, U, 0, m, 0.0, UtU, 0, m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(Math.abs(UtU[i * m + j] - expected)).isLessThan(EPSILON);
            }
        }

        double[] V = gsvd.V();
        double[] VtV = new double[p * p];
        BLAS.dgemm(BLAS.Transpose.Trans, BLAS.Transpose.NoTrans, p, p, p, 1.0, V, 0, p, V, 0, p, 0.0, VtV, 0, p);
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(Math.abs(VtV[i * p + j] - expected)).isLessThan(EPSILON);
            }
        }

        double[] Q = gsvd.Q();
        double[] QtQ = new double[n * n];
        BLAS.dgemm(BLAS.Transpose.Trans, BLAS.Transpose.NoTrans, n, n, n, 1.0, Q, 0, n, Q, 0, n, 0.0, QtQ, 0, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(Math.abs(QtQ[i * n + j] - expected)).isLessThan(EPSILON);
            }
        }
    }
}
