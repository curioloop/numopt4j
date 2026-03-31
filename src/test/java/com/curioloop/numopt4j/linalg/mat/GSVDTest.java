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

    private static int queryWorkspace(int m, int p, int n) {
        double[] tmp = new double[1];
        BLAS.dggsvp3(BLAS.GsvdJob.Compute, BLAS.GsvdJob.Compute, BLAS.GsvdJob.Compute,
                m, p, n, null, 0, n, null, 0, n, 0, 0,
                null, 0, m, null, 0, p, null, 0, n,
                null, null, tmp, 0, -1);
        return Math.max(n + (int) tmp[0], Math.max(2 * n + 2, 6));
    }

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
        Decomposition.Workspace ws1 = GSVD.workspace();
        Decomposition.Workspace ws2 = GSVD.workspace();
        Decomposition.Workspace ws3 = GSVD.workspace();

        assertThat(ws1.work()).isNull();
        assertThat(ws2.work()).isNull();
        assertThat(ws3.work()).isNull();
    }

    @Test
    void testWorkspaceMatchesRequestedKind() {
        GSVD.Pool none = GSVD.workspace();
        GSVD.Pool onlyU = GSVD.workspace();
        GSVD.Pool onlyV = GSVD.workspace();
        GSVD.Pool onlyQ = GSVD.workspace();
        GSVD.Pool all = GSVD.workspace();

        none.ensure(4, 3, 2, GSVD.GSVD_NONE);
        onlyU.ensure(4, 3, 2, GSVD.GSVD_U);
        onlyV.ensure(4, 3, 2, GSVD.GSVD_V);
        onlyQ.ensure(4, 3, 2, GSVD.GSVD_Q);
        all.ensure(4, 3, 2, GSVD.GSVD_ALL);

        int expectedWork = queryWorkspace(4, 3, 2);
        assertThat(none.work().length).isEqualTo(expectedWork);
        assertThat(onlyU.work().length).isEqualTo(expectedWork);
        assertThat(onlyV.work().length).isEqualTo(expectedWork);
        assertThat(onlyQ.work().length).isEqualTo(expectedWork);
        assertThat(all.work().length).isEqualTo(expectedWork);

        assertThat(none.U).isNull();
        assertThat(none.V).isNull();
        assertThat(none.Q).isNull();
        assertThat(onlyU.U).hasSize(16);
        assertThat(onlyU.V).isNull();
        assertThat(onlyU.Q).isNull();
        assertThat(onlyV.U).isNull();
        assertThat(onlyV.V).hasSize(9);
        assertThat(onlyV.Q).isNull();
        assertThat(onlyQ.U).isNull();
        assertThat(onlyQ.V).isNull();
        assertThat(onlyQ.Q).hasSize(4);
        assertThat(all.U).hasSize(16);
        assertThat(all.V).hasSize(9);
        assertThat(all.Q).hasSize(4);
    }

    @Test
    void testNullWorkspaceRespectsKind() {
        double[] A = {
            1.0, 0.0,
            0.0, 1.0
        };
        double[] B = {
            1.0, 0.0,
            0.0, 1.0
        };

        GSVD none = GSVD.decompose(A.clone(), 2, 2, B.clone(), 2, GSVD.GSVD_NONE);
        GSVD onlyU = GSVD.decompose(A.clone(), 2, 2, B.clone(), 2, GSVD.GSVD_U);

        assertThat(none.pool().work().length).isEqualTo(queryWorkspace(2, 2, 2));
        assertThat(none.pool().U).isNull();
        assertThat(none.pool().V).isNull();
        assertThat(none.pool().Q).isNull();

        assertThat(onlyU.pool().work().length).isEqualTo(queryWorkspace(2, 2, 2));
        assertThat(onlyU.pool().U).hasSize(4);
        assertThat(onlyU.pool().V).isNull();
        assertThat(onlyU.pool().Q).isNull();
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
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, m, m, m, 1.0, U, 0, m, U, 0, m, 0.0, UtU, 0, m);

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
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, p, p, p, 1.0, V, 0, p, V, 0, p, 0.0, VtV, 0, p);

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
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, n, n, n, 1.0, Q, 0, n, Q, 0, n, 0.0, QtQ, 0, n);

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
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, m, m, m, 1.0, U, 0, m, U, 0, m, 0.0, UtU, 0, m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(Math.abs(UtU[i * m + j] - expected)).isLessThan(EPSILON);
            }
        }

        double[] V = gsvd.V();
        double[] VtV = new double[p * p];
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, p, p, p, 1.0, V, 0, p, V, 0, p, 0.0, VtV, 0, p);
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(Math.abs(VtV[i * p + j] - expected)).isLessThan(EPSILON);
            }
        }

        double[] Q = gsvd.Q();
        double[] QtQ = new double[n * n];
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, n, n, n, 1.0, Q, 0, n, Q, 0, n, 0.0, QtQ, 0, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(Math.abs(QtQ[i * n + j] - expected)).isLessThan(EPSILON);
            }
        }
    }
}
