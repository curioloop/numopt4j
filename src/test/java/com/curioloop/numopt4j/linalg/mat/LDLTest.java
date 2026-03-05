/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class LDLTest {

    private static final double EPSILON = 1e-10;

    @Test
    void testBasicDecompositionLower() {
        double[] A = {
            4.0, 0.0,
            2.0, 3.0
        };
        
        Cholesky ldl = Cholesky.decompose(A, 2, 'L', true, null);
        
        assertThat(ldl.ok()).isTrue();
        assertThat(ldl.uplo()).isEqualTo('L');
        assertThat(ldl.isPivoting()).isTrue();
    }

    @Test
    void testBasicDecompositionUpper() {
        double[] A = {
            4.0, 2.0,
            0.0, 3.0
        };
        
        Cholesky ldl = Cholesky.decompose(A, 2, 'U', true, null);
        
        assertThat(ldl.ok()).isTrue();
        assertThat(ldl.uplo()).isEqualTo('U');
        assertThat(ldl.isPivoting()).isTrue();
    }

    @Test
    void testIndefiniteMatrix() {
        double[] A = {
            1.0, 0.0,
            0.0, -1.0
        };
        
        Cholesky ldl = Cholesky.decompose(A, 2, 'L', true, null);
        
        assertThat(ldl.ok()).isTrue();
        assertThat(ldl.extract(Decomposition.Part.D)).isNotNull();
    }

    @Test
    void testPivot() {
        double[] A = {
            1.0, 2.0,
            2.0, 1.0
        };
        
        Cholesky ldl = Cholesky.decompose(A, 2, 'L', true, null);
        
        assertThat(ldl.ok()).isTrue();
        assertThat(ldl.piv()).isNotNull();
        assertThat(ldl.piv().length).isEqualTo(2);
    }

    @Test
    void testIdentityMatrix() {
        double[] A = {
            1.0, 0.0,
            0.0, 1.0
        };
        
        Cholesky ldl = Cholesky.decompose(A, 2, 'L', true, null);
        
        assertThat(ldl.ok()).isTrue();
        double[] D = ldl.extract(Decomposition.Part.D).data;
        assertThat(D[0]).isCloseTo(1.0, offset(EPSILON));
        assertThat(D[3]).isCloseTo(1.0, offset(EPSILON));
    }

    @Test
    void testBlockDiagonalD() {
        double[] A = {
            1.0, 2.0,
            2.0, 1.0
        };
        
        Cholesky ldl = Cholesky.decompose(A, 2, 'L', true, null);
        
        assertThat(ldl.ok()).isTrue();
        
        int[] piv = ldl.piv();
        assertThat(Math.abs(piv[0])).isGreaterThan(0);
    }

    @Test
    void testPositiveDefinite() {
        double[] A = {
            4.0, 2.0,
            2.0, 3.0
        };
        
        Cholesky ldl = Cholesky.decompose(A, 2, 'L', true, null);
        
        assertThat(ldl.ok()).isTrue();
        
        double[] D = ldl.extract(Decomposition.Part.D).data;
        assertThat(D[0]).isGreaterThan(0);
        assertThat(D[3]).isGreaterThan(0);
    }

    @Test
    void testLargerMatrix() {
        double[] A = {
            4.0, 0.0, 0.0,
            2.0, 5.0, 0.0,
            1.0, 3.0, 6.0
        };
        
        Cholesky ldl = Cholesky.decompose(A, 3, 'L', true, null);
        
        assertThat(ldl.ok()).isTrue();
        assertThat(ldl.n()).isEqualTo(3);
    }

    @Test
    void testSolveIndefiniteLower() {
        double[] A = {
            1.0, 2.0,
            2.0, -1.0
        };
        double[] A_orig = A.clone();
        double[] b = {3.0, 1.0};
        double[] b_orig = b.clone();
        
        Cholesky ldl = Cholesky.decompose(A, 2, 'L', true, null);
        assertThat(ldl.ok()).isTrue();
        
        double[] x = ldl.solve(b, null);
        
        double ax0 = A_orig[0] * x[0] + A_orig[1] * x[1];
        double ax1 = A_orig[2] * x[0] + A_orig[3] * x[1];
        
        assertThat(ax0).isCloseTo(b_orig[0], offset(1e-6));
        assertThat(ax1).isCloseTo(b_orig[1], offset(1e-6));
    }

    @Test
    void testSolveIndefiniteUpper() {
        double[] A = {
            1.0, 2.0,
            2.0, -1.0
        };
        double[] A_orig = A.clone();
        double[] b = {3.0, 1.0};
        double[] b_orig = b.clone();
        
        Cholesky ldl = Cholesky.decompose(A, 2, 'U', true, null);
        assertThat(ldl.ok()).isTrue();
        
        double[] x = ldl.solve(b, null);
        
        double ax0 = A_orig[0] * x[0] + A_orig[1] * x[1];
        double ax1 = A_orig[2] * x[0] + A_orig[3] * x[1];
        
        assertThat(ax0).isCloseTo(b_orig[0], offset(1e-6));
        assertThat(ax1).isCloseTo(b_orig[1], offset(1e-6));
    }

    @Test
    void testDeterminantIndefinite() {
        double[] A = {
            1.0, 2.0,
            2.0, 1.0
        };
        
        Cholesky ldl = Cholesky.decompose(A, 2, 'L', true, null);
        assertThat(ldl.ok()).isTrue();
        
        double det = ldl.determinant();
        double expectedDet = 1.0 * 1.0 - 2.0 * 2.0;
        
        assertThat(det).isCloseTo(expectedDet, offset(1e-6));
    }

    @Test
    void testWorkspaceReuse() {
        int n = 5;
        Decomposition.Workspace ws = Cholesky.workspace(n, true);
        
        for (int i = 0; i < 3; i++) {
            double[] A = createRandomSymmetric(n);
            Cholesky ldl = Cholesky.decompose(A, n, 'L', true, ws);
            assertThat(ldl.ok()).isTrue();
        }
    }

    @Test
    void testConsistencyWithCholeskyOnPositiveDefinite() {
        double[] A = {
            4.0, 2.0,
            2.0, 3.0
        };
        
        double[] A1 = A.clone();
        double[] A2 = A.clone();
        
        Cholesky chol = Cholesky.decompose(A1, 2, 'L', false, null);
        Cholesky ldl = Cholesky.decompose(A2, 2, 'L', true, null);
        
        assertThat(chol.ok()).isTrue();
        assertThat(ldl.ok()).isTrue();
        
        double detChol = chol.determinant();
        double detLDL = ldl.determinant();
        
        assertThat(detLDL).isCloseTo(detChol, offset(1e-6));
    }

    private double[] createRandomSymmetric(int n) {
        java.util.Random rand = new java.util.Random(42);
        double[] A = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                A[i * n + j] = rand.nextDouble() * 2 - 1;
                A[j * n + i] = A[i * n + j];
            }
        }
        return A;
    }
}
