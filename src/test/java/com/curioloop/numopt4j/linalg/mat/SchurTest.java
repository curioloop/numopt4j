/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.Select;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SchurTest {

    private static final double EPS = 1e-10;

    @Test
    @DisplayName("Schur: basic 3x3 matrix")
    void testBasic3x3() {
        int n = 3;
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok(), "Schur decomposition should succeed");
        assertEquals(n, schur.n());

        double[] wr = schur.eigenvalues();
        double[] wi = schur.eigenvaluesImag();

        double[] expectedWr = {16.1168, -1.1168, 0};

        for (int i = 0; i < n; i++) {
            assertEquals(expectedWr[i], wr[i], Math.abs(expectedWr[i]) * 1e-4 + 1e-4,
                "Real part of eigenvalue " + i);
            assertEquals(0, wi[i], EPS, "Imaginary part of eigenvalue " + i);
        }
    }

    @Test
    @DisplayName("Schur: verify Schur decomposition properties")
    void testSchurDecomposition() {
        int n = 4;
        double[] AOrig = {
            4, 1, 2, 3,
            1, 5, 1, 2,
            2, 1, 6, 1,
            3, 2, 1, 7
        };
        double[] A = AOrig.clone();

        Schur schur = Schur.decompose(A, n);
        assertTrue(schur.ok(), "Schur decomposition should succeed");

        double[] T = schur.T();

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                assertEquals(0, T[i * n + j], 1e-10,
                    "T should be upper triangular at (" + i + "," + j + ")");
            }
        }

        double trace = 0;
        for (int i = 0; i < n; i++) {
            trace += AOrig[i * n + i];
        }
        double traceT = 0;
        for (int i = 0; i < n; i++) {
            traceT += T[i * n + i];
        }
        assertEquals(trace, traceT, Math.abs(trace) * 1e-10, "Trace should be preserved");
    }

    @Test
    @DisplayName("Schur: matrix with complex eigenvalues")
    void testComplexEigenvalues() {
        int n = 2;
        double[] A = {
            0, -1,
            1, 0
        };

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok(), "Schur decomposition should succeed");

        double[] wr = schur.eigenvalues();
        double[] wi = schur.eigenvaluesImag();

        assertEquals(0, wr[0], EPS, "Real part should be 0");
        assertEquals(1, Math.abs(wi[0]), EPS, "Imaginary part should be 1");
        assertEquals(0, wr[1], EPS, "Real part should be 0");
        assertEquals(1, Math.abs(wi[1]), EPS, "Imaginary part should be 1");
        assertEquals(wi[0], -wi[1], EPS, "Complex eigenvalues should be conjugate pairs");
    }

    @Test
    @DisplayName("Schur: identity matrix")
    void testIdentityMatrix() {
        int n = 3;
        double[] A = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        };

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok(), "Schur decomposition should succeed");

        double[] wr = schur.eigenvalues();
        double[] wi = schur.eigenvaluesImag();

        for (int i = 0; i < n; i++) {
            assertEquals(1, wr[i], EPS, "Eigenvalue should be 1");
            assertEquals(0, wi[i], EPS, "Imaginary part should be 0");
        }
    }

    @Test
    @DisplayName("Schur: diagonal matrix")
    void testDiagonalMatrix() {
        int n = 3;
        double[] A = {
            3, 0, 0,
            0, 2, 0,
            0, 0, 1
        };

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok(), "Schur decomposition should succeed");

        double[] wr = schur.eigenvalues();

        double[] expected = {1, 2, 3};
        java.util.Arrays.sort(wr);
        for (int i = 0; i < n; i++) {
            assertEquals(expected[i], wr[i], EPS, "Eigenvalue " + i);
        }
    }

    @Test
    @DisplayName("Schur: stable matrix (all eigenvalues in LHP)")
    void testStableMatrix() {
        int n = 3;
        double[] A = {
            -1, 0, 0,
            0, -2, 0,
            0, 0, -3
        };

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());
        assertTrue(schur.isStable(), "Matrix with negative eigenvalues should be stable");
        assertFalse(schur.isDiscreteStable(), "Eigenvalues outside unit circle");
    }

    @Test
    @DisplayName("Schur: unstable matrix (eigenvalues in RHP)")
    void testUnstableMatrix() {
        int n = 3;
        double[] A = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());
        assertFalse(schur.isStable(), "Matrix with positive eigenvalues should be unstable");
    }

    @Test
    @DisplayName("Schur: discrete stable matrix (eigenvalues inside unit circle)")
    void testDiscreteStableMatrix() {
        int n = 2;
        double[] A = {
            0.5, 0,
            0, 0.3
        };

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());
        assertTrue(schur.isDiscreteStable(), "Matrix with eigenvalues inside unit circle should be discrete stable");
        assertFalse(schur.isStable(), "Eigenvalues are positive");
    }

    @Test
    @DisplayName("Schur: sort with LHP selector")
    void testSortLHP() {
        int n = 3;
        double[] A = {
            1, 0, 0,
            0, -1, 0,
            0, 0, -2
        };

        Schur schur = Schur.decompose(A.clone(), n, true, Schur.LHP);
        assertTrue(schur.ok());
        assertEquals(2, schur.sdim(), "Two eigenvalues in LHP");

        double[] wr = schur.eigenvalues();
        assertTrue(wr[0] < 0, "First eigenvalue should be in LHP");
        assertTrue(wr[1] < 0, "Second eigenvalue should be in LHP");
    }

    @Test
    @DisplayName("Schur: sort with RHP selector")
    void testSortRHP() {
        int n = 3;
        double[] A = {
            1, 0, 0,
            0, -1, 0,
            0, 0, -2
        };

        Schur schur = Schur.decompose(A.clone(), n, true, Schur.RHP);
        assertTrue(schur.ok());
        assertEquals(1, schur.sdim(), "One eigenvalue in RHP");

        double[] wr = schur.eigenvalues();
        assertTrue(wr[0] >= 0, "First eigenvalue should be in RHP");
    }

    @Test
    @DisplayName("Schur: sort with IUC selector")
    void testSortIUC() {
        int n = 3;
        double[] A = {
            0.5, 0, 0,
            0, 1.5, 0,
            0, 0, 0.8
        };

        Schur schur = Schur.decompose(A.clone(), n, true, Schur.IUC);
        assertTrue(schur.ok());
        assertEquals(2, schur.sdim(), "Two eigenvalues inside unit circle");
    }

    @Test
    @DisplayName("Schur: sort with OUC selector")
    void testSortOUC() {
        int n = 3;
        double[] A = {
            0.5, 0, 0,
            0, 1.5, 0,
            0, 0, 0.8
        };

        Schur schur = Schur.decompose(A.clone(), n, true, Schur.OUC);
        assertTrue(schur.ok());
        assertEquals(1, schur.sdim(), "One eigenvalue outside unit circle");
    }

    @Test
    @DisplayName("Schur: custom selector")
    void testCustomSelector() {
        int n = 3;
        double[] A = {
            1, 0, 0,
            0, 2, 0,
            0, 0, 3
        };

        Select greaterThanOne = (wr, wi) -> wr > 1.5;
        Schur schur = Schur.decompose(A.clone(), n, true, greaterThanOne);
        assertTrue(schur.ok());
        assertEquals(2, schur.sdim(), "Two eigenvalues > 1.5");
    }

    @Test
    @DisplayName("Schur: invariant subspace")
    void testInvariantSubspace() {
        int n = 3;
        double[] A = {
            -1, 0, 0,
            0, -2, 0,
            0, 0, 1
        };

        Schur schur = Schur.decompose(A.clone(), n, true, Schur.LHP);
        assertTrue(schur.ok());
        assertEquals(2, schur.sdim());

        double[] stableSubspace = schur.invariantSubspace();
        assertNotNull(stableSubspace);
        assertEquals(n * 2, stableSubspace.length);
    }

    @Test
    @DisplayName("Schur: unstable subspace")
    void testUnstableSubspace() {
        int n = 3;
        double[] A = {
            -1, 0, 0,
            0, -2, 0,
            0, 0, 1
        };

        Schur schur = Schur.decompose(A.clone(), n, true, Schur.LHP);
        assertTrue(schur.ok());
        assertEquals(2, schur.sdim());

        double[] unstableSubspace = schur.unstableSubspace();
        assertNotNull(unstableSubspace);
        assertEquals(n * 1, unstableSubspace.length);
    }

    @Test
    @DisplayName("Schur: no Z computation")
    void testNoZComputation() {
        int n = 3;
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };

        Schur schur = Schur.decompose(A.clone(), n, false);
        assertTrue(schur.ok());
        assertNull(schur.Z(), "Z should be null when computeZ is false");
        assertNotNull(schur.eigenvalues());
    }

    @Test
    @DisplayName("Schur: Z matrix exists when computeZ is true")
    void testZMatrixExists() {
        int n = 4;
        double[] A = {
            4, 1, 2, 3,
            1, 5, 1, 2,
            2, 1, 6, 1,
            3, 2, 1, 7
        };

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());

        double[] Z = schur.Z();
        assertNotNull(Z, "Z should not be null when computeZ is true");
        assertEquals(n * n, Z.length, "Z should have n*n elements");
    }

    @Test
    @DisplayName("Schur: workspace size")
    void testWorkspaceSize() {
        int n = 10;
        Decomposition.Workspace ws = Schur.workspace(n);
        assertTrue(ws.work().length >= 3 * n);
        assertTrue(ws.work().length >= n + 10);
        assertTrue(ws.bwork().length >= n);
    }

    @Test
    @DisplayName("Schur: predefined selectors")
    void testPredefinedSelectors() {
        assertNotNull(Schur.LHP);
        assertNotNull(Schur.RHP);
        assertNotNull(Schur.IUC);
        assertNotNull(Schur.OUC);

        assertTrue(Schur.LHP.select(-1, 0));
        assertFalse(Schur.LHP.select(1, 0));

        assertTrue(Schur.RHP.select(1, 0));
        assertFalse(Schur.RHP.select(-1, 0));

        assertTrue(Schur.IUC.select(0.5, 0));
        assertFalse(Schur.IUC.select(1.5, 0));

        assertTrue(Schur.OUC.select(1.5, 0));
        assertFalse(Schur.OUC.select(0.5, 0));
    }

    @Test
    @DisplayName("Schur: 1x1 matrix")
    void test1x1Matrix() {
        int n = 1;
        double[] A = {5.0};

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());
        assertEquals(5.0, schur.eigenvalues()[0], EPS);
        assertEquals(0.0, schur.eigenvaluesImag()[0], EPS);
    }

    @Test
    @DisplayName("Schur: trace preservation")
    void testTracePreservation() {
        int n = 3;
        double[] A = {
            2, 1, 0,
            1, 3, 1,
            0, 1, 2
        };

        double traceOrig = A[0] + A[4] + A[8];

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());

        double[] T = schur.T();
        double traceT = 0;
        for (int i = 0; i < n; i++) {
            traceT += T[i * n + i];
        }

        assertEquals(traceOrig, traceT, Math.abs(traceOrig) * 1e-10);
    }

    // ==================== Lyapunov equation tests ====================

    /** Compute A*X + X*A^T and return the residual matrix. */
    private static double[] continuousResidual(double[] A, double[] X, int n) {
        double[] R = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double v = 0;
                for (int k = 0; k < n; k++) {
                    v += A[i * n + k] * X[k * n + j]   // A*X
                       + X[i * n + k] * A[j * n + k];  // X*A^T
                }
                R[i * n + j] = v;
            }
        }
        return R;
    }

    /** Compute A*X*A^T - X and return the residual matrix. */
    private static double[] discreteResidual(double[] A, double[] X, int n) {
        double[] AX = new double[n * n];
        double[] R  = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++) {
                    AX[i * n + j] += A[i * n + k] * X[k * n + j];
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double v = 0;
                for (int k = 0; k < n; k++) {
                    v += AX[i * n + k] * A[j * n + k]; // (A*X)*A^T
                }
                R[i * n + j] = v - X[i * n + j];
            }
        }
        return R;
    }

    private static double maxAbs(double[] M, double[] ref, int n) {
        double err = 0;
        for (int i = 0; i < n * n; i++) {
            err = Math.max(err, Math.abs(M[i] - ref[i]));
        }
        return err;
    }

    @Test
    @DisplayName("Lyapunov continuous: stable 2x2 diagonal A")
    void testContinuousLyapunov2x2() {
        int n = 2;
        double[] A = { -1, 0,  0, -2 };
        double[] Q = {  1, 0,  0,  1 };

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.lyapunov(Q));

        assertEquals(-0.5,  Q[0], 1e-10);
        assertEquals( 0.0,  Q[1], 1e-10);
        assertEquals( 0.0,  Q[2], 1e-10);
        assertEquals(-0.25, Q[3], 1e-10);
    }

    @Test
    @DisplayName("Lyapunov continuous: residual check on 3x3 stable matrix")
    void testContinuousLyapunovResidual3x3() {
        int n = 3;
        double[] A = {
            -3,  1,  0,
             0, -2,  1,
             0,  0, -1
        };
        double[] Q = {
             2, -1,  0,
            -1,  3, -1,
             0, -1,  2
        };
        double[] Aorig = A.clone();
        double[] Qorig = Q.clone();

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.lyapunov(Q));

        double[] R = continuousResidual(Aorig, Q, n);
        assertEquals(0, maxAbs(R, Qorig, n), 1e-9);
    }

    @Test
    @DisplayName("Lyapunov continuous: sign=-1 gives AX + XA^T = -Q")
    void testContinuousLyapunovNegSign() {
        int n = 2;
        double[] A = { -1, 0,  0, -2 };
        double[] Q = {  1, 0,  0,  1 };
        double[] Aorig = A.clone();
        double[] Qorig = Q.clone();

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.lyapunov(Q, -1.0));

        double[] R = continuousResidual(Aorig, Q, n);
        for (int i = 0; i < n * n; i++) {
            assertEquals(-Qorig[i], R[i], 1e-10);
        }
    }

    @Test
    @DisplayName("Lyapunov continuous: 4x4 general stable matrix")
    void testContinuousLyapunov4x4() {
        int n = 4;
        double[] A = {
            -4,  1,  0,  0,
             0, -3,  1,  0,
             0,  0, -2,  1,
             0,  0,  0, -1
        };
        double[] Q = {
             4, 0, 0, 0,
             0, 3, 0, 0,
             0, 0, 2, 0,
             0, 0, 0, 1
        };
        double[] Aorig = A.clone();
        double[] Qorig = Q.clone();

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.lyapunov(Q));

        double[] R = continuousResidual(Aorig, Q, n);
        assertEquals(0, maxAbs(R, Qorig, n), 1e-9);
    }

    @Test
    @DisplayName("Lyapunov discrete: stable 2x2 diagonal A")
    void testDiscreteLyapunov2x2() {
        int n = 2;
        double[] A = { 0.5, 0,  0, 0.3 };
        double[] Q = {  1,  0,  0,  1  };
        double[] Aorig = A.clone();
        double[] Qorig = Q.clone();

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.discreteLyapunov(A.clone(), Q));

        double[] R = discreteResidual(Aorig, Q, n);
        assertEquals(0, maxAbs(R, Qorig, n), 1e-9);
    }

    @Test
    @DisplayName("Lyapunov discrete: residual check on 3x3 stable matrix")
    void testDiscreteLyapunovResidual3x3() {
        int n = 3;
        double[] A = {
             0.5, 0.1,  0.0,
             0.0, 0.4,  0.1,
             0.0, 0.0,  0.3
        };
        double[] Q = {
             1, 0, 0,
             0, 1, 0,
             0, 0, 1
        };
        double[] Aorig = A.clone();
        double[] Qorig = Q.clone();

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.discreteLyapunov(A.clone(), Q));

        double[] R = discreteResidual(Aorig, Q, n);
        assertEquals(0, maxAbs(R, Qorig, n), 1e-9);
    }

    @Test
    @DisplayName("Lyapunov discrete: 4x4 general stable matrix")
    void testDiscreteLyapunov4x4() {
        int n = 4;
        double[] A = {
             0.8,  0.1,  0.0,  0.0,
             0.0,  0.6,  0.1,  0.0,
             0.0,  0.0,  0.4,  0.1,
             0.0,  0.0,  0.0,  0.2
        };
        double[] Q = {
             2, 0, 0, 0,
             0, 2, 0, 0,
             0, 0, 2, 0,
             0, 0, 0, 2
        };
        double[] Aorig = A.clone();
        double[] Qorig = Q.clone();

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.discreteLyapunov(A.clone(), Q));

        double[] R = discreteResidual(Aorig, Q, n);
        assertEquals(0, maxAbs(R, Qorig, n), 1e-9);
    }

    @Test
    @DisplayName("Lyapunov: n=1 edge case")
    void testLyapunovSize1() {
        // Continuous: a*x + x*a = q  =>  2*a*x = q  =>  x = q/(2*a)
        double[] A = { -3.0 };
        double[] Q = {  6.0 };
        Schur schur = Schur.decompose(A.clone(), 1, true);
        assertTrue(schur.lyapunov(Q));
        assertEquals(6.0 / (2.0 * -3.0), Q[0], 1e-10); // = -1.0

        // Discrete: a*x*a - x = q
        double[] A2 = { 0.5 };
        double[] Q2 = { 1.0 };
        double[] A2orig = A2.clone();
        double[] Q2orig = Q2.clone();
        Schur schur2 = Schur.decompose(A2.clone(), 1, true);
        assertTrue(schur2.discreteLyapunov(A2.clone(), Q2));
        double[] R = discreteResidual(A2orig, Q2, 1);
        assertEquals(0, maxAbs(R, Q2orig, 1), 1e-10);
    }

    // ==================== Instance method tests for lyapunov ====================

    @Test
    @DisplayName("Lyapunov continuous (instance): stable 2x2 diagonal A")
    void testContinuousLyapunov2x2_instance() {
        int n = 2;
        double[] A = { -1, 0,  0, -2 };
        double[] Q = {  1, 0,  0,  1 };

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());
        assertTrue(schur.lyapunov(Q));

        // X[0,0] = 1/(2*(-1)) = -0.5, X[1,1] = 1/(2*(-2)) = -0.25
        assertEquals(-0.5,  Q[0], 1e-10);
        assertEquals( 0.0,  Q[1], 1e-10);
        assertEquals( 0.0,  Q[2], 1e-10);
        assertEquals(-0.25, Q[3], 1e-10);
    }

    @Test
    @DisplayName("Lyapunov continuous (instance): residual check on 3x3 stable matrix")
    void testContinuousLyapunovResidual3x3_instance() {
        int n = 3;
        double[] A = {
            -3,  1,  0,
             0, -2,  1,
             0,  0, -1
        };
        double[] Q = {
             2, -1,  0,
            -1,  3, -1,
             0, -1,  2
        };
        double[] Aorig = A.clone();
        double[] Qorig = Q.clone();

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());
        assertTrue(schur.lyapunov(Q));

        double[] R = continuousResidual(Aorig, Q, n);
        double machEps = Math.ulp(1.0);
        double qNorm = 0;
        for (double v : Qorig) qNorm = Math.max(qNorm, Math.abs(v));
        assertEquals(0, maxAbs(R, Qorig, n), 10.0 * n * machEps * qNorm + 1e-12);
    }

    @Test
    @DisplayName("Lyapunov continuous (instance): sign=-1 gives AX + XA^T = -Q")
    void testContinuousLyapunovNegSign_instance() {
        int n = 2;
        double[] A = { -1, 0,  0, -2 };
        double[] Q = {  1, 0,  0,  1 };
        double[] Aorig = A.clone();
        double[] Qorig = Q.clone();

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());
        assertTrue(schur.lyapunov(Q, -1.0));

        double[] R = continuousResidual(Aorig, Q, n);
        for (int i = 0; i < n * n; i++) {
            assertEquals(-Qorig[i], R[i], 1e-10);
        }
    }

    @Test
    @DisplayName("Lyapunov continuous (instance): 4x4 general stable matrix")
    void testContinuousLyapunov4x4_instance() {
        int n = 4;
        double[] A = {
            -4,  1,  0,  0,
             0, -3,  1,  0,
             0,  0, -2,  1,
             0,  0,  0, -1
        };
        double[] Q = {
             4, 0, 0, 0,
             0, 3, 0, 0,
             0, 0, 2, 0,
             0, 0, 0, 1
        };
        double[] Aorig = A.clone();
        double[] Qorig = Q.clone();

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());
        assertTrue(schur.lyapunov(Q));

        double[] R = continuousResidual(Aorig, Q, n);
        double machEps = Math.ulp(1.0);
        double qNorm = 0;
        for (double v : Qorig) qNorm = Math.max(qNorm, Math.abs(v));
        assertEquals(0, maxAbs(R, Qorig, n), 10.0 * n * machEps * qNorm + 1e-12);
    }

    @Test
    @DisplayName("Lyapunov continuous (instance): no Z throws IllegalStateException")
    void testContinuousLyapunov_noZ() {
        int n = 2;
        double[] A = { -1, 0,  0, -2 };
        double[] Q = {  1, 0,  0,  1 };

        // computeZ=false: Z is null, should throw
        Schur schur = Schur.decompose(A.clone(), n, false);
        assertTrue(schur.ok());
        assertNull(schur.Z(), "Z should be null when computeZ=false");

        assertThrows(IllegalStateException.class, () -> schur.lyapunov(Q));
        assertThrows(IllegalStateException.class, () -> schur.lyapunov(Q, 1.0));
    }

    @Test
    @DisplayName("Lyapunov continuous (instance): n=1 boundary case")
    void testContinuousLyapunov_n1_instance() {
        // Continuous: a*x + x*a = q  =>  2*a*x = q  =>  x = q/(2*a)
        int n = 1;
        double[] A = { -3.0 };
        double[] Q = {  6.0 };

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());
        assertTrue(schur.lyapunov(Q));

        // x = 6.0 / (2 * -3.0) = -1.0
        assertEquals(-1.0, Q[0], 1e-10);

        // Verify residual: A*X + X*A^T = Q_orig
        double[] Aorig = { -3.0 };
        double[] Qorig = { 6.0 };
        double[] R = continuousResidual(Aorig, Q, n);
        double machEps = Math.ulp(1.0);
        assertEquals(0, maxAbs(R, Qorig, n), 10.0 * n * machEps * Math.abs(Qorig[0]) + 1e-12);
    }

    // ==================== Instance method tests for discreteLyapunov ====================

    @Test
    @DisplayName("Lyapunov discrete (instance): stable 2x2 diagonal A")
    void testDiscreteLyapunov2x2_instance() {
        int n = 2;
        double[] A = { 0.5, 0,  0, 0.3 };
        double[] Q = {  1,  0,  0,  1  };
        double[] Aorig = A.clone();
        double[] Qorig = Q.clone();

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());
        assertTrue(schur.discreteLyapunov(A.clone(), Q));

        double[] R = discreteResidual(Aorig, Q, n);
        assertEquals(0, maxAbs(R, Qorig, n), 1e-9);
    }

    @Test
    @DisplayName("Lyapunov discrete (instance): residual check on 3x3 stable matrix")
    void testDiscreteLyapunovResidual3x3_instance() {
        int n = 3;
        double[] A = {
             0.5, 0.1,  0.0,
             0.0, 0.4,  0.1,
             0.0, 0.0,  0.3
        };
        double[] Q = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };
        double[] Aorig = A.clone();
        double[] Qorig = Q.clone();

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());
        assertTrue(schur.discreteLyapunov(A.clone(), Q));

        double[] R = discreteResidual(Aorig, Q, n);
        assertEquals(0, maxAbs(R, Qorig, n), 1e-9);
    }

    @Test
    @DisplayName("Lyapunov discrete (instance): 4x4 general stable matrix")
    void testDiscreteLyapunov4x4_instance() {
        int n = 4;
        double[] A = {
             0.8,  0.1,  0.0,  0.0,
             0.0,  0.6,  0.1,  0.0,
             0.0,  0.0,  0.4,  0.1,
             0.0,  0.0,  0.0,  0.2
        };
        double[] Q = { 2,0,0,0, 0,2,0,0, 0,0,2,0, 0,0,0,2 };
        double[] Aorig = A.clone();
        double[] Qorig = Q.clone();

        Schur schur = Schur.decompose(A.clone(), n, true);
        assertTrue(schur.ok());
        assertTrue(schur.discreteLyapunov(A.clone(), Q));

        double[] R = discreteResidual(Aorig, Q, n);
        assertEquals(0, maxAbs(R, Qorig, n), 1e-9);
    }

    @Test
    @DisplayName("Lyapunov discrete (instance): no Z throws IllegalStateException")
    void testDiscreteLyapunov_noZ() {
        int n = 2;
        double[] A = { 0.5, 0,  0, 0.3 };
        double[] Q = {  1,  0,  0,  1  };

        Schur schur = Schur.decompose(A.clone(), n, false);
        assertTrue(schur.ok());
        assertNull(schur.Z());

        assertThrows(IllegalStateException.class, () -> schur.discreteLyapunov(A.clone(), Q));
    }
}

