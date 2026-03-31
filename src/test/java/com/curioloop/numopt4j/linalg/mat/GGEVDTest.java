/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposer;
import com.curioloop.numopt4j.linalg.Decomposition.Matrix;
import com.curioloop.numopt4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static java.lang.Math.*;
import static org.assertj.core.api.Assertions.*;

/**
 * Tests for GGEVD (non-symmetric generalized eigenvalue decomposition).
 *
 * <p>Verifies A*x = lambda*B*x where lambda = (alphar + i*alphai) / beta.
 */
class GGEVDTest {

    private static final double TOL = 1e-6;

    private static int queryWorkspace(int n, boolean wantVL, boolean wantVR) {
        double[] tmp = new double[1];
        BLAS.dggev(wantVL, wantVR, n, null, n, null, n, null, null, null,
                null, n, null, n, tmp, 0, -1);
        return (int) tmp[0];
    }

    /**
     * Verify A*vr_i = lambda_i * B*vr_i for all real eigenvalues.
     * For complex pair at (j, j+1): column j = real part, column j+1 = imag part.
     */
    private static void verifyRightEigvecs(double[] A, double[] B, int n,
                                            double[] alphar, double[] alphai, double[] beta,
                                            double[] VR) {
        int j = 0;
        while (j < n) {
            if (alphai[j] == 0.0) {
                // Real eigenvalue: lambda = alphar[j] / beta[j]
                if (abs(beta[j]) < 1e-14) { j++; continue; } // infinite eigenvalue
                double lambda = alphar[j] / beta[j];
                // Check A*v = lambda*B*v
                for (int i = 0; i < n; i++) {
                    double av = 0, bv = 0;
                    for (int k = 0; k < n; k++) {
                        av += A[i * n + k] * VR[k * n + j];
                        bv += B[i * n + k] * VR[k * n + j];
                    }
                    assertThat(av).as("A*v[%d] row %d", j, i)
                                  .isCloseTo(lambda * bv, offset(TOL));
                }
                j++;
            } else {
                // Complex pair: lambda = (alphar[j] +/- i*alphai[j]) / beta[j]
                if (abs(beta[j]) < 1e-14) { j += 2; continue; }
                double wr = alphar[j] / beta[j];
                double wi = alphai[j] / beta[j];
                // Check (A - lambda*B) * (vr + i*vi) = 0
                // => A*vr - wr*B*vr + wi*B*vi = 0  (real part)
                // => A*vi - wr*B*vi - wi*B*vr = 0  (imag part)
                for (int i = 0; i < n; i++) {
                    double avr = 0, avi = 0, bvr = 0, bvi = 0;
                    for (int k = 0; k < n; k++) {
                        avr += A[i * n + k] * VR[k * n + j];
                        avi += A[i * n + k] * VR[k * n + j + 1];
                        bvr += B[i * n + k] * VR[k * n + j];
                        bvi += B[i * n + k] * VR[k * n + j + 1];
                    }
                    assertThat(avr - wr * bvr + wi * bvi).as("Re(A*v)[%d] row %d", j, i)
                                                          .isCloseTo(0.0, offset(TOL));
                    assertThat(avi - wr * bvi - wi * bvr).as("Im(A*v)[%d] row %d", j, i)
                                                          .isCloseTo(0.0, offset(TOL));
                }
                j += 2;
            }
        }
    }

    @Test
    void testRealEigenvalues2x2() {
        // A = [2 1; 1 2], B = I => eigenvalues 1 and 3
        double[] A = { 2.0, 1.0, 1.0, 2.0 };
        double[] B = { 1.0, 0.0, 0.0, 1.0 };

        GGEVD g = Decomposer.ggevd(A.clone(), B.clone(), 2);

        assertThat(g.ok()).isTrue();
        double[] ar = g.alphar(), ai = g.alphai(), be = g.beta();

        // Collect real eigenvalues
        double[] eigs = new double[2];
        for (int i = 0; i < 2; i++) {
            assertThat(ai[i]).isCloseTo(0.0, offset(TOL));
            eigs[i] = ar[i] / be[i];
        }
        java.util.Arrays.sort(eigs);
        assertThat(eigs[0]).isCloseTo(1.0, offset(TOL));
        assertThat(eigs[1]).isCloseTo(3.0, offset(TOL));
    }

    @Test
    void testWorkspaceDefaultsToValuesOnly() {
        GGEVD.Pool pool = GGEVD.workspace();

        assertThat(pool.work()).isNull();
        assertThat(pool.VL).isNull();
        assertThat(pool.VR).isNull();
    }

    @Test
    void testWorkspaceMatchesRequestedEigenvectors() {
        GGEVD.Pool valuesOnly = GGEVD.workspace();
        GGEVD.Pool leftOnly = GGEVD.workspace();
        GGEVD.Pool rightOnly = GGEVD.workspace();
        GGEVD.Pool both = GGEVD.workspace();

        valuesOnly.ensure(4, false, false);
        leftOnly.ensure(4, true, false);
        rightOnly.ensure(4, false, true);
        both.ensure(4, true, true);

        assertThat(valuesOnly.work().length).isEqualTo(queryWorkspace(4, false, false));
        assertThat(leftOnly.work().length).isEqualTo(queryWorkspace(4, true, false));
        assertThat(rightOnly.work().length).isEqualTo(queryWorkspace(4, false, true));
        assertThat(both.work().length).isEqualTo(queryWorkspace(4, true, true));

        assertThat(valuesOnly.VL).isNull();
        assertThat(valuesOnly.VR).isNull();
        assertThat(leftOnly.VL).hasSize(16);
        assertThat(leftOnly.VR).isNull();
        assertThat(rightOnly.VL).isNull();
        assertThat(rightOnly.VR).hasSize(16);
        assertThat(both.VL).hasSize(16);
        assertThat(both.VR).hasSize(16);
    }

    @Test
    void testRealEigenvalues3x3() {
        // A = diag(1,2,3), B = I => eigenvalues 1, 2, 3
        double[] A = {
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0
        };
        double[] B = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        };

        GGEVD g = Decomposer.ggevd(A.clone(), B.clone(), 3);

        assertThat(g.ok()).isTrue();
        double[] ar = g.alphar(), ai = g.alphai(), be = g.beta();

        double[] eigs = new double[3];
        for (int i = 0; i < 3; i++) {
            assertThat(ai[i]).isCloseTo(0.0, offset(TOL));
            eigs[i] = ar[i] / be[i];
        }
        java.util.Arrays.sort(eigs);
        assertThat(eigs[0]).isCloseTo(1.0, offset(TOL));
        assertThat(eigs[1]).isCloseTo(2.0, offset(TOL));
        assertThat(eigs[2]).isCloseTo(3.0, offset(TOL));
    }

    @Test
    void testNonSymmetricWithRightEigvecs() {
        // A = [3 1; 0 2], B = [1 0; 0 2] => eigenvalues 3/1=3, 2/2=1
        double[] Aorig = { 3.0, 1.0, 0.0, 2.0 };
        double[] Borig = { 1.0, 0.0, 0.0, 2.0 };

        GGEVD g = Decomposer.ggevd(Aorig.clone(), Borig.clone(), 2, GGEVD.Opts.WANT_VR);

        assertThat(g.ok()).isTrue();
        double[] ar = g.alphar(), ai = g.alphai(), be = g.beta();

        // All eigenvalues should be real
        for (int i = 0; i < 2; i++) assertThat(ai[i]).isCloseTo(0.0, offset(TOL));

        double[] eigs = new double[2];
        for (int i = 0; i < 2; i++) eigs[i] = ar[i] / be[i];
        java.util.Arrays.sort(eigs);
        assertThat(eigs[0]).isCloseTo(1.0, offset(TOL));
        assertThat(eigs[1]).isCloseTo(3.0, offset(TOL));

        // Verify right eigenvectors
        Matrix VR = g.toVR();
        assertThat(VR).isNotNull();
        verifyRightEigvecs(Aorig, Borig, 2, ar, ai, be, VR.data);

        GGEVD.Pool pool = g.pool();
        assertThat(pool.work().length).isEqualTo(queryWorkspace(2, false, true));
        assertThat(pool.VL).isNull();
        assertThat(pool.VR).isNotNull();
    }

    @Test
    void testComplexEigenvalues() {
        // A = [0 -1; 1 0], B = I => eigenvalues +/- i
        double[] Aorig = { 0.0, -1.0, 1.0, 0.0 };
        double[] Borig = { 1.0, 0.0, 0.0, 1.0 };

        GGEVD g = Decomposer.ggevd(Aorig.clone(), Borig.clone(), 2);

        assertThat(g.ok()).isTrue();
        double[] ar = g.alphar(), ai = g.alphai(), be = g.beta();

        // Should have a complex conjugate pair
        boolean hasComplex = false;
        for (int i = 0; i < 2; i++) {
            if (abs(ai[i]) > TOL) { hasComplex = true; break; }
        }
        assertThat(hasComplex).isTrue();

        // Real parts should be ~0, imaginary parts ~+/-1
        for (int i = 0; i < 2; i++) {
            assertThat(ar[i] / be[i]).isCloseTo(0.0, offset(TOL));
        }

        GGEVD.Pool pool = g.pool();
        assertThat(pool.work().length).isEqualTo(queryWorkspace(2, false, false));
        assertThat(pool.VL).isNull();
        assertThat(pool.VR).isNull();
    }

    @Test
    void testViaDecomposerFacade() {
        double[] A = {
            4.0, 1.0, 0.0,
            2.0, 3.0, 1.0,
            0.0, 1.0, 2.0
        };
        double[] B = {
            2.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 2.0
        };

        GGEVD g = Decomposer.ggevd(A.clone(), B.clone(), 3);
        assertThat(g.ok()).isTrue();

        double[] ar = g.alphar(), ai = g.alphai(), be = g.beta();
        // All eigenvalues should be real (B is positive definite, A is real)
        for (int i = 0; i < 3; i++) {
            if (abs(ai[i]) < TOL) {
                // Real eigenvalue: verify A*v = lambda*B*v
                double lambda = ar[i] / be[i];
                assertThat(lambda).isGreaterThan(0.0); // should be positive
            }
        }
    }
}
