/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.blas.BLAS;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class GEVDTest {

    private static final double EPSILON = 1e-8;

    // Verify A*v_i = lambda_i * B*v_i for all eigenpairs
    private static void verifyType1(double[] Aorig, double[] Borig, int n, double[] w, double[] V) {
        for (int i = 0; i < n; i++) {
            double[] Av = new double[n];
            double[] Bv = new double[n];
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++) {
                    Av[j] += Aorig[j * n + k] * V[k * n + i];
                    Bv[j] += Borig[j * n + k] * V[k * n + i];
                }
            for (int j = 0; j < n; j++)
                assertThat(Av[j]).isCloseTo(w[i] * Bv[j], offset(1e-6));
        }
    }

    // Verify A*B*v_i = lambda_i * v_i for all eigenpairs
    private static void verifyType2(double[] Aorig, double[] Borig, int n, double[] w, double[] V) {
        for (int i = 0; i < n; i++) {
            double[] Bv = new double[n];
            double[] ABv = new double[n];
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    Bv[j] += Borig[j * n + k] * V[k * n + i];
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    ABv[j] += Aorig[j * n + k] * Bv[k];
            for (int j = 0; j < n; j++)
                assertThat(ABv[j]).isCloseTo(w[i] * V[j * n + i], offset(1e-6));
        }
    }

    // Verify B*A*v_i = lambda_i * v_i for all eigenpairs
    private static void verifyType3(double[] Aorig, double[] Borig, int n, double[] w, double[] V) {
        for (int i = 0; i < n; i++) {
            double[] Av = new double[n];
            double[] BAv = new double[n];
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    Av[j] += Aorig[j * n + k] * V[k * n + i];
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    BAv[j] += Borig[j * n + k] * Av[k];
            for (int j = 0; j < n; j++)
                assertThat(BAv[j]).isCloseTo(w[i] * V[j * n + i], offset(1e-6));
        }
    }

    @Test
    void testType1Basic() {
        double[] A = { 2.0, 1.0, 1.0, 2.0 };
        double[] B = { 1.0, 0.0, 0.0, 1.0 };

        GEVD eg = GEVD.decompose(A.clone(), B.clone(), 2, BLAS.Uplo.Lower);

        assertThat(eg.ok()).isTrue();
        assertThat(eg.toS().data[0]).isCloseTo(1.0, offset(EPSILON));
        assertThat(eg.toS().data[1]).isCloseTo(3.0, offset(EPSILON));
    }

    @Test
    void testType1WithNonIdentityB() {
        double[] Aorig = { 2.0, 1.0, 1.0, 2.0 };
        double[] Borig = { 4.0, 0.0, 0.0, 1.0 };

        GEVD eg = GEVD.decompose(Aorig.clone(), Borig.clone(), 2, BLAS.Uplo.Lower);

        assertThat(eg.ok()).isTrue();
        verifyType1(Aorig, Borig, 2, eg.toS().data, eg.toV().data);
    }

    @Test
    void testType2IdentityB() {
        // A*B*x = lambda*x with B=I reduces to A*x = lambda*x
        double[] Aorig = { 2.0, 1.0, 1.0, 2.0 };
        double[] Borig = { 1.0, 0.0, 0.0, 1.0 };

        GEVD eg = GEVD.decompose(Aorig.clone(), Borig.clone(), 2, BLAS.Uplo.Lower, 2, null);

        assertThat(eg.ok()).isTrue();
        assertThat(eg.type()).isEqualTo(2);
        // With B=I, eigenvalues of A*B = A, same as standard eigenvalues
        assertThat(eg.toS().data[0]).isCloseTo(1.0, offset(EPSILON));
        assertThat(eg.toS().data[1]).isCloseTo(3.0, offset(EPSILON));
        verifyType2(Aorig, Borig, 2, eg.toS().data, eg.toV().data);
    }

    @Test
    void testType2WithNonIdentityB() {
        double[] Aorig = { 2.0, 1.0, 1.0, 2.0 };
        double[] Borig = { 4.0, 0.0, 0.0, 1.0 };

        GEVD eg = GEVD.decompose(Aorig.clone(), Borig.clone(), 2, BLAS.Uplo.Lower, 2, null);

        assertThat(eg.ok()).isTrue();
        verifyType2(Aorig, Borig, 2, eg.toS().data, eg.toV().data);
    }

    @Test
    void testType3IdentityB() {
        // B*A*x = lambda*x with B=I reduces to A*x = lambda*x
        double[] Aorig = { 2.0, 1.0, 1.0, 2.0 };
        double[] Borig = { 1.0, 0.0, 0.0, 1.0 };

        GEVD eg = GEVD.decompose(Aorig.clone(), Borig.clone(), 2, BLAS.Uplo.Lower, 3, null);

        assertThat(eg.ok()).isTrue();
        assertThat(eg.type()).isEqualTo(3);
        assertThat(eg.toS().data[0]).isCloseTo(1.0, offset(EPSILON));
        assertThat(eg.toS().data[1]).isCloseTo(3.0, offset(EPSILON));
        verifyType3(Aorig, Borig, 2, eg.toS().data, eg.toV().data);
    }

    @Test
    void testType3WithNonIdentityB() {
        double[] Aorig = { 2.0, 1.0, 1.0, 2.0 };
        double[] Borig = { 4.0, 0.0, 0.0, 1.0 };

        GEVD eg = GEVD.decompose(Aorig.clone(), Borig.clone(), 2, BLAS.Uplo.Lower, 3, null);

        assertThat(eg.ok()).isTrue();
        verifyType3(Aorig, Borig, 2, eg.toS().data, eg.toV().data);
    }

    @Test
    void testConditionNumber() {
        double[] A = { 2.0, 1.0, 1.0, 2.0 };
        double[] B = { 1.0, 0.0, 0.0, 1.0 };

        GEVD eg = GEVD.decompose(A.clone(), B.clone(), 2, BLAS.Uplo.Lower);

        assertThat(eg.ok()).isTrue();
        assertThat(eg.cond()).isCloseTo(3.0, offset(EPSILON));
    }

    @Test
    void testLargerMatrixType1() {
        double[] Aorig = {
            6.0, 3.0, 1.0,
            3.0, 5.0, 2.0,
            1.0, 2.0, 4.0
        };
        double[] Borig = {
            2.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0
        };

        GEVD eg = GEVD.decompose(Aorig.clone(), Borig.clone(), 3, BLAS.Uplo.Lower);

        assertThat(eg.ok()).isTrue();
        double[] eigenvalues = eg.toS().data;
        for (int i = 1; i < 3; i++)
            assertThat(eigenvalues[i]).isGreaterThanOrEqualTo(eigenvalues[i - 1]);
        verifyType1(Aorig, Borig, 3, eigenvalues, eg.toV().data);
    }

    @Test
    void testLargerMatrixType2() {
        double[] Aorig = {
            6.0, 3.0, 1.0,
            3.0, 5.0, 2.0,
            1.0, 2.0, 4.0
        };
        double[] Borig = {
            2.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0
        };

        GEVD eg = GEVD.decompose(Aorig.clone(), Borig.clone(), 3, BLAS.Uplo.Lower, 2, null);

        assertThat(eg.ok()).isTrue();
        verifyType2(Aorig, Borig, 3, eg.toS().data, eg.toV().data);
    }

    @Test
    void testLargerMatrixType3() {
        double[] Aorig = {
            6.0, 3.0, 1.0,
            3.0, 5.0, 2.0,
            1.0, 2.0, 4.0
        };
        double[] Borig = {
            2.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0
        };

        GEVD eg = GEVD.decompose(Aorig.clone(), Borig.clone(), 3, BLAS.Uplo.Lower, 3, null);

        assertThat(eg.ok()).isTrue();
        verifyType3(Aorig, Borig, 3, eg.toS().data, eg.toV().data);
    }

    @Test
    void testUpperTriangle() {
        // Same problem as testType1WithNonIdentityB but using upper triangle
        double[] Aorig = { 2.0, 1.0, 1.0, 2.0 };
        double[] Borig = { 4.0, 0.0, 0.0, 1.0 };

        GEVD eg = GEVD.decompose(Aorig.clone(), Borig.clone(), 2, BLAS.Uplo.Upper);

        assertThat(eg.ok()).isTrue();
        verifyType1(Aorig, Borig, 2, eg.toS().data, eg.toV().data);
    }
}
