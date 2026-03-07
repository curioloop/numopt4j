/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg;

import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.linalg.mat.Cholesky;
import com.curioloop.numopt4j.linalg.mat.GEVD;
import com.curioloop.numopt4j.linalg.mat.LU;
import com.curioloop.numopt4j.linalg.mat.QR;
import com.curioloop.numopt4j.linalg.mat.SVD;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

/**
 * Tests for the Decomposer facade and the refactored decomposition API.
 * Feature: linalg-api-refactor
 */
class DecomposerTest {

    private static final double EPSILON = 1e-8;

    // =========================================================================
    // 8.1 Unit tests
    // =========================================================================

    @Test
    void luFacadeBasicCall() {
        double[] A = {4, 3, 6, 3};
        LU lu = Decomposer.lu(A, 2);
        assertThat(lu.ok()).isTrue();
    }

    @Test
    void luAllocNonNull() {
        double[] A = {4, 3, 6, 3};
        LU lu = Decomposer.lu(A, 2);
        LU.Pool pool = lu.work();
        assertThat(pool).isNotNull();
    }

    @Test
    void luWorkCovariantNocast() {
        double[] A = {4, 3, 6, 3};
        LU lu = Decomposer.lu(A, 2);
        // work() returns LU.Pool directly — no cast needed
        LU.Pool pool = lu.work();
        assertThat(pool).isNotNull();
        assertThat(pool).isSameAs(lu.work());
    }

    @Test
    void luSingularMatrixToMethodsReturnNull() {
        double[] A = {1, 2, 2, 4}; // singular
        LU lu = Decomposer.lu(A, 2);
        assertThat(lu.ok()).isFalse();
        assertThat(lu.toL()).isNull();
        assertThat(lu.toU()).isNull();
        assertThat(lu.toP()).isNull();
    }

    @Test
    void qrFacadeBasicCall() {
        double[] A = {1, 2, 3, 4, 5, 6};
        QR qr = Decomposer.qr(A, 3, 2);
        assertThat(qr.ok()).isTrue();
    }

    @Test
    void qrAllocNonNull() {
        double[] A = {1, 2, 3, 4};
        QR qr = Decomposer.qr(A, 2, 2);
        assertThat(qr.work()).isNotNull();
    }

    @Test
    void qrWorkCovariantNocast() {
        double[] A = {1, 2, 3, 4};
        QR qr = Decomposer.qr(A, 2, 2);
        QR.Pool pool = qr.work();
        assertThat(pool).isNotNull();
        assertThat(pool).isSameAs(qr.work());
    }

    @Test
    void qrNonPivotingToPReturnsNull() {
        double[] A = {1, 2, 3, 4};
        QR qr = Decomposer.qr(A, 2, 2); // no PIVOTING opt
        assertThat(qr.ok()).isTrue();
        assertThat(qr.toP()).isNull();
    }

    @Test
    void qrPivotingToPNotNull() {
        double[] A = {1, 2, 3, 4};
        QR qr = Decomposer.qr(A, 2, 2, QR.Opts.PIVOTING);
        assertThat(qr.ok()).isTrue();
        assertThat(qr.toP()).isNotNull();
    }

    @Test
    void svdFacadeBasicCall() {
        double[] A = {1, 2, 3, 4};
        SVD svd = Decomposer.svd(A, 2, 2);
        assertThat(svd.ok()).isTrue();
        assertThat(svd.toU()).isNotNull();
        assertThat(svd.toVT()).isNotNull();
    }

    @Test
    void svdAllocNonNull() {
        double[] A = {1, 2, 3, 4};
        SVD svd = Decomposer.svd(A, 2, 2);
        assertThat(svd.work()).isNotNull();
    }

    @Test
    void svdWorkCovariantNocast() {
        double[] A = {1, 2, 3, 4};
        SVD svd = Decomposer.svd(A, 2, 2);
        SVD.Pool pool = svd.work();
        assertThat(pool).isNotNull();
        assertThat(pool).isSameAs(svd.work());
    }

    @Test
    void svdNoOptsDefaultsToAll() {
        double[] A = {1, 2, 3, 4};
        SVD svd = Decomposer.svd(A, 2, 2); // no opts → SVD_ALL
        assertThat(svd.ok()).isTrue();
        assertThat(svd.toU()).isNotNull();
        assertThat(svd.toVT()).isNotNull();
    }

    @Test
    void svdWantUOnlyVTNull() {
        double[] A = {1, 2, 3, 4};
        SVD svd = Decomposer.svd(A, 2, 2, SVD.Opts.WANT_U);
        assertThat(svd.ok()).isTrue();
        assertThat(svd.toU()).isNotNull();
        assertThat(svd.toVT()).isNull();
    }

    @Test
    void svdKindNoneToUAndVTReturnNull() {
        // Pass no opts but use the int-kind overload directly for SVD_NONE
        double[] A = {1, 2, 3, 4};
        SVD svd = SVD.decompose(A, 2, 2, SVD.SVD_NONE, (SVD.Pool) null);
        assertThat(svd.ok()).isTrue();
        assertThat(svd.toU()).isNull();
        assertThat(svd.toVT()).isNull();
    }

    @Test
    void choleskyFacadeBasicCall() {
        double[] A = {4, 2, 2, 3};
        Cholesky chol = Decomposer.cholesky(A, 2); // default: lower, no pivoting
        assertThat(chol.ok()).isTrue();
    }

    @Test
    void choleskyUpperOpt() {
        double[] A = {4, 2, 2, 3};
        Cholesky chol = Decomposer.cholesky(A, 2, Cholesky.Opts.UPPER);
        assertThat(chol.ok()).isTrue();
        assertThat(chol.uplo()).isEqualTo(BLAS.Uplo.Upper);
    }

    @Test
    void choleskyAllocNonNull() {
        double[] A = {4, 2, 2, 3};
        Cholesky chol = Decomposer.cholesky(A, 2);
        assertThat(chol.work()).isNotNull();
    }

    @Test
    void choleskyWorkCovariantNocast() {
        double[] A = {4, 2, 2, 3};
        Cholesky chol = Decomposer.cholesky(A, 2);
        Cholesky.Pool pool = chol.work();
        assertThat(pool).isNotNull();
        assertThat(pool).isSameAs(chol.work());
    }

    @Test
    void choleskyNonPivotingToDReturnsNull() {
        double[] A = {4, 2, 2, 3};
        Cholesky chol = Decomposer.cholesky(A, 2); // no PIVOTING opt
        assertThat(chol.ok()).isTrue();
        assertThat(chol.toD()).isNull();
    }

    @Test
    void choleskyPivotingOptToDNotNull() {
        double[] A = {4, 2, 2, 3};
        Cholesky ldl = Decomposer.cholesky(A, 2, Cholesky.Opts.PIVOTING);
        assertThat(ldl.ok()).isTrue();
        assertThat(ldl.toD()).isNotNull();
    }

    @Test
    void workspaceReuseReferenceEquality() {
        double[] A1 = {4, 3, 6, 3};
        LU lu1 = Decomposer.lu(A1, 2);
        LU.Pool ws = lu1.work();

        double[] A2 = {2, 1, 1, 3};
        LU lu2 = Decomposer.lu(A2, 2, ws);
        assertThat(lu2.work()).isSameAs(ws);
    }

    @Test
    void qrWorkspaceReuseWithOpts() {
        double[] A1 = {1, 2, 3, 4};
        QR qr1 = Decomposer.qr(A1, 2, 2, QR.Opts.PIVOTING);
        QR.Pool ws = qr1.work();

        double[] A2 = {5, 6, 7, 8};
        QR qr2 = Decomposer.qr(A2, 2, 2, ws, QR.Opts.PIVOTING);
        assertThat(qr2.work()).isSameAs(ws);
    }

    @Test
    void svdWorkspaceReuseWithOpts() {
        double[] A1 = {1, 2, 3, 4};
        SVD svd1 = Decomposer.svd(A1, 2, 2, SVD.Opts.WANT_U);
        SVD.Pool ws = svd1.work();

        double[] A2 = {5, 6, 7, 8};
        SVD svd2 = Decomposer.svd(A2, 2, 2, ws, SVD.Opts.WANT_U);
        assertThat(svd2.work()).isSameAs(ws);
    }

    @Test
    void choleskyWorkspaceReuseWithOpts() {
        double[] A1 = {4, 2, 2, 3};
        Cholesky c1 = Decomposer.cholesky(A1, 2, Cholesky.Opts.PIVOTING);
        Cholesky.Pool ws = c1.work();

        double[] A2 = {9, 3, 3, 4};
        Cholesky c2 = Decomposer.cholesky(A2, 2, ws, Cholesky.Opts.PIVOTING);
        assertThat(c2.work()).isSameAs(ws);
    }

    // =========================================================================
    // GEVD
    // =========================================================================

    // A = [2,1; 1,2], B = I  →  standard eigenvalues 1 and 3
    private static final double[] A2 = {2.0, 1.0, 1.0, 2.0};
    private static final double[] B2 = {1.0, 0.0, 0.0, 1.0};

    @Test
    void gevdFacadeBasicCall() {
        GEVD eg = Decomposer.gevd(A2.clone(), B2.clone(), 2);
        assertThat(eg.ok()).isTrue();
        assertThat(eg.toS()).isNotNull();
        assertThat(eg.toV()).isNotNull();
    }

    @Test
    void gevdDefaultIsType1Lower() {
        GEVD eg = Decomposer.gevd(A2.clone(), B2.clone(), 2);
        assertThat(eg.ok()).isTrue();
        assertThat(eg.type()).isEqualTo(1);
        assertThat(eg.toS().data[0]).isCloseTo(1.0, offset(EPSILON));
        assertThat(eg.toS().data[1]).isCloseTo(3.0, offset(EPSILON));
    }

    @Test
    void gevdUpperOpt() {
        GEVD eg = Decomposer.gevd(A2.clone(), B2.clone(), 2, GEVD.Opts.UPPER);
        assertThat(eg.ok()).isTrue();
        assertThat(eg.toS().data[0]).isCloseTo(1.0, offset(EPSILON));
        assertThat(eg.toS().data[1]).isCloseTo(3.0, offset(EPSILON));
    }

    @Test
    void gevdType2Opt() {
        GEVD eg = Decomposer.gevd(A2.clone(), B2.clone(), 2, GEVD.Opts.TYPE2);
        assertThat(eg.ok()).isTrue();
        assertThat(eg.type()).isEqualTo(2);
        // B=I → A·B = A, same eigenvalues
        assertThat(eg.toS().data[0]).isCloseTo(1.0, offset(EPSILON));
        assertThat(eg.toS().data[1]).isCloseTo(3.0, offset(EPSILON));
    }

    @Test
    void gevdType3Opt() {
        GEVD eg = Decomposer.gevd(A2.clone(), B2.clone(), 2, GEVD.Opts.TYPE3);
        assertThat(eg.ok()).isTrue();
        assertThat(eg.type()).isEqualTo(3);
        assertThat(eg.toS().data[0]).isCloseTo(1.0, offset(EPSILON));
        assertThat(eg.toS().data[1]).isCloseTo(3.0, offset(EPSILON));
    }

    @Test
    void gevdAllocNonNull() {
        GEVD eg = Decomposer.gevd(A2.clone(), B2.clone(), 2);
        assertThat(eg.work()).isNotNull();
    }

    @Test
    void gevdWorkCovariantNocast() {
        GEVD eg = Decomposer.gevd(A2.clone(), B2.clone(), 2);
        GEVD.Pool pool = eg.work();
        assertThat(pool).isNotNull();
        assertThat(pool).isSameAs(eg.work());
    }

    @Test
    void gevdNonPdBReturnsNotOk() {
        double[] Abad = {1.0, 0.0, 0.0, -1.0}; // not positive-definite
        GEVD eg = Decomposer.gevd(A2.clone(), Abad, 2);
        assertThat(eg.ok()).isFalse();
        assertThat(eg.toS()).isNull();
        assertThat(eg.toV()).isNull();
    }

    @Test
    void gevdWorkspaceReuse() {
        GEVD eg1 = Decomposer.gevd(A2.clone(), B2.clone(), 2);
        GEVD.Pool ws = eg1.work();

        double[] A2b = {4.0, 1.0, 1.0, 3.0};
        double[] B2b = {2.0, 0.0, 0.0, 2.0};
        GEVD eg2 = Decomposer.gevd(A2b, B2b, 2, ws);
        assertThat(eg2.ok()).isTrue();
        assertThat(eg2.work()).isSameAs(ws);
    }

    @Test
    void gevdEigenvectorsOrthonormal() {
        // For type 1 with B=I, eigenvectors should be orthonormal
        GEVD eg = Decomposer.gevd(A2.clone(), B2.clone(), 2);
        assertThat(eg.ok()).isTrue();
        double[] Q = eg.toV().data;
        // Q^T * Q = I  (columns are orthonormal)
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                double dot = Q[0 * 2 + i] * Q[0 * 2 + j] + Q[1 * 2 + i] * Q[1 * 2 + j];
                assertThat(dot).isCloseTo(i == j ? 1.0 : 0.0, offset(EPSILON));
            }
        }
    }
}
