/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg;

import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.linalg.mat.Cholesky;
import com.curioloop.numopt4j.linalg.mat.Eigen;
import com.curioloop.numopt4j.linalg.mat.GEVD;
import com.curioloop.numopt4j.linalg.mat.GGEVD;
import com.curioloop.numopt4j.linalg.mat.GSVD;
import com.curioloop.numopt4j.linalg.mat.LQ;
import com.curioloop.numopt4j.linalg.mat.LU;
import com.curioloop.numopt4j.linalg.mat.QR;
import com.curioloop.numopt4j.linalg.mat.SVD;
import com.curioloop.numopt4j.linalg.mat.Schur;

/**
 * Unified facade for matrix decompositions.
 *
 * <p>Each method accepts an optional varargs {@code Pool} for workspace reuse,
 * and an optional varargs {@code Opts} enum for algorithm configuration.</p>
 *
 * <pre>{@code
 * // Basic usage — defaults
 * LU  lu   = Decomposer.lu(A, n);
 * QR  qr   = Decomposer.qr(A, m, n);
 * SVD svd  = Decomposer.svd(A, m, n);
 * Cholesky chol = Decomposer.cholesky(A, n);
 *
 * // With options
 * QR  qrp  = Decomposer.qr(A, m, n, QR.Opts.PIVOTING);
 * SVD svdU = Decomposer.svd(A, m, n, SVD.Opts.WANT_U, SVD.Opts.FULL_U);
 * Cholesky ldl = Decomposer.cholesky(A, n, Cholesky.Opts.UPPER, Cholesky.Opts.PIVOTING);
 *
 * // Workspace reuse
 * LU.Pool ws = lu.alloc();
 * for (double[] b : rhs) {
 *     LU result = Decomposer.lu(b, n, ws);
 * }
 * }</pre>
 */
public final class Decomposer {

    private Decomposer() {}

    // =========================================================================
    // LU
    // =========================================================================

    /**
     * Performs LU decomposition of an n×n matrix.
     *
     * @param A  input matrix (row-major, length >= n*n), overwritten in place
     * @param n  matrix dimension
     * @param ws optional workspace pool for reuse
     */
    public static LU lu(double[] A, int n, LU.Pool... ws) {
        LU.Pool pool = (ws != null && ws.length > 0) ? ws[0] : null;
        return LU.decompose(A, n, pool);
    }

    // =========================================================================
    // QR  —  options: QR.Opts.PIVOTING
    // =========================================================================

    /**
     * Performs QR decomposition of an m×n matrix.
     *
     * <p>Options: {@link QR.Opts#PIVOTING} enables column pivoting.</p>
     *
     * @param A    input matrix (row-major, length >= m*n), overwritten in place
     * @param m    number of rows
     * @param n    number of columns
     * @param opts zero or more {@link QR.Opts} values
     */
    public static QR qr(double[] A, int m, int n, QR.Opts... opts) {
        boolean pivoting = contains(opts, QR.Opts.PIVOTING);
        return QR.decompose(A, m, n, pivoting, (QR.Pool) null);
    }

    /**
     * Performs QR decomposition with workspace reuse.
     *
     * @param A    input matrix (row-major, length >= m*n), overwritten in place
     * @param m    number of rows
     * @param n    number of columns
     * @param ws   workspace pool for reuse
     * @param opts zero or more {@link QR.Opts} values
     */
    public static QR qr(double[] A, int m, int n, QR.Pool ws, QR.Opts... opts) {
        boolean pivoting = contains(opts, QR.Opts.PIVOTING);
        return QR.decompose(A, m, n, pivoting, ws);
    }

    // =========================================================================
    // SVD  —  options: SVD.Opts.WANT_U / WANT_V / FULL_U / FULL_V
    // =========================================================================

    /**
     * Performs SVD of an m×n matrix.
     *
     * <p>Default (no opts): computes both U and Vᵀ in thin form (SVD_ALL).</p>
     * <p>Options: {@link SVD.Opts#WANT_U}, {@link SVD.Opts#WANT_V},
     * {@link SVD.Opts#FULL_U}, {@link SVD.Opts#FULL_V}.</p>
     *
     * @param A    input matrix (row-major, length >= m*n), overwritten in place
     * @param m    number of rows
     * @param n    number of columns
     * @param opts zero or more {@link SVD.Opts} values
     */
    public static SVD svd(double[] A, int m, int n, SVD.Opts... opts) {
        int kind = svdKind(opts);
        return SVD.decompose(A, m, n, kind, (SVD.Pool) null);
    }

    /**
     * Performs SVD with workspace reuse.
     *
     * @param A    input matrix (row-major, length >= m*n), overwritten in place
     * @param m    number of rows
     * @param n    number of columns
     * @param ws   workspace pool for reuse
     * @param opts zero or more {@link SVD.Opts} values
     */
    public static SVD svd(double[] A, int m, int n, SVD.Pool ws, SVD.Opts... opts) {
        int kind = svdKind(opts);
        return SVD.decompose(A, m, n, kind, ws);
    }

    // =========================================================================
    // Cholesky  —  options: Cholesky.Opts.LOWER / UPPER / PIVOTING
    // =========================================================================
    /**
     * Performs Cholesky (or LDLᵀ) decomposition of an n×n symmetric matrix.
     *
     * <p>Default (no opts): standard Cholesky using lower triangle.</p>
     * <p>Options: {@link Cholesky.Opts#UPPER} uses upper triangle;
     * {@link Cholesky.Opts#PIVOTING} switches to pivoted LDLᵀ.</p>
     *
     * @param A    input matrix (row-major, length >= n*n), overwritten in place
     * @param n    matrix dimension
     * @param opts zero or more {@link Cholesky.Opts} values
     */
    public static Cholesky cholesky(double[] A, int n, Cholesky.Opts... opts) {
        BLAS.Uplo uplo = contains(opts, Cholesky.Opts.UPPER) ? BLAS.Uplo.Upper : BLAS.Uplo.Lower;
        boolean pivoting = contains(opts, Cholesky.Opts.PIVOTING);
        return Cholesky.decompose(A, n, uplo, pivoting, null);
    }

    /**
     * Performs Cholesky (or LDLᵀ) decomposition with workspace reuse.
     *
     * @param A    input matrix (row-major, length >= n*n), overwritten in place
     * @param n    matrix dimension
     * @param ws   workspace pool for reuse
     * @param opts zero or more {@link Cholesky.Opts} values
     */
    public static Cholesky cholesky(double[] A, int n, Cholesky.Pool ws, Cholesky.Opts... opts) {
        BLAS.Uplo uplo = contains(opts, Cholesky.Opts.UPPER) ? BLAS.Uplo.Upper : BLAS.Uplo.Lower;
        boolean pivoting = contains(opts, Cholesky.Opts.PIVOTING);
        return Cholesky.decompose(A, n, uplo, pivoting, ws);
    }

    // =========================================================================
    // GEVD  —  options: GEVD.Opts.LOWER / UPPER / TYPE1 / TYPE2 / TYPE3 / WANT_V
    // =========================================================================

    /**
     * Performs generalized symmetric-definite eigenvalue decomposition.
     *
     * <p>Default (no opts): lower triangle, type 1 (A·x = λ·B·x), eigenvalues only.</p>
     * <p>Options: {@link GEVD.Opts#UPPER} uses upper triangle;
     * {@link GEVD.Opts#TYPE2} solves A·B·x = λ·x;
     * {@link GEVD.Opts#TYPE3} solves B·A·x = λ·x;
     * {@link GEVD.Opts#WANT_V} computes eigenvectors.</p>
     *
     * @param A    symmetric matrix (n×n, row-major), overwritten with eigenvectors if WANT_V
     * @param B    symmetric positive-definite matrix (n×n, row-major), overwritten with Cholesky factor
     * @param n    matrix dimension
     * @param opts zero or more {@link GEVD.Opts} values
     */
    public static GEVD gevd(double[] A, double[] B, int n, GEVD.Opts... opts) {
        BLAS.Uplo uplo = contains(opts, GEVD.Opts.UPPER) ? BLAS.Uplo.Upper : BLAS.Uplo.Lower;
        int type = contains(opts, GEVD.Opts.TYPE3) ? 3 : contains(opts, GEVD.Opts.TYPE2) ? 2 : 1;
        boolean valuesOnly = !contains(opts, GEVD.Opts.WANT_V);
        return GEVD.decompose(A, B, n, uplo, type, valuesOnly, null);
    }

    /**
     * Performs generalized symmetric-definite eigenvalue decomposition with workspace reuse.
     *
     * @param A    symmetric matrix (n×n, row-major), overwritten with eigenvectors if WANT_V
     * @param B    symmetric positive-definite matrix (n×n, row-major), overwritten with Cholesky factor
     * @param n    matrix dimension
     * @param ws   workspace pool for reuse
     * @param opts zero or more {@link GEVD.Opts} values
     */
    public static GEVD gevd(double[] A, double[] B, int n, GEVD.Pool ws, GEVD.Opts... opts) {
        BLAS.Uplo uplo = contains(opts, GEVD.Opts.UPPER) ? BLAS.Uplo.Upper : BLAS.Uplo.Lower;
        int type = contains(opts, GEVD.Opts.TYPE3) ? 3 : contains(opts, GEVD.Opts.TYPE2) ? 2 : 1;
        boolean valuesOnly = !contains(opts, GEVD.Opts.WANT_V);
        return GEVD.decompose(A, B, n, uplo, type, valuesOnly, ws);
    }

    // =========================================================================
    // GGEVD  —  options: GGEVD.Opts.WANT_VL / WANT_VR
    // =========================================================================

    /**
     * Performs generalized non-symmetric eigenvalue decomposition: A·x = λ·B·x.
     *
     * <p>Default (no opts): eigenvalues only.</p>
     * <p>Options: {@link GGEVD.Opts#WANT_VL}, {@link GGEVD.Opts#WANT_VR}.</p>
     *
     * @param A    general matrix (n×n, row-major), overwritten
     * @param B    general matrix (n×n, row-major), overwritten
     * @param n    matrix dimension
     * @param opts zero or more {@link GGEVD.Opts} values
     */
    public static GGEVD ggevd(double[] A, double[] B, int n, GGEVD.Opts... opts) {
        return GGEVD.decompose(A, B, n, opts);
    }

    /**
     * Performs generalized non-symmetric eigenvalue decomposition with workspace reuse.
     *
     * @param A    general matrix (n×n, row-major), overwritten
     * @param B    general matrix (n×n, row-major), overwritten
     * @param n    matrix dimension
     * @param ws   workspace pool for reuse
     * @param opts zero or more {@link GGEVD.Opts} values
     */
    public static GGEVD ggevd(double[] A, double[] B, int n, GGEVD.Pool ws, GGEVD.Opts... opts) {
        return GGEVD.decompose(A, B, n, ws, opts);
    }

    // =========================================================================
    // Eigen  —  options: Eigen.Opts.SYMMETRIC_LOWER / SYMMETRIC_UPPER / WANT_LEFT / WANT_RIGHT
    // =========================================================================

    /**
     * Performs eigenvalue decomposition of an n×n matrix.
     *
     * <p>Default (no opts): general (non-symmetric), eigenvalues only.</p>
     * <p>Options: {@link Eigen.Opts#SYMMETRIC_LOWER}, {@link Eigen.Opts#SYMMETRIC_UPPER},
     * {@link Eigen.Opts#WANT_LEFT}, {@link Eigen.Opts#WANT_RIGHT}.</p>
     *
     * @param A    input matrix (n×n, row-major), overwritten
     * @param n    matrix dimension
     * @param opts zero or more {@link Eigen.Opts} values
     */
    public static Eigen eigen(double[] A, int n, Eigen.Opts... opts) {
        return Eigen.decompose(A, n, null, opts);
    }

    /**
     * Performs eigenvalue decomposition with workspace reuse.
     *
     * @param A    input matrix (n×n, row-major), overwritten
     * @param n    matrix dimension
     * @param ws   workspace pool for reuse
     * @param opts zero or more {@link Eigen.Opts} values
     */
    public static Eigen eigen(double[] A, int n, Eigen.Pool ws, Eigen.Opts... opts) {
        return Eigen.decompose(A, n, ws, opts);
    }

    // =========================================================================
    // Schur  —  options: Schur.Opts.WANT_Z / SORT_LHP / SORT_RHP / SORT_IUC / SORT_OUC
    // =========================================================================

    /**
     * Performs real Schur decomposition: A = Z·T·Zᵀ.
     *
     * <p>Default (no opts): computes T only, no Z, no eigenvalue sorting.</p>
     * <p>Options: {@link Schur.Opts#WANT_Z} computes orthogonal matrix Z;
     * {@link Schur.Opts#SORT_LHP}, {@link Schur.Opts#SORT_RHP},
     * {@link Schur.Opts#SORT_IUC}, {@link Schur.Opts#SORT_OUC}.</p>
     *
     * @param A    input matrix (n×n, row-major), overwritten with Schur form T
     * @param n    matrix dimension
     * @param opts zero or more {@link Schur.Opts} values
     */
    public static Schur schur(double[] A, int n, Schur.Opts... opts) {
        return Schur.decompose(A, n, null, opts);
    }

    /**
     * Performs real Schur decomposition with workspace reuse.
     *
     * @param A    input matrix (n×n, row-major), overwritten with Schur form T
     * @param n    matrix dimension
     * @param ws   workspace pool for reuse
     * @param opts zero or more {@link Schur.Opts} values
     */
    public static Schur schur(double[] A, int n, Schur.Pool ws, Schur.Opts... opts) {
        return Schur.decompose(A, n, ws, opts);
    }

    // =========================================================================
    // LQ
    // =========================================================================

    /**
     * Performs LQ decomposition of an m×n matrix (m ≤ n).
     *
     * @param A  input matrix (row-major, length >= m*n), overwritten in place
     * @param m  number of rows
     * @param n  number of columns
     * @param ws optional workspace pool for reuse
     */
    public static LQ lq(double[] A, int m, int n, LQ.Pool... ws) {
        LQ.Pool pool = (ws != null && ws.length > 0) ? ws[0] : null;
        return LQ.decompose(A, m, n, pool);
    }

    // =========================================================================
    // GSVD  —  options: GSVD.Opts.WANT_U / WANT_V / WANT_Q
    // =========================================================================

    /**
     * Performs generalized SVD of A (m×n) and B (p×n).
     *
     * <p>Default (no opts): computes U, V, and Q.</p>
     * <p>Options: {@link GSVD.Opts#WANT_U}, {@link GSVD.Opts#WANT_V}, {@link GSVD.Opts#WANT_Q}.</p>
     *
     * @param A    matrix A (m×n, row-major), overwritten
     * @param m    rows of A
     * @param n    columns of A and B
     * @param B    matrix B (p×n, row-major), overwritten
     * @param p    rows of B
     * @param opts zero or more {@link GSVD.Opts} values
     */
    public static GSVD gsvd(double[] A, int m, int n, double[] B, int p, GSVD.Opts... opts) {
        return GSVD.decompose(A, m, n, B, p, null, opts);
    }

    /**
     * Performs generalized SVD with workspace reuse.
     *
     * @param A    matrix A (m×n, row-major), overwritten
     * @param m    rows of A
     * @param n    columns of A and B
     * @param B    matrix B (p×n, row-major), overwritten
     * @param p    rows of B
     * @param ws   workspace pool for reuse
     * @param opts zero or more {@link GSVD.Opts} values
     */
    public static GSVD gsvd(double[] A, int m, int n, double[] B, int p, GSVD.Pool ws, GSVD.Opts... opts) {
        return GSVD.decompose(A, m, n, B, p, ws, opts);
    }

    // =========================================================================
    // Helpers
    // =========================================================================
    private static <E extends Enum<E>> boolean contains(E[] opts, E target) {
        if (opts == null) return false;
        for (E o : opts) if (o == target) return true;
        return false;
    }

    private static int svdKind(SVD.Opts[] opts) {
        if (opts == null || opts.length == 0) return SVD.SVD_ALL;
        int kind = SVD.SVD_NONE;
        for (SVD.Opts o : opts) {
            switch (o) {
                case WANT_U: kind |= SVD.SVD_WANT_U; break;
                case FULL_U: kind |= SVD.SVD_WANT_U | SVD.SVD_FULL_U; break;
                case WANT_V: kind |= SVD.SVD_WANT_V; break;
                case FULL_V: kind |= SVD.SVD_WANT_V | SVD.SVD_FULL_V; break;
            }
        }
        return kind;
    }
}
