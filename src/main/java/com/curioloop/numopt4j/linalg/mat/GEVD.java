/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.linalg.blas.Dsyev;

/**
 * Generalized symmetric-definite eigenvalue decomposition.
 *
 * <p>Solves one of three generalized eigenproblems:
 * <ul>
 *   <li>Type 1: A·x = λ·B·x</li>
 *   <li>Type 2: A·B·x = λ·x</li>
 *   <li>Type 3: B·A·x = λ·x</li>
 * </ul>
 * where A is symmetric and B is symmetric positive-definite.
 *
 * <p>Algorithm (follows LAPACK {@code DSYGV}):
 * <ol>
 *   <li>Cholesky factorize B = L·Lᵀ (lower) or B = Uᵀ·U (upper) via {@code dpotrf}.</li>
 *   <li>Reduce to standard form via {@code dsygst}: overwrites A with the equivalent
 *       standard symmetric eigenproblem C·y = λ·y.</li>
 *   <li>Symmetrize the result (dsygst only updates one triangle).</li>
 *   <li>Solve the standard problem via {@code dsyev}.</li>
 *   <li>Back-transform eigenvectors to the original basis.</li>
 * </ol>
 *
 * <p>Back-transforms (lower Cholesky, B = L·Lᵀ):
 * <ul>
 *   <li>Type 1: x = L⁻ᵀ·y  →  {@code dtrsm(Left, Lower, Trans)}</li>
 *   <li>Type 2: x = L⁻ᵀ·y  →  {@code dtrsm(Left, Lower, Trans)}</li>
 *   <li>Type 3: x = L·y     →  {@code dtrmm(Left, Lower, NoTrans)}</li>
 * </ul>
 *
 * <p>Back-transforms (upper Cholesky, B = Uᵀ·U):
 * <ul>
 *   <li>Type 1: x = U⁻¹·y  →  {@code dtrsm(Left, Upper, NoTrans)}</li>
 *   <li>Type 2: x = U⁻¹·y  →  {@code dtrsm(Left, Upper, NoTrans)}</li>
 *   <li>Type 3: x = Uᵀ·y   →  {@code dtrmm(Left, Upper, Trans)}</li>
 * </ul>
 */
public final class GEVD implements Decomposition {

    /** Decomposition options for GEVD. */
    public enum Opts {
        /** Use lower triangle of the input matrices (default). */
        LOWER,
        /** Use upper triangle of the input matrices. */
        UPPER,
        /** Solve A·x = λ·B·x (type 1, default). */
        TYPE1,
        /** Solve A·B·x = λ·x (type 2). */
        TYPE2,
        /** Solve B·A·x = λ·x (type 3). */
        TYPE3,
        /** Compute eigenvectors (back-transformation). Default: values only. */
        WANT_V
    }

    private static final double EPSILON = BLAS.dlamch('E');

    public static final class Pool extends Workspace {

        public double[] eigenvalues;

        private Pool() {}

        public Pool ensure(int n) {
            if (eigenvalues == null || eigenvalues.length < n) {
                eigenvalues = new double[n];
            }
            double[] tmp = new double[1];
            Dsyev.dsyev('V', 'L', n, null, n, null, 0, tmp, 0, -1);
            ensureWork((int) tmp[0]);
            return this;
        }
    }

    // -------------------------------------------------------------------------

    private Pool pool;
    private double[] A;
    private int n;
    private int type;
    private boolean ok;

    private GEVD() {}

    public static Workspace workspace(int n) {
        return new Pool().ensure(n);
    }

    /**
     * Decomposes with default options: lower triangle, type 1 (A·x = λ·B·x).
     *
     * @param A  symmetric matrix (n×n, row-major), overwritten with eigenvectors
     * @param B  symmetric positive-definite matrix (n×n, row-major), overwritten with Cholesky factor
     * @param n  matrix dimension
     * @param uplo which triangle of A and B to use
     */
    public static GEVD decompose(double[] A, double[] B, int n, BLAS.Uplo uplo) {
        return decompose(A, B, n, uplo, 1, false, null);
    }

    /**
     * Decomposes with explicit type and optional workspace.
     *
     * @param A    symmetric matrix (n×n, row-major), overwritten with eigenvectors
     * @param B    symmetric positive-definite matrix (n×n, row-major), overwritten with Cholesky factor
     * @param n    matrix dimension
     * @param uplo which triangle of A and B to use
     * @param type problem type: 1 for A·x=λ·B·x, 2 for A·B·x=λ·x, 3 for B·A·x=λ·x
     * @param ws   optional workspace pool for reuse; allocated internally if null
     */
    public static GEVD decompose(double[] A, double[] B, int n, BLAS.Uplo uplo, int type, Pool ws) {
        return decompose(A, B, n, uplo, type, false, ws);
    }

    /**
     * Decomposes with explicit type, values-only flag, and optional workspace.
     *
     * @param A          symmetric matrix (n×n, row-major), overwritten with eigenvectors (or unchanged if valuesOnly)
     * @param B          symmetric positive-definite matrix (n×n, row-major), overwritten with Cholesky factor
     * @param n          matrix dimension
     * @param uplo       which triangle of A and B to use
     * @param type       problem type: 1, 2, or 3
     * @param valuesOnly true to skip eigenvector back-transformation
     * @param ws         optional workspace pool for reuse; allocated internally if null
     */
    public static GEVD decompose(double[] A, double[] B, int n, BLAS.Uplo uplo, int type, boolean valuesOnly, Pool ws) {
        GEVD eg = new GEVD();
        eg.doDecompose(A, B, n, uplo, type, valuesOnly, ws);
        return eg;
    }

    private void doDecompose(double[] A, double[] B, int n, BLAS.Uplo uplo, int type, boolean valuesOnly, Pool ws) {
        if (A == null || A.length < n * n)
            throw new IllegalArgumentException("Matrix A must have length >= n*n");
        if (B == null || B.length < n * n)
            throw new IllegalArgumentException("Matrix B must have length >= n*n");
        if (n <= 0)
            throw new IllegalArgumentException("Matrix dimension must be positive");
        if (uplo == null)
            throw new NullPointerException("uplo must not be null");
        if (type < 1 || type > 3)
            throw new IllegalArgumentException("Type must be 1, 2, or 3");

        this.A = A;
        this.n = n;
        this.type = type;
        this.ok = false;

        if (ws == null) ws = (Pool) workspace(n);
        this.pool = ws;
        this.pool.ensure(n);

        boolean lower = (uplo == BLAS.Uplo.Lower);

        // Step 1: Cholesky factorize B = L*L^T (lower) or U^T*U (upper)
        if (BLAS.dpotrf(uplo, n, B, 0, n) != 0) {
            return; // B not positive-definite
        }

        // Step 2: Reduce to standard form via dsygst
        BLAS.dsygst(type, uplo, n, A, 0, n, B, 0, n);

        // Step 3: Symmetrize (dsygst only updates one triangle)
        if (lower) {
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    A[i * n + j] = A[j * n + i];
        } else {
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    A[j * n + i] = A[i * n + j];
        }

        // Step 4: Solve standard symmetric eigenproblem C*y = lambda*y
        char jobz = valuesOnly ? 'N' : 'V';
        this.ok = Dsyev.dsyev(jobz, lower ? 'L' : 'U', n, A, n,
                pool.eigenvalues, 0, pool.work(), 0, pool.work().length) == 0;

        if (!ok || valuesOnly) return;

        // Step 5: Back-transform eigenvectors
        if (lower) {
            if (type == 1 || type == 2) {
                // x = L^{-T} * y  =>  solve L^T * X = Y
                BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Transpose.Trans, BLAS.Diag.NonUnit,
                        n, n, 1.0, B, 0, n, A, 0, n);
            } else {
                // type 3: x = L * y
                BLAS.dtrmm(BLAS.Side.Left, BLAS.Uplo.Lower, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit,
                        n, n, 1.0, B, 0, n, A, 0, n);
            }
        } else {
            if (type == 1 || type == 2) {
                // x = U^{-1} * y  =>  solve U * X = Y
                BLAS.dtrsm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit,
                        n, n, 1.0, B, 0, n, A, 0, n);
            } else {
                // type 3: x = U^T * y
                BLAS.dtrmm(BLAS.Side.Left, BLAS.Uplo.Upper, BLAS.Transpose.Trans, BLAS.Diag.NonUnit,
                        n, n, 1.0, B, 0, n, A, 0, n);
            }
        }
    }

    // -------------------------------------------------------------------------

    public boolean ok() { return ok; }
    public int n() { return n; }
    public int type() { return type; }

    /** Returns eigenvalues as a raw double[] (ascending order), or null if decomposition failed. */
    public double[] eigenvalues() { return ok && pool != null ? pool.eigenvalues : null; }

    public double cond() {
        if (!ok || pool == null) return Double.NaN;
        double[] ev = pool.eigenvalues;
        double max = Math.abs(ev[0]), min = Math.abs(ev[0]);
        for (int i = 1; i < n; i++) {
            double a = Math.abs(ev[i]);
            if (a > max) max = a;
            if (a < min) min = a;
        }
        return min < EPSILON ? Double.POSITIVE_INFINITY : max / min;
    }

    @Override
    public Pool work() { return pool; }

    /** Returns eigenvalues as an n×1 matrix (ascending order), or null if decomposition failed. */
    public Matrix toS() {
        if (!ok || pool == null) return null;
        double[] dst = new double[n];
        System.arraycopy(pool.eigenvalues, 0, dst, 0, n);
        return new Matrix(n, 1, false, dst);
    }

    /**
     * Returns eigenvectors as an n×n matrix, or null if decomposition failed.
     *
     * <p>Storage convention: column-major eigenvectors in a row-major array.
     * The j-th eigenvector (corresponding to eigenvalue {@code toS().data[j]}) occupies
     * column j: element {@code (i, j)} is at {@code data[i * n + j]}.
     * Equivalently, {@code toV().get(i, j)} returns the i-th component of the j-th eigenvector.
     */
    public Matrix toV() {
        if (!ok || A == null) return null;
        return new Matrix(n, n, false, A);
    }

}
