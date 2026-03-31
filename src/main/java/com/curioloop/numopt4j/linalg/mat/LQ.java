/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;

public final class LQ implements Decomposition {
    private static final double CONDITION_TOLERANCE = 1e16;

    /** Configuration options for LQ decomposition (currently no variants). */
    public enum Opts {}

    public static final class Pool extends Decomposition.Workspace {
        /** Householder reflection factors. */
        public double[] tau;

        /**
         * Ensure all buffers are allocated for LQ decomposition of an m×n matrix.
         */
        public Pool ensure(int m, int n) {
            if (m <= 0 || n <= 0) {
                throw new IllegalArgumentException("Dimensions must be positive");
            }
            if (m > n) {
                throw new IllegalArgumentException("For LQ decomposition, m must be <= n");
            }
            if (tau == null || tau.length < m) {
                tau = new double[m];
            }

            double[] tmp = new double[1];
            BLAS.dgelqf(m, n, null, 0, n, null, 0, tmp, 0, -1);
            int lqWork = (int) tmp[0];
            BLAS.dormlq(BLAS.Side.Left, BLAS.Trans.Trans, n, 1, m,
                    null, 0, n, null, 0, null, 0, 1, tmp, 0, -1);
            ensureWork(Math.max(Math.max(lqWork, (int) tmp[0]), 3 * m));
            ensureIwork(m);
            return this;
        }
    }

    private Pool pool;
    private double[] LQ;
    private int m;
    private int n;
    private boolean ok;
    private double condition;

    private LQ() {}

    /**
     * Creates an empty workspace pool for LQ decomposition.
     */
    public static Pool workspace() {
        return new Pool();
    }

    /**
     * Factorizes a square or wide matrix using LQ decomposition.
     */
    public static LQ decompose(double[] A, int m, int n) {
        return decompose(A, m, n, (Pool) null);
    }

    /**
     * Factorizes a square or wide matrix using LQ decomposition with workspace reuse.
     */
    public static LQ decompose(double[] A, int m, int n, Pool ws) {
        LQ lq = new LQ();
        lq.doDecompose(A, m, n, ws);
        return lq;
    }

    private void doDecompose(double[] A, int m, int n, Pool ws) {
        if (m <= 0 || n <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive");
        }
        if (m > n) {
            throw new IllegalArgumentException("For LQ decomposition, m must be <= n");
        }
        if (A == null || A.length < m * n) {
            throw new IllegalArgumentException("Matrix A must have length >= m*n");
        }

        this.LQ = A;
        this.m = m;
        this.n = n;
        this.ok = false;
        this.condition = Double.NaN;
        this.pool = ws == null ? new Pool() : ws;
        pool.ensure(m, n);

        int result = BLAS.dgelqf(m, n, LQ, 0, n, pool.tau, 0, pool.work(), 0, pool.work().length);
        if (result != 0) {
            throw new IllegalStateException("LQ factorization failed with dgelqf status=" + result);
        }

        this.ok = true;
    }

    /**
     * Solves the minimum-norm system {@code A*x = b} for a wide or square matrix.
     *
    * <p>To minimize allocations, this method uses {@code b} as a working buffer and may
    * overwrite it. Pass a distinct {@code x} buffer if the original right-hand side must be
    * preserved.</p>
     *
     * @param b right-hand side vector of length at least {@code m}
     * @param x destination vector of length at least {@code n}; may be {@code null}
     * @return the minimum-norm solution vector {@code x}
     * @throws IllegalStateException if the decomposition is unavailable
     * @throws ArithmeticException if the factor is singular or ill-conditioned
     */
    public double[] solve(double[] b, double[] x) {
        requireDecomposition();
        if (b == null || b.length < m) {
            throw new IllegalArgumentException("Vector b must have length >= " + m);
        }
        if (x == null || x.length < n) {
            x = new double[n];
        }

        requireWellConditioned();
        double[] rhs = b;
        if (!BLAS.dtrtrs(BLAS.Uplo.Lower, BLAS.Trans.NoTrans, BLAS.Diag.NonUnit,
            m, 1, LQ, 0, n, rhs, 0, 1)) {
            throw new ArithmeticException("LQ factor is singular");
        }

        BLAS.dset(n, 0.0, x, 0, 1);
        BLAS.dcopy(m, rhs, 0, 1, x, 0, 1);

        int result = BLAS.dormlq(BLAS.Side.Left, BLAS.Trans.Trans, n, 1, m, LQ, 0, n,
            pool.tau, 0, x, 0, 1, pool.work(), 0, pool.work().length);
        if (result != 0) {
            throw new IllegalStateException("Failed to apply Q^T with dormlq status=" + result);
        }

        return x;
    }

    /**
     * Solves the transpose system {@code A^T*x = b} in the least-squares sense.
     *
    * <p>To minimize allocations, this method uses {@code b} as a working buffer and may
    * overwrite it. Pass a distinct {@code x} buffer if the original right-hand side must be
    * preserved.</p>
     *
     * @param b right-hand side vector of length at least {@code n}
     * @param x destination vector of length at least {@code m}; may be {@code null}
     * @return the solution vector {@code x}
     * @throws IllegalStateException if the decomposition is unavailable
     * @throws ArithmeticException if the factor is singular or ill-conditioned
     */
    public double[] solveTranspose(double[] b, double[] x) {
        requireDecomposition();
        if (b == null || b.length < n) {
            throw new IllegalArgumentException("Vector b must have length >= " + n);
        }
        if (x == null || x.length < m) {
            x = new double[m];
        }

        requireWellConditioned();
        double[] rhs = b;

    int result = BLAS.dormlq(BLAS.Side.Left, BLAS.Trans.NoTrans, n, 1, m, LQ, 0, n,
        pool.tau, 0, rhs, 0, 1, pool.work(), 0, pool.work().length);
    if (result != 0) {
        throw new IllegalStateException("Failed to apply Q with dormlq status=" + result);
    }

    if (!BLAS.dtrtrs(BLAS.Uplo.Lower, BLAS.Trans.Trans, BLAS.Diag.NonUnit,
        m, 1, LQ, 0, n, rhs, 0, 1)) {
            throw new ArithmeticException("LQ factor is singular");
        }

        BLAS.dcopy(m, rhs, 0, 1, x, 0, 1);

        return x;
    }

    /**
     * Compatibility alias for {@link #solve(double[], double[])}.
     *
     * <p>For the standard {@code m <= n} LQ domain, solving {@code A*x = b} already returns the
     * minimum-norm solution, so this method delegates directly to {@link #solve(double[], double[])}.
     * It therefore shares the same in-place right-hand-side overwrite behavior.</p>
     */
    public double[] leastSquares(double[] b, double[] x) {
        return solve(b, x);
    }

    /**
     * Returns the full lower-trapezoidal factor {@code L} as an {@code m x n} matrix.
     */
    public Matrix toL() {
        requireDecomposition();
        double[] dst = new double[m * n];
        for (int i = 0; i < m; i++) {
            int limit = Math.min(i + 1, n);
            for (int j = 0; j < limit; j++) {
                dst[i * n + j] = LQ[i * n + j];
            }
        }
        return new Matrix(m, n, false, dst);
    }

    /**
     * Returns the full orthogonal factor {@code Q} as an {@code n x n} matrix.
     */
    public Matrix toQ() {
        requireDecomposition();
        double[] dst = new double[n * n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(LQ, i * n, dst, i * n, n);
        }
        BLAS.dorglq(n, n, m, dst, 0, n, pool.tau, 0, pool.work(), 0);
        return new Matrix(n, n, false, dst);
    }

    /**
     * Returns the condition number estimate associated with the lower-triangular LQ factor.
     */
    public double cond() {
        requireDecomposition();
        return ensureCondition();
    }

    private double computeCondition() {
        if (m == 0) {
            return Double.NaN;
        }
        double rcond = BLAS.dtrcon(BLAS.Norm.One, BLAS.Uplo.Lower, BLAS.Diag.NonUnit,
                m, LQ, n, pool.work(), pool.iwork());
        if (rcond == 0) {
            return Double.POSITIVE_INFINITY;
        }
        return 1.0 / rcond;
    }

    private void requireDecomposition() {
        if (!ok) {
            throw new IllegalStateException("LQ decomposition not completed or failed");
        }
    }

    private void requireWellConditioned() {
        double condition = ensureCondition();
        if (!Double.isFinite(condition) || condition > CONDITION_TOLERANCE) {
            throw new ArithmeticException("LQ factor is singular or ill-conditioned");
        }
    }

    private double ensureCondition() {
        if (Double.isNaN(condition)) {
            condition = computeCondition();
        }
        return condition;
    }

    @Override
    public boolean ok() {
        return ok;
    }

    @Override
    public Pool pool() {
        return pool;
    }

    /** Returns the number of rows in the factorized matrix. */
    public int m() {
        return m;
    }

    /** Returns the number of columns in the factorized matrix. */
    public int n() {
        return n;
    }

    /** Returns the stored Householder scalar factors. */
    public double[] tau() {
        return pool.tau;
    }

}
