/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.linalg.blas.Dsyev;

public final class GEVD implements Decomposition {

    private static final double EPSILON = BLAS.dlamch('E');

    /**
     * Pool extends Workspace with an extra {@code eigenvalues} array.
     *
     * <p>Work layout: the single {@code work} array serves two purposes in sequence:
     * <ol>
     *   <li>type-2/type-3: first {@code n*n} elements are used as a temporary matrix
     *       for the symmetrisation step (A^T·A or A·A^T).</li>
     *   <li>After the symmetrisation result is copied back into A, the same {@code work}
     *       array is passed to {@code dsyev} as its scratch buffer (length >= 3n-1).</li>
     * </ol>
     * Because the two uses are strictly sequential, sharing one array of length
     * {@code max(3n-1, n*n)} is safe.
     */
    public static final class Pool extends Workspace {

        public double[] eigenvalues;

        private Pool() {}

        public Pool ensure(int n) {
            if (eigenvalues == null || eigenvalues.length < n) {
                eigenvalues = new double[n];
            }
            // work must fit both the n×n temp matrix AND the dsyev scratch (3n-1)
            int needed = Math.max(Math.max(1, 3 * n - 1), n * n);
            ensureWork(needed);
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

    public static GEVD decompose(double[] A, double[] B, int n, char uplo) {
        return decompose(A, B, n, uplo, 1, null);
    }

    public static GEVD decompose(double[] A, double[] B, int n, char uplo, int type, Workspace ws) {
        GEVD eg = new GEVD();
        eg.doDecompose(A, B, n, uplo, type, ws);
        return eg;
    }

    private void doDecompose(double[] A, double[] B, int n, char uplo, int type, Workspace ws) {
        if (A == null || A.length < n * n)
            throw new IllegalArgumentException("Matrix A must have length >= n*n");
        if (B == null || B.length < n * n)
            throw new IllegalArgumentException("Matrix B must have length >= n*n");
        if (n <= 0)
            throw new IllegalArgumentException("Matrix dimension must be positive");
        if (type < 1 || type > 3)
            throw new IllegalArgumentException("Type must be 1, 2, or 3");

        this.A = A;
        this.n = n;
        this.type = type;
        this.ok = false;

        if (ws == null) {
            ws = workspace(n);
        }
        if (!(ws instanceof Pool)) {
            throw new IllegalArgumentException("Workspace must be an instance of GEVD.Pool");
        }
        this.pool = (Pool) ws;
        this.pool.ensure(n);

        boolean lower = (uplo == 'L' || uplo == 'l');

        if (BLAS.dpotrf(lower ? BLAS.Uplo.Lower : BLAS.Uplo.Upper, n, B, 0, n) != 0) {
            return;
        }

        switch (type) {
            case 1: solveType1(A, B, n, lower); break;
            case 2: solveType2(A, B, n, lower); break;
            case 3: solveType3(A, B, n, lower); break;
        }
    }

    private void solveType1(double[] A, double[] B, int n, boolean lower) {
        BLAS.dtrsm(BLAS.Side.Right, lower ? BLAS.Uplo.Lower : BLAS.Uplo.Upper, BLAS.Transpose.Trans, BLAS.Diag.NonUnit, n, n, 1.0, B, 0, n, A, 0, n);
        BLAS.dtrsm(BLAS.Side.Left, lower ? BLAS.Uplo.Lower : BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit, n, n, 1.0, B, 0, n, A, 0, n);

        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                A[j * n + i] = A[i * n + j];

        this.ok = Dsyev.dsyev('V', lower ? 'L' : 'U', n, A, n, pool.eigenvalues, 0, pool.work(), 0, pool.work().length) == 0;

        if (ok) {
            BLAS.dtrsm(BLAS.Side.Left, lower ? BLAS.Uplo.Lower : BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit, n, n, 1.0, B, 0, n, A, 0, n);
        }
    }

    private void solveType2(double[] A, double[] B, int n, boolean lower) {
        BLAS.dtrsm(BLAS.Side.Left, lower ? BLAS.Uplo.Lower : BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit, n, n, 1.0, B, 0, n, A, 0, n);

        // Use work[0..n*n) as temp matrix for A^T·A, then copy back into A.
        // After copy, work is reused by dsyev (needs >= 3n-1 elements, which is <= n*n for n>=3).
        double[] tempWork = pool.work();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int k = 0; k < n; k++) s += A[i * n + k] * A[j * n + k];
                tempWork[i * n + j] = s;
            }

        System.arraycopy(tempWork, 0, A, 0, n * n);

        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                A[j * n + i] = A[i * n + j];

        this.ok = Dsyev.dsyev('V', lower ? 'L' : 'U', n, A, n, pool.eigenvalues, 0, pool.work(), 0, pool.work().length) == 0;
    }

    private void solveType3(double[] A, double[] B, int n, boolean lower) {
        BLAS.dtrsm(BLAS.Side.Right, lower ? BLAS.Uplo.Lower : BLAS.Uplo.Upper, BLAS.Transpose.Trans, BLAS.Diag.NonUnit, n, n, 1.0, B, 0, n, A, 0, n);

        double[] tempWork = pool.work();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int k = 0; k < n; k++) s += A[k * n + i] * A[k * n + j];
                tempWork[i * n + j] = s;
            }

        System.arraycopy(tempWork, 0, A, 0, n * n);

        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                A[j * n + i] = A[i * n + j];

        this.ok = Dsyev.dsyev('V', lower ? 'L' : 'U', n, A, n, pool.eigenvalues, 0, pool.work(), 0, pool.work().length) == 0;

        if (ok) {
            BLAS.dtrsm(BLAS.Side.Left, lower ? BLAS.Uplo.Lower : BLAS.Uplo.Upper, BLAS.Transpose.NoTrans, BLAS.Diag.NonUnit, n, n, 1.0, B, 0, n, A, 0, n);
        }
    }

    // -------------------------------------------------------------------------

    public boolean ok() { return ok; }
    public int n() { return n; }
    public int type() { return type; }

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
    public Workspace work() { return pool; }

    @Override
    public Matrix extract(Part part) {
        return extract(part, null);
    }

    @Override
    public Matrix extract(Part part, double[] dst) {
        if (!ok) return null;
        switch (part) {
            case Q: {
                if (A == null) return null;
                int size = n * n;
                if (dst == null || dst.length < size) dst = new double[size];
                System.arraycopy(A, 0, dst, 0, size);
                return new Matrix(n, n, false, dst);
            }
            case S: {
                if (pool == null) return null;
                if (dst == null || dst.length < n) dst = new double[n];
                System.arraycopy(pool.eigenvalues, 0, dst, 0, n);
                return new Matrix(n, 1, false, dst);
            }
            default:
                return null;
        }
    }

    @Override
    public int rows(Part part) {
        if (!ok) throw new IllegalStateException("Decomposition failed");
        switch (part) {
            case Q: case S: return n;
            default: throw new UnsupportedOperationException("Part " + part + " not supported");
        }
    }

    @Override
    public int cols(Part part) {
        if (!ok) throw new IllegalStateException("Decomposition failed");
        switch (part) {
            case Q: return n;
            case S: return 1;
            default: throw new UnsupportedOperationException("Part " + part + " not supported");
        }
    }
}
