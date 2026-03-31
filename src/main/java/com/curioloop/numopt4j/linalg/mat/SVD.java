/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;

import static java.lang.Math.max;
import static java.lang.Math.min;

public final class SVD implements Decomposition {

    private static final double EPSILON = BLAS.dlamch('E');

    public static final int SVD_NONE   = 0;
    public static final int SVD_WANT_U = 1;
    public static final int SVD_WANT_V = 2;
    public static final int SVD_FULL_U = 4;
    public static final int SVD_FULL_V = 8;
    public static final int SVD_ALL    = SVD_WANT_U | SVD_WANT_V;

    /** Decomposition options for SVD. */
    public enum Opts {
        /** Compute thin U (m × min(m,n)). Default when no opts given. */
        WANT_U,
        /** Compute full U (m × m). */
        FULL_U,
        /** Compute thin Vᵀ (min(m,n) × n). Default when no opts given. */
        WANT_V,
        /** Compute full Vᵀ (n × n). */
        FULL_V
    }

    public static final class Pool extends Decomposition.Workspace {
        public double[] s;
        public double[] UV;

        public Pool ensureS(int minMN) {
            if (s == null || s.length < minMN) s = new double[minMN];
            return this;
        }

        public Pool ensureUV(int size) {
            if (size > 0 && (UV == null || UV.length < size)) UV = new double[size];
            return this;
        }
    }

    private Pool pool;
    private double[] s;
    private double[] U;
    private double[] VT;
    private int m;
    private int n;
    private int kind;
    private boolean ok;

    private SVD() {}

    public static Pool workspace() {
        return new Pool();
    }

    /**
     * Decompose with default kind = SVD_ALL.
     */
    public static SVD decompose(double[] A, int m, int n) {
        return decompose(A, m, n, SVD_ALL, (Pool) null);
    }

    /**
     * Decompose with explicit kindMask and optional workspace.
     */
    public static SVD decompose(double[] A, int m, int n, int kindMask, Pool ws) {
        SVD svd = new SVD();
        svd.m = m;
        svd.n = n;
        svd.kind = kindMask;
        svd.doDecompose(A, m, n, kindMask, ws);
        return svd;
    }

    private void doDecompose(double[] A, int m, int n, int kind, Pool wsIn) {
        int minMN = min(m, n);

        boolean wantU     = (kind & SVD_WANT_U) != 0;
        boolean wantVT    = (kind & SVD_WANT_V) != 0;
        boolean wantFullU = (kind & SVD_FULL_U) != 0;
        boolean wantFullV = (kind & SVD_FULL_V) != 0;

        boolean overwriteU = false;
        boolean overwriteVT = false;
        if (wantU && wantVT) {
            if (m >= n) {
                overwriteU = !wantFullU;
                overwriteVT = wantFullU;
            } else {
                overwriteU = wantFullV;
                overwriteVT = !wantFullV;
            }
        } else if (wantU) {
            overwriteU = !wantFullU || m < n;
        } else if (wantVT) {
            overwriteVT = !wantFullV || m >= n;
        }

        if (wsIn == null) wsIn = new Pool();
        this.pool = wsIn;
        pool.ensureS(minMN);
        int uSize = wantU && !overwriteU ? (wantFullU ? m * m : m * minMN) : 0;
        int vtSize = wantVT && !overwriteVT ? (wantFullV ? n * n : minMN * n) : 0;
        pool.ensureUV(max(uSize, vtSize));

        char jobU = wantU ? (overwriteU ? 'O' : (wantFullU ? 'A' : 'S')) : 'N';
        char jobVT = wantVT ? (overwriteVT ? 'O' : (wantFullV ? 'A' : 'S')) : 'N';
        int ldu = wantU ? max(1, wantFullU ? m : minMN) : 1;
        int ldvt = wantVT ? max(1, n) : 1;

        double[] tmp = new double[1];
        BLAS.dgesvd(jobU, jobVT, m, n, null, 0, max(1, n), null, 0,
                null, 0, ldu, null, 0, ldvt, tmp, 0, -1);
        pool.ensureWork(max(1, (int) tmp[0]));

        double[] outS = pool.s;
        double[] outU = null;
        if (wantU) {
            outU = overwriteU ? A : pool.UV;
        }
        double[] outVT = null;
        if (wantVT) {
            outVT = overwriteVT ? A : pool.UV;
        }

        int info = BLAS.dgesvd(jobU, jobVT, m, n, A, 0, max(1, n), outS, 0,
                outU, 0, ldu, outVT, 0, ldvt, pool.work(), 0, pool.work().length);

        if (info == 0 && overwriteU && m < n) {
            for (int row = 1; row < m; row++) {
                System.arraycopy(A, row * n, A, row * m, m);
            }
        }

        this.s = outS;
        this.U = outU;
        this.VT = outVT;
        this.ok = info == 0;
    }

    @Override
    public Pool pool() {
        return pool;
    }

    @Override
    public boolean ok() {
        return ok;
    }

    public double[] singularValues() { return ok ? s : null; }

    public double[] U()  { return ok ? U : null; }
    public double[] VT() { return ok ? VT : null; }

    /** Returns the left singular vectors matrix U. Returns null if decomposition failed or U was not requested. */
    public Matrix toU() {
        if (!ok || U == null) return null;
        int cols = uCols();
        double[] dst = new double[m * cols];
        System.arraycopy(U, 0, dst, 0, m * cols);
        return new Matrix(m, cols, false, dst);
    }

    /** Returns the right singular vectors matrix Vᵀ. Returns null if decomposition failed or V was not requested. */
    public Matrix toVT() {
        if (!ok || VT == null) return null;
        int rows = vtRows();
        double[] dst = new double[rows * n];
        System.arraycopy(VT, 0, dst, 0, rows * n);
        return new Matrix(rows, n, false, dst);
    }

    public int uCols() {
        if (U == null) return 0;
        return (kind & SVD_FULL_U) != 0 ? m : min(m, n);
    }

    public int vtRows() {
        if (VT == null) return 0;
        return (kind & SVD_FULL_V) != 0 ? n : min(m, n);
    }

    public int m() { return m; }
    public int n() { return n; }
    public int kind() { return ok ? kind : -1; }

    /**
     * Returns the effective rank using the default relative cutoff {@code max(m,n) * eps}.
     */
    public int rank() {
        if (s == null || s.length == 0) return 0;
        return rank(max(m, n) * EPSILON);
    }

    /**
     * Returns the effective rank using the relative cutoff {@code rcond * s[0]}.
     *
     * @param rcond relative singular value cutoff; must be nonnegative
     * @return the number of singular values strictly larger than {@code rcond * s[0]}
     */
    public int rank(double rcond) {
        if (rcond < 0) {
            throw new IllegalArgumentException("rcond must be nonnegative");
        }
        if (s == null || s.length == 0 || s[0] <= 0) return 0;
        double threshold = rcond * s[0];
        int r = 0;
        for (double value : s) {
            if (value > threshold) {
                r++;
            } else {
                break;
            }
        }
        return r;
    }

    public double cond() {
        if (s == null || s.length == 0) return Double.NaN;
        double sMin = s[s.length - 1];
        if (sMin <= 0) return Double.POSITIVE_INFINITY;
        return s[0] / sMin;
    }

    public double norm2() {
        if (s == null || s.length == 0) return Double.NaN;
        return s[0];
    }

    /**
     * Solves for the minimum-norm vector using the default effective rank.
     *
     * <p>This method requires that both left and right singular vectors were requested during
     * factorization. It throws if the decomposition is unavailable or if the right-hand side has
     * incompatible length.</p>
     */
    public double[] solve(double[] b, double[] x) {
        int rank = rank();
        return solve(b, x, rank);
    }

    /**
     * Solves for the minimum-norm vector using the leading {@code rank} singular values.
     *
     * @param b right-hand side vector of length at least {@code m}
     * @param x destination vector of length at least {@code n}; may be {@code null}
     * @param rank effective rank in the range {@code [0, min(m,n)]}
     * @return the solution vector
     */
    public double[] solve(double[] b, double[] x, int rank) {
        if (!ok) {
            throw new IllegalStateException("SVD decomposition not completed or failed");
        }
        if (U == null || VT == null) {
            throw new IllegalStateException("SVD solve requires both U and VT to be computed");
        }
        if (b == null || b.length < m) {
            throw new IllegalArgumentException("Vector b must have length >= " + m);
        }
        int minMN = min(m, n);
        if (rank < 0 || rank > minMN) {
            throw new IllegalArgumentException("rank must be in [0, " + minMN + "]");
        }
        if (x == null || x.length < n) {
            x = new double[n];
        }
        double[] rhs = x == b ? b.clone() : b;
        int uCols = uCols();
        BLAS.dset(n, 0.0, x, 0, 1);
        for (int i = 0; i < rank; i++) {
            double coeff = BLAS.ddot(m, U, i, uCols, rhs, 0, 1) / s[i];
            BLAS.daxpy(n, coeff, VT, i * n, 1, x, 0, 1);
        }
        return x;
    }

}
