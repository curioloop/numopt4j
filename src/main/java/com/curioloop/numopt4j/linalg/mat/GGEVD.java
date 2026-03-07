/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;

/**
 * Generalized non-symmetric eigenvalue decomposition.
 *
 * <p>Solves the generalized eigenvalue problem:
 * <pre>
 *   A * x = lambda * B * x
 * </pre>
 * where A and B are general (non-symmetric) real matrices.
 *
 * <p>Eigenvalues are returned as {@code (alphar[i] + i*alphai[i]) / beta[i]}.
 * Real eigenvalues have {@code alphai[i] = 0}; complex eigenvalues come in
 * conjugate pairs.
 *
 * <p>Algorithm (follows LAPACK {@code DGGEV}):
 * <ol>
 *   <li>Balance A via {@code dgebal}.</li>
 *   <li>Reduce A to upper Hessenberg form via {@code dgehrd} + {@code dorghr}.</li>
 *   <li>QR-factorize B to get upper triangular T.</li>
 *   <li>Reduce (H, T) to generalized Hessenberg form via {@code dgghrd}.</li>
 *   <li>QZ iteration via {@code dhgeqz} to compute Schur form and eigenvalues.</li>
 *   <li>Compute eigenvectors via {@code dtgevc}.</li>
 *   <li>Back-transform eigenvectors via {@code dgebak}.</li>
 * </ol>
 */
public final class GGEVD implements Decomposition {

    public static final class Pool extends Workspace {

        public double[] alphar;
        public double[] alphai;
        public double[] beta;
        public double[] VL;
        public double[] VR;

        private Pool() {}

        public Pool ensure(int n, boolean wantVL, boolean wantVR) {
            if (alphar == null || alphar.length < n) {
                alphar = new double[n];
                alphai = new double[n];
                beta   = new double[n];
            }
            if (wantVL && (VL == null || VL.length < n * n)) VL = new double[n * n];
            if (wantVR && (VR == null || VR.length < n * n)) VR = new double[n * n];
            double[] qry = new double[1];
            BLAS.dggev(false, false, n, null, n, null, n, null, null, null,
                       null, n, null, n, qry, 0, -1);
            // Extra 2*n*n for tmpVL/tmpVR scratch (carved from work, no separate alloc)
            ensureWork((int) qry[0] + 2 * n * n);
            return this;
        }

        /** Ensures both VL and VR are allocated (suitable for any combination of opts). */
        public Pool ensure(int n) {
            return ensure(n, true, true);
        }
    }

    /** Configuration options for GGEVD. */
    public enum Opts {
        /** Compute left eigenvectors. Default: not computed. */
        WANT_VL,
        /** Compute right eigenvectors. Default: not computed. */
        WANT_VR
    }

    // -------------------------------------------------------------------------

    private Pool pool;
    private int n;
    private boolean ok;
    private boolean hasVL;
    private boolean hasVR;

    private GGEVD() {}

    public static Pool workspace(int n) {
        return new Pool().ensure(n, true, true);
    }

    /**
     * Decomposes A*x = lambda*B*x, computing both left and right eigenvectors.
     *
     * @param A  general matrix (n×n, row-major), overwritten
     * @param B  general matrix (n×n, row-major), overwritten
     * @param n  matrix dimension
     * @param opts zero or more {@link Opts} values; default computes eigenvalues only
     */
    public static GGEVD decompose(double[] A, double[] B, int n, Opts... opts) {
        boolean wantVL = opts != null && contains(opts, Opts.WANT_VL);
        boolean wantVR = opts != null && contains(opts, Opts.WANT_VR);
        return decompose(A, B, n, wantVL, wantVR, null);
    }

    /**
     * Decomposes A*x = lambda*B*x, computing eigenvalues only.
     *
     * @param A  general matrix (n×n, row-major), overwritten
     * @param B  general matrix (n×n, row-major), overwritten
     * @param n  matrix dimension
     */
    public static GGEVD decompose(double[] A, double[] B, int n) {
        return decompose(A, B, n, false, false, null);
    }

    /**
     * Decomposes A*x = lambda*B*x with control over which eigenvectors to compute.
     *
     * @param A      general matrix (n×n, row-major), overwritten
     * @param B      general matrix (n×n, row-major), overwritten
     * @param n      matrix dimension
     * @param ws     workspace pool for reuse
     * @param opts   zero or more {@link Opts} values; default computes eigenvalues only
     */
    public static GGEVD decompose(double[] A, double[] B, int n, Pool ws, Opts... opts) {
        boolean wantVL = opts != null && contains(opts, Opts.WANT_VL);
        boolean wantVR = opts != null && contains(opts, Opts.WANT_VR);
        return decompose(A, B, n, wantVL, wantVR, ws);
    }

    private static boolean contains(Opts[] opts, Opts target) {
        if (opts == null) return false;
        for (Opts o : opts) if (o == target) return true;
        return false;
    }

    /**
     * Decomposes A*x = lambda*B*x with control over which eigenvectors to compute.
     *
     * @param A      general matrix (n×n, row-major), overwritten
     * @param B      general matrix (n×n, row-major), overwritten
     * @param n      matrix dimension
     * @param wantVL true to compute left eigenvectors
     * @param wantVR true to compute right eigenvectors
     * @param ws     optional workspace pool for reuse; allocated internally if null
     */
    public static GGEVD decompose(double[] A, double[] B, int n,
                                   boolean wantVL, boolean wantVR, Pool ws) {
        GGEVD g = new GGEVD();
        g.doDecompose(A, B, n, wantVL, wantVR, ws);
        return g;
    }

    private void doDecompose(double[] A, double[] B, int n,
                              boolean wantVL, boolean wantVR, Pool ws) {
        if (A == null || A.length < n * n)
            throw new IllegalArgumentException("Matrix A must have length >= n*n");
        if (B == null || B.length < n * n)
            throw new IllegalArgumentException("Matrix B must have length >= n*n");
        if (n <= 0)
            throw new IllegalArgumentException("Matrix dimension must be positive");

        this.n = n;
        this.ok = false;
        this.hasVL = wantVL;
        this.hasVR = wantVR;

        if (ws == null) ws = (Pool) workspace(n);
        this.pool = ws;
        this.pool.ensure(n, wantVL, wantVR);

        int info = BLAS.dggev(wantVL, wantVR, n,
                               A, n, B, n,
                               pool.alphar, pool.alphai, pool.beta,
                               pool.VL, n, pool.VR, n,
                               pool.work(), 0, pool.work().length);
        this.ok = (info == 0);
    }

    // -------------------------------------------------------------------------

    public boolean ok() { return ok; }
    public int n() { return n; }

    /**
     * Returns the real parts of alpha (numerators of eigenvalues).
     * Eigenvalue i = (alphar[i] + i*alphai[i]) / beta[i].
     */
    public double[] alphar() { return ok ? pool.alphar : null; }

    /**
     * Returns the imaginary parts of alpha.
     * Zero for real eigenvalues; complex pairs come with alphai[i] > 0, alphai[i+1] < 0.
     */
    public double[] alphai() { return ok ? pool.alphai : null; }

    /**
     * Returns the beta values (denominators of eigenvalues).
     * beta[i] = 0 indicates an infinite eigenvalue.
     */
    public double[] beta() { return ok ? pool.beta : null; }

    /**
     * Returns eigenvalues as a flat array of interleaved (real, imag) pairs, length 2n.
     * Infinite eigenvalues (beta[i]=0) are represented as (±Inf, 0).
     * Returns null if decomposition failed.
     */
    public double[] eigenvalues() {
        if (!ok) return null;
        double[] ev = new double[n * 2];
        for (int i = 0; i < n; i++) {
            if (pool.beta[i] == 0.0) {
                ev[i * 2]     = pool.alphar[i] >= 0 ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
                ev[i * 2 + 1] = 0.0;
            } else {
                ev[i * 2]     = pool.alphar[i] / pool.beta[i];
                ev[i * 2 + 1] = pool.alphai[i] / pool.beta[i];
            }
        }
        return ev;
    }

    /**
     * Returns eigenvalues as an n×1 complex matrix (interleaved [re0,im0,...]), or null if failed.
     * Consistent with {@link Eigen#toS()}, {@link Schur#toS()}.
     */
    public Matrix toS() {
        double[] ev = eigenvalues();
        if (ev == null) return null;
        return new Matrix(n, 1, true, ev);
    }

    /**
     * Returns the right eigenvectors as an n×n matrix, or null if not computed.
     *
     * <p>For a real eigenvalue at index j: column j holds the eigenvector.
     * For a complex pair at indices j, j+1: column j = real part, column j+1 = imaginary part.
     */
    public Matrix toVR() {
        if (!ok || !hasVR || pool.VR == null) return null;
        return new Matrix(n, n, false, pool.VR);
    }

    public Matrix toVL() {
        if (!ok || !hasVL || pool.VL == null) return null;
        return new Matrix(n, n, false, pool.VL);
    }

    @Override
    public Pool work() { return pool; }
}
