/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.linalg.blas.Dsyev;

public final class Eigen implements Decomposition {

    public static final int EIGEN_NONE = 0;
    public static final int EIGEN_LEFT = 1;
    public static final int EIGEN_RIGHT = 2;
    public static final int EIGEN_BOTH = EIGEN_LEFT | EIGEN_RIGHT;

    /** Configuration options for Eigen decomposition. */
    public enum Opts {
        /** Input is symmetric; use lower triangle (dsyev with 'L'). */
        SYMMETRIC_LOWER,
        /** Input is symmetric; use upper triangle (dsyev with 'U'). */
        SYMMETRIC_UPPER,
        /** Compute left eigenvectors (general non-symmetric only). */
        WANT_LEFT,
        /** Compute right eigenvectors. For symmetric input, also enables eigenvector computation. */
        WANT_RIGHT,
    }

    public static final class Pool extends Decomposition.Workspace {
        public double[] wr;
        public double[] wi;
        public double[] vl;
        public double[] vr;

        /** Allocates/grows result arrays on demand (lazy-init, grow-only). */
        public Pool ensure(int n, boolean symmetric, int kind) {
            // wr: always ensure length >= n
            if (wr == null || wr.length < n) {
                wr = new double[n];
            }

            // wi: non-symmetric → length >= n; symmetric → not allocated (null)
            if (!symmetric) {
                if (wi == null || wi.length < n) {
                    wi = new double[n];
                }
            } else {
                wi = null;
            }

            // vr: EIGEN_RIGHT flag → length >= n*n; otherwise not allocated (null)
            if ((kind & EIGEN_RIGHT) != 0) {
                int vrSize = n * n;
                if (vr == null || vr.length < vrSize) {
                    vr = new double[vrSize];
                }
            } else {
                vr = null;
            }

            // vl: EIGEN_LEFT flag → length >= n*n; otherwise not allocated (null)
            if ((kind & EIGEN_LEFT) != 0) {
                int vlSize = n * n;
                if (vl == null || vl.length < vlSize) {
                    vl = new double[vlSize];
                }
            } else {
                vl = null;
            }

            return this;
        }

        /** Returns true if all allocated arrays satisfy minimum size requirements for the given parameters. */
        public boolean isCompatible(int n, boolean symmetric, int kind) {
            if (wr == null || wr.length < n) return false;

            if (!symmetric) {
                if (wi == null || wi.length < n) return false;
            }

            if ((kind & EIGEN_RIGHT) != 0) {
                if (vr == null || vr.length < n * n) return false;
            }

            if ((kind & EIGEN_LEFT) != 0) {
                if (vl == null || vl.length < n * n) return false;
            }

            return true;
        }
    }

    private Pool pool;
    private double[] A;
    private int n;
    private int kind;
    private boolean symmetric;
    private boolean ok;

    private Eigen() {}

    private static void ensureGeneralPool(Pool pool, int n, boolean wantLeft, boolean wantRight, int kind) {
        double[] tmp = new double[1];
        BLAS.dgeev(wantLeft, wantRight, n, null, n, null, null, null, n, null, n, tmp, 0, -1);
        pool.ensureWork((int) tmp[0]);
        pool.ensure(n, false, kind);
    }

    public static Pool workspace(int n) {
        return workspace(n, false, EIGEN_RIGHT);
    }

    public static Pool workspace(int n, boolean symmetric, int kind) {
        Pool pool = new Pool();
        if (symmetric) {
            // Query dsyev for optimal work size
            double[] tmp = new double[1];
            Dsyev.dsyev('V', 'L', n, null, n, null, 0, tmp, 0, -1);
            pool.ensureWork((int) tmp[0]);
            pool.ensure(n, true, kind);
        } else {
            boolean wantLeft  = (kind & EIGEN_LEFT)  != 0;
            boolean wantRight = (kind & EIGEN_RIGHT) != 0;
            ensureGeneralPool(pool, n, wantLeft, wantRight, kind);
        }
        return pool;
    }

    /** Opts-based entry point. */
    public static Eigen decompose(double[] A, int n, Workspace ws, Opts... opts) {
        boolean symLower = contains(opts, Opts.SYMMETRIC_LOWER);
        boolean symUpper = contains(opts, Opts.SYMMETRIC_UPPER);
        boolean symmetric = symLower || symUpper;
        char uplo = symUpper ? 'U' : 'L';
        boolean wantLeft  = contains(opts, Opts.WANT_LEFT);
        boolean wantRight = contains(opts, Opts.WANT_RIGHT);
        int kind = EIGEN_NONE;
        if (wantLeft)  kind |= EIGEN_LEFT;
        if (wantRight) kind |= EIGEN_RIGHT;
        return decompose(A, n, symmetric, uplo, kind, ws);
    }

    private static boolean contains(Opts[] opts, Opts target) {
        if (opts == null) return false;
        for (Opts o : opts) if (o == target) return true;
        return false;
    }

    public static Eigen decompose(double[] A, int n, boolean symmetric, char uplo, int kind, Workspace ws) {
        Eigen e = new Eigen();
        if (symmetric) {
            e.doSymmetric(A, n, uplo, (kind & EIGEN_RIGHT) != 0, ws);
        } else {
            e.doGeneral(A, n, kind, ws);
        }
        return e;
    }

    private void doSymmetric(double[] A, int n, char uplo, boolean wantVectors, Workspace ws) {
        if (A == null || A.length < n * n) {
            throw new IllegalArgumentException("Matrix A must have length >= n*n");
        }
        if (n <= 0) {
            throw new IllegalArgumentException("Matrix dimension must be positive");
        }

        this.A = A;
        this.n = n;
        this.symmetric = true;
        this.kind = wantVectors ? EIGEN_RIGHT : EIGEN_NONE;
        this.ok = false;

        if (ws != null && !(ws instanceof Pool)) {
            throw new IllegalArgumentException("Workspace must be an instance of Eigen.Pool");
        }

        if (ws == null) {
            ws = workspace(n, true, wantVectors ? EIGEN_RIGHT : EIGEN_NONE);
        }
        this.pool = (Pool) ws;

        pool.ensure(n, true, wantVectors ? EIGEN_RIGHT : EIGEN_NONE);
        // Query dsyev for optimal work size
        double[] tmp = new double[1];
        Dsyev.dsyev(wantVectors ? 'V' : 'N', uplo, n, null, n, null, 0, tmp, 0, -1);
        pool.ensureWork((int) tmp[0]);

        if (n == 1) {
            pool.wr[0] = A[0];
            if (wantVectors) {
                A[0] = 1.0;
            }
            this.ok = true;
            return;
        }

        char jobz = wantVectors ? 'V' : 'N';
        this.ok = Dsyev.dsyev(jobz, uplo, n, A, n, pool.wr, 0, pool.work(), 0, pool.work().length) == 0;
    }

    private void doGeneral(double[] A, int n, int kind, Workspace ws) {
        if (A == null || A.length < n * n) {
            throw new IllegalArgumentException("Matrix A must have length >= n*n");
        }
        if (n <= 0) {
            throw new IllegalArgumentException("Matrix dimension must be positive");
        }

        boolean wantLeft = (kind & EIGEN_LEFT) != 0;
        boolean wantRight = (kind & EIGEN_RIGHT) != 0;

        this.A = A;
        this.n = n;
        this.symmetric = false;
        this.kind = kind;
        this.ok = false;

        if (ws != null && !(ws instanceof Pool)) {
            throw new IllegalArgumentException("Workspace must be an instance of Eigen.Pool");
        }

        if (ws == null) {
            ws = workspace(n, false, kind);
        }
        this.pool = (Pool) ws;

        ensureGeneralPool(pool, n, wantLeft, wantRight, kind);

        if (n == 1) {
            pool.wr[0] = A[0];
            pool.wi[0] = 0.0;
            if (pool.vr != null) pool.vr[0] = 1.0;
            if (pool.vl != null) pool.vl[0] = 1.0;
            this.ok = true;
            return;
        }

        int info = BLAS.dgeev(wantLeft, wantRight, n, A, n, pool.wr, pool.wi, pool.vl, n, pool.vr, n, pool.work(), 0, pool.work().length);
        this.ok = (info == 0);
    }

    public double[] eigenvalues() {
        return ok && pool != null ? pool.wr : null;
    }

    public double[] eigenvaluesImag() {
        return ok && pool != null ? pool.wi : null;
    }

    public int kind() {
        return ok ? kind : -1;
    }

    public boolean ok() {
        return ok;
    }

    /** Returns right eigenvectors as a flat n×n row-major array, or null if not computed. */
    public double[] eigenvectors() {
        if (!ok || (kind & EIGEN_RIGHT) == 0) return null;
        int size = n * n;
        double[] dst = new double[size];
        System.arraycopy(symmetric ? A : pool.vr, 0, dst, 0, size);
        return dst;
    }

    /** Returns left eigenvectors as a flat n×n row-major array, or null if not computed. */
    public double[] leftEigenvectors() {
        if (!ok || (kind & EIGEN_LEFT) == 0) return null;
        int size = n * n;
        double[] dst = new double[size];
        System.arraycopy(pool.vl, 0, dst, 0, size);
        return dst;
    }

    public int n() {
        return n;
    }

    public boolean isSymmetric() {
        return symmetric;
    }

    public double cond() {
        if (!ok || pool == null || pool.wr == null) return Double.NaN;

        // Use a relative threshold to identify near-zero eigenvalues
        double maxAbs = 0;
        for (int i = 0; i < n; i++) {
            double abs = symmetric ? Math.abs(pool.wr[i])
                                   : Math.hypot(pool.wr[i], pool.wi != null ? pool.wi[i] : 0.0);
            if (abs > maxAbs) maxAbs = abs;
        }
        if (maxAbs == 0) return Double.NaN;

        double tol = maxAbs * n * BLAS.dlamch('E');
        double min = Double.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            double abs = symmetric ? Math.abs(pool.wr[i])
                                   : Math.hypot(pool.wr[i], pool.wi != null ? pool.wi[i] : 0.0);
            if (abs <= tol) return Double.POSITIVE_INFINITY;
            if (abs < min) min = abs;
        }

        return maxAbs / min;
    }

    @Override
    public Pool pool() {
        return pool;
    }

    /**
     * Returns eigenvalues as a matrix.
     * Symmetric: n×1 real matrix. General: n×1 complex matrix (interleaved [wr0,wi0,...]).
     */
    public Matrix toS() {
        if (!ok || pool == null || pool.wr == null) return null;
        if (symmetric) {
            return new Matrix(n, 1, false, pool.wr);
        } else {
            double[] dst = new double[n * 2];
            for (int i = 0; i < n; i++) {
                dst[i * 2]     = pool.wr[i];
                dst[i * 2 + 1] = pool.wi != null ? pool.wi[i] : 0.0;
            }
            return new Matrix(n, 1, true, dst);
        }
    }

    /** Returns right eigenvectors as an n×n matrix, or null if not computed. */
    public Matrix toV() {
        if (!ok || (kind & EIGEN_RIGHT) == 0) return null;
        return new Matrix(n, n, false, symmetric ? A : pool.vr);
    }

    public double[] eigenvector(int j) {
        return eigenvector(j, null);
    }

    public double[] eigenvector(int j, double[] dst) {
        if (!ok || j < 0 || j >= n) return null;
        if (symmetric && (kind & EIGEN_RIGHT) == 0) return null;
        if (dst == null || dst.length < n * 2) {
            dst = new double[n * 2];
        }
        if (symmetric) {
            for (int i = 0; i < n; i++) {
                dst[i * 2] = A[i * n + j];
                dst[i * 2 + 1] = 0;
            }
            return dst;
        }
        if ((kind & EIGEN_RIGHT) == 0) return null;

        double[] wi = pool.wi;
        double[] vr = pool.vr;
        if (wi[j] == 0) {
            for (int i = 0; i < n; i++) {
                dst[i * 2] = vr[i * n + j];
                dst[i * 2 + 1] = 0;
            }
        } else if (wi[j] > 0) {
            if (j + 1 >= n) return null; // complex pair must have a conjugate at j+1
            for (int i = 0; i < n; i++) {
                dst[i * 2] = vr[i * n + j];
                dst[i * 2 + 1] = vr[i * n + j + 1];
            }
        } else {
            for (int i = 0; i < n; i++) {
                dst[i * 2] = vr[i * n + j - 1];
                dst[i * 2 + 1] = -vr[i * n + j];
            }
        }
        return dst;
    }
}
