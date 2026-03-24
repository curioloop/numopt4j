/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg;

import com.curioloop.numopt4j.linalg.reg.OLS;
import com.curioloop.numopt4j.linalg.reg.WLS;

/**
 * Facade for ordinary and weighted least squares regression.
 *
 * <p>Solves the linear model y = Xβ + ε, with optional per-observation weights.
 *
 * <pre>{@code
 * // OLS with SVD solver
 * OLS r = Regressor.ols(y, X, n, k, Regressor.Opts.PINV);
 *
 * // OLS with QR solver
 * OLS r = Regressor.ols(y, X, n, k, Regressor.Opts.QR);
 *
 * // WLS
 * WLS r = Regressor.wls(y, X, weights, n, k, Regressor.Opts.PINV);
 *
 * // Workspace reuse across multiple fits
 * Regressor.Pool ws = new Regressor.Pool();
 * for (double[] yi : series) {
 *     OLS r = Regressor.ols(yi, X, n, k, ws, Regressor.Opts.PINV);
 * }
 * }</pre>
 *
 * <p>Data layout: 𝗫 is row-major n×k, each row is one observation.
 * Neither 𝒚 nor 𝗫 is modified by any method in this class.
 */
public final class Regressor {

    private Regressor() {}

    // =========================================================================
    // Opts
    // =========================================================================

    /** Algorithm options for least squares fitting. */
    public enum Opts {
        /** Use QR factorization (faster when X is full rank). */
        QR,
        /** Use SVD/pinv solver (numerically robust, handles rank-deficient X). */
        PINV,
        /** Skip automatic constant-column detection (treat kConst as 0). */
        NO_CONST_DETECT
    }

    // =========================================================================
    // Pool
    // =========================================================================

    /**
     * Reusable workspace for least squares computations.
     *
     * <p>Buffers are grown on demand and reused across multiple {@code ols}/{@code wls} calls,
     * eliminating per-fit allocations for the LAPACK work array, input copies, and pivot indices.
     * Shared by both the constant-detection SVDs and the main solver.
     */
    public static final class Pool {

        public double[] work;  // LAPACK floating-point scratch (SVD/QR work array)
        public double[] yCopy; // copy of endogenous vector (solver input)
        public double[] xCopy; // copy of exogenous matrix  (solver input, possibly whitened)
        public int[]    ipiv;  // pivot indices for dgetrf/dgetri (QR path)

        public Pool() {}

        public Pool ensureWork(int size) {
            if (work == null || work.length < size) work = new double[size];
            return this;
        }

        public Pool ensureData(int n, int nk) {
            if (yCopy == null || yCopy.length < n)  yCopy = new double[n];
            if (xCopy == null || xCopy.length < nk) xCopy = new double[nk];
            return this;
        }

        public Pool ensureIpiv(int size) {
            if (ipiv == null || ipiv.length < size) ipiv = new int[size];
            return this;
        }
    }

    // =========================================================================
    // OLS
    // =========================================================================

    /**
     * Fits an ordinary least squares model.
     *
     * @param y    endogenous vector (length >= n, not modified)
     * @param X    exogenous matrix, row-major n×k (not modified)
     * @param n    number of observations
     * @param k    number of regressors
     * @param opts zero or more {@link Opts} values
     * @return fitted OLS result
     */
    public static OLS ols(double[] y, double[] X, int n, int k, Opts... opts) {
        return ols(y, X, n, k, null, opts);
    }

    /**
     * Fits an ordinary least squares model with workspace reuse.
     *
     * @param y    endogenous vector (length >= n, not modified)
     * @param X    exogenous matrix, row-major n×k (not modified)
     * @param n    number of observations
     * @param k    number of regressors
     * @param ws   reusable workspace (may be null)
     * @param opts zero or more {@link Opts} values
     * @return fitted OLS result
     */
    public static OLS ols(double[] y, double[] X, int n, int k, Pool ws, Opts... opts) {
        boolean hasQR = contains(opts, Opts.QR), hasPINV = contains(opts, Opts.PINV);
        if (!hasQR && !hasPINV) throw new IllegalArgumentException("Must specify Opts.QR or Opts.PINV");
        if (hasQR && hasPINV)  throw new IllegalArgumentException("Cannot specify both Opts.QR and Opts.PINV");
        boolean useQR = hasQR;
        boolean noConst = contains(opts, Opts.NO_CONST_DETECT);
        if (ws == null) ws = new Pool();
        return new OLS(y, X, n, k, useQR, noConst).fit(ws);
    }

    // =========================================================================
    // WLS
    // =========================================================================

    /**
     * Fits a weighted least squares model.
     *
     * @param y       endogenous vector (length >= n, not modified)
     * @param X       exogenous matrix, row-major n×k (not modified)
     * @param weights observation weights (length >= n, all positive, not modified)
     * @param n       number of observations
     * @param k       number of regressors
     * @param opts    zero or more {@link Opts} values
     * @return fitted WLS result
     */
    public static WLS wls(double[] y, double[] X, double[] weights, int n, int k, Opts... opts) {
        return wls(y, X, weights, n, k, null, opts);
    }

    /**
     * Fits a weighted least squares model with workspace reuse.
     *
     * @param y       endogenous vector (length >= n, not modified)
     * @param X       exogenous matrix, row-major n×k (not modified)
     * @param weights observation weights (length >= n, all positive, not modified)
     * @param n       number of observations
     * @param k       number of regressors
     * @param ws      reusable workspace (may be null)
     * @param opts    zero or more {@link Opts} values
     * @return fitted WLS result
     */
    public static WLS wls(double[] y, double[] X, double[] weights, int n, int k, Pool ws, Opts... opts) {
        boolean hasQR = contains(opts, Opts.QR), hasPINV = contains(opts, Opts.PINV);
        if (!hasQR && !hasPINV) throw new IllegalArgumentException("Must specify Opts.QR or Opts.PINV");
        if (hasQR && hasPINV)  throw new IllegalArgumentException("Cannot specify both Opts.QR and Opts.PINV");
        boolean useQR = hasQR;
        boolean noConst = contains(opts, Opts.NO_CONST_DETECT);
        if (ws == null) ws = new Pool();
        return new WLS(y, X, weights, n, k, useQR, noConst).fit(ws);
    }

    // =========================================================================
    // Helper
    // =========================================================================

    private static boolean contains(Opts[] opts, Opts target) {
        if (opts == null) return false;
        for (Opts o : opts) if (o == target) return true;
        return false;
    }
}
