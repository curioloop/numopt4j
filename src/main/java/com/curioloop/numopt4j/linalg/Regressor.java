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
 * // OLS with SVD solver (X is overwritten in-place; clone if needed)
 * OLS ols = Regressor.ols(y, X, n, k, Regressor.Opts.PINV);
 *
 * // OLS with QR solver
 * OLS ols = Regressor.ols(y, X, n, k, Regressor.Opts.QR);
 *
 * // WLS (X is overwritten in-place; y is never modified)
 * WLS wls = Regressor.wls(y, X, weights, n, k, Regressor.Opts.PINV);
 *
 * // Workspace reuse across multiple fits
 * OLS.Pool ws = new OLS.Pool();
 * for (double[] Xi : series) {
 *     OLS r = Regressor.ols(y, Xi.clone(), n, k, ws, Regressor.Opts.PINV);
 * }
 * }</pre>
 *
 * <p>Data layout: X is row-major n×k, each row is one observation.
 * <b>Both OLS and WLS overwrite X in-place</b>; y is never modified.
 * WLS additionally writes whitened y~ into {@link WLS.Pool#yWhiten}.
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
        /**
         * Declare that X contains a constant column (intercept); sets kConst = 1.
         * Skips automatic detection. Equivalent to statsmodels hasconst=True.
         */
        HAS_CONST,
        /**
         * Declare that X has no constant column; sets kConst = 0.
         * Skips automatic detection. Equivalent to statsmodels hasconst=False.
         */
        NO_CONST
    }

    // =========================================================================
    // OLS
    // =========================================================================

    /**
     * Fits an ordinary least squares model.
     *
     * <p><b>Warning:</b> X is overwritten in-place. Clone X before calling if
     * the original data must be preserved.
     *
     * @param y    endogenous vector (length >= n, not modified)
     * @param X    exogenous matrix, row-major n×k (<b>overwritten in-place</b>)
     * @param n    number of observations
     * @param k    number of regressors
     * @param opts one or more {@link Opts} values; {@code QR} or {@code PINV} is required
     * @return fitted OLS result
     */
    public static OLS ols(double[] y, double[] X, int n, int k, Opts... opts) {
        return ols(y, X, n, k, (OLS.Pool) null, opts);
    }

    /**
     * Fits an ordinary least squares model with workspace reuse.
     *
     * @param y    endogenous vector (length >= n, not modified)
     * @param X    exogenous matrix, row-major n×k (<b>overwritten in-place</b>)
     * @param n    number of observations
     * @param k    number of regressors
     * @param ws   reusable workspace (may be null; {@link Pool} is accepted)
     * @param opts one or more {@link Opts} values; {@code QR} or {@code PINV} is required
     * @return fitted OLS result
     */
    public static OLS ols(double[] y, double[] X, int n, int k, OLS.Pool ws, Opts... opts) {
        boolean hasQR = contains(opts, Opts.QR), hasPINV = contains(opts, Opts.PINV);
        if (!hasQR && !hasPINV) throw new IllegalArgumentException("Must specify Opts.QR or Opts.PINV");
        if (hasQR && hasPINV)   throw new IllegalArgumentException("Cannot specify both Opts.QR and Opts.PINV");
        int kConst = contains(opts, Opts.HAS_CONST) ? 1 : contains(opts, Opts.NO_CONST) ? 0 : -1;
        if (ws == null) ws = new OLS.Pool();
        return new OLS(y, X, n, k, hasQR, kConst).fit(ws);
    }

    // =========================================================================
    // WLS
    // =========================================================================

    /**
     * Fits a weighted least squares model.
     *
     * @param y       endogenous vector (length >= n, not modified)
     * @param X       exogenous matrix, row-major n×k (<b>overwritten in-place</b>)
     * @param weights observation weights (length >= n, all positive, not modified)
     * @param n       number of observations
     * @param k       number of regressors
     * @param opts    one or more {@link Opts} values; {@code QR} or {@code PINV} is required
     * @return fitted WLS result
     */
    public static WLS wls(double[] y, double[] X, double[] weights, int n, int k, Opts... opts) {
        return wls(y, X, weights, n, k, (WLS.Pool) null, opts);
    }

    /**
     * Fits a weighted least squares model with workspace reuse.
     *
     * @param y       endogenous vector (length >= n, not modified)
     * @param X       exogenous matrix, row-major n×k (<b>overwritten in-place</b>)
     * @param weights observation weights (length >= n, all positive, not modified)
     * @param n       number of observations
     * @param k       number of regressors
     * @param ws      reusable workspace (may be null; {@link Pool} is accepted)
     * @param opts    one or more {@link Opts} values; {@code QR} or {@code PINV} is required
     * @return fitted WLS result
     */
    public static WLS wls(double[] y, double[] X, double[] weights, int n, int k, WLS.Pool ws, Opts... opts) {
        boolean hasQR = contains(opts, Opts.QR), hasPINV = contains(opts, Opts.PINV);
        if (!hasQR && !hasPINV) throw new IllegalArgumentException("Must specify Opts.QR or Opts.PINV");
        if (hasQR && hasPINV)   throw new IllegalArgumentException("Cannot specify both Opts.QR and Opts.PINV");
        int kConst = contains(opts, Opts.HAS_CONST) ? 1 : contains(opts, Opts.NO_CONST) ? 0 : -1;
        if (ws == null) ws = new WLS.Pool();
        return new WLS(y, X, weights, n, k, hasQR, kConst).fit(ws);
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
