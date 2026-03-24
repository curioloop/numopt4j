/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.reg;

import static java.lang.Math.sqrt;

/**
 * Weighted Least Squares (WLS) linear regression.
 *
 * <p>Solves the weighted linear model:
 * <pre>  y = Xβ + ε,  ε ~ N(0, σ²W⁻¹)</pre>
 * where W = diag(w₁, …, wₙ) is the diagonal weight matrix.
 *
 * <p>Equivalent to OLS on the whitened system:
 * <pre>  ỹ = W^½y,  X̃ = W^½X</pre>
 * so that β̂ = (X̃ᵀX̃)⁻¹X̃ᵀỹ = (XᵀWX)⁻¹XᵀWy.
 *
 * <p>Data layout: same as {@link OLS} — X is row-major n×k.
 * <b>X is overwritten in-place</b> (scaled by sqrt(w)) before being passed to the solver.
 * y is never modified; the whitened copy ỹ is held in {@link Pool#yWhiten}.
 * After the solver, fitted and residual are unscaled back to the original scale.
 */
public class WLS extends OLS {

    private final double[] w;

    // =========================================================================
    // Pool
    // =========================================================================

    /**
     * Reusable workspace for WLS computations.
     *
     * <p>Extends {@link OLS.Pool} with one WLS-specific buffer:
     * <ul>
     *   <li>{@code yWhiten} — whitened ỹ = √W·y, length n</li>
     * </ul>
     * whitened X̃ = √W·X is scaled in-place (same as OLS). The inherited {@code fitted}/{@code residual}
     * buffers hold the <em>unwhitened</em> ŷ = X·β̂ and e = y - ŷ.
     */
    public static class Pool extends OLS.Pool {

        public double[] yWhiten; // ỹ = √W·y, length n

        public Pool() {}

        @Override
        public Pool ensure(int n, int k) {
            if (yWhiten == null || yWhiten.length < n)            yWhiten = new double[n];
            if (beta == null || beta.length < k)                  beta = new double[k];
            if (unscaledCov == null || unscaledCov.length < k*k)  unscaledCov = new double[k*k];
            if (fitted   == null || fitted.length   < n) fitted   = new double[n];
            if (residual == null || residual.length < n) residual = new double[n];
            return this;
        }
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    public WLS(double[] y, double[] X, double[] weights, int n, int k, boolean useQR, boolean noConst) {
        super(y, X, n, k, useQR, noConst);
        if (weights.length < n) throw new IllegalArgumentException("weights too short");
        this.w = weights;
    }

    public WLS(double[] y, double[] X, double[] weights, int n, int k, boolean useQR) {
        this(y, X, weights, n, k, useQR, false);
    }

    public WLS(double[] y, double[] X, double[] weights, int n, int k) {
        this(y, X, weights, n, k, false, false);
    }

    @Override public double[] weights() { return w; }

    @Override
    public WLS.Pool alloc() {
        return new WLS.Pool();
    }

    // =========================================================================
    // fit
    // =========================================================================

    @Override
    public WLS fit() {
        return fit(new Pool());
    }

    /**
     * Fits the model with workspace reuse.
     *
     * <p>Scales X in-place by sqrt(w) and writes ỹ into {@code ws.yWhiten},
     * then delegates to the OLS solver on the whitened system.
     * After the solver, fitted and residual are divided by sqrt(w) to restore
     * the original scale. <b>X is overwritten.</b>
     */
    @Override
    public WLS fit(OLS.Pool ws) {
        return fitInternal((ws instanceof Pool) ? (Pool) ws : new Pool());
    }

    public WLS fit(Pool ws) {
        return fitInternal(ws == null ? new Pool() : ws);
    }

    private WLS fitInternal(Pool ws) {
        ws.ensure(nObs, nParams);
        if (kConst < 0) kConst = detectConst(X, nObs, nParams, ws);
        whiten(ws.yWhiten, nObs, nParams);
        if (useQR) solveQR(ws.yWhiten, X, ws);
        else       solveSVD(ws.yWhiten, X, ws);
        unwhitenFittedResidual(ws, nObs);
        return this;
    }

    // =========================================================================
    // Whiten / unwhiten
    // =========================================================================

    // ỹ[i] = sqrt(w[i]) * y[i] into yW,  X[i,j] *= sqrt(w[i]) in-place
    private void whiten(double[] yW, int n, int k) {
        for (int i = 0; i < n; i++) {
            double sqrtW = sqrt(w[i]);
            yW[i] = y[i] * sqrtW;
            for (int j = 0; j < k; j++) X[i * k + j] *= sqrtW;
        }
    }

    // After OLS solver: divide fitted/residual by sqrt(w[i]) to restore original scale
    private void unwhitenFittedResidual(Pool ws, int n) {
        for (int i = 0; i < n; i++) {
            double invSqrtW = 1.0 / sqrt(w[i]);
            ws.fitted[i]   *= invSqrtW;
            ws.residual[i] *= invSqrtW;
        }
    }
}
