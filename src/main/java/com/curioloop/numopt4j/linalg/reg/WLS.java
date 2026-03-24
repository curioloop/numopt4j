/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.reg;

import com.curioloop.numopt4j.linalg.Regressor;

import static java.lang.Math.sqrt;

/**
 * Weighted Least Squares (WLS) linear regression.
 *
 * <p>Solves the weighted linear model:
 * <pre>  y = Xβ + ε,  ε ~ N(0, σ²W⁻¹)</pre>
 * where W = diag(w₁, …, wₙ) is the diagonal weight matrix.
 *
 * <p>Equivalent to OLS on the whitened system:
 * <pre>  ỹ = W^½y,  X̃ = W^½X</pre>
 * so that β̂ = (X̃ᵀX̃)⁻¹X̃ᵀỹ = (XᵀWX)⁻¹XᵀWy.
 *
 * <p>Data layout: same as {@link OLS} — X is row-major n×k.
 */
public class WLS extends OLS {

    private final double[] w;     // observation weights wᵢ, length n
    private final double[] origY; // original (unwhitened) y
    private final double[] origX; // original (unwhitened) X

    public WLS(double[] y, double[] X, double[] weights, int n, int k, boolean useQR, boolean noConst) {
        super(y, X, n, k, useQR, noConst);
        if (weights.length < n) throw new IllegalArgumentException("weights too short");
        this.w = weights;
        this.origY = y;
        this.origX = X;
    }

    public WLS(double[] y, double[] X, double[] weights, int n, int k, boolean useQR) {
        this(y, X, weights, n, k, useQR, false);
    }

    public WLS(double[] y, double[] X, double[] weights, int n, int k) {
        this(y, X, weights, n, k, false, false);
    }

    @Override public double[] endog()    { return origY; }
    @Override public double[] exog()    { return origX; }
    @Override public double[] weights() { return w; }

    /** Fits the model by whitening y and X with √W, then delegating to the OLS solver. */
    @Override
    public WLS fit() {
        return fit(new Regressor.Pool());
    }

    /** Fits the model with workspace reuse. */
    @Override
    public WLS fit(Regressor.Pool ws) {
        int n = nObs, k = nParams;
        ws.ensureData(n, n * k);
        if (kConst < 0) kConst = detectConst(origX, n, k, ws);
        // Whiten: ỹᵢ = √wᵢ · yᵢ,  X̃ᵢⱼ = √wᵢ · Xᵢⱼ
        whiten(ws.yCopy, ws.xCopy, n, k);
        if (useQR) solveQR(ws.yCopy, ws.xCopy, ws);
        else       solveSVD(ws.yCopy, ws.xCopy, ws);
        return this;
    }

    private void whiten(double[] yW, double[] XW, int n, int k) {
        for (int i = 0; i < n; i++) {
            double sqrtW = sqrt(w[i]);
            yW[i] = origY[i] * sqrtW;
            for (int j = 0; j < k; j++) XW[i * k + j] = origX[i * k + j] * sqrtW;
        }
    }
}
