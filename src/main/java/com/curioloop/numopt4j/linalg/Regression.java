/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg;

import static java.lang.Math.*;

import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.linalg.reg.OLS;
import com.curioloop.numopt4j.linalg.reg.Prediction;
import com.curioloop.numopt4j.linalg.reg.WLS;

/**
 * Statistical result of an OLS or WLS fit.
 *
 * <p>All quantities are computed lazily and cached on first access.
 * Subclasses ({@link OLS}, {@link WLS}) extend this class and populate
 * the solver outputs ({@code beta}, {@code normCov}, {@code rank}, {@code cond})
 * during {@link OLS#fit()}.
 *
 * <h2>Key statistics</h2>
 * <ul>
 *   <li>{@link #parameters()} — β̂</li>
 *   <li>{@link #paramCov()}   — Cov(β̂) = σ̂²·(XᵀX)⁻¹</li>
 *   <li>{@link #bse()}        — standard errors √diag(Cov(β̂))</li>
 *   <li>{@link #ssr()}        — sum of squared (whitened) residuals ‖ỹ - X̃β̂‖²</li>
 *   <li>{@link #scale()}      — σ̂² = SSR / (n - rank)</li>
 *   <li>{@link #r2(boolean)}  — R² (optionally adjusted)</li>
 *   <li>{@link #mse()}        — model / residual / total MSE</li>
 *   <li>{@link #logLike()}    — Gaussian log-likelihood</li>
 *   <li>{@link #condNum()}    — condition number √(σ_max²/σ_min²)</li>
 *   <li>{@link #predict(double[], int, double[])} — prediction with confidence intervals</li>
 * </ul>
 */
public abstract class Regression {

    // ---- solver outputs (populated by OLS/WLS.fit) ----
    protected double[] beta;     // parameter estimates β̂, length k
    protected double[] unscaledCov;  // (XᵀX)⁻¹ or (RᵀR)⁻¹, k×k row-major
    protected int      rank;     // numerical rank of X
    protected double   cond;     // condition number √(σ_max²/σ_min²)

    // ---- lazy cache ----
    private double[] fitted;
    private double[] residual;
    private double[] whitenFitted;
    private double[] whitenResidual;
    private double   ssr    = Double.NaN;
    private double   tss    = Double.NaN;
    private double   tssCen = Double.NaN;
    private double   llf    = Double.NaN;

    // ---- abstract accessors (implemented by OLS/WLS) ----

    public abstract int      nObs();
    public abstract int      nParams();
    public abstract int      kConst();
    protected abstract int      dfModel();
    protected abstract int      dfResidual();
    protected abstract double[] endog();
    protected abstract double[] exog();
    protected abstract double[] weights();


    // ---- parameters ----

    /** Parameter estimates β̂ (length k). */
    public double[] parameters() { return beta; }

    /** Numerical rank of X. */
    public int rank() { return rank; }

    /** Condition number √(σ_max²/σ_min²). */
    public double condNum() { return cond; }

    // ---- fitted values and residuals ----

    /**
     * Fitted values ŷ = Xβ̂.
     *
     * @param whiten if true and weights are present, returns √W·ŷ
     */
    public double[] fitted(boolean whiten) {
        double[] w = weights();
        if (!whiten || w == null) {
            if (fitted == null) fitted = computeFitted();
            return fitted;
        } else {
            if (whitenFitted == null) {
                double[] f = fitted(false);
                int n = nObs();
                whitenFitted = new double[n];
                for (int i = 0; i < n; i++) whitenFitted[i] = f[i] * sqrt(w[i]);
            }
            return whitenFitted;
        }
    }

    private double[] computeFitted() {
        int n = nObs(), k = nParams();
        double[] f = new double[n];
        BLAS.dgemv(BLAS.Trans.NoTrans, n, k, 1.0, exog(), 0, k, beta, 0, 1, 0.0, f, 0, 1);
        return f;
    }

    /**
     * Residuals e = y - ŷ.
     *
     * @param whiten if true and weights are present, returns √W·e
     */
    public double[] residual(boolean whiten) {
        double[] w = weights();
        if (!whiten || w == null) {
            if (residual == null) residual = computeResidual();
            return residual;
        } else {
            if (whitenResidual == null) {
                double[] e = residual(false);
                int n = nObs();
                whitenResidual = new double[n];
                for (int i = 0; i < n; i++) whitenResidual[i] = e[i] * sqrt(w[i]);
            }
            return whitenResidual;
        }
    }

    private double[] computeResidual() {
        double[] y = endog();
        double[] f = fitted(false);
        int n = nObs();
        double[] e = y.clone();  // e = y
        // e = y - f  via daxpy: e += -1 * f
        BLAS.daxpy(n, -1.0, f, 0, 1, e, 0, 1);
        return e;
    }

    // ---- sum of squares ----

    /** Sum of squared (whitened) residuals: SSR = ‖ỹ - X̃β̂‖² = ẽᵀẽ. */
    public double ssr() {
        if (Double.isNaN(ssr)) {
            double[] e = residual(true);
            ssr = BLAS.ddot(e.length, e, 0, 1, e, 0, 1);
        }
        return ssr;
    }

    /**
     * Total sum of squares.
     *
     * @param centered if true, uses centered TSS (subtract mean); otherwise uncentered
     */
    public double tss(boolean centered) {
        if (centered  && !Double.isNaN(tssCen)) return tssCen;
        if (!centered && !Double.isNaN(tss))    return tss;

        double[] y = endog();
        double[] w = weights();
        int n = nObs();

        double mean = 0;
        if (centered) {
            if (w != null) {
                mean = BLAS.ddot(n, w, 0, 1, y, 0, 1) / BLAS.dasum(n, w, 0, 1);
            } else {
                double sum = 0;
                for (int i = 0; i < n; i++) sum += y[i];
                mean = sum / n;
            }
        }

        double s = 0;
        if (w != null) {
            for (int i = 0; i < n; i++) { double d = y[i] - mean; s += w[i] * d * d; }
        } else if (centered) {
            for (int i = 0; i < n; i++) { double d = y[i] - mean; s += d * d; }
        } else {
            // uncentered, no weights: TSS = y·y
            s = BLAS.ddot(n, y, 0, 1, y, 0, 1);
        }

        if (centered) tssCen = s; else tss = s;
        return s;
    }

    /** Explained sum of squares (ESS = TSS - SSR). */
    public double ess() {
        return tss(kConst() > 0) - ssr();
    }

    // ---- scale and covariance ----

    /** Scale factor σ̂² = SSR / (n - rank). */
    public double scale() {
        return ssr() / dfResidual();
    }

    /**
     * Parameter covariance matrix Cov(β̂) = σ̂²·(XᵀX)⁻¹ (k×k, row-major).
     * Returns a new array each call.
     */
    public double[] paramCov() {
        double[] cov = unscaledCov.clone();
        BLAS.dscal(cov.length, scale(), cov, 0, 1);
        return cov;
    }

    /** Standard errors of parameter estimates: bse = √diag(Cov(β̂)). */
    public double[] bse() {
        double s = scale();
        int k = nParams();
        double[] se = new double[k];
        for (int i = 0; i < k; i++) se[i] = sqrt(s * unscaledCov[i * k + i]);
        return se;
    }

    // ---- goodness of fit ----

    /**
     * R² coefficient of determination.
     *
     * @param adjusted if true, returns adjusted R²
     */
    public double r2(boolean adjusted) {
        double r2 = 1.0 - ssr() / tss(kConst() > 0);
        if (adjusted) {
            int n = nObs(), kc = kConst();
            return 1.0 - ((double)(n - kc) / dfResidual()) * (1.0 - r2);
        }
        return r2;
    }

    /**
     * Mean squared errors.
     *
     * @return double[3] = {MSE_model, MSE_residual, MSE_total}
     */
    public double[] mse() {
        double mseModel    = ess() / dfModel();
        double mseResidual = ssr() / dfResidual();
        double mseTotal    = tss(kConst() > 0) / (dfModel() + dfResidual());
        return new double[]{mseModel, mseResidual, mseTotal};
    }

    // ---- information criteria ----

    /**
     * Gaussian log-likelihood:
     * <pre>  llf = -n/2·log(SSR) - n/2·(1 + log(π/n)) + ½·∑log(wᵢ)  (WLS term)</pre>
     */
    public double logLike() {
        if (Double.isNaN(llf)) {
            double n2 = nObs() / 2.0;
            double l  = -n2 * log(ssr()) - n2 * (1.0 + log(PI / n2));
            double[] w = weights();
            if (w != null) {
                double sumLogW = 0;
                for (double wi : w) sumLogW += log(wi);
                l += 0.5 * sumLogW;
            }
            llf = l;
        }
        return llf;
    }

    /** AIC = -2·llf + 2·(dfModel + kConst). */
    public double aic() {
        return -2.0 * logLike() + 2.0 * (dfModel() + kConst());
    }

    /** BIC = -2·llf + log(n)·(dfModel + kConst). */
    public double bic() {
        return -2.0 * logLike() + log(nObs()) * (dfModel() + kConst());
    }

    // ---- prediction ----

    /**
     * Predicts for new observations.
     *
     * <p>For each row xᵢ of newX:
     * <pre>
     *   mean[i]        = xᵢᵀβ̂
     *   paramVar[i]    = xᵢᵀ·Cov(β̂)·xᵢ
     *   residualVar[i] = σ̂²/wᵢ
     * </pre>
     *
     * @param newX    new exogenous matrix, row-major m×k (not modified)
     * @param m       number of new observations
     * @param weights observation weights for residual variance scaling (null = uniform)
     * @return {@link Prediction} containing mean, paramVar, residualVar
     */
    public Prediction predict(double[] newX, int m, double[] weights) {
        return new Prediction(newX, unscaledCov, beta, scale(), weights, m, nParams(), dfResidual());
    }
}
