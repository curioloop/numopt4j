/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.reg;

import com.curioloop.numopt4j.linalg.blas.BLAS;

import static java.lang.Math.*;

/**
 * Prediction result with lazy-computed variance components.
 *
 * <p>{@link #mean()} is available immediately after construction.
 * {@link #paramVar()} and {@link #residualVar()} are computed on first access and cached.
 *
 * <p>Obtain via {@link com.curioloop.numopt4j.linalg.Regression#predict(double[], int, double[])}.
 */
public final class Prediction {

    // ---- inputs held for lazy computation ----
    private final double[] newX;        // m×k row-major
    private final double[] unscaledCov; // k×k (XᵀX)⁻¹, upper triangle
    private final double[] weights;     // observation weights (null = uniform)
    private final int      m;
    private final int      k;
    private final double   scale;       // σ̂²
    private final int dfResidual;       // degrees of freedom for t-quantile

    private final double[] mean;  // ŷ = newX·β̂ (length m)
    private double[] paramVar;    // x·Cov(β̂)·xᵀ
    private double[] residualVar; // σ̂²/wᵢ


    public Prediction(double[] newX, double[] unscaledCov, double[] beta,
                      double scale, double[] weights, int m, int k, int dfResidual) {
        this.newX        = newX;
        this.unscaledCov = unscaledCov;
        this.weights     = weights;
        this.m           = m;
        this.k           = k;
        this.scale       = scale;
        this.dfResidual  = dfResidual;

        // mean is cheap (single dgemv) — compute eagerly
        this.mean = new double[m];
        BLAS.dgemv(BLAS.Trans.NoTrans, m, k, 1.0, newX, 0, k, beta, 0, 1, 0.0, mean, 0, 1);
    }

    /** Predicted mean ŷ = newX·β̂ (length m). */
    public double[] mean() { return mean; }

    /**
     * Parameter variance xᵢᵀ·Cov(β̂)·xᵢ per observation (length m).
     * Computed lazily on first call via dsymv + ddot loop.
     */
    public double[] paramVar() {
        if (paramVar == null) {
            paramVar = new double[m];
            double[] tmp = new double[k];
            for (int i = 0; i < m; i++) {
                int off = i * k;
                BLAS.dsymv(BLAS.Uplo.Upper, k, 1.0, unscaledCov, 0, k, newX, off, 1, 0.0, tmp, 0, 1);
                paramVar[i] = scale * BLAS.ddot(k, tmp, 0, 1, newX, off, 1);
            }
        }
        return paramVar;
    }

    /**
     * Residual variance σ̂²/wᵢ per observation (length m).
     * Computed lazily on first call.
     */
    public double[] residualVar() {
        if (residualVar == null) {
            residualVar = new double[m];
            if (weights != null) {
                for (int i = 0; i < m; i++) residualVar[i] = scale / weights[i];
            } else {
                java.util.Arrays.fill(residualVar, scale);
            }
        }
        return residualVar;
    }

    /**
     * Computes prediction intervals.
     *
     * <p>std[i] = sqrt(residualVar[i] + paramVar[i])
     * <p>lb[i]  = mean[i] - Q·std[i],  ub[i] = mean[i] + Q·std[i]
     * <p>where Q = t_{dfResidual, 1-alpha/2}
     *
     * @param alpha significance level (e.g. 0.05 for 95%)
     * @return double[3][] — [0]=lower bounds, [1]=upper bounds, [2]=std errors
     */
    public double[][] confInt(double alpha) {
        double[] pv  = paramVar();
        double[] rv  = residualVar();
        double   q   = tQuantile(dfResidual, 1.0 - alpha / 2.0);
        double[] lb  = new double[m];
        double[] ub  = new double[m];
        double[] std = new double[m];
        for (int i = 0; i < m; i++) {
            std[i] = sqrt(rv[i] + pv[i]);
            lb[i]  = mean[i] - q * std[i];
            ub[i]  = mean[i] + q * std[i];
        }
        return new double[][]{lb, ub, std};
    }

    // ---- t-distribution quantile ----

    /**
     * t-distribution quantile via Cornish-Fisher initial estimate + Newton-Raphson refinement.
     * Converges to machine precision in 2-3 iterations for typical df/p values.
     */
    public static double tQuantile(int df, double p) {
        if (p <= 0) return Double.NEGATIVE_INFINITY;
        if (p >= 1) return Double.POSITIVE_INFINITY;
        if (p == 0.5) return 0.0;

        double z = normalQuantile(p);
        double z2 = z * z;
        double df1 = df, df2 = df * df, df3 = df * df * df;
        double x = z
            + (z2 * z + z) / (4.0 * df1)
            + (5 * z2 * z2 * z + 16 * z2 * z + 3 * z) / (96.0 * df2)
            + (3 * z2 * z2 * z2 * z + 19 * z2 * z2 * z + 17 * z2 * z - 15 * z) / (384.0 * df3);

        double v = df;
        double logC = lgamma((v + 1) / 2.0) - lgamma(v / 2.0) - 0.5 * log(v * PI);

        for (int iter = 0; iter < 6; iter++) {
            double fx  = tCdf(df, x) - p;
            double fpx = exp(logC + log1p(x * x / v) * (-(v + 1) / 2.0));
            if (fpx == 0) break;
            double dx = fx / fpx;
            x -= dx;
            if (abs(dx) < 1e-15 * abs(x)) break;
        }
        return x;
    }

    private static double tCdf(int df, double x) {
        double v = df;
        double z = v / (v + x * x);
        double ib = regularizedIncompleteBeta(v / 2.0, 0.5, z);
        return x >= 0 ? 1.0 - ib / 2.0 : ib / 2.0;
    }

    private static double regularizedIncompleteBeta(double a, double b, double x) {
        if (x <= 0) return 0;
        if (x >= 1) return 1;
        if (x > (a + 1) / (a + b + 2))
            return 1.0 - regularizedIncompleteBeta(b, a, 1.0 - x);
        double lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
        double front = exp(log(x) * a + log1p(-x) * b - lbeta) / a;
        return front * betaCF(a, b, x);
    }

    private static double betaCF(double a, double b, double x) {
        final int    MAX_ITER = 200;
        final double EPS = 1e-15, TINY = 1e-300;
        double c = 1.0, d = 1.0 - (a + b) * x / (a + 1);
        if (abs(d) < TINY) d = TINY;
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= MAX_ITER; m++) {
            int    m2 = 2 * m;
            double aa = (double) m * (b - m) * x / ((a + m2 - 1) * (a + m2));
            d = 1.0 + aa * d; if (abs(d) < TINY) d = TINY; d = 1.0 / d;
            c = 1.0 + aa / c; if (abs(c) < TINY) c = TINY;
            h *= d * c;
            aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1));
            d = 1.0 + aa * d; if (abs(d) < TINY) d = TINY; d = 1.0 / d;
            c = 1.0 + aa / c; if (abs(c) < TINY) c = TINY;
            double delta = d * c;
            h *= delta;
            if (abs(delta - 1.0) < EPS) break;
        }
        return h;
    }

    private static final double[] LANCZOS = {
        0.99999999999980993, 676.5203681218851,   -1259.1392167224028,
        771.32342877765313,  -176.61502916214059,  12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    };

    private static double lgamma(double z) {
        if (z < 0.5) return log(PI / sin(PI * z)) - lgamma(1.0 - z);
        z -= 1;
        double x = LANCZOS[0];
        for (int i = 1; i < 9; i++) x += LANCZOS[i] / (z + i);
        double t = z + 7.5;
        return 0.5 * log(2 * PI) + (z + 0.5) * log(t) - t + log(x);
    }

    private static final double NQ_C0 = 2.515517, NQ_C1 = 0.802853, NQ_C2 = 0.010328;
    private static final double NQ_D1 = 1.432788, NQ_D2 = 0.189269, NQ_D3 = 0.001308;

    public static double normalQuantile(double p) {
        if (p <= 0) return Double.NEGATIVE_INFINITY;
        if (p >= 1) return Double.POSITIVE_INFINITY;
        if (p < 0.5) return -normalQuantile(1 - p);
        double t = sqrt(-2.0 * log1p(-p));
        return t - (NQ_C0 + NQ_C1 * t + NQ_C2 * t * t)
                 / (1 + NQ_D1 * t + NQ_D2 * t * t + NQ_D3 * t * t * t);
    }
}
