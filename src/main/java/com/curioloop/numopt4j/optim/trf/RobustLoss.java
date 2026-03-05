/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

/**
 * Built-in robust loss functions for Trust Region Reflective (TRF) least-squares.
 *
 * <p>The cost function minimized by TRF is:</p>
 * <pre>
 *   F(x) = 0.5 * C² * Σ ρ(fᵢ(x)² / C²)
 * </pre>
 * <p>where ρ is the loss function and C is the scale parameter ({@code lossScale}).</p>
 *
 * <p>For non-linear losses, the Jacobian and residuals are scaled in-place before
 * each trust-region step using the {@code scale_for_robust_loss_function} formula
 * from scipy:</p>
 * <pre>
 *   J_scale[i] = sqrt(ρ'(zᵢ) + 2·ρ''(zᵢ)·zᵢ)   where zᵢ = fᵢ²/C²
 *   f[i]       ← f[i] · ρ'(zᵢ) / J_scale[i]
 *   J[i,:]     ← J[i,:] · J_scale[i]
 * </pre>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * TRFProblem.create()
 *     .residuals(fn, m)
 *     .initialPoint(x0)
 *     .loss(RobustLoss.SOFT_L1)
 *     .lossScale(1.0)
 *     .solve();
 * }</pre>
 *
 * @see TRFProblem#loss(RobustLoss)
 * @see TRFProblem#lossScale(double)
 */
public enum RobustLoss {

    /**
     * Standard least-squares loss: ρ(z) = z.
     * <p>No scaling is applied. This is the default.</p>
     */
    LINEAR,

    /**
     * Soft L1 loss: ρ(z) = 2·((1+z)^0.5 - 1).
     * <p>Smooth approximation of absolute-value loss. Usually a good first choice
     * for robust least squares.</p>
     * <pre>
     *   ρ'(z)  = (1+z)^{-0.5}
     *   ρ''(z) = -0.5·(1+z)^{-1.5}
     * </pre>
     */
    SOFT_L1,

    /**
     * Huber loss: ρ(z) = z if z ≤ 1, else 2·z^0.5 - 1.
     * <p>Quadratic near zero, linear for large residuals. Similar to SOFT_L1.</p>
     * <pre>
     *   ρ'(z)  = 1          if z ≤ 1
     *   ρ'(z)  = z^{-0.5}   if z > 1
     *   ρ''(z) = 0           if z ≤ 1
     *   ρ''(z) = -0.5·z^{-1.5} if z > 1
     * </pre>
     */
    HUBER,

    /**
     * Cauchy loss: ρ(z) = ln(1+z).
     * <p>Severely weakens outlier influence, but may cause difficulties in optimization.</p>
     * <pre>
     *   ρ'(z)  = 1/(1+z)
     *   ρ''(z) = -1/(1+z)²
     * </pre>
     */
    CAUCHY,

    /**
     * Arctan loss: ρ(z) = arctan(z).
     * <p>Limits maximum loss on a single residual. Similar properties to CAUCHY.</p>
     * <pre>
     *   ρ'(z)  = 1/(1+z²)
     *   ρ''(z) = -2z/(1+z²)²
     * </pre>
     */
    ARCTAN;

    /**
     * Scales the Jacobian (row-major, m×n) and residual vector in-place for robust loss.
     *
     * <p>Implements scipy's {@code scale_for_robust_loss_function}:</p>
     * <pre>
     *   z[i]        = (f[i] / C)²
     *   J_scale[i]  = sqrt(max(ρ'(z[i]) + 2·ρ''(z[i])·z[i], ε))
     *   f[i]       ← f[i] · ρ'(z[i]) / J_scale[i]
     *   J[i,:]     ← J[i,:] · J_scale[i]
     * </pre>
     *
     * <p>No-op when called on {@link #LINEAR}.</p>
     *
     * @param fvec     residual vector f (length m), modified in-place
     * @param fjac     Jacobian matrix J (row-major, length m*n), modified in-place
     * @param m        number of residuals
     * @param n        number of parameters
     * @param fScale   soft margin C (lossScale parameter)
     */
    public void scaleJF(double[] fvec, double[] fjac, int m, int n, double fScale) {
        if (this == LINEAR) return;
        final double eps = TRFConstants.EPSMCH;
        final double invC2 = 1.0 / (fScale * fScale);
        for (int i = 0; i < m; i++) {
            double fi = fvec[i];
            double z  = fi * fi * invC2;
            double rho1, rho2;
            switch (this) {
                case SOFT_L1: {
                    double t = 1.0 + z;
                    double sqrtT = Math.sqrt(t);
                    rho1 = 1.0 / sqrtT;
                    rho2 = -0.5 / (t * sqrtT);
                    break;
                }
                case HUBER: {
                    if (z <= 1.0) {
                        rho1 = 1.0; rho2 = 0.0;
                    } else {
                        double sqrtZ = Math.sqrt(z);
                        rho1 = 1.0 / sqrtZ;
                        rho2 = -0.5 / (z * sqrtZ);
                    }
                    break;
                }
                case CAUCHY: {
                    double t = 1.0 + z;
                    rho1 = 1.0 / t;
                    rho2 = -1.0 / (t * t);
                    break;
                }
                case ARCTAN: {
                    double t = 1.0 + z * z;
                    rho1 = 1.0 / t;
                    rho2 = -2.0 * z / (t * t);
                    break;
                }
                default:
                    continue;
            }
            double jScale = rho1 + 2.0 * rho2 * z;
            if (jScale < eps) jScale = eps;
            jScale = Math.sqrt(jScale);
            // scale residual: f[i] *= rho1 / jScale
            fvec[i] = fi * rho1 / jScale;
            // scale Jacobian row i: J[i,:] *= jScale
            int rowBase = i * n;
            for (int j = 0; j < n; j++) fjac[rowBase + j] *= jScale;
        }
    }

    /**
     * Computes the robust cost: 0.5 * C² * Σ ρ(fᵢ²/C²).
     *
     * <p>For {@link #LINEAR} this equals the standard 0.5·‖f‖².</p>
     *
     * @param fvec   residual vector (length m)
     * @param m      number of residuals
     * @param fScale soft margin C
     * @return scalar cost value
     */
    public double cost(double[] fvec, int m, double fScale) {
        if (this == LINEAR) {
            double s = 0.0;
            for (int i = 0; i < m; i++) s += fvec[i] * fvec[i];
            return 0.5 * s;
        }
        double invC2 = 1.0 / (fScale * fScale);
        double sum = 0.0;
        for (int i = 0; i < m; i++) {
            double z = fvec[i] * fvec[i] * invC2;
            double rho0;
            switch (this) {
                case SOFT_L1: rho0 = 2.0 * (Math.sqrt(1.0 + z) - 1.0); break;
                case HUBER:   rho0 = (z <= 1.0) ? z : 2.0 * Math.sqrt(z) - 1.0; break;
                case CAUCHY:  rho0 = Math.log1p(z); break;
                case ARCTAN:  rho0 = Math.atan(z); break;
                default:      rho0 = z; break;
            }
            sum += rho0;
        }
        return 0.5 * fScale * fScale * sum;
    }
}
