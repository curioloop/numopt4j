/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.ode.rk;

import com.curioloop.numopt4j.quad.ode.ODEPool;

/**
 * Workspace for explicit Runge-Kutta solvers (RK23, RK45, DOP853).
 *
 * <p>Holds all reusable buffers needed by {@link RungeKuttaCore}, including the stage
 * slope matrix {@code K}, error buffer, and dense-output interpolation state.</p>
 *
 * <p>For DOP853, {@code KExtended} contains both the main stages and the extra stages
 * needed for dense output; {@code K} is a view of the first {@code (nStages+1)*n} elements.</p>
 */
public class RungeKuttaPool extends ODEPool {

    /**
     * Stage slope matrix, shape {@code (nStages+1) × n}, row-major.
     * {@code K[s*n + i]} = i-th component of the s-th stage slope.
     * For DOP853, this is a view of the front of {@link #KExtended}.
     */
    public double[] K;

    /**
     * Error estimation buffer, length n.
     * Aliased to {@link #yOld} (non-overlapping usage: error estimation completes
     * before {@code yOld} is written in {@code computeDenseOutput}).
     */
    public double[] errBuf;

    /**
     * Extended stage matrix for DOP853, shape {@code nStagesExtended × n}, row-major.
     * Contains both main stages and extra stages for dense output.
     * {@code null} for RK23/RK45.
     */
    public double[] KExtended;

    /** State at the start of the last accepted step, length n. Used for dense output. */
    public double[] yOld;

    /** Dense output polynomial order (set by {@link #ensure}). */
    public int interpOrder;

    public RungeKuttaPool() {}

    /**
     * Ensures all buffers are allocated for the given method and system size.
     * Expands buffers as needed; never shrinks.
     *
     * @param nStages          number of main RK stages (excluding error-estimation stage)
     * @param n                system dimension
     * @param nStagesExtended  total extended stage count for DOP853 dense output; 0 for RK23/RK45
     * @param interpOrder      dense output polynomial order
     * @return {@code this} for chaining
     */
    public RungeKuttaPool ensure(int nStages, int n, int nStagesExtended, int interpOrder) {
        ensureBase(n);
        this.interpOrder = interpOrder;
        int kSize = (nStages + 1) * n;
        if (nStagesExtended > 0) {
            // DOP853: KExtended holds all stages; K is a view of the first kSize elements
            int extSize = nStagesExtended * n;
            if (KExtended == null || KExtended.length < extSize) KExtended = new double[extSize];
            K = KExtended;
        } else {
            if (K == null || K.length < kSize) K = new double[kSize];
            KExtended = null;
        }
        if (yOld == null || yOld.length < n) yOld = new double[n];
        // errBuf reuses yOld: error estimation finishes before yOld is written
        errBuf = yOld;
        int qSize = n * interpOrder;
        if (interpCoeffs == null || interpCoeffs.length < qSize) interpCoeffs = new double[qSize];
        return this;
    }

    // -----------------------------------------------------------------------
    // Snapshot-based interpolation (used by Trajectory.DenseOutput)
    // -----------------------------------------------------------------------

    /**
     * Interpolates from a coefficient snapshot saved during integration.
     *
     * <p>Snapshot format: {@code [yOld[0..n-1], interpCoeffs[0..interpOrder*n-1]]}.
     * <ul>
     *   <li>RK23/RK45: standard Horner evaluation on the {@code Q} matrix.</li>
     *   <li>DOP853: alternating-multiply Horner on the {@code F} matrix ({@code interpOrder = 7}).</li>
     * </ul>
     * </p>
     *
     * @param t        query time
     * @param snapshot coefficient snapshot
     * @param tOld     step start time
     * @param tCur     step end time
     * @param out      output array of length n, written in-place
     */
    @Override
    public void interpolate(double t, double[] snapshot, double tOld, double tCur, double[] out) {
        double h = tCur - tOld;
        double x = (t - tOld) / h;
        if (interpOrder == 7) {
            // DOP853: alternating Horner
            // y = yOld + F[6]*x*(1-x)*x*(1-x)*x*(1-x)*x  (alternating multiply)
            for (int i = 0; i < n; i++) {
                double val = 0.0;
                for (int fi = 0; fi < interpOrder; fi++) {
                    int ri = interpOrder - 1 - fi;
                    val += snapshot[n + ri * n + i];
                    val *= (fi % 2 == 0) ? x : (1.0 - x);
                }
                out[i] = snapshot[i] + val;
            }
        } else {
            // RK23/RK45: standard Horner on Q matrix
            // y = yOld + h * Q · [x, x², ..., x^interpOrder]
            // Horner: Σⱼ Q[i][j] * x^(j+1) = x * (Q[i][0] + x*(Q[i][1] + ... + x*Q[i][p-1]))
            for (int i = 0; i < n; i++) {
                double val = 0;
                for (int j = interpOrder - 1; j >= 0; j--) {
                    val = val * x + snapshot[n + i * interpOrder + j];
                }
                out[i] = snapshot[i] + h * val * x;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Live interpolation (used during integration for event detection / t_eval)
    // -----------------------------------------------------------------------

    /**
     * In-place interpolation using the pool's live {@link #interpCoeffs} state.
     * Corresponds to scipy {@code RkDenseOutput._call_impl}:
     * <pre>
     *   x = (t - tOld) / (tCur - tOld)
     *   y = yOld + h · Q · [x, x², ..., x^interpOrder]
     * </pre>
     *
     * <p>Horner expansion:
     * <pre>
     *   Σⱼ₌₀^{p-1} Q[i][j] · x^(j+1) = x · (Q[i][0] + x·(Q[i][1] + ... + x·Q[i][p-1]))
     * </pre>
     * </p>
     *
     * @param t   query time
     * @param out output array of length n, written in-place
     */
    public void interpolate(double t, double[] out) {
        double h    = tCur - tOld;
        double x    = (t - tOld) / h;
        int    cols = interpOrder;
        for (int i = 0; i < n; i++) {
            double val = 0;
            for (int j = cols - 1; j >= 0; j--) {
                val = val * x + interpCoeffs[i * cols + j];
            }
            out[i] = yOld[i] + h * val * x;
        }
    }
}
