/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.ode;

import com.curioloop.numopt4j.quad.Trajectory;
import com.curioloop.numopt4j.quad.ode.implicit.ImplicitPool;
import com.curioloop.numopt4j.quad.ode.rk.RungeKuttaPool;

/**
 * Base workspace class shared by all ODE solvers.
 *
 * <p>Holds the minimum set of reusable buffers needed by every solver:
 * a temporary state vector, a temporary derivative vector, and the
 * dense-output interpolation state written after each accepted step.</p>
 *
 * <p>Subclasses ({@link RungeKuttaPool},
 * {@link ImplicitPool}) add method-specific buffers.</p>
 *
 * <p>Workspaces can be pre-allocated and passed to {@link IVPIntegral#integrate(IVPPool)}
 * to avoid repeated allocation when solving many problems of the same size.</p>
 */
public abstract class IVPPool {

    /** System dimension (set by {@link #ensureBase}). */
    public int n;

    /** Temporary state vector, length n. Reused across solver internals. */
    public double[] yTmp;

    /** Temporary derivative vector f(t, y), length n. Reused across solver internals. */
    public double[] fTmp;

    // -----------------------------------------------------------------------
    // Dense-output interpolation state (overwritten after each accepted step)
    // -----------------------------------------------------------------------

    /** Step start time, written by the solver after each accepted step. */
    public double tOld;

    /** Step end time, written by the solver after each accepted step. */
    public double tCur;

    /**
     * Interpolation coefficient snapshot for the current step.
     * Format is method-specific; overwritten after each accepted step.
     * Used by {@link Trajectory.DenseOutput} and event detection.
     */
    public double[] interpCoeffs;

    /**
     * Ensures base buffers are allocated for a system of dimension {@code n}.
     * Called by subclass {@code ensure(n)} before allocating method-specific buffers.
     *
     * @param n system dimension
     */
    protected void ensureBase(int n) {
        this.n = n;
        if (yTmp == null || yTmp.length < n) yTmp = new double[n];
        if (fTmp == null || fTmp.length < n) fTmp = new double[n];
    }

    /**
     * Interpolates the solution at time {@code t} using a previously saved coefficient snapshot,
     * without modifying the pool's live state.
     *
     * <p>Called by {@link Trajectory.DenseOutput#interpolate} to evaluate the dense output
     * at arbitrary time points after integration is complete.</p>
     *
     * @param t        query time, must be within {@code [tOld, tCur]} (or reversed for backward integration)
     * @param snapshot coefficient snapshot saved by the solver for this step
     * @param tOld     step start time
     * @param tCur     step end time
     * @param out      output array of length n, written in-place
     */
    public abstract void interpolate(double t, double[] snapshot, double tOld, double tCur, double[] out);
}
