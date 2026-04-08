/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.ode.implicit;

import com.curioloop.numopt4j.quad.ode.ODEPool;

/**
 * Shared workspace base class for implicit ODE solvers (BDF and Radau).
 *
 * <p>Extends {@link ODEPool} with the buffers needed by all implicit methods:
 * <ul>
 *   <li>{@link #jacBuf} — n×n Jacobian matrix (row-major).</li>
 *   <li>{@link #newtonBuf} — Newton iteration increment vector.</li>
 *   <li>{@link #scaleBuf} — per-component error scaling vector.</li>
 *   <li>{@link #jacFactor} — adaptive step-size factors for numerical Jacobian (scipy {@code num_jac}).</li>
 * </ul>
 * </p>
 *
 * <p>Concrete subclasses ({@link BDFPool}, {@link RadauPool}) add method-specific buffers.</p>
 */
public abstract class ImplicitPool extends ODEPool {

    /**
     * Jacobian buffer, n×n, row-major: {@code jacBuf[i*n+j] = ∂fᵢ/∂yⱼ}.
     * Written by {@link ImplicitCore#computeJacobian}.
     */
    public double[] jacBuf;

    /**
     * Newton iteration increment vector, length n.
     * Reused across Newton iterations to avoid allocation.
     */
    public double[] newtonBuf;

    /**
     * Per-component error scaling vector, length n.
     * Typically {@code scale[i] = atol + rtol * |y[i]|}.
     */
    public double[] scaleBuf;

    /**
     * Per-component adaptive step-size factor for the numerical Jacobian (scipy {@code num_jac}).
     * Initialised to {@code sqrt(eps)} and updated adaptively after each Jacobian evaluation.
     * Persistent across steps — must not be aliased to other workspace arrays.
     */
    public double[] jacFactor;

    /**
     * Ensures all base implicit buffers are allocated for a system of dimension {@code n}.
     * Called by subclass {@code ensure(n)}.
     *
     * @param n system dimension
     * @return {@code this} for chaining
     */
    public ImplicitPool ensure(int n) {
        ensureBase(n);
        if (jacBuf    == null || jacBuf.length    < n * n) jacBuf    = new double[n * n];
        if (newtonBuf == null || newtonBuf.length < n)     newtonBuf = new double[n];
        if (scaleBuf  == null || scaleBuf.length  < n)     scaleBuf  = new double[n];
        if (jacFactor == null || jacFactor.length < n)     jacFactor = ImplicitCore.numJacInitFactor(n);
        return this;
    }

    /**
     * In-place interpolation using the pool's live {@link #interpCoeffs} state.
     * Used during integration for event detection and t_eval evaluation.
     *
     * @param t   query time
     * @param out output array of length n, written in-place
     */
    public void interpolate(double t, double[] out) {
        throw new UnsupportedOperationException("interpolate not implemented");
    }

    @Override
    public abstract void interpolate(double t, double[] snapshot, double tOld, double tCur, double[] out);
}
