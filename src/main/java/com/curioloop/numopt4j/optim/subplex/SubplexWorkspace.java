/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.subplex;

import com.curioloop.numopt4j.optim.Bound;

import java.util.Arrays;

/**
 * Pre-allocated workspace for the Subplex algorithm.
 *
 * <p>Stores both Nelder-Mead simplex arrays (sized for subspace dimension ≤ nsmax)
 * and Subplex outer-loop arrays (sized for full problem dimension n).
 * Can be reused across multiple {@code solve()} calls via
 * {@link SubplexProblem#solve(SubplexWorkspace)}.</p>
 *
 * <p>Use the no-arg constructor and call {@link #ensure(int)} before each solve.</p>
 */
public final class SubplexWorkspace {

    // ── NM inner arrays (sized for subspace dimension ≤ nsmax) ────────────

    /** Simplex vertices, stored flat: sim[i*ns + j]. Size (ns+1)*ns. */
    double[] sim;

    /** Function values at each vertex. Size ns+1. */
    double[] fsim;

    /** Centroid of best ns vertices. Size ns. */
    double[] xbar;

    /** Reflected point. Size ns. */
    double[] xr;

    /** Scratch point: shared by expand, contract, and sort (never simultaneous). Size ns. */
    double[] xc;

    // ── Subplex outer-loop arrays (sized for full dimension n) ────────────

    /** Previous x for progress tracking. Size n. */
    double[] xprev;

    /** Progress vector dx = x - xprev. Size n. */
    double[] dx;

    /** Permutation indices sorted by |dx|. Size n. */
    int[] perm;

    /** Subspace x coordinates. Size nsmax. */
    double[] xs;

    /** Subspace step sizes. Size nsmax. */
    double[] xsstep;

    /** Subspace bounds. Size nsmax. May be null if unbounded. */
    Bound[] subBounds;

    /** NM subspace capacity. */
    private int nmCapacity;

    /** Full problem dimension capacity. */
    private int fullCapacity;

    /**
     * Ensures the workspace can handle full dimension {@code n}.
     * Reallocates all arrays if capacity is insufficient, then calls {@link #reset()}.
     *
     * @param n required full problem dimension
     */
    public void ensure(int n) {
        int nsmax = Math.min(5, n);
        if (n > fullCapacity || nsmax > nmCapacity) {
            this.fullCapacity = n;
            this.nmCapacity = nsmax;
            // NM arrays
            sim = new double[(nsmax + 1) * nsmax];
            fsim = new double[nsmax + 1];
            xbar = new double[nsmax];
            xr = new double[nsmax];
            xc = new double[nsmax];
            // Outer-loop arrays
            xprev = new double[n];
            dx = new double[n];
            perm = new int[n];
            xs = new double[nsmax];
            xsstep = new double[nsmax];
            subBounds = new Bound[nsmax];
        }
        reset();
    }

    /**
     * Resets all arrays to their initial state without reallocating memory.
     */
    public void reset() {
        resetNm();
        resetFull();
    }

    /**
     * Resets NM arrays for a fresh inner solve.
     */
    void resetNm() {
        if (sim == null) return;
        Arrays.fill(sim, 0, (nmCapacity + 1) * nmCapacity, 0.0);
        Arrays.fill(fsim, 0, nmCapacity + 1, 0.0);
        Arrays.fill(xbar, 0, nmCapacity, 0.0);
        Arrays.fill(xr, 0, nmCapacity, 0.0);
        Arrays.fill(xc, 0, nmCapacity, 0.0);
    }

    /**
     * Resets outer-loop arrays for a fresh Subplex solve.
     */
    void resetFull() {
        if (dx == null) return;
        Arrays.fill(dx, 0, fullCapacity, 0.0);
    }
}
