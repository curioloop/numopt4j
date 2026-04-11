/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss;

import java.util.Arrays;

/**
 * Reusable workspace for Gaussian rule generation and fixed-point quadrature.
 *
 * <p>The arena holds only the generated nodes and weights (2·n doubles).
 * Temporary buffers required by the Golub-Welsch eigendecomposition
 * (n×n eigenvector matrix, diagonal, off-diagonal, LAPACK work array)
 * are allocated inside {@link GaussRule#generate} and released after use,
 * so the pool's steady-state footprint is O(n) rather than O(n²).</p>
 *
 * <p>A {@code (points, rule)} cache avoids redundant regeneration when
 * {@link #ensure(int, GaussRule)} is called repeatedly with the same arguments.</p>
 */
public final class GaussPool {

    double[] arena;
    int points;
    GaussRule cachedRule;

    public GaussPool() {}

    /**
     * Ensures the arena is sized for {@code points}, then generates nodes and weights
     * for {@code rule} if the cached result does not already match.
     *
     * <p>The cache is keyed on {@code (points, rule)} using {@link Object#equals}.
     * Stateless singleton rules (Legendre, Laguerre, Hermite) hit the cache by identity;
     * parameterized rules (Jacobi, GeneralizedLaguerre, GeneralizedHermite) must implement
     * {@code equals}/{@code hashCode} for the cache to be effective.</p>
     */
    public GaussPool ensure(int points, GaussRule rule) {
        ensureArena(points);
        if (!rule.equals(cachedRule)) {
            rule.generate(this);
            cachedRule = rule;
        }
        return this;
    }

    /** Ensures the arena can hold nodes and weights for the given number of points. */
    void ensureArena(int points) {
        int required = 2 * points; // [nodes | weights]
        if (this.points == points && arena != null && arena.length >= required) return;
        if (arena == null || arena.length < required) arena = new double[required];
        this.points = points;
        cachedRule = null;
    }

    public double[] arena()        { return arena; }
    public int points()            { return points; }
    public double nodeAt(int i)    { return arena[i]; }
    public double weightAt(int i)  { return arena[points + i]; }
    public int nodesOffset()       { return 0; }
    public int weightsOffset()     { return points; }

    /** Returns a snapshot of the most recently generated nodes. */
    public double[] nodes() {
        return arena == null ? null : Arrays.copyOfRange(arena, 0, points);
    }

    /** Returns a snapshot of the most recently generated weights. */
    public double[] weights() {
        return arena == null ? null : Arrays.copyOfRange(arena, points, 2 * points);
    }
}
