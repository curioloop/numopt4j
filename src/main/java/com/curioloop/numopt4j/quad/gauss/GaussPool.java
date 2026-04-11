/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss;

import java.util.Arrays;

/**
 * Reusable workspace for Gaussian rule generation and fixed-point quadrature.
 *
 * <p>Backed by a single contiguous arena; offsets are recomputed on each {@link #ensure} call.
 * Reuse across multiple calls of the same or smaller point count avoids repeated allocation.</p>
 */
public final class GaussPool {

    double[] arena;
    int points;
    int nodesOffset;
    int weightsOffset;
    int matrixOffset;
    int spectrumOffset;
    int offDiagonalOffset;
    int workOffset;
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
     *
     * @return {@code true} if nodes/weights were recomputed, {@code false} if the cache was hit
     */
    public GaussPool ensure(int points, GaussRule rule) {
        ensure(points);
        if (!rule.equals(cachedRule)) {
            rule.generate(this);
            cachedRule = rule;
        }
        return this;
    }

    /** Ensures the arena can hold buffers for a rule with the given number of points. */
    private void ensure(int points) {
        if (this.points == points && arena != null) return; // fast path: already sized
        long matrixSize = (long) points * points;
        long total = 2L * points + matrixSize + points + Math.max(0, points - 1L) + Math.max(1L, 2L * points);
        if (total > Integer.MAX_VALUE) throw new IllegalArgumentException("points too large for quadrature workspace");

        nodesOffset = 0;
        weightsOffset = nodesOffset + points;
        matrixOffset = weightsOffset + points;
        spectrumOffset = matrixOffset + points * points;
        offDiagonalOffset = spectrumOffset + points;
        workOffset = offDiagonalOffset + Math.max(0, points - 1);
        int required = workOffset + Math.max(1, 2 * points);
        if (arena == null || arena.length < required) arena = new double[required];
        this.points = points;
        cachedRule = null; // arena resized, cached nodes/weights are invalid
    }

    public double[] arena()        { return arena; }
    public int points()            { return points; }
    public double nodeAt(int i)    { return arena[nodesOffset + i]; }
    public double weightAt(int i)  { return arena[weightsOffset + i]; }
    public int nodesOffset()       { return nodesOffset; }
    public int weightsOffset()     { return weightsOffset; }
    public int matrixOffset()      { return matrixOffset; }
    public int spectrumOffset()    { return spectrumOffset; }
    public int offDiagonalOffset() { return offDiagonalOffset; }
    public int workOffset()        { return workOffset; }

    /** Returns a snapshot of the most recently generated nodes. */
    public double[] nodes() {
        return arena == null ? null : Arrays.copyOfRange(arena, nodesOffset, nodesOffset + points);
    }

    /** Returns a snapshot of the most recently generated weights. */
    public double[] weights() {
        return arena == null ? null : Arrays.copyOfRange(arena, weightsOffset, weightsOffset + points);
    }
}
