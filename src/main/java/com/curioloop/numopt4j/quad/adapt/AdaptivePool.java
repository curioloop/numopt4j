/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.adapt;

/**
 * Reusable workspace for adaptive GK15 quadrature and all adaptive-derived families.
 *
 * <p>Backed by a single contiguous arena plus a max-heap index array.
 * Reuse across multiple calls avoids repeated allocation.</p>
 *
 * <p>After a call to {@link com.curioloop.numopt4j.quad.adapt.GK15Core} or
 * {@link com.curioloop.numopt4j.quad.adapt.GaussLobattoCore}, the fields
 * {@link #resultValue}, {@link #resultError}, {@link #resultIterations} and
 * {@link #resultEvaluations} hold the raw output so the caller can build a
 * {@link com.curioloop.numopt4j.quad.Quadrature} without an extra allocation
 * per segment.</p>
 */
public final class AdaptivePool {

    double[] arena;
    int[] heap;
    int intervals;
    int intervalLeftOffset;
    int intervalRightOffset;
    int intervalEstimateOffset;
    int intervalErrorOffset;

    // Mutable result slots — written by GK15Core / GaussLobattoCore
    double resultValue;
    double resultError;
    int    resultIterations;
    int    resultEvaluations;

    public double resultValue()      { return resultValue; }
    public double resultError()      { return resultError; }
    public int    resultIterations() { return resultIterations; }
    public int    resultEvaluations(){ return resultEvaluations; }

    public AdaptivePool() {}

    /** Ensures the arena can hold the requested number of active intervals. */
    public AdaptivePool ensure(int intervals) {
        if (this.intervals == intervals && arena != null) return this; // fast path: already sized
        intervalLeftOffset = 0;
        intervalRightOffset = intervals;
        intervalEstimateOffset = intervals * 2;
        intervalErrorOffset = intervals * 3;
        int required = intervals * 4;
        if (arena == null || arena.length < required) arena = new double[required];
        if (heap == null || heap.length < intervals) heap = new int[intervals];
        this.intervals = intervals;
        return this;
    }

    /** Returns the heap array (used by adaptive quadrature for O(log n) interval selection). */
    public int[] heap()                 { return heap; }
    public double[] arena()             { return arena; }
    public int intervals()              { return intervals; }
    public int intervalLeftOffset()     { return intervalLeftOffset; }
    public int intervalRightOffset()    { return intervalRightOffset; }
    public int intervalEstimateOffset() { return intervalEstimateOffset; }
    public int intervalErrorOffset()    { return intervalErrorOffset; }
}
