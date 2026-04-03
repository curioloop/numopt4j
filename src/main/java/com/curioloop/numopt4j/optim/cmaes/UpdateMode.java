/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.cmaes;

/**
 * Covariance matrix update mode for CMA-ES.
 *
 * <ul>
 *   <li>{@link #ACTIVE_CMA}  — full covariance matrix with Active CMA negative-weight update
 *       (default; best convergence on ill-conditioned problems)</li>
 *   <li>{@link #CLASSIC_CMA} — full covariance matrix without Active CMA (classic CMA-ES)</li>
 *   <li>{@link #SEP_CMA}     — diagonal covariance only (sep-CMA-ES); O(n) per iteration,
 *       suitable for high-dimensional separable problems; skips O(n²) allocations</li>
 * </ul>
 */
public enum UpdateMode {
    /** Full covariance matrix with Active CMA negative-weight update. Default mode. */
    ACTIVE_CMA(false, true),
    /** Full covariance matrix without Active CMA (classic CMA-ES). */
    CLASSIC_CMA(false, false),
    /** Diagonal covariance only — sep-CMA-ES, O(n) per iteration. */
    SEP_CMA(true, false);

    /** Whether to use diagonal-only covariance (sep-CMA-ES). */
    public final boolean separable;
    /** Whether to apply Active CMA negative-weight updates. */
    public final boolean activeCMA;

    UpdateMode(boolean separable, boolean activeCMA) {
        this.separable = separable;
        this.activeCMA = activeCMA;
    }
}
