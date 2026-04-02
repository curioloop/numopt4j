/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.cmaes;

/**
 * Restart strategy for CMA-ES optimizer.
 *
 * <ul>
 *   <li>{@link #NONE}  — single run, no restart</li>
 *   <li>{@link #IPOP}  — Increasing Population restart: lambda *= incPopSize each restart</li>
 *   <li>{@link #BIPOP} — Bi-Population restart: alternates large and small population regimes</li>
 * </ul>
 */
public enum RestartStrategy {
    /** No restart — run once and return. */
    NONE,
    /** IPOP restart — multiply lambda by incPopSize on each restart. */
    IPOP,
    /** BIPOP restart — alternate between large-population and small-population regimes. */
    BIPOP
}
