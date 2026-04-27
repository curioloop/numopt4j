/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.de;

/**
 * Reusable workspace for double-exponential quadrature tables.
 *
 * <p>Each rule integrates against two logical table segments:</p>
 * <ul>
 *   <li>base — the shared baked Boost-compatible rows shipped in
 *       {@link DoubleExponentialsTables}</li>
 *   <li>refine — the cached extension block for rows generated beyond the baked depth</li>
 * </ul>
 *
 * <p>The pool stores one mutable refine segment at a time together with the currently selected DE
 * rule. The baked base segment stays shared on {@link DEOpts}. The refine cache can point either
 * to a shared immutable default block or to a workspace-owned extension built for deeper custom
 * requests. Reusing the
 * same pool with a different DE rule replaces the previous refine cache.</p>
 *
 * <p>Pass a single {@code DEPool} to repeated {@link DoubleExponentialIntegral#integrate(DEPool)}
 * or low-level {@link DoubleExponentialCore} calls to avoid rebuilding the refined rows on each
 * integration.</p>
 */
public final class DEPool {

    DEOpts opts;
    DETable refine;
    int refineRows;

    public DEPool() {
    }

    DETable select(DEOpts opts) {
        if (opts == null) {
            throw new IllegalArgumentException("DE rule cannot be null.");
        }
        if (this.opts != opts) {
            this.opts = opts;
            this.refine = null;
            this.refineRows = 0;
        }
        return opts.baseTable();
    }
}