/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;

/**
 * Specifies the kernel function and domain for an oscillatory integral.
 *
 * <ul>
 *   <li>{@link #COS} — ∫_{min}^{max} f(x)·cos(ω·x) dx  (finite interval)</li>
 *   <li>{@link #SIN} — ∫_{min}^{max} f(x)·sin(ω·x) dx  (finite interval)</li>
 *   <li>{@link #COS_UPPER} — ∫_{min}^{+∞} f(x)·cos(ω·x) dx  (semi-infinite)</li>
 *   <li>{@link #SIN_UPPER} — ∫_{min}^{+∞} f(x)·sin(ω·x) dx  (semi-infinite)</li>
 * </ul>
 */
public enum OscillatoryOpts {
    COS(false, false),
    SIN(true, false),
    COS_UPPER(false, true),
    SIN_UPPER(true, true);

    final boolean sine;
    final boolean upper;

    OscillatoryOpts(boolean sine, boolean upper) {
        this.sine = sine;
        this.upper = upper;
    }
}
