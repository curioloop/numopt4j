/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;

/**
 * Specifies the domain for an improper integral via variable substitution.
 *
 * <ul>
 *   <li>{@link #UPPER}      — ∫_{min}^{+∞} f(x) dx,  via x = min + t/(1−t)</li>
 *   <li>{@link #LOWER}      — ∫_{−∞}^{max} f(x) dx,  via x = max − t/(1−t)</li>
 *   <li>{@link #WHOLE_LINE} — ∫_{−∞}^{+∞} f(x) dx,   via x = t/(1−t²)</li>
 * </ul>
 *
 * <p>For {@link #UPPER} the {@code min} parameter of the factory is used as the finite lower bound;
 * for {@link #LOWER} the {@code max} parameter is used as the finite upper bound;
 * for {@link #WHOLE_LINE} both bounds are ignored.</p>
 */
public enum ImproperOpts {
    UPPER, LOWER, WHOLE_LINE
}
