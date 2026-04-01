/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.sampled;

/**
 * Kernel type for Filon quadrature.
 *
 * <ul>
 *   <li>{@link #COSINE} — ∫_{a}^{b} f(x)·cos(t·x) dx</li>
 *   <li>{@link #SINE}   — ∫_{a}^{b} f(x)·sin(t·x) dx</li>
 * </ul>
 */
public enum FilonOpts { COSINE, SINE }
