/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.sampled;

/**
 * Integration rule for sampled-data quadrature.
 *
 * <ul>
 *   <li>{@link #TRAPEZOIDAL} — composite trapezoidal rule, O(h²) accuracy</li>
 *   <li>{@link #SIMPSON}     — composite Simpson's rule, O(h⁴) accuracy for odd sample counts</li>
 *   <li>{@link #ROMBERG}     — Romberg extrapolation, requires 2^k+1 equally spaced samples;
 *                              not supported for {@link CumulativeIntegral}</li>
 * </ul>
 */
public enum SampledRule { TRAPEZOIDAL, SIMPSON, ROMBERG }
