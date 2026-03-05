/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

/**
 * Numerical constants for the Trust Region Reflective (TRF) optimization algorithm.
 */
final class TRFConstants {

    private TRFConstants() {}

    /** Machine epsilon */
    static final double EPSMCH = Math.ulp(1.0);

    static final double P1    = 0.1;
    static final double P5    = 0.5;
    static final double P25   = 0.25;
    static final double P75   = 0.75;
    static final double P0001 = 1e-4;

    static final double DEFAULT_FTOL   = 1e-8;
    static final double DEFAULT_XTOL   = 1e-8;
    static final double DEFAULT_GTOL   = 1e-8;
    static final double DEFAULT_FACTOR = 100.0;
}
