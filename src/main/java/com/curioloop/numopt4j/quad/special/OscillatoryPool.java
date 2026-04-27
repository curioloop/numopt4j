/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;

import com.curioloop.numopt4j.quad.adapt.AdaptivePool;

import java.util.function.DoubleUnaryOperator;

/**
 * Specialized workspace for semi-infinite oscillatory quadrature.
 *
 * <p>Extends {@link AdaptivePool} with its own contiguous scratch arena for the
 * oscillatory partial sums, epsilon rows, and extrapolation result used by
 * {@link OscillatoryCore#integrateUpper}. Passing this pool to
 * {@link OscillatoryIntegral#integrate(OscillatoryPool)} keeps the adaptive arena semantics
 * unchanged while also reusing the oscillatory-specific scratch.</p>
 */
public final class OscillatoryPool extends AdaptivePool {

    private static final int EXTRAPOLATION_WIDTH = 2;

    private double[] seriesArena;
    private final WeightedIntegrand weightedIntegrand = new WeightedIntegrand();

    @Override
    public OscillatoryPool ensure(int intervals) {
        super.ensure(intervals);
        return this;
    }

    OscillatoryPool ensureSeries(int maxCycles) {
        int required = maxCycles * 3 + EXTRAPOLATION_WIDTH;
        if (seriesArena == null || seriesArena.length < required) {
            seriesArena = new double[required];
        }
        return this;
    }

    double[] seriesArena() {
        return seriesArena;
    }

    WeightedIntegrand weightedIntegrand() {
        return weightedIntegrand;
    }

    static final class WeightedIntegrand implements DoubleUnaryOperator {
        private DoubleUnaryOperator function;
        private double omega;
        private boolean sine;

        WeightedIntegrand configure(DoubleUnaryOperator function, double omega, boolean sine) {
            this.function = function;
            this.omega = omega;
            this.sine = sine;
            return this;
        }

        @Override
        public double applyAsDouble(double x) {
            return function.applyAsDouble(x) * (sine ? Math.sin(omega * x) : Math.cos(omega * x));
        }
    }
}