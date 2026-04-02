/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss;

import com.curioloop.numopt4j.quad.Checks;
import com.curioloop.numopt4j.quad.Integral;

import java.util.function.DoubleUnaryOperator;

/**
 * Builder for fixed-point Gauss-Legendre quadrature on a finite interval [min, max].
 *
 * <p>Applies the affine map x = c + h·t (c = (min+max)/2, h = (max−min)/2, t ∈ [−1,1])
 * and evaluates the n-point Legendre rule:
 *   ∫_{min}^{max} f(x) dx ≈ h · Σᵢ wᵢ · f(c + h·tᵢ)
 * where tᵢ and wᵢ are the Legendre nodes and weights on [−1,1].</p>
 *
 * <p>Exact for polynomials of degree ≤ 2n−1.  No error estimate is produced.</p>
 *
 * <p>Minimum required setters: {@code .function()}, {@code .bounds()}, {@code .points()}.</p>
 */
public class FixedIntegral implements Integral<Double, GaussPool> {

    private DoubleUnaryOperator function;
    private double min = Double.NaN;
    private double max = Double.NaN;
    private int points;
    private transient GaussPool workspace;

    public FixedIntegral() {}

    public FixedIntegral function(DoubleUnaryOperator function) {
        this.function = function;
        return this;
    }

    public FixedIntegral bounds(double min, double max) {
        this.min = min;
        this.max = max;
        return this;
    }

    /** Sets the number of quadrature points. Exact for polynomials of degree ≤ 2n−1. */
    public FixedIntegral points(int points) {
        Checks.validatePoints(points);
        this.points = points;
        return this;
    }

    /**
     * Sets the quadrature rule. Only {@link GaussRule#legendre()} is accepted
     * for interval-mapped fixed quadrature; any other rule will throw.
     */
    public FixedIntegral rule(GaussRule rule) {
        if (rule == null) throw new IllegalArgumentException("rule must not be null");
        if (!(rule instanceof com.curioloop.numopt4j.quad.gauss.rule.LegendreRule)) {
            throw new IllegalArgumentException(
                    "fixed quadrature requires a plain-measure rule on [-1, 1]; use GaussRule.legendre()");
        }
        return this;
    }

    @Override
    public GaussPool alloc() {
        requirePoints();
        if (workspace == null) workspace = new GaussPool();
        workspace.ensure(points);
        return workspace;
    }

    @Override
    public Double integrate(GaussPool workspace) {
        Checks.validateFunction(function);
        Checks.validateFiniteInterval(min, max);
        requirePoints();
        GaussPool pool = workspace != null ? workspace.ensure(points) : alloc();
        GaussRule.legendre().generate(points, pool);

        double center = 0.5 * (min + max);
        double halfWidth = 0.5 * (max - min);
        double sum = 0.0;
        for (int i = 0; i < points; i++) {
            sum += pool.weightAt(i) * function.applyAsDouble(center + halfWidth * pool.nodeAt(i));
        }
        return halfWidth * sum;
    }

    private void requirePoints() {
        if (points <= 0) throw new IllegalStateException(
                "Missing required parameter: points. Call .points(n) before .integrate().");
    }
}
