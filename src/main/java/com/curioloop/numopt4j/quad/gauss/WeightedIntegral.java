/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss;

import com.curioloop.numopt4j.quad.Checks;
import com.curioloop.numopt4j.quad.Integral;

import java.util.function.DoubleUnaryOperator;

/**
 * Builder for quadrature on a rule's natural domain and weight function.
 *
 * <p>Evaluates the n-point rule directly on its canonical domain without any affine mapping:
 *   ∫ w(x)·f(x) dx ≈ Σᵢ wᵢ · f(xᵢ)
 * where xᵢ and wᵢ are the rule's nodes and weights (which already absorb the weight function w).</p>
 *
 * <p>Supported rules: {@link GaussRule#Legendre} (∫_{−1}^{1} f dx),
 * {@link GaussRule#Hermite} (∫ e^{−x²} f dx),
 * {@link GaussRule#Laguerre} (∫_{0}^{+∞} e^{−x} f dx),
 * and any {@link JacobiRule} (∫_{−1}^{1} (1−x)^α (1+x)^β f dx).</p>
 *
 * <p>Minimum required setters: {@code .function()}, {@code .points()}, {@code .rule()}.</p>
 */
public class WeightedIntegral implements Integral<Double, GaussPool> {

    private DoubleUnaryOperator function;
    private int points;
    private GaussRule rule;
    private transient GaussPool workspace;

    public WeightedIntegral() {}

    public WeightedIntegral function(DoubleUnaryOperator function) {
        this.function = function;
        return this;
    }

    public WeightedIntegral points(int points) {
        Checks.validatePoints(points);
        this.points = points;
        return this;
    }

    public WeightedIntegral rule(GaussRule rule) {
        if (rule == null) throw new IllegalArgumentException("rule must not be null");
        this.rule = rule;
        return this;
    }

    @Override
    public GaussPool alloc() {
        requireReady();
        if (workspace == null) workspace = new GaussPool();
        workspace.ensure(points);
        return workspace;
    }

    @Override
    public Double integrate() {
        return integrate((GaussPool) null);
    }

    @Override
    public Double integrate(GaussPool workspace) {
        Checks.validateFunction(function);
        requireReady();
        GaussPool pool = workspace != null ? workspace.ensure(points) : alloc();
        rule.generate(points, pool);

        double sum = 0.0;
        for (int i = 0; i < points; i++) {
            sum += pool.weightAt(i) * function.applyAsDouble(pool.nodeAt(i));
        }
        return sum;
    }

    private void requireReady() {
        if (points <= 0) throw new IllegalStateException(
                "Missing required parameter: points. Call .points(n) before .integrate().");
        if (rule == null) throw new IllegalStateException(
                "Missing required parameter: rule. Call .rule(rule) before .integrate().");
    }
}
