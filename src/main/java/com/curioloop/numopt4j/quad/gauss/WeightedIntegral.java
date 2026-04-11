/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss;
import java.util.Objects;

import com.curioloop.numopt4j.quad.Checks;
import com.curioloop.numopt4j.quad.Integral;
import com.curioloop.numopt4j.quad.gauss.rule.JacobiRule;

import java.util.function.DoubleUnaryOperator;

/**
 * Builder for quadrature on a rule's natural domain and weight function.
 *
 * <p>Evaluates the n-point rule directly on its canonical domain without any affine mapping:
 *   ∫ w(x)·f(x) dx ≈ Σᵢ wᵢ · f(xᵢ)
 * where xᵢ and wᵢ are the rule's nodes and weights (which already absorb the weight function w).</p>
 *
 * <p>Supported rules: {@link GaussRule#legendre()} (∫_{−1}^{1} f dx),
 * {@link GaussRule#hermite()} (∫ e^{−x²} f dx),
 * {@link GaussRule#laguerre()} (∫_{0}^{+∞} e^{−x} f dx),
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
        Objects.requireNonNull(rule, "rule must not be null");
        this.rule = rule;
        return this;
    }

    @Override
    public Double integrate(GaussPool workspace) {
        Checks.validateFunction(function);
        requireReady();
        if (workspace == null) {
            if (this.workspace == null) this.workspace = new GaussPool();
            workspace = this.workspace;
        }
        GaussPool pool = workspace.ensure(points, rule);

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
