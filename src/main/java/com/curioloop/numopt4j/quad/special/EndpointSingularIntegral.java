/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;
import java.util.Objects;

import com.curioloop.numopt4j.quad.Checks;
import com.curioloop.numopt4j.quad.Integral;
import com.curioloop.numopt4j.quad.Quadrature;
import com.curioloop.numopt4j.quad.gauss.GaussPool;

import java.util.function.DoubleUnaryOperator;

/**
 * Builder for endpoint-singular integrals of the form
 *   ∫_{a}^{b} (x−a)^α (b−x)^β · log-factor · f(x) dx,  α,β > −1.
 *
 * <p>The algorithm is selected by {@link EndpointOpts}:</p>
 * <ul>
 *   <li>{@link EndpointOpts#ALGEBRAIC}  — Gauss-Jacobi rule refinement (no log factor)</li>
 *   <li>{@link EndpointOpts#LOG_LEFT}   — tanh-sinh quadrature with ln(x−a) factor</li>
 *   <li>{@link EndpointOpts#LOG_RIGHT}  — tanh-sinh quadrature with ln(b−x) factor</li>
 *   <li>{@link EndpointOpts#LOG_BOTH}   — tanh-sinh quadrature with ln(x−a)·ln(b−x) factor</li>
 * </ul>
 *
 * <p>Minimum required setters: {@code .function()}, {@code .bounds()}, {@code .exponents()},
 * {@code .tolerances()}.</p>
 */
public class EndpointSingularIntegral implements Integral<Quadrature, GaussPool> {

    private DoubleUnaryOperator function;
    private double min = Double.NaN;
    private double max = Double.NaN;
    private double alpha = 0.0;
    private double beta = 0.0;
    private EndpointOpts opts = EndpointOpts.ALGEBRAIC;
    private double absTol = 1e-10;
    private double relTol = 1e-10;
    private int maxRefinements = Checks.DEFAULT_MAX_REFINEMENTS;
    private transient GaussPool workspace;

    EndpointSingularIntegral() {}

    /** Creates a builder pre-configured with the given log opts. */
    public EndpointSingularIntegral(EndpointOpts opts) {
        Objects.requireNonNull(opts, "opts must not be null");
        this.opts = opts;
    }

    public EndpointSingularIntegral function(DoubleUnaryOperator function) {
        this.function = function;
        return this;
    }

    public EndpointSingularIntegral bounds(double min, double max) {
        this.min = min;
        this.max = max;
        return this;
    }

    /** Sets the left and right algebraic singularity exponents α and β (both must be > −1). */
    public EndpointSingularIntegral exponents(double alpha, double beta) {
        this.alpha = alpha;
        this.beta = beta;
        return this;
    }

    public EndpointSingularIntegral opts(EndpointOpts opts) {
        Objects.requireNonNull(opts, "opts must not be null");
        this.opts = opts;
        return this;
    }

    public EndpointSingularIntegral tolerances(double absTol, double relTol) {
        if (absTol < 0.0) throw new IllegalArgumentException("absTol must be >= 0");
        if (relTol < 0.0) throw new IllegalArgumentException("relTol must be >= 0");
        this.absTol = absTol;
        this.relTol = relTol;
        return this;
    }

    /** Sets the maximum number of refinement levels (default {@value Checks#DEFAULT_MAX_REFINEMENTS}). */
    public EndpointSingularIntegral maxRefinements(int maxRefinements) {
        if (maxRefinements <= 0) throw new IllegalArgumentException("maxRefinements must be > 0");
        this.maxRefinements = maxRefinements;
        return this;
    }

    @Override
    public Quadrature integrate(GaussPool workspace) {
        Checks.validateFunction(function);
        Checks.validateFiniteInterval(min, max);
        Checks.validateEndpointExponents(alpha, beta);
        Checks.validateTolerances(absTol, relTol);
        Checks.validateMaxRefinements(maxRefinements);
        if (workspace == null) {
            if (this.workspace == null) this.workspace = new GaussPool();
            workspace = this.workspace;
        }
        GaussPool pool = workspace;

        if (opts == EndpointOpts.ALGEBRAIC) {
            return EndpointSingularCore.algebraic(function, min, max, alpha, beta, absTol, relTol, maxRefinements, pool);
        }

        DoubleUnaryOperator weighted = x -> {
            double left = x - min, right = max - x;
            double v = function.applyAsDouble(x) * Math.pow(left, alpha) * Math.pow(right, beta);
            if (opts.logLeft)  v *= Math.log(left);
            if (opts.logRight) v *= Math.log(right);
            return v;
        };
        return EndpointSingularCore.tanhSinh(weighted, min, max, absTol, relTol, maxRefinements);
    }

}
