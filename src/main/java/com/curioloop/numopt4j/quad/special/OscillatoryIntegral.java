/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;
import java.util.Objects;

import com.curioloop.numopt4j.quad.Integrator;
import com.curioloop.numopt4j.quad.Checks;
import com.curioloop.numopt4j.quad.Integral;
import com.curioloop.numopt4j.quad.Quadrature;
import com.curioloop.numopt4j.quad.adapt.AdaptivePool;
import com.curioloop.numopt4j.quad.adapt.AdaptiveIntegral;

import java.util.function.DoubleUnaryOperator;

/**
 * Builder for oscillatory integrals ∫ f(x)·w(ω·x) dx.
 *
 * <p>The kernel {@code w} and domain are specified by {@link OscillatoryOpts}:</p>
 * <ul>
 *   <li>{@link OscillatoryOpts#COS}       — ∫_{min}^{max} f(x)·cos(ω·x) dx</li>
 *   <li>{@link OscillatoryOpts#SIN}       — ∫_{min}^{max} f(x)·sin(ω·x) dx</li>
 *   <li>{@link OscillatoryOpts#COS_UPPER} — ∫_{min}^{+∞} f(x)·cos(ω·x) dx</li>
 *   <li>{@link OscillatoryOpts#SIN_UPPER} — ∫_{min}^{+∞} f(x)·sin(ω·x) dx</li>
 * </ul>
 *
 * <p>Finite-interval variants use adaptive GK15 directly on the weighted integrand.
 * Semi-infinite variants use the Longman/QUADPACK cycle-by-cycle strategy with
 * Wynn ε-algorithm acceleration (see {@link OscillatoryCore}).</p>
 *
 * <p>When ω = 0 and the kernel is sine, the result is exactly zero.
 * When ω = 0 and the kernel is cosine, the integral reduces to a plain adaptive integral.</p>
 *
 * <p>Minimum required setters: {@code .function()}, {@code .omega()}, bound(s), {@code .tolerances()}.</p>
 */
public class OscillatoryIntegral implements Integral<Quadrature, AdaptivePool> {

    private DoubleUnaryOperator function;
    private double min = Double.NaN;
    private double max = Double.NaN;
    private double omega = Double.NaN;
    private OscillatoryOpts opts;
    private double absTol = 1e-10;
    private double relTol = 1e-10;
    private int maxCycles = Checks.DEFAULT_MAX_CYCLES;
    private int maxSubdivisions = Checks.DEFAULT_MAX_SUBDIVISIONS;
    private int maxEvaluations = Checks.DEFAULT_MAX_EVALUATIONS;
    private transient AdaptivePool workspace;

    OscillatoryIntegral() {}

    /** Creates a builder pre-configured with the given opts. */
    public OscillatoryIntegral(OscillatoryOpts opts) {
        Objects.requireNonNull(opts, "opts must not be null");
        this.opts = opts;
    }

    public OscillatoryIntegral function(DoubleUnaryOperator function) {
        this.function = function;
        return this;
    }

    /** Sets the finite lower bound for {@link OscillatoryOpts#COS_UPPER} / {@link OscillatoryOpts#SIN_UPPER}. */
    public OscillatoryIntegral lowerBound(double min) {
        this.min = min;
        return this;
    }

    /** Sets the finite upper bound for finite-interval opts. */
    public OscillatoryIntegral upperBound(double max) {
        this.max = max;
        return this;
    }

    /** Sets the angular frequency ω. Must be finite; may be zero. */
    public OscillatoryIntegral omega(double omega) {
        this.omega = omega;
        return this;
    }

    public OscillatoryIntegral opts(OscillatoryOpts opts) {
        Objects.requireNonNull(opts, "opts must not be null");
        this.opts = opts;
        return this;
    }

    public OscillatoryIntegral tolerances(double absTol, double relTol) {
        if (absTol < 0.0) throw new IllegalArgumentException("absTol must be >= 0");
        if (relTol < 0.0) throw new IllegalArgumentException("relTol must be >= 0");
        this.absTol = absTol;
        this.relTol = relTol;
        return this;
    }

    /**
     * Sets the maximum number of oscillation cycles for semi-infinite integration
     * (default {@value Checks#DEFAULT_MAX_CYCLES}).  Ignored for finite-interval opts.
     */
    public OscillatoryIntegral maxCycles(int maxCycles) {
        if (maxCycles <= 0) throw new IllegalArgumentException("maxCycles must be > 0");
        this.maxCycles = maxCycles;
        return this;
    }

    public OscillatoryIntegral maxSubdivisions(int maxSubdivisions) {
        if (maxSubdivisions <= 0) throw new IllegalArgumentException("maxSubdivisions must be > 0");
        this.maxSubdivisions = maxSubdivisions;
        return this;
    }

    public OscillatoryIntegral maxEvaluations(int maxEvaluations) {
        if (maxEvaluations <= 0) throw new IllegalArgumentException("maxEvaluations must be > 0");
        this.maxEvaluations = maxEvaluations;
        return this;
    }

    @Override
    public Quadrature integrate(AdaptivePool workspace) {
        Checks.validateFunction(function);
        if (opts == null) throw new IllegalStateException("Missing required parameter: opts. Call .opts(OscillatoryOpts) before .integrate().");
        Checks.validateFrequency(omega);
        Checks.validateTolerances(absTol, relTol);
        Checks.validateAdaptiveLimits(maxSubdivisions, maxEvaluations);
        if (opts.upper) {
            if (!Double.isFinite(min)) throw new IllegalArgumentException("min must be finite");
        } else {
            Checks.validateFiniteInterval(min, max);
        }
        if (workspace == null) {
            if (this.workspace == null) this.workspace = new AdaptivePool();
            workspace = this.workspace;
        }
        AdaptivePool pool = workspace.ensure(maxSubdivisions);
        return opts.upper ? integrateUpper(pool) : integrateFinite(pool);
    }

    private Quadrature integrateFinite(AdaptivePool pool) {
        if (omega == 0.0) {
            return opts.sine
                    ? new Quadrature(0.0, 0.0, Quadrature.Status.CONVERGED, 0, 0)
                    : AdaptiveIntegral.integrate(function, min, max, null, absTol, relTol, maxSubdivisions, maxEvaluations, pool);
        }
        DoubleUnaryOperator weighted = x -> function.applyAsDouble(x)
                * (opts.sine ? Math.sin(omega * x) : Math.cos(omega * x));
        return AdaptiveIntegral.integrate(weighted, min, max, null, absTol, relTol, maxSubdivisions, maxEvaluations, pool);
    }

    private Quadrature integrateUpper(AdaptivePool pool) {
        Checks.validateMaxCycles(maxCycles);
        if (omega == 0.0) {
            return opts.sine
                    ? new Quadrature(0.0, 0.0, Quadrature.Status.CONVERGED, 0, 0)
                    : Integrator.adaptive().function(t -> { double d = 1.0 - t; return function.applyAsDouble(min + t / d) / (d * d); }).bounds(0.0, 1.0)
                            .tolerances(absTol, relTol).maxSubdivisions(maxSubdivisions).maxEvaluations(maxEvaluations).integrate(pool);
        }
        return OscillatoryCore.integrateUpper(function, min, omega, opts.sine, absTol, relTol,
                maxCycles, maxSubdivisions, maxEvaluations, pool);
    }

}
