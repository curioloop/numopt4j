/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad;

import com.curioloop.numopt4j.quad.ode.IVPIntegral;
import com.curioloop.numopt4j.quad.ode.IVPMethod;
import com.curioloop.numopt4j.quad.ode.ODE;
import com.curioloop.numopt4j.quad.adapt.AdaptiveIntegral;
import com.curioloop.numopt4j.quad.gauss.FixedIntegral;
import com.curioloop.numopt4j.quad.gauss.WeightedIntegral;
import com.curioloop.numopt4j.quad.sampled.CumulativeIntegral;
import com.curioloop.numopt4j.quad.sampled.FilonIntegral;
import com.curioloop.numopt4j.quad.sampled.FilonOpts;
import com.curioloop.numopt4j.quad.sampled.SampledIntegral;
import com.curioloop.numopt4j.quad.sampled.SampledRule;
import com.curioloop.numopt4j.quad.special.EndpointSingularIntegral;
import com.curioloop.numopt4j.quad.special.ImproperIntegral;
import com.curioloop.numopt4j.quad.special.ImproperOpts;
import com.curioloop.numopt4j.quad.special.EndpointOpts;
import com.curioloop.numopt4j.quad.special.OscillatoryIntegral;
import com.curioloop.numopt4j.quad.special.OscillatoryOpts;
import com.curioloop.numopt4j.quad.special.PrincipalValueIntegral;

/**
 * Unified facade for one-dimensional numerical integration.
 *
 * <p>Each factory method returns a builder ({@link Integral} implementation) that
 * accepts parameters via a fluent setter chain and exposes {@code alloc()} /
 * {@code integrate()} for workspace reuse.</p>
 *
 * <p>Quick reference:</p>
 * <pre>{@code
 * // Fixed Gauss-Legendre on [a, b]
 * double v = Integrator.fixed().function(f).bounds(a, b).points(8).integrate();
 *
 * // Adaptive GK15 on [a, b]
 * Quadrature r = Integrator.adaptive().function(f).bounds(a, b).tolerances(1e-10, 1e-10).integrate();
 *
 * // Oscillatory: ∫ f(x)·cos(ω·x) dx from a to +∞
 * Quadrature r = Integrator.oscillatory(OscillatoryOpts.COS_UPPER)
 *     .function(f).lowerBound(a).omega(omega).tolerances(1e-10, 1e-10).integrate();
 *
 * // Improper: ∫ f(x) dx from a to +∞  (adaptive)
 * Quadrature r = Integrator.improper(ImproperOpts.UPPER)
 *     .function(f).lowerBound(a).tolerances(1e-10, 1e-10).integrate();
 *
 * // Sampled data
 * double total = Integrator.sampled(SampledRule.SIMPSON).samples(y, dx).integrate();
 *
 * // ODE IVP: dy/dt = -y, y(0) = 1
 * Trajectory sol = Integrator.ode(IVPMethod.RK45)
 *     .equation((t, y, dydt) -> dydt[0] = -y[0])
 *     .bounds(0.0, 5.0).initialState(1.0).integrate();
 * }</pre>
 */
public final class Integrator {

    private Integrator() {}

    // -----------------------------------------------------------------------
    // Sampled-data integration
    // -----------------------------------------------------------------------

    /**
     * Returns a builder for computing the scalar total integral of sampled data
     * using the given rule (trapezoidal, Simpson, or Romberg).
     */
    public static SampledIntegral sampled(SampledRule opts) {
        return new SampledIntegral().rule(opts);
    }

    /**
     * Returns a builder for computing the cumulative integral array of sampled data
     * using the given rule (trapezoidal or Simpson).
     */
    public static CumulativeIntegral cumulative(SampledRule opts) {
        return new CumulativeIntegral().rule(opts);
    }

    /**
     * Returns a builder for Filon quadrature of highly oscillatory integrals
     * ∫_{a}^{b} f(x)·cos(t·x) dx or ∫_{a}^{b} f(x)·sin(t·x) dx.
     *
     * <p>Filon's method handles the trigonometric factor analytically, making it
     * far more efficient than adaptive GK15 when t is large.
     * Requires {@code .function()}, {@code .bounds()}, {@code .frequency()},
     * and {@code .intervals()}.</p>
     *
     * @see FilonOpts
     */
    public static FilonIntegral filon(FilonOpts opts) {
        return new FilonIntegral(opts);
    }

    // -----------------------------------------------------------------------
    // Function-based integration — builder factories
    // -----------------------------------------------------------------------

    /**
     * Returns a blank builder for fixed-point Gauss-Legendre quadrature on a finite interval.
     * Requires {@code .function()}, {@code .bounds()}, and {@code .points()} before integrating.
     */
    public static FixedIntegral fixed() { return new FixedIntegral(); }

    /**
     * Returns a blank builder for quadrature on a rule's natural domain and weight function.
     * Requires {@code .function()}, {@code .points()}, and {@code .rule()} before integrating.
     */
    public static WeightedIntegral weighted() { return new WeightedIntegral(); }

    /**
     * Returns a blank builder for adaptive Gauss-Kronrod (GK15) quadrature on a finite interval.
     * Requires {@code .function()}, {@code .bounds()}, and {@code .tolerances()} before integrating.
     */
    public static AdaptiveIntegral adaptive() { return new AdaptiveIntegral(); }

    /**
     * Returns a builder pre-configured with the given oscillatory kernel and domain opts.
     * Requires {@code .function()}, {@code .omega()}, bound(s), and {@code .tolerances()}.
     *
     * @see OscillatoryOpts
     */
    public static OscillatoryIntegral oscillatory(OscillatoryOpts opts) {
        return new OscillatoryIntegral(opts);
    }

    /**
     * Returns a blank builder for a Cauchy principal value integral.
     * Requires {@code .function()}, {@code .bounds()}, {@code .pole()}, and {@code .tolerances()}.
     */
    public static PrincipalValueIntegral principalValue() { return new PrincipalValueIntegral(); }

    /**
     * Returns a builder pre-configured with the given endpoint opts (algebraic or log-weighted).
     * Requires {@code .function()}, {@code .bounds()}, {@code .exponents()}, and {@code .tolerances()}.
     *
     * @see EndpointOpts
     */
    public static EndpointSingularIntegral endpointSingular(EndpointOpts opts) {
        return new EndpointSingularIntegral(opts);
    }

    /**
     * Returns an adaptive GK15 builder pre-configured with the given improper domain opts.
     * Requires {@code .function()}, bound(s) per opts, and {@code .tolerances()}.
     *
     * @see ImproperOpts
     */
    public static ImproperIntegral.Adaptive improper(ImproperOpts opts) { return new ImproperIntegral.Adaptive(opts); }

    /**
     * Returns a fixed Gauss-Legendre builder pre-configured with the given improper domain opts.
     * Requires {@code .function()}, bound(s) per opts, and {@code .points()}.
     *
     * @see ImproperOpts
     */
    public static ImproperIntegral.Fixed improperFixed(ImproperOpts opts) { return new ImproperIntegral.Fixed(opts); }

    // -----------------------------------------------------------------------
    // ODE initial value problem
    // -----------------------------------------------------------------------

    /**
     * Returns a blank {@link IVPIntegral} builder pre-configured with the given solver method.
     * Requires {@code .equation()}, {@code .bounds()}, and {@code .initialState()} before integrating.
     *
     * @param method solver method (e.g. {@link IVPMethod#RK45}, {@link IVPMethod#BDF})
     * @see IVPIntegral
     */
    public static IVPIntegral ode(IVPMethod method) {
        return new IVPIntegral(method);
    }

}

