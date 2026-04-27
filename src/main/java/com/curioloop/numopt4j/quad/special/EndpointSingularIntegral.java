/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;
import java.util.Objects;

import com.curioloop.numopt4j.quad.Checks;
import com.curioloop.numopt4j.quad.Integral;
import com.curioloop.numopt4j.quad.Quadrature;
import com.curioloop.numopt4j.quad.de.DEPool;
import com.curioloop.numopt4j.quad.gauss.GaussPool;
import com.curioloop.numopt4j.quad.gauss.GaussRule;

import java.util.function.DoubleUnaryOperator;

/**
 * Builder for endpoint-singular integrals of the form
 *   ∫_{a}^{b} (x−a)^α (b−x)^β · log-factor · f(x) dx,  α,β > −1.
 *
 * <p>The algorithm is selected by {@link EndpointOpts}:</p>
 * <ul>
 *   <li>{@link EndpointOpts#ALGEBRAIC}  — Gauss-Jacobi rule refinement (no log factor)</li>
 *   <li>{@link EndpointOpts#LOG_LEFT}   — double-exponential quadrature with ln(x−a) factor</li>
 *   <li>{@link EndpointOpts#LOG_RIGHT}  — double-exponential quadrature with ln(b−x) factor</li>
 *   <li>{@link EndpointOpts#LOG_BOTH}   — double-exponential quadrature with ln(x−a)·ln(b−x) factor</li>
 * </ul>
 *
 * <p>Minimum required setters: {@code .function()}, {@code .bounds()}, {@code .exponents()},
 * {@code .tolerances()}.</p>
 *
 * <p>Workspace semantics follow the selected algorithm: algebraic solves reuse a
 * {@link GaussPool} through {@link #integrate(GaussPool)}, while logarithmic solves reuse a
 * {@link DEPool} through {@link #integrateLogarithmic(DEPool)}. Calling {@link #integrate()} on a
 * logarithmic configuration keeps an internal DE workspace for repeated solves on the same builder.</p>
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
    private transient GaussRule algebraicRule;
    private transient GaussPool algebraicWorkspace;
    private transient DEPool logarithmicWorkspace;

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
        this.algebraicRule = null;
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
        validateState();
        if (opts != EndpointOpts.ALGEBRAIC) {
            return integrateLogarithmicValidated(null);
        }
        if (workspace == null) {
            if (this.algebraicWorkspace == null) this.algebraicWorkspace = new GaussPool();
            workspace = this.algebraicWorkspace;
        }
        if (algebraicRule == null) {
            algebraicRule = GaussRule.jacobi(beta, alpha);
        }
        return EndpointSingularCore.algebraic(function, min, max, alpha, beta,
                absTol, relTol, maxRefinements, algebraicRule, workspace);
    }

    /**
     * Executes a logarithmic endpoint-singular solve with an explicit DE workspace.
     *
     * <p>This applies only to {@link EndpointOpts#LOG_LEFT}, {@link EndpointOpts#LOG_RIGHT}, and
     * {@link EndpointOpts#LOG_BOTH}. Reuse a single {@link DEPool} across repeated solves to retain
     * the DE refine table across calls.</p>
     */
    public Quadrature integrateLogarithmic(DEPool workspace) {
        validateState();
        return integrateLogarithmicValidated(workspace);
    }

    private Quadrature integrateLogarithmicValidated(DEPool workspace) {
        if (opts == EndpointOpts.ALGEBRAIC) {
            throw new IllegalStateException(
                "integrateLogarithmic(DEPool) requires LOG_LEFT, LOG_RIGHT, or LOG_BOTH opts"
            );
        }
        if (workspace == null) {
            if (this.logarithmicWorkspace == null) this.logarithmicWorkspace = new DEPool();
            workspace = this.logarithmicWorkspace;
        }
        return EndpointSingularCore.logarithmic(
            function,
            min,
            max,
            alpha,
            beta,
            opts,
            absTol,
            relTol,
            maxRefinements,
            workspace
        );
    }

    private void validateState() {
        Checks.validateFunction(function);
        Checks.validateFiniteInterval(min, max);
        Checks.validateEndpointExponents(alpha, beta);
        Checks.validateTolerances(absTol, relTol);
        Checks.validateMaxRefinements(maxRefinements);
    }

}
