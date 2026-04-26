/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;

import com.curioloop.numopt4j.quad.Checks;
import com.curioloop.numopt4j.quad.Integral;
import com.curioloop.numopt4j.quad.Quadrature;
import com.curioloop.numopt4j.quad.adapt.AdaptivePool;
import com.curioloop.numopt4j.quad.adapt.AdaptiveIntegral;

import java.util.function.DoubleUnaryOperator;

/**
 * Builder for Cauchy principal value integrals.
 *
 * <p>Computes the Cauchy principal value:
 *   P.V. ∫_{a}^{b} f(x)/(x−c) dx
 * where {@code c} is the pole.</p>
 *
 * <p>When c ∈ (a,b), the integral is regularized via the Sokhotski–Plemelj identity:
 *   P.V. ∫_{a}^{b} f(x)/(x−c) dx
 *     = ∫_{a}^{b} [f(x)−f(c)]/(x−c) dx  +  f(c)·ln|(b−c)/(c−a)|
 * The first term is a regular integral (the singularity is removable) computed
 * adaptively with a breakpoint at c.  The second term is the analytic log contribution.</p>
 *
 * <p>When c ∉ [a,b], the integrand f(x)/(x−c) is smooth and is integrated directly.</p>
 *
 * <p>Minimum required setters: {@code .function()}, {@code .bounds()}, {@code .pole()},
 * {@code .tolerances()}.</p>
 */
public class PrincipalValueIntegral implements Integral<Quadrature, AdaptivePool> {

    private DoubleUnaryOperator function;
    private double min = Double.NaN;
    private double max = Double.NaN;
    private double pole = Double.NaN;
    private double absTol = 1e-10;
    private double relTol = 1e-10;
    private int maxSubdivisions = Checks.DEFAULT_MAX_SUBDIVISIONS;
    private int maxEvaluations = Checks.DEFAULT_MAX_EVALUATIONS;
    private transient AdaptivePool workspace;

    public PrincipalValueIntegral() {}

    public PrincipalValueIntegral function(DoubleUnaryOperator function) {
        this.function = function;
        return this;
    }

    public PrincipalValueIntegral bounds(double min, double max) {
        this.min = min;
        this.max = max;
        return this;
    }

    /** Sets the location of the Cauchy pole. Must be finite; must not coincide with a bound. */
    public PrincipalValueIntegral pole(double pole) {
        this.pole = pole;
        return this;
    }

    public PrincipalValueIntegral tolerances(double absTol, double relTol) {
        if (absTol < 0.0) throw new IllegalArgumentException("absTol must be >= 0");
        if (relTol < 0.0) throw new IllegalArgumentException("relTol must be >= 0");
        this.absTol = absTol;
        this.relTol = relTol;
        return this;
    }

    public PrincipalValueIntegral maxSubdivisions(int maxSubdivisions) {
        if (maxSubdivisions <= 0) throw new IllegalArgumentException("maxSubdivisions must be > 0");
        this.maxSubdivisions = maxSubdivisions;
        return this;
    }

    public PrincipalValueIntegral maxEvaluations(int maxEvaluations) {
        if (maxEvaluations <= 0) throw new IllegalArgumentException("maxEvaluations must be > 0");
        this.maxEvaluations = maxEvaluations;
        return this;
    }

    @Override
    public Quadrature integrate(AdaptivePool workspace) {
        Checks.validateFunction(function);
        Checks.validateFiniteInterval(min, max);
        Checks.validatePole(pole, min, max);
        Checks.validateTolerances(absTol, relTol);
        Checks.validateAdaptiveLimits(maxSubdivisions, maxEvaluations);
        if (workspace == null) {
            if (this.workspace == null) this.workspace = new AdaptivePool();
            workspace = this.workspace;
        }
        AdaptivePool pool = workspace.ensure(maxSubdivisions);

        if (pole < min || pole > max) {
            return AdaptiveIntegral.integrate(x -> function.applyAsDouble(x) / (x - pole),
                    min, max, null, absTol, relTol, maxSubdivisions, maxEvaluations, pool);
        }

        double valueAtPole = function.applyAsDouble(pole);
        if (!Double.isFinite(valueAtPole)) {
            return new Quadrature(Double.NaN, Double.NaN, Quadrature.Status.ABNORMAL_TERMINATION, 0, 1);
        }

        Quadrature regularized = AdaptiveIntegral.integrate(
                x -> (function.applyAsDouble(x) - valueAtPole) / (x - pole),
                min, max, new double[]{pole}, absTol, relTol, maxSubdivisions, maxEvaluations, pool);

        double logContribution = valueAtPole * (Math.log(max - pole) - Math.log(pole - min));
        if (!Double.isFinite(logContribution)) {
            return new Quadrature(Double.NaN, Double.NaN,
                Quadrature.Status.ABNORMAL_TERMINATION,
                regularized.getIterations(), regularized.getEvaluations() + 1);
        }
        return new Quadrature(regularized.getValue() + logContribution,
                regularized.getEstimatedError(), regularized.getStatus(),
                regularized.getIterations(), regularized.getEvaluations() + 1);
    }
}
