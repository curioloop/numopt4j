/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.de;

import com.curioloop.numopt4j.quad.Integral;
import com.curioloop.numopt4j.quad.Quadrature;

import java.util.Objects;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Fluent builder for Boost-style double-exponential quadrature rules.
 *
 * <p>This builder provides the same configuration style as the other function-based
 * quadrature builders exposed by {@code Integrator}, while delegating to the low-level
 * {@link DoubleExponentialCore} kernels for the actual rule implementation.</p>
 *
 * <p>It configures one of the three transformed integrals</p>
 * <pre>
 * TANH_SINH: integral_a^b f(x) dx
 * EXP_SINH:  integral_a^(+inf) f(x) dx   or   integral_(-inf)^b f(x) dx
 * SINH_SINH: integral_(-inf)^(+inf) f(x) dx
 * </pre>
 * and reports convergence when the underlying DE sequence satisfies
 * {@code |I_n - I_(n-1)| <= tolerance * L1_n}.
 *
 * <p>Minimum required setters depend on {@link DEOpts}:</p>
 * <ul>
 *   <li>{@link DEOpts#TANH_SINH} — {@code .function()}, {@code .bounds()}</li>
 *   <li>{@link DEOpts#EXP_SINH} — {@code .function()}, {@code .bounds()} with a half-infinite domain</li>
 *   <li>{@link DEOpts#SINH_SINH} — {@code .function()} only</li>
 * </ul>
 */
public class DoubleExponentialIntegral implements Integral<Quadrature, DEPool> {

    private DEOpts opts;
    private DoubleUnaryOperator unaryFunction;
    private DoubleBinaryOperator complementAwareFunction;
    private double lower = Double.NaN;
    private double upper = Double.NaN;
    private double tolerance = Double.NaN;
    private Integer maxRefinements;
    private double minComplement = Double.NaN;
    private transient DEPool workspace;

    public DoubleExponentialIntegral() {}

    public DoubleExponentialIntegral(DEOpts opts) {
        opts(opts);
    }

    public DoubleExponentialIntegral function(DoubleUnaryOperator function) {
        this.unaryFunction = Objects.requireNonNull(function, "function must not be null");
        this.complementAwareFunction = null;
        return this;
    }

    /**
     * Supplies a complement-aware finite-interval integrand {@code g(x, c)} with
     * {@code c = min(x-a, b-x)}.
     *
     * <p>This is only meaningful for finite-interval tanh-sinh, where endpoint-singular models
     * are often more stable when expressed in terms of the boundary distance rather than the raw
     * abscissa.</p>
     */
    public DoubleExponentialIntegral function(DoubleBinaryOperator function) {
        this.complementAwareFunction = Objects.requireNonNull(function, "function must not be null");
        this.unaryFunction = null;
        return this;
    }

    public DoubleExponentialIntegral bounds(double lower, double upper) {
        this.lower = lower;
        this.upper = upper;
        return this;
    }

    public DoubleExponentialIntegral opts(DEOpts opts) {
        this.opts = Objects.requireNonNull(opts, "opts must not be null");
        return this;
    }

    public DoubleExponentialIntegral tolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    public DoubleExponentialIntegral maxRefinements(int maxRefinements) {
        this.maxRefinements = maxRefinements;
        return this;
    }

    public DoubleExponentialIntegral minComplement(double minComplement) {
        this.minComplement = minComplement;
        return this;
    }

    /**
     * Executes the configured DE rule and maps the low-level result to {@link Quadrature}.
     *
     * <p>The status is {@code CONVERGED} exactly when the DE kernel reports</p>
     * <pre>
     * |I_n - I_(n-1)| <= tolerance * L1_n.
     * </pre>
     * Pass a reusable {@link DEPool} to retain refined DE tables across repeated solves.
     */
    @Override
    public Quadrature integrate(DEPool workspace) {
        if (opts == null) {
            throw new IllegalStateException(
                    "Missing required parameter: opts. Call .opts(DEOpts) before .integrate().");
        }
        if (workspace == null) {
            if (this.workspace == null) this.workspace = new DEPool();
            workspace = this.workspace;
        }

        double effectiveTolerance = Double.isNaN(tolerance)
                ? DoubleExponentialCore.DEFAULT_TOLERANCE
                : tolerance;
        int effectiveMaxRefinements = maxRefinements != null
                ? maxRefinements
                : defaultMaxRefinements(opts);
        double effectiveMinComplement = Double.isNaN(minComplement)
                ? DoubleExponentialCore.DEFAULT_MIN_COMPLEMENT
                : minComplement;

        Quadrature result = switch (opts) {
            case TANH_SINH -> integrateTanhSinh(
                effectiveTolerance,
                effectiveMaxRefinements,
                effectiveMinComplement,
                workspace
            );
            case EXP_SINH -> {
                yield DoubleExponentialCore.expSinh(
                    requireUnaryFunction(),
                    lower,
                    upper,
                    effectiveTolerance,
                    effectiveMaxRefinements,
                    workspace
                );
            }
            case SINH_SINH -> {
                validateWholeLineBounds();
                yield DoubleExponentialCore.sinhSinh(
                    requireUnaryFunction(),
                    effectiveTolerance,
                    effectiveMaxRefinements,
                    workspace
                );
            }
        };
        return result;
    }

    private Quadrature integrateTanhSinh(
        double effectiveTolerance,
        int effectiveMaxRefinements,
        double effectiveMinComplement,
        DEPool workspace
    ) {
        if (complementAwareFunction != null) {
            validateFiniteBounds();
            return DoubleExponentialCore.tanhSinh(
                complementAwareFunction,
                lower,
                upper,
                effectiveTolerance,
                effectiveMaxRefinements,
                effectiveMinComplement,
                workspace
            );
        }
        return DoubleExponentialCore.tanhSinh(
            requireUnaryFunction(),
            lower,
            upper,
            effectiveTolerance,
            effectiveMaxRefinements,
            effectiveMinComplement,
            workspace
        );
    }

    private DoubleUnaryOperator requireUnaryFunction() {
        if (unaryFunction == null) {
            throw new IllegalStateException(
                    "Missing required parameter: function. Call .function() before .integrate().");
        }
        return unaryFunction;
    }

    private void validateFiniteBounds() {
        if (!Double.isFinite(lower) || !Double.isFinite(upper) || !(lower < upper)) {
            throw new IllegalArgumentException(
                    "Complement-aware tanh-sinh requires finite bounds with lower < upper");
        }
    }

    private void validateWholeLineBounds() {
        if (Double.isNaN(lower) && Double.isNaN(upper)) {
            return;
        }
        if (!(lower <= -Double.MAX_VALUE && upper >= Double.MAX_VALUE)) {
            throw new IllegalArgumentException(
                    "sinh-sinh targets the whole real line; omit bounds or use (-Infinity, +Infinity)");
        }
    }

    private static int defaultMaxRefinements(DEOpts opts) {
        return switch (opts) {
            case TANH_SINH -> DoubleExponentialCore.DEFAULT_TANH_SINH_REFINEMENTS;
            case EXP_SINH, SINH_SINH -> DoubleExponentialCore.DEFAULT_HALF_LINE_REFINEMENTS;
        };
    }
}