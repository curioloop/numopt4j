/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.special;

import com.curioloop.numopt4j.quad.Integral;
import com.curioloop.numopt4j.quad.Integrator;
import com.curioloop.numopt4j.quad.Checks;
import com.curioloop.numopt4j.quad.Quadrature;
import com.curioloop.numopt4j.quad.adapt.AdaptivePool;
import com.curioloop.numopt4j.quad.gauss.GaussPool;
import com.curioloop.numopt4j.quad.gauss.GaussRule;

import java.util.function.DoubleUnaryOperator;

/**
 * Abstract base for improper integral builders.
 *
 * <p>The type parameter {@code T} is the concrete subtype, enabling fluent setter chaining
 * without unchecked casts in subclasses (CRTP / self-referential generic pattern).</p>
 *
 * <p>Use the concrete subclasses:</p>
 * <ul>
 *   <li>{@link Fixed}    — fixed-point Gauss-Legendre, workspace {@link GaussPool}</li>
 *   <li>{@link Adaptive} — adaptive GK15, workspace {@link AdaptivePool}</li>
 * </ul>
 *
 * <p>The domain is specified by {@link ImproperOpts}:</p>
 * <ul>
 *   <li>{@link ImproperOpts#UPPER}      — ∫_{a}^{+∞} f(x) dx</li>
 *   <li>{@link ImproperOpts#LOWER}      — ∫_{−∞}^{b} f(x) dx</li>
 *   <li>{@link ImproperOpts#WHOLE_LINE} — ∫_{−∞}^{+∞} f(x) dx</li>
 * </ul>
 *
 * <p>Variable substitutions used:</p>
 * <ul>
 *   <li>UPPER:      x = a + t/(1−t),  t ∈ [0,1),  dx/dt = 1/(1−t)²</li>
 *   <li>LOWER:      x = b − t/(1−t),  t ∈ [0,1),  dx/dt = 1/(1−t)²</li>
 *   <li>WHOLE_LINE: x = t/(1−t²),     t ∈ (−1,1), dx/dt = (1+t²)/(1−t²)²</li>
 * </ul>
 * Each substitution maps the semi-infinite or infinite domain to a finite interval
 * on which standard Gauss-Legendre or adaptive GK15 quadrature is applied.
 */
public abstract class ImproperIntegral<T extends ImproperIntegral<T>> {

    protected DoubleUnaryOperator function;
    protected ImproperOpts opts;
    protected double min = Double.NaN;
    protected double max = Double.NaN;

    protected ImproperIntegral() {}

    protected ImproperIntegral(ImproperOpts opts) {
        if (opts == null) throw new IllegalArgumentException("opts must not be null");
        this.opts = opts;
    }

    // -----------------------------------------------------------------------
    // Shared setters — return T (the concrete subtype) without unchecked cast
    // -----------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    public T function(DoubleUnaryOperator function) {
        this.function = function;
        return (T) this;
    }

    @SuppressWarnings("unchecked")
    public T opts(ImproperOpts opts) {
        if (opts == null) throw new IllegalArgumentException("opts must not be null");
        this.opts = opts;
        return (T) this;
    }

    /** Sets the finite lower bound for {@link ImproperOpts#UPPER}. */
    @SuppressWarnings("unchecked")
    public T lowerBound(double min) {
        this.min = min;
        return (T) this;
    }

    /** Sets the finite upper bound for {@link ImproperOpts#LOWER}. */
    @SuppressWarnings("unchecked")
    public T upperBound(double max) {
        this.max = max;
        return (T) this;
    }

    // -----------------------------------------------------------------------
    // Shared validation helpers
    // -----------------------------------------------------------------------

    protected void validateCommon() {
        Checks.validateFunction(function);
        if (opts == null) throw new IllegalStateException(
                "Missing required parameter: opts. Call .opts(ImproperOpts) before .integrate().");
    }

    protected void validateBounds() {
        switch (opts) {
            case UPPER:
                if (!Double.isFinite(min)) throw new IllegalArgumentException("min must be finite for UPPER");
                break;
            case LOWER:
                if (!Double.isFinite(max)) throw new IllegalArgumentException("max must be finite for LOWER");
                break;
            default: // WHOLE_LINE — no bounds needed
        }
    }

    // -----------------------------------------------------------------------
    // Fixed-point subclass
    // -----------------------------------------------------------------------

    /**
     * Fixed-point Gauss-Legendre approximation of an improper integral.
     * Fast but provides no error estimate.
     *
     * <p>The Legendre nodes tᵢ ∈ [−1,1] are mapped to [0,1] via s = (1+tᵢ)/2,
     * then the substitution is applied:</p>
     * <ul>
     *   <li>UPPER: x = a + s/(1−s),  dx/ds = 1/(1−s)²
     *     ∫_{a}^{+∞} f(x) dx = ∫_{0}^{1} f(a+s/(1−s))/(1−s)² ds
     *     ≈ (1/2) · Σᵢ wᵢ · f(a + sᵢ/(1−sᵢ)) / (1−sᵢ)²</li>
     *   <li>LOWER: x = b − s/(1−s),  dx/ds = 1/(1−s)²
     *     ∫_{−∞}^{b} f(x) dx = ∫_{0}^{1} f(b−s/(1−s))/(1−s)² ds
     *     ≈ (1/2) · Σᵢ wᵢ · f(b − sᵢ/(1−sᵢ)) / (1−sᵢ)²</li>
     *   <li>WHOLE_LINE: x = t/(1−t²),  dx/dt = (1+t²)/(1−t²)²
     *     ∫_{−∞}^{+∞} f(x) dx = ∫_{−1}^{1} f(t/(1−t²)) · (1+t²)/(1−t²)² dt
     *     ≈ Σᵢ wᵢ · f(tᵢ/(1−tᵢ²)) · (1+tᵢ²)/(1−tᵢ²)²</li>
     * </ul>
     * The factor 1/2 in UPPER/LOWER comes from the Legendre weight rescaling
     * from [−1,1] to [0,1]: w_{[0,1]} = w_{[−1,1]} / 2.
     */
    public static final class Fixed extends ImproperIntegral<Fixed>
            implements Integral<Quadrature, GaussPool> {

        private int points;
        private transient GaussPool workspace;

        public Fixed() {}

        public Fixed(ImproperOpts opts) { super(opts); }

        public Fixed points(int points) {
            Checks.validatePoints(points);
            this.points = points;
            return this;
        }

        @Override
        public GaussPool alloc() {
            if (workspace == null) workspace = new GaussPool();
            workspace.ensure(points);
            return workspace;
        }

        @Override
        public Quadrature integrate(GaussPool workspace) {
            validateCommon();
            validateBounds();
            if (points <= 0) throw new IllegalStateException(
                    "Missing required parameter: points. Call .points(n) before .integrate().");
            GaussPool pool = workspace != null ? workspace.ensure(points) : alloc();
            GaussRule.Legendre.generate(points, pool);
            double sum = 0.0;
            switch (opts) {
                case UPPER:
                    for (int i = 0; i < points; i++) {
                        double t = 0.5 + 0.5 * pool.nodeAt(i), d = 1.0 - t;
                        sum += pool.weightAt(i) * function.applyAsDouble(min + t / d) / (d * d);
                    }
                    return result(0.5 * sum);
                case LOWER:
                    for (int i = 0; i < points; i++) {
                        double t = 0.5 + 0.5 * pool.nodeAt(i), d = 1.0 - t;
                        sum += pool.weightAt(i) * function.applyAsDouble(max - t / d) / (d * d);
                    }
                    return result(0.5 * sum);
                default: // WHOLE_LINE
                    for (int i = 0; i < points; i++) {
                        double t = pool.nodeAt(i), d = 1.0 - t * t;
                        sum += pool.weightAt(i) * function.applyAsDouble(t / d) * (1.0 + t * t) / (d * d);
                    }
                    return result(sum);
            }
        }

        private Quadrature result(double value) {
            return new Quadrature(value, 0.0, Quadrature.Status.CONVERGED, 1, points);
        }
    }

    // -----------------------------------------------------------------------
    // Adaptive subclass
    // -----------------------------------------------------------------------

    /**
     * Adaptive GK15 approximation of an improper integral with error control.
     *
     * <p>Applies the same variable substitutions as {@link Fixed}, then integrates
     * the transformed function on the finite interval using adaptive GK15:</p>
     * <ul>
     *   <li>UPPER:      ∫_{0}^{1} f(a+t/(1−t))/(1−t)² dt  via adaptive on [0,1]</li>
     *   <li>LOWER:      ∫_{0}^{1} f(b−t/(1−t))/(1−t)² dt  via adaptive on [0,1]</li>
     *   <li>WHOLE_LINE: ∫_{−1}^{1} f(t/(1−t²))·(1+t²)/(1−t²)² dt  via adaptive on [−1,1]</li>
     * </ul>
     */
    public static final class Adaptive extends ImproperIntegral<Adaptive>
            implements Integral<Quadrature, AdaptivePool> {

        private double absTol = 1e-10;
        private double relTol = 1e-10;
        private int maxSubdivisions = Checks.DEFAULT_MAX_SUBDIVISIONS;
        private int maxEvaluations  = Checks.DEFAULT_MAX_EVALUATIONS;
        private transient AdaptivePool workspace;

        public Adaptive() {}

        public Adaptive(ImproperOpts opts) { super(opts); }

        public Adaptive tolerances(double absTol, double relTol) {
            if (absTol < 0.0) throw new IllegalArgumentException("absTol must be >= 0");
            if (relTol < 0.0) throw new IllegalArgumentException("relTol must be >= 0");
            this.absTol = absTol;
            this.relTol = relTol;
            return this;
        }

        public Adaptive maxSubdivisions(int maxSubdivisions) {
            if (maxSubdivisions <= 0) throw new IllegalArgumentException("maxSubdivisions must be > 0");
            this.maxSubdivisions = maxSubdivisions;
            return this;
        }

        public Adaptive maxEvaluations(int maxEvaluations) {
            if (maxEvaluations <= 0) throw new IllegalArgumentException("maxEvaluations must be > 0");
            this.maxEvaluations = maxEvaluations;
            return this;
        }

        @Override
        public AdaptivePool alloc() {
            if (workspace == null) workspace = new AdaptivePool();
            workspace.ensure(maxSubdivisions);
            return workspace;
        }

        @Override
        public Quadrature integrate(AdaptivePool workspace) {
            validateCommon();
            validateBounds();
            Checks.validateTolerances(absTol, relTol);
            Checks.validateAdaptiveLimits(maxSubdivisions, maxEvaluations);
            AdaptivePool pool = workspace != null ? workspace.ensure(maxSubdivisions) : alloc();
            switch (opts) {
                case UPPER:
                    return Integrator.adaptive()
                            .function(t -> { double d = 1.0 - t; return function.applyAsDouble(min + t / d) / (d * d); })
                            .bounds(0.0, 1.0).tolerances(absTol, relTol)
                            .maxSubdivisions(maxSubdivisions).maxEvaluations(maxEvaluations)
                            .integrate(pool);
                case LOWER:
                    return Integrator.adaptive()
                            .function(t -> { double d = 1.0 - t; return function.applyAsDouble(max - t / d) / (d * d); })
                            .bounds(0.0, 1.0).tolerances(absTol, relTol)
                            .maxSubdivisions(maxSubdivisions).maxEvaluations(maxEvaluations)
                            .integrate(pool);
                default: // WHOLE_LINE
                    return Integrator.adaptive()
                            .function(t -> { double d = 1.0 - t * t; return function.applyAsDouble(t / d) * (1.0 + t * t) / (d * d); })
                            .bounds(-1.0, 1.0).tolerances(absTol, relTol)
                            .maxSubdivisions(maxSubdivisions).maxEvaluations(maxEvaluations)
                            .integrate(pool);
            }
        }
    }
}
