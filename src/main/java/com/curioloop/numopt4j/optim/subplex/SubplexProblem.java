/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.subplex;
import java.util.Objects;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.Optimization;
import com.curioloop.numopt4j.optim.Univariate;

/**
 * Fluent API for the Subplex algorithm (Rowan 1990).
 *
 * <p>Subplex decomposes high-dimensional derivative-free optimization into
 * low-dimensional Nelder-Mead subproblems, making it practical for N &gt; 10
 * where plain Nelder-Mead degrades.</p>
 *
 * <h2>Basic Usage</h2>
 * <pre>{@code
 * Optimization result = Minimizer.subplex()
 *     .objective((x, n) -> { double s = 0; for (double v : x) s += v*v; return s; })
 *     .initialPoint(new double[20])
 *     .solve();
 * }</pre>
 *
 * <h2>With Step Sizes</h2>
 * <pre>{@code
 * // Specify per-dimension step sizes for better initial exploration
 * Optimization result = Minimizer.subplex()
 *     .objective(fn)
 *     .initialPoint(x0)
 *     .initialStep(0.1, 10.0, 0.01)  // different scales per dimension
 *     .functionTolerance(1e-8)
 *     .solve();
 * }</pre>
 *
 * @see Minimizer#subplex()
 */
public final class SubplexProblem
        extends Minimizer<Univariate.Objective, SubplexWorkspace, SubplexProblem> {

    private double parameterTolerance = 1e-4;
    private double functionTolerance = 1e-4;
    private int maxEvaluations = 0; // 0 = auto: N * 200
    private double[] initialStep;

    public SubplexProblem() {}

    private void validate() {
        requireObjective();
        requireInitialPoint();
        if (initialStep != null && initialStep.length != dimension) {
            throw new IllegalArgumentException(
                    "initialStep must have " + dimension + " elements, got " + initialStep.length);
        }
    }

    @Override
    public Optimization solve(SubplexWorkspace workspace) {
        validate();
        SubplexWorkspace ws = resolveWorkspace(workspace, SubplexWorkspace::new);
        ws.ensure(dimension);

        double[] x = initialPoint.clone();
        int maxEval = (maxEvaluations > 0) ? maxEvaluations : dimension * 200;

        // Build xstep: user-supplied or default (NLopt-style heuristic)
        double[] xstep;
        if (initialStep != null) {
            xstep = initialStep.clone();
        } else {
            xstep = defaultStep(x, bounds, dimension);
        }

        Optimization r = SubplexCore.optimize(
                dimension, x, objective, bounds, xstep,
                maxEval, parameterTolerance, functionTolerance, ws);

        return new Optimization(Double.NaN, x, r.getCost(), r.getStatus(),
                r.getIterations(), r.getEvaluations());
    }

    // ── Getters ──────────────────────────────────────────────────────────────

    /**
     * Creates a new {@link SubplexWorkspace} for use with {@link #solve(SubplexWorkspace)}.
     * Memory is allocated lazily on the first {@code solve()} call.
     */
    public static SubplexWorkspace workspace() {
        return new SubplexWorkspace();
    }

    public double parameterTolerance() { return parameterTolerance; }
    public double functionTolerance() { return functionTolerance; }
    public int maxEvaluations() { return maxEvaluations; }
    public double[] initialStep() { return initialStep; }

    // ── Fluent setters ────────────────────────────────────────────────────────

    public SubplexProblem objective(Univariate.Objective f) {
        Objects.requireNonNull(f, "objective function must not be null");
        this.objective = f;
        return this;
    }

    public SubplexProblem parameterTolerance(double value) {
        if (value <= 0 || Double.isNaN(value))
            throw new IllegalArgumentException("parameterTolerance must be positive, got " + value);
        this.parameterTolerance = value;
        return this;
    }

    public SubplexProblem functionTolerance(double value) {
        if (value <= 0 || Double.isNaN(value))
            throw new IllegalArgumentException("functionTolerance must be positive, got " + value);
        this.functionTolerance = value;
        return this;
    }

    public SubplexProblem maxEvaluations(int value) {
        if (value <= 0) throw new IllegalArgumentException("maxEvaluations must be positive, got " + value);
        this.maxEvaluations = value;
        return this;
    }

    /**
     * Sets per-dimension initial step sizes.
     *
     * <p>Controls the scale of initial exploration in each dimension.
     * Important when variables have very different scales.
     * Default: NLopt-style heuristic based on bounds and initial point.</p>
     *
     * @param step step sizes (one per dimension)
     * @return this problem instance
     */
    public SubplexProblem initialStep(double... step) {
        this.initialStep = step;
        return this;
    }

    /**
     * Computes default initial step sizes using NLopt's heuristic:
     * <ol>
     *   <li>If bounded: min of (ub-lb)*0.25, (ub-x)*0.75, (x-lb)*0.75</li>
     *   <li>If half-bounded: distance to bound * 1.1</li>
     *   <li>Fallback: |x| or 1.0 if x=0</li>
     * </ol>
     */
    private static double[] defaultStep(double[] x, Bound[] bounds, int n) {
        double[] step = new double[n];
        for (int i = 0; i < n; i++) {
            double s = Double.MAX_VALUE;
            Bound b = Bound.of(bounds, i, Bound.unbounded());
            double lb = b.getLower(), ub = b.getUpper();
            boolean hasLb = b.hasLower(), hasUb = b.hasUpper();

            // Bounded: use fraction of range
            if (hasLb && hasUb && ub > lb && (ub - lb) * 0.25 < s)
                s = (ub - lb) * 0.25;
            if (hasUb && ub > x[i] && ub - x[i] < s)
                s = (ub - x[i]) * 0.75;
            if (hasLb && x[i] > lb && x[i] - lb < s)
                s = (x[i] - lb) * 0.75;

            // Half-bounded fallback
            if (s >= Double.MAX_VALUE) {
                if (hasUb && Math.abs(ub - x[i]) < s) s = (ub - x[i]) * 1.1;
                if (hasLb && Math.abs(x[i] - lb) < s) s = (x[i] - lb) * 1.1;
            }

            // Unbounded fallback
            if (s >= Double.MAX_VALUE || s == 0.0) s = Math.abs(x[i]);
            if (s == 0.0) s = 1.0;

            step[i] = s;
        }
        return step;
    }
}
