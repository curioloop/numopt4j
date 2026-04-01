package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.Optimization;
import com.curioloop.numopt4j.optim.RootFinder;

import java.util.function.DoubleUnaryOperator;

/**
 * One-dimensional root finder using Brent's method.
 *
 * <p>No workspace is required; {@link #alloc()} returns {@code null}.</p>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * Optimization r = RootFinder.brentq(Math::sin)
 *     .bracket(Bound.between(3.0, 4.0))
 *     .solve();
 * }</pre>
 */
public final class BrentqProblem extends RootFinder<DoubleUnaryOperator, Void, BrentqProblem> {

    private Bound bracket;
    private double xtol = BrentqSolver.DEFAULT_XTOL;
    private double rtol = BrentqSolver.DEFAULT_RTOL;
    private int maxIterations = BrentqSolver.DEFAULT_MAXITER;

    public BrentqProblem() {}

    /** Sets the scalar function whose root is sought. */
    public BrentqProblem function(DoubleUnaryOperator f) {
        if (f == null) throw new IllegalArgumentException("function must not be null");
        this.function = f;
        return this;
    }

    /**
     * Sets the search bracket via a {@link Bound} object.
     * Requires {@code bound.hasBoth()} and {@code f(lower) * f(upper) <= 0}.
     */
    public BrentqProblem bracket(Bound bound) {
        if (bound == null || !bound.hasBoth())
            throw new IllegalArgumentException("bracket must be a bounded interval with both lower and upper limits");
        this.bracket = bound;
        return this;
    }

    /** Returns the search bracket, or {@code null} if not set. */
    public Bound bracket() { return bracket; }

    /** Sets absolute tolerance. */
    public BrentqProblem absoluteTolerance(double xtol) {
        if (xtol < 0) throw new IllegalArgumentException("xtol must be >= 0");
        this.xtol = xtol;
        return this;
    }

    /** Sets relative tolerance. */
    public BrentqProblem relativeTolerance(double rtol) {
        if (rtol < 0) throw new IllegalArgumentException("rtol must be >= 0");
        this.rtol = rtol;
        return this;
    }

    public double absoluteTolerance() { return xtol; }
    public double relativeTolerance() { return rtol; }

    /** Sets the maximum number of iterations. */
    public BrentqProblem maxIterations(int k) {
        if (k <= 0) throw new IllegalArgumentException("maxIterations must be > 0");
        this.maxIterations = k;
        return this;
    }

    public int maxIterations() { return maxIterations; }

    /** No-op — Brentq requires no workspace. Returns {@code null}. */
    @Override
    public Void alloc() { return null; }

    @Override
    public Optimization solve(Void ws) {
        if (function == null)
            throw new IllegalStateException(
                "Missing required parameter: function. Call .function(f) before .solve().");
        if (bracket == null)
            throw new IllegalStateException(
                "Missing required parameter: bracket. Call .bracket(a, b) before .solve().");
        return BrentqSolver.solve(function, bracket.getLower(), bracket.getUpper(), xtol, rtol, maxIterations);
    }
}
