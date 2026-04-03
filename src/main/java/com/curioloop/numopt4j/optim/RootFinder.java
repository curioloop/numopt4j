package com.curioloop.numopt4j.optim;

import java.util.function.DoubleUnaryOperator;

import com.curioloop.numopt4j.optim.root.BrentqProblem;
import com.curioloop.numopt4j.optim.root.BroydenProblem;
import com.curioloop.numopt4j.optim.root.HYBRProblem;

/**
 * Root-finding facade and abstract base for solver-specific builders.
 *
 * <p>Use the static factory methods as the primary entry point:</p>
 * <pre>{@code
 * // 1-D: Brent's method
 * Optimization r = RootFinder.brentq(x -> Math.sin(x))
 *     .bracket(Bound.between(3.0, 4.0)).solve();
 *
 * // N-D: Powell hybrid (requires Jacobian, numerical by default)
 * Optimization r = RootFinder.hybr((x, f) -> { f[0] = x[0] - 1; }, 1)
 *     .initialPoint(0.0).solve();
 *
 * // N-D: Broyden (Jacobian-free)
 * Optimization r = RootFinder.broyden((x, f) -> { f[0] = x[0] - 1; }, 1)
 *     .initialPoint(0.0).solve();
 * }</pre>
 *
 * <p>Each concrete subclass corresponds to one solver and its workspace type:</p>
 * <ul>
 *   <li>{@link BrentqProblem} — 1-D Brent's method, {@code W = Void}</li>
 *   <li>{@link HYBRProblem}   — N-D Powell hybrid method, {@code W = HYBRWorkspace}</li>
 *   <li>{@link BroydenProblem}— N-D Broyden's method, {@code W = BroydenWorkspace}</li>
 * </ul>
 *
 * @param <F> equation / function type
 * @param <W> workspace type ({@code Void} for stateless solvers)
 * @param <S> self type for fluent builder chaining
 */
public abstract class RootFinder<F, W, S extends RootFinder<F, W, S>> implements Problem<W> {

    /** Problem dimension (number of variables) */
    protected int dimension = 0;

    /** Initial point (x₀) */
    protected double[] initialPoint;

    /** Equation / function to solve */
    protected F function;

    /** Cached workspace for reuse across multiple solve calls */
    protected transient W workspace;

    protected RootFinder() {}

    // ── Static factory methods (facade) ──────────────────────────────────────

    /**
     * Creates a {@link BrentqProblem} for one-dimensional root finding via Brent's method.
     *
     * @param f scalar function whose root is sought
     * @return configured {@link BrentqProblem} builder
     */
    public static BrentqProblem brentq(DoubleUnaryOperator f) {
        if (f == null) throw new IllegalArgumentException("function must not be null");
        return new BrentqProblem().function(f);
    }

    /**
     * Creates a {@link HYBRProblem} for multi-dimensional root finding
     * via the Powell hybrid method (MINPACK {@code hybrd}).
     * <p>Requires a Jacobian (numerical by default: {@link NumericalJacobian#FORWARD}).
     * Use {@link HYBRProblem#jacobian} to switch to central differences.</p>
     *
     * @param fn system of equations F(x) = 0
     * @param n  number of equations / unknowns
     * @return configured {@link HYBRProblem} builder
     */
    public static HYBRProblem hybr(Multivariate.Objective fn, int n) {
        return new HYBRProblem().equations(fn, n);
    }

    /**
     * Creates a {@link BroydenProblem} for multi-dimensional root finding
     * via Broyden's method ({@code broyden1}).
     * <p>Jacobian-free: maintains a rank-1 inverse-Jacobian approximation.
     * Suitable when the Jacobian is expensive or unavailable.</p>
     *
     * @param fn system of equations F(x) = 0
     * @param n  number of equations / unknowns
     * @return configured {@link BroydenProblem} builder
     */
    public static BroydenProblem broyden(Multivariate.Objective fn, int n) {
        return new BroydenProblem().equations(fn, n);
    }

    /**
     * Sets the initial point. Also infers {@code n} when not yet known.
     *
     * @param x0 initial point values
     * @return this builder (typed to the concrete subclass)
     */
    @SuppressWarnings("unchecked")
    public S initialPoint(double... x0) {
        if (x0 == null || x0.length == 0)
            throw new IllegalArgumentException("initialPoint must not be null or empty");
        this.initialPoint = x0;
        if (dimension == 0) dimension = x0.length;
        return (S) this;
    }

    /** Returns the problem dimension (number of variables). */
    public int dimension() { return dimension; }

    /** Returns the initial point. */
    public double[] initialPoint() { return initialPoint; }
}
