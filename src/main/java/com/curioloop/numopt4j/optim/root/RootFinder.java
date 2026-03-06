package com.curioloop.numopt4j.optim.root;

import com.curioloop.numopt4j.optim.Multivariate;
import com.curioloop.numopt4j.optim.NumericalJacobian;
import com.curioloop.numopt4j.optim.OptimizationProblem;
import com.curioloop.numopt4j.optim.OptimizationResult;
import java.util.function.BiConsumer;
import java.util.function.DoubleUnaryOperator;

/**
 * Fluent builder API for root-finding problems.
 *
 * <p>Mirrors the design of {@code scipy.optimize.root} and {@code scipy.optimize.brentq},
 * providing a unified entry point for one-dimensional and multi-dimensional solvers.</p>
 *
 * <h2>One-dimensional usage (Brentq)</h2>
 * <pre>{@code
 * OptimizationResult result = RootFinder.create()
 *     .function(x -> Math.sin(x))
 *     .bracket(3.0, 4.0)
 *     .solve();
 * }</pre>
 *
 * <h2>Multi-dimensional usage (HYBR / Broyden)</h2>
 * <pre>{@code
 * OptimizationResult result = RootFinder.create()
 *     .equations((x, f) -> { f[0] = x[0]*x[0] - 2; f[1] = x[1] - x[0]; }, 2)
 *     .initialPoint(1.0, 1.0)
 *     .solve();
 *
 * // With workspace reuse
 * RootFinder finder = RootFinder.create()
 *     .equations(fn, n)
 *     .initialPoint(x0);
 * RootWorkspace ws = finder.alloc();
 * for (double[] pt : points) {
 *     finder.initialPoint(pt).solve(ws);
 * }
 * }</pre>
 *
 * <p>Algorithm auto-selection:</p>
 * <ul>
 *   <li>{@code function + bracket} → {@link RootMethod#BRENTQ}</li>
 *   <li>{@code equations + initialPoint} → {@link RootMethod#HYBR} (default)</li>
 *   <li>Override with {@link #method(RootMethod)}</li>
 * </ul>
 *
 * @see BrentqSolver
 * @see HYBRSolver
 * @see BroydenSolver
 */
public final class RootFinder implements OptimizationProblem<OptimizationResult, RootWorkspace> {

    // ── One-dimensional fields ────────────────────────────────────────────────

    private DoubleUnaryOperator function;
    private double bracketA = Double.NaN;
    private double bracketB = Double.NaN;

    // ── Multi-dimensional fields ──────────────────────────────────────────────

    private BiConsumer<double[], double[]> equations;
    private NumericalJacobian jacobian = NumericalJacobian.FORWARD; // default for HYBR
    private int n = 0;
    private double[] initialPoint;

    // ── Common parameters ─────────────────────────────────────────────────────

    private double xtol = BrentqSolver.DEFAULT_XTOL;
    private double rtol = BrentqSolver.DEFAULT_RTOL;
    private double ftol = HYBRSolver.DEFAULT_FTOL;
    private int maxIterations = 0;   // 0 = use solver default
    private RootMethod method = null; // null = auto-select

    // ── Cached workspace ──────────────────────────────────────────────────────

    private transient RootWorkspace workspace;

    private RootFinder() {}

    /** Creates a new {@code RootFinder} builder. */
    public static RootFinder create() {
        return new RootFinder();
    }

    // ── One-dimensional fluent setters ────────────────────────────────────────

    /**
     * Sets the scalar function whose root is sought.
     * Selects {@link RootMethod#BRENTQ} automatically when combined with {@link #bracket}.
     *
     * @param f scalar function
     * @return this builder
     */
    public RootFinder function(DoubleUnaryOperator f) {
        this.function = f;
        return this;
    }

    /**
     * Sets the bracket {@code [a, b]} for one-dimensional root finding.
     * Requires {@code f(a) * f(b) <= 0}.
     *
     * @param a left endpoint
     * @param b right endpoint
     * @return this builder
     */
    public RootFinder bracket(double a, double b) {
        this.bracketA = a;
        this.bracketB = b;
        return this;
    }

    // ── Multi-dimensional fluent setters ──────────────────────────────────────

    /**
     * Sets the system of equations {@code F(x) = 0} for multi-dimensional solving.
     * Selects {@link RootMethod#HYBR} automatically when combined with {@link #initialPoint}.
     * Use {@link #jacobian(NumericalJacobian)} to override the default {@link NumericalJacobian#FORWARD}.
     *
     * @param fn system function: {@code (x, f) -> void}
     * @param n  number of equations / unknowns
     * @return this builder
     */
    public RootFinder equations(BiConsumer<double[], double[]> fn, int n) {
        if (n < 1) throw new IllegalArgumentException("n must be >= 1, got " + n);
        this.equations = fn;
        this.n = n;
        return this;
    }

    /**
     * Sets the numerical Jacobian method used by {@link RootMethod#HYBR}.
     * Has no effect for {@link RootMethod#BROYDEN}, which builds its own Jacobian approximation.
     *
     * @param jac numerical Jacobian method ({@link NumericalJacobian#FORWARD} or {@link NumericalJacobian#CENTRAL})
     * @return this builder
     */
    public RootFinder jacobian(NumericalJacobian jac) {
        this.jacobian = jac != null ? jac : NumericalJacobian.FORWARD;
        return this;
    }

    /**
     * Sets the initial point for multi-dimensional solvers.
     * Also infers {@code n} when {@link #equations} has not been called yet.
     *
     * @param x0 initial point values
     * @return this builder
     */
    public RootFinder initialPoint(double... x0) {
        if (x0 == null || x0.length == 0)
            throw new IllegalArgumentException("initialPoint must not be null or empty");
        this.initialPoint = x0;
        if (n == 0) n = x0.length;
        return this;
    }

    // ── Common fluent setters ─────────────────────────────────────────────────

    /**
     * Sets absolute and relative tolerances for one-dimensional (Brentq) solving.
     * Convergence when {@code |f(x)| <= xtol + rtol * |x|}.
     *
     * @param xtol absolute tolerance (>= 0)
     * @param rtol relative tolerance (>= 0)
     * @return this builder
     */
    public RootFinder tolerance(double xtol, double rtol) {
        if (xtol < 0) throw new IllegalArgumentException("xtol must be >= 0");
        if (rtol < 0) throw new IllegalArgumentException("rtol must be >= 0");
        this.xtol = xtol;
        this.rtol = rtol;
        return this;
    }

    /**
     * Sets the function-value tolerance for multi-dimensional solvers.
     * Convergence when {@code ||F(x)||₂ <= ftol}.
     *
     * @param ftol function tolerance (> 0)
     * @return this builder
     */
    public RootFinder functionTolerance(double ftol) {
        if (ftol <= 0) throw new IllegalArgumentException("ftol must be > 0");
        this.ftol = ftol;
        return this;
    }

    /**
     * Sets the maximum number of iterations / function evaluations.
     * When 0 (default), each solver uses its own default.
     *
     * @param k maximum iterations (> 0)
     * @return this builder
     */
    public RootFinder maxIterations(int k) {
        if (k <= 0) throw new IllegalArgumentException("maxIterations must be > 0");
        this.maxIterations = k;
        return this;
    }

    /**
     * Explicitly selects the root-finding algorithm.
     * When not set, the algorithm is chosen automatically based on configured inputs.
     *
     * @param m algorithm to use
     * @return this builder
     */
    public RootFinder method(RootMethod m) {
        this.method = m;
        return this;
    }

    // ── Workspace management ──────────────────────────────────────────────────

    /**
     * Allocates (or reuses) a {@link RootWorkspace} compatible with the current problem size.
     * Only applicable for multi-dimensional solvers.
     *
     * @return a workspace of dimension {@code n}
     * @throws IllegalStateException if {@code n} is not yet known
     */
    public RootWorkspace alloc() {
        if (n < 1) throw new IllegalStateException("Problem dimension n is unknown; call equations(fn, n) or initialPoint(x0) first");
        if (workspace == null) {
            workspace = new RootWorkspace(n);
        } else {
            workspace.ensureCapacity(n);
        }
        return workspace;
    }

    // ── Solve ─────────────────────────────────────────────────────────────────

    /**
     * Validates parameters and solves the root-finding problem.
     *
     * <p>Optionally accepts a pre-allocated {@link RootWorkspace} for reuse.
     * If none is provided, an internal workspace is allocated automatically.</p>
     *
     * @param ws optional pre-allocated workspace (ignored for Brentq)
     * @return the root-finding result
     * @throws IllegalStateException if required parameters are missing
     */
    public OptimizationResult solve(RootWorkspace... ws) {
        RootMethod chosen = resolveMethod();
        validate(chosen);

        switch (chosen) {
            case BRENTQ: {
                int maxIter = (maxIterations > 0) ? maxIterations : BrentqSolver.DEFAULT_MAXITER;
                return BrentqSolver.solve(function, bracketA, bracketB, xtol, rtol, maxIter);
            }
            case HYBR: {
                RootWorkspace workspace = resolveWorkspace(ws);
                int maxfev = (maxIterations > 0) ? maxIterations : HYBRSolver.DEFAULT_MAXFEV_FACTOR * (n + 1);
                Multivariate eval = buildMultivariate();
                return HYBRSolver.solve(eval, initialPoint, ftol, maxfev, workspace.hybr());
            }
            case BROYDEN: {
                RootWorkspace workspace = resolveWorkspace(ws);
                int maxIter = (maxIterations > 0) ? maxIterations : 100 * (n + 1);
                return BroydenSolver.solve(equations, initialPoint, ftol, maxIter, workspace.broyden());
            }
            default:
                throw new IllegalStateException("Unknown method: " + chosen);
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private RootMethod resolveMethod() {
        if (method != null) return method;
        if (function != null) return RootMethod.BRENTQ;
        return RootMethod.HYBR;
    }

    private void validate(RootMethod chosen) {
        switch (chosen) {
            case BRENTQ:
                if (function == null)
                    throw new IllegalStateException("Missing required parameter: function. Call .function(f) before .solve().");
                if (Double.isNaN(bracketA) || Double.isNaN(bracketB))
                    throw new IllegalStateException("Missing required parameter: bracket. Call .bracket(a, b) before .solve().");
                break;
            case HYBR:
            case BROYDEN:
                if (equations == null)
                    throw new IllegalStateException("Missing required parameter: equations. Call .equations(fn, n) before .solve().");
                if (initialPoint == null)
                    throw new IllegalStateException("Missing required parameter: initialPoint. Call .initialPoint(x0) before .solve().");
                break;
        }
    }

    private RootWorkspace resolveWorkspace(RootWorkspace[] ws) {
        if (ws != null && ws.length > 0 && ws[0] != null) return ws[0];
        return alloc();
    }

    /**
     * Builds a {@link Multivariate} for {@link HYBRSolver} using the configured numerical Jacobian.
     */
    private Multivariate buildMultivariate() {
        return jacobian.wrap(equations, n, n, true);
    }
}
