/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.ode;

import com.curioloop.numopt4j.quad.Integral;
import com.curioloop.numopt4j.quad.Trajectory;
import com.curioloop.numopt4j.quad.ode.implicit.BDFCore;
import com.curioloop.numopt4j.quad.ode.implicit.BDFPool;
import com.curioloop.numopt4j.quad.ode.implicit.RadauCore;
import com.curioloop.numopt4j.quad.ode.implicit.RadauPool;
import com.curioloop.numopt4j.quad.ode.rk.RungeKuttaCore;
import com.curioloop.numopt4j.quad.ode.rk.RungeKuttaPool;

import java.util.Objects;

/**
 * Fluent facade for solving ODE initial value problems, corresponding to scipy {@code solve_ivp}.
 *
 * <p>Solves the IVP:
 * <pre>
 *   dy/dt = f(t, y),   y(t₀) = y₀,   t ∈ [t₀, tf]
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * Trajectory sol = new ODEIntegral(IVPMethod.RK45)
 *     .equation((t, y, dydt) -> { dydt[0] = -y[0]; })
 *     .bounds(0.0, 5.0)
 *     .initialState(1.0)
 *     .tolerances(1e-6, 1e-9)
 *     .integrate();
 *
 * // Access solution
 * double[] t = sol.timeSeries.t;
 * double[] y = sol.timeSeries.y;   // column-major: y[i*m + j] = equation i at time j
 * }</pre>
 *
 * <h2>Method Selection</h2>
 * <ul>
 *   <li>{@link IVPMethod#RK45} (default) — good general-purpose choice for non-stiff problems.</li>
 *   <li>{@link IVPMethod#RK23} — lower-order, faster for loose tolerances.</li>
 *   <li>{@link IVPMethod#DOP853} — high-order, for smooth problems with tight tolerances.</li>
 *   <li>{@link IVPMethod#BDF} / {@link IVPMethod#Radau} — implicit methods for stiff problems.</li>
 * </ul>
 *
 * @see ODE
 * @see Trajectory
 * @see ODEEvent
 */
public class IVPIntegral implements Integral<Trajectory, IVPPool> {

    // -----------------------------------------------------------------------
    // Required parameters
    // -----------------------------------------------------------------------

    private final IVPMethod method;
    private ODE          equation;      // full ODE with optional analytic Jacobian
    private ODE.Equation rhs;  // RHS-only (no Jacobian); set when user passes ODE.Equation
    private double t0 = Double.NaN, tf = Double.NaN;
    private double[] y0;

    // -----------------------------------------------------------------------
    // Optional parameters (with defaults)
    // -----------------------------------------------------------------------

    private double     rtol     = 1e-3;
    private double[]   atol     = {1e-6};
    private double     maxStep  = Double.MAX_VALUE;
    private double     firstStep = Double.NaN;
    private double[]   evalAt;
    private boolean    denseOutput = false;
    private ODEEvent[] eventDetectors;

    public IVPIntegral(IVPMethod method) {
        this.method = Objects.requireNonNull(method);
    }

    // -----------------------------------------------------------------------
    // Builder setters
    // -----------------------------------------------------------------------

    /**
     * Sets the ODE system (RHS + optional analytic Jacobian).
     * Use this overload for stiff problems where an analytic Jacobian is available.
     */
    public IVPIntegral equation(ODE f) { this.equation = f; this.rhs = null; return this; }

    /**
     * Sets the ODE right-hand side as a lambda-friendly {@link ODE.Equation}.
     * The Jacobian will be approximated numerically when needed.
     */
    public IVPIntegral equation(ODE.Equation f) {
        this.rhs = f;
        this.equation = null;
        return this;
    }

    /**
     * Sets the integration interval {@code [t₀, tf]}.
     * Use {@code tf < t0} for backward integration.
     */
    public IVPIntegral bounds(double t0, double tf) { this.t0 = t0; this.tf = tf; return this; }

    /**
     * Sets the initial state {@code y(t₀) = y₀}.
     * The array is copied; the original is not modified.
     */
    public IVPIntegral initialState(double... y0) { this.y0 = y0.clone(); return this; }

    /**
     * Sets relative and absolute tolerances.
     * {@code atol} may be a scalar (applied to all components) or a per-component vector of length n.
     * Default: {@code rtol = 1e-3}, {@code atol = 1e-6}.
     */
    public IVPIntegral tolerances(double rtol, double... atol) {
        if (rtol <= 0) throw new IllegalArgumentException("rtol must be positive");
        for (double a : atol) if (a <= 0) throw new IllegalArgumentException("atol must be positive");
        this.rtol = rtol;
        this.atol = atol.clone();
        return this;
    }

    /**
     * Sets the maximum allowed step size.
     * Default: {@code Double.MAX_VALUE} (no limit).
     */
    public IVPIntegral maxStep(double h) { this.maxStep = h; return this; }

    /**
     * Sets the initial step size.
     * Default: {@code NaN} (auto-estimated via Hairer's formula).
     */
    public IVPIntegral firstStep(double h) { this.firstStep = h; return this; }

    /**
     * Sets specific time points at which the solution is evaluated.
     * Must be monotonically increasing (forward) or decreasing (backward),
     * and lie within {@code [t₀, tf]}.
     * When set, the output {@link Trajectory.TimeSeries} contains exactly these points.
     */
    public IVPIntegral evalAt(double[] ts) { this.evalAt = ts == null ? null : ts.clone(); return this; }

    /**
     * Enables or disables dense output.
     * When {@code true}, {@link Trajectory#denseOutput} is populated with a piecewise
     * polynomial interpolant that can be evaluated at any time in {@code [t₀, tf]}.
     */
    public IVPIntegral denseOutput(boolean b) { this.denseOutput = b; return this; }

    /**
     * Registers event detectors to monitor during integration.
     * Each detector defines a scalar indicator function whose zero crossings are detected.
     */
    public IVPIntegral detectors(ODEEvent... ds) { this.eventDetectors = ds; return this; }

    // -----------------------------------------------------------------------
    // integrate
    // -----------------------------------------------------------------------

    /** Integrates using a freshly allocated workspace. */
    public Trajectory integrate() { return integrate(null); }

    /**
     * Returns a pre-allocated workspace for the given method.
     * Pass to {@link #integrate(IVPPool)} to avoid repeated allocation across multiple solves.
     *
     * @param method solver method
     * @return a fresh, empty workspace of the appropriate type
     */
    public static IVPPool workspace(IVPMethod method) {
        switch (method) {
            case RK23: case RK45: case DOP853: return new RungeKuttaPool();
            case BDF:   return new BDFPool();
            case Radau: return new RadauPool();
            default: throw new IllegalArgumentException("Unknown method: " + method);
        }
    }

    /**
     * Integrates using the provided workspace (or allocates one if {@code null}).
     *
     * <p>Passing a pre-allocated workspace avoids repeated allocation when solving
     * many problems of the same size and method.</p>
     *
     * @param workspace pre-allocated workspace matching the chosen method, or {@code null}
     * @return integration result
     * @throws IllegalStateException    if required parameters are missing
     * @throws IllegalArgumentException if parameters are invalid
     */
    @Override
    public Trajectory integrate(IVPPool workspace) {
        // Validate parameters
        if (equation == null && rhs == null) throw new IllegalStateException("equation is required");
        if (Double.isNaN(t0) || Double.isNaN(tf)) throw new IllegalStateException("bounds(t0, tf) is required");
        if (y0 == null) throw new IllegalStateException("initialState is required");
        if (t0 == tf) throw new IllegalArgumentException("t0 must not equal tf");
        if (rtol <= 0 || atol[0] <= 0) throw new IllegalArgumentException("rtol and atol must be positive");
        if (atol.length != 1 && atol.length != y0.length)
            throw new IllegalArgumentException(
                    "atol length must be 1 (scalar) or match system dimension " + y0.length + ", got " + atol.length);
        validateEvalAt();

        // Create solver
        IVPCore<?> solver = createSolver(workspace);

        // Initialize event handler
        EventHandler eventHandler = (eventDetectors != null && eventDetectors.length > 0)
                ? new EventHandler(eventDetectors, t0, y0) : null;

        int n = y0.length;
        TimeSeriesBuffer out = new TimeSeriesBuffer(n);

        // Dense output buffer (only used when denseOutput=true)
        DenseOutputBuffer denseOutBuf = denseOutput ? new DenseOutputBuffer(n) : null;

        // t_eval handling
        boolean forward = tf > t0;
        double[] tEvalSorted = evalAt;
        int tEvalIdx = 0;

        // Initial point (only when not using t_eval)
        if (evalAt == null) out.add(t0, y0);

        double[] yTmp = (solver.ws.yTmp != null && solver.ws.yTmp.length >= n)
                ? solver.ws.yTmp : new double[n];
        int status = -2;  // not finished

        while (status == -2) {
            if (!solver.step()) { status = -1; break; }

            double tOld = solver.tOld;
            double tCur = solver.t;
            double[] yCur = solver.y;
            boolean finished = forward ? tCur >= tf : tCur <= tf;

            // Collect output points
            if (evalAt != null) {
                if (forward) {
                    while (tEvalIdx < tEvalSorted.length && tEvalSorted[tEvalIdx] <= tCur) {
                        double te = tEvalSorted[tEvalIdx++];
                        if (te >= tOld) { solver.interpolate(te, yTmp); out.add(te, yTmp); }
                    }
                } else {
                    while (tEvalIdx < tEvalSorted.length && tEvalSorted[tEvalIdx] >= tCur) {
                        double te = tEvalSorted[tEvalIdx++];
                        if (te <= tOld) { solver.interpolate(te, yTmp); out.add(te, yTmp); }
                    }
                }
            } else {
                out.add(tCur, yCur);
            }

            // Dense output snapshot
            if (denseOutBuf != null) {
                double[] q = solver.ws.interpCoeffs;
                if (q != null) {
                    if (solver instanceof RungeKuttaCore) {
                        // RK snapshot = [yOld[0..n-1], interpCoeffs...]
                        double[] snapshot = new double[n + q.length];
                        solver.interpolate(tOld, snapshot);
                        System.arraycopy(q, 0, snapshot, n, q.length);
                        q = snapshot;
                    } else {
                        // BDF / Radau: interpCoeffs is self-contained
                        q = q.clone();
                    }
                    denseOutBuf.add(tOld, tCur, q);
                }
            }

            // Event detection
            if (eventHandler != null) {
                Trajectory.EventPoint evResult = eventHandler.process(solver, tOld, tCur, yCur);
                if (evResult != null) {
                    if (evalAt == null && out.size > 0) out.replaceLast(evResult.t, evResult.y);
                    status = 1;
                    break;
                }
            }

            if (finished) status = 0;
        }

        // Build result
        Trajectory.TimeSeries ts = out.build();

        Trajectory.DenseOutput des = (denseOutBuf != null && denseOutBuf.size > 0)
                ? denseOutBuf.build(t0, tf, solver.ws) : null;

        Trajectory.EventPoint[][] events = (eventHandler != null) ? eventHandler.getRecords() : null;
        return new Trajectory(Trajectory.Status.of(status), solver.nfev, solver.njev, solver.nlu, ts, des, events);
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    private void validateEvalAt() {
        if (evalAt == null) return;
        double lo = Math.min(t0, tf), hi = Math.max(t0, tf);
        for (double te : evalAt) {
            if (te < lo || te > hi) throw new IllegalArgumentException(
                    "evalAt values must be within [t0, tf]");
        }
        boolean forward = tf > t0;
        for (int i = 1; i < evalAt.length; i++) {
            if (forward && evalAt[i] <= evalAt[i - 1])
                throw new IllegalArgumentException("evalAt must be monotonically increasing");
            if (!forward && evalAt[i] >= evalAt[i - 1])
                throw new IllegalArgumentException("evalAt must be monotonically decreasing");
        }
    }

    private IVPCore<?> createSolver(IVPPool workspace) {
        switch (method) {
            case BDF: {
                BDFPool ws = workspace instanceof BDFPool ? (BDFPool) workspace : new BDFPool();
                return new BDFCore(rhs, equation, t0, y0, tf, rtol, atol, maxStep, firstStep, ws);
            }
            case Radau: {
                RadauPool ws = workspace instanceof RadauPool ? (RadauPool) workspace : new RadauPool();
                return new RadauCore(rhs, equation, t0, y0, tf, rtol, atol, maxStep, firstStep, ws);
            }
            default:
                RungeKuttaPool ws = workspace instanceof RungeKuttaPool ? (RungeKuttaPool) workspace : new RungeKuttaPool();
                ODE.Equation fun = rhs != null ? rhs : (t, y, dydt) -> equation.evaluate(t, y, dydt, null);
                switch (method) {
                    case RK23: return RungeKuttaCore.rk23(fun, t0, y0, tf, rtol, atol, maxStep, firstStep, ws);
                    case RK45: return RungeKuttaCore.rk45(fun, t0, y0, tf, rtol, atol, maxStep, firstStep, ws);
                    case DOP853: return RungeKuttaCore.dop853(fun, t0, y0, tf, rtol, atol, maxStep, firstStep, ws);
                }
                throw new IllegalArgumentException("Unknown method: " + method);
        }
    }

    // -----------------------------------------------------------------------
    // DenseOutputBuffer — accumulates per-step snapshots
    // -----------------------------------------------------------------------

    /**
     * Resizable buffer that accumulates dense-output snapshots during integration,
     * then builds an {@link Trajectory.DenseOutput} at the end.
     *
     * <p>Step boundaries are stored flat: {@code tBuf[2i] = tOld_i}, {@code tBuf[2i+1] = tCur_i}.</p>
     */
    private static final class DenseOutputBuffer {
        private final int      n;
        private double[]       tBuf;    // flat: [tOld₀, tCur₀, tOld₁, tCur₁, ...]
        private double[][]     coeffs;  // coeffs[i] = snapshot for step i
        int size;

        DenseOutputBuffer(int n) {
            this.n      = n;
            tBuf        = new double[64 * 2];
            coeffs      = new double[64][];
        }

        void add(double tOld, double tCur, double[] snapshot) {
            if (size == coeffs.length) {
                tBuf   = java.util.Arrays.copyOf(tBuf,   size * 4);
                coeffs = java.util.Arrays.copyOf(coeffs, size * 2);
            }
            tBuf[size * 2]     = tOld;
            tBuf[size * 2 + 1] = tCur;
            coeffs[size]       = snapshot;
            size++;
        }

        Trajectory.DenseOutput build(double t0, double tf, IVPPool pool) {
            return new Trajectory.DenseOutput(
                    java.util.Arrays.copyOf(tBuf,   size * 2),
                    java.util.Arrays.copyOf(coeffs, size),
                    t0, tf, n, pool);
        }
    }

    // -----------------------------------------------------------------------
    // TimeSeriesBuffer — accumulates output time points and states
    // -----------------------------------------------------------------------

    /**
     * Resizable buffer for output time points and state vectors, avoiding boxing overhead.
     *
     * <p>State is stored row-major internally ({@code yBuf[j*n .. j*n+n-1] = state at step j})
     * and transposed to column-major on {@link #build()}.</p>
     */
    private static final class TimeSeriesBuffer {
        private final int n;
        double[] tBuf;
        double[] yBuf;  // row-major: yBuf[j*n .. j*n+n-1] = state at step j
        int size;

        TimeSeriesBuffer(int n) {
            this.n = n;
            tBuf   = new double[64];
            yBuf   = new double[64 * n];
        }

        void add(double t, double[] y) {
            if (size == tBuf.length) {
                tBuf = java.util.Arrays.copyOf(tBuf, size * 2);
                yBuf = java.util.Arrays.copyOf(yBuf, size * 2 * n);
            }
            tBuf[size] = t;
            System.arraycopy(y, 0, yBuf, size * n, n);
            size++;
        }

        void replaceLast(double t, double[] y) {
            tBuf[size - 1] = t;
            System.arraycopy(y, 0, yBuf, (size - 1) * n, n);
        }

        /**
         * Builds the final {@link Trajectory.TimeSeries}, transposing from row-major to column-major.
         * Column-major layout: {@code yArr[i*m + j] = equation i at time j}.
         */
        Trajectory.TimeSeries build() {
            int m = size;
            double[] tArr = java.util.Arrays.copyOf(tBuf, m);
            double[] yArr = new double[n * m];
            for (int j = 0; j < m; j++)
                for (int i = 0; i < n; i++)
                    yArr[i * m + j] = yBuf[j * n + i];
            return new Trajectory.TimeSeries(tArr, yArr, n);
        }
    }

}
