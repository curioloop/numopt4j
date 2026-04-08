/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad;

import com.curioloop.numopt4j.quad.ode.ODEEvent;
import com.curioloop.numopt4j.quad.ode.ODEIntegral;
import com.curioloop.numopt4j.quad.ode.ODEPool;

/**
 * Result of an ODE initial value problem integration performed by {@link ODEIntegral}.
 *
 * <p>Carries:
 * <ul>
 *   <li>{@link #timeSeries} — time points and corresponding state vectors.</li>
 *   <li>{@link #denseOutput} — piecewise polynomial interpolant (non-null only when
 *       {@link ODEIntegral#denseOutput(boolean) denseOutput(true)} was requested).</li>
 *   <li>{@link #events} — per-detector event records (non-null only when detectors were registered).</li>
 *   <li>Solver statistics: {@link #getFunctionEvaluations()}, {@link #getJacobianEvaluations()},
 *       {@link #getLuDecompositions()}.</li>
 * </ul>
 * </p>
 *
 * <p>Use {@link #isSuccessful()} to check whether integration completed without error,
 * and {@link #getStatus()} for the detailed termination reason.</p>
 */
public class Trajectory {

    // -----------------------------------------------------------------------
    // Status
    // -----------------------------------------------------------------------

    /**
     * Integration termination status.
     *
     * <p>Statuses with {@link #isError()} == false indicate the integration produced
     * a usable result.  {@link #FAILED} indicates a hard numerical failure.</p>
     */
    public enum Status {

        /** Solver successfully reached the end of the integration interval. */
        SUCCESS(0, "Solver successfully reached the end of the integration interval", null),

        /** A terminal event occurred before the end of the interval. */
        EVENT(1, "A terminal event occurred before the end of the interval", null),

        /** Integration step failed: step size became too small. */
        FAILED(-1, "Integration step failed: step size became too small",
               "Consider relaxing tolerances or checking for stiff behaviour");

        private final int    code;
        private final String description;
        private final String suggestion;

        Status(int code, String description, String suggestion) {
            this.code        = code;
            this.description = description;
            this.suggestion  = suggestion;
        }

        /** Numeric status code ({@code 0} = success, {@code 1} = event, {@code -1} = failed). */
        public int    getCode()        { return code; }
        /** Human-readable description of this status. */
        public String getDescription() { return description; }
        /** Suggested remediation, or {@code null} when not applicable. */
        public String getSuggestion()  { return suggestion; }
        /** Returns {@code true} for hard numerical failures ({@code code < 0}). */
        public boolean isError()       { return code < 0; }

        /**
         * Returns the status whose code matches {@code code},
         * or {@link #FAILED} if no match is found.
         */
        public static Status of(int code) {
            switch (code) {
                case  0: return SUCCESS;
                case  1: return EVENT;
                default: return FAILED;
            }
        }
    }

    // -----------------------------------------------------------------------
    // TimeSeries
    // -----------------------------------------------------------------------

    /**
     * Time-series output: the sequence of time points and corresponding state vectors
     * produced by the integration.
     *
     * <p>State data is stored column-major so that all values of a single equation
     * across all time points are contiguous in memory:
     * <pre>
     *   y[i * length + j]  =  value of equation i at time point j
     * </pre>
     * This layout matches scipy's {@code sol.y} and is efficient for per-equation analysis.</p>
     */
    public static final class TimeSeries {

        /**
         * Time point sequence.
         * Monotonically increasing (forward) or decreasing (backward).
         */
        public final double[] t;

        /**
         * State data, column-major: {@code y[i*t.length + j]} = value of equation i at time point j.
         * Total length = {@link #dim} × {@code t.length}.
         */
        public final double[] y;

        /** Number of equations (state dimension). */
        public final int dim;

        public TimeSeries(double[] t, double[] y, int dim) {
            this.t = t; this.y = y; this.dim = dim;
        }
    }

    // -----------------------------------------------------------------------
    // EventPoint
    // -----------------------------------------------------------------------

    /**
     * Immutable record of a single event occurrence detected during integration.
     *
     * <p>The event time {@link #t} is located precisely using Brent's method within the
     * step interval where the sign change was detected. The state {@link #y} is obtained
     * by interpolating the dense output at that time.</p>
     *
     * <p>All detected events are collected in {@link Trajectory#events}:
     * {@code events[i][j]} is the j-th occurrence of the i-th {@link ODEEvent}.</p>
     */
    public static final class EventPoint {
        /** Precisely located event time. */
        public final double t;
        /** Interpolated state vector at the event time, length n. */
        public final double[] y;

        public EventPoint(double t, double[] y) { this.t = t; this.y = y; }
    }

    // -----------------------------------------------------------------------
    // DenseOutput
    // -----------------------------------------------------------------------

    /**
     * Piecewise polynomial dense-output interpolant built from per-step coefficient snapshots.
     *
     * <p>Allows evaluating the solution at any time within {@code [t₀, tf]} (or {@code [tf, t₀]}
     * for backward integration), not just at the discrete output points in {@link TimeSeries}.</p>
     *
     * <p>Internally, binary-searches for the step containing the query time, then delegates
     * to the pool's method-specific interpolation logic.</p>
     *
     * <p>Only available when {@link ODEIntegral#denseOutput(boolean) denseOutput(true)} was set.</p>
     */
    public static final class DenseOutput {

        /** Flat step-boundary array: {@code [tOld₀, tCur₀, tOld₁, tCur₁, ...]}. */
        private final double[]   tBounds;
        /** Per-step coefficient snapshots; format is method-specific. */
        private final double[][] coeffs;
        private final double  tMin, tMax;
        private final boolean forward;
        private final int     n;
        private final ODEPool pool;

        public DenseOutput(double[] tBounds, double[][] coeffs,
                    double t0, double tf, int n, ODEPool pool) {
            this.tBounds = tBounds;
            this.coeffs  = coeffs;
            this.tMin    = Math.min(t0, tf);
            this.tMax    = Math.max(t0, tf);
            this.forward = tf > t0;
            this.n       = n;
            this.pool    = pool;
        }

        /**
         * Evaluates the interpolated solution at time {@code t}, writing the result into {@code out}.
         * Zero allocation.
         *
         * @param t   query time, must be within {@code [t₀, tf]}
         * @param out output array of length ≥ n, written in-place
         * @throws IllegalArgumentException if {@code t} is out of range or {@code out} is too short
         */
        public void interpolate(double t, double[] out) {
            if (t < tMin || t > tMax)
                throw new IllegalArgumentException("t=" + t + " is out of range [" + tMin + ", " + tMax + "]");
            if (out == null || out.length < n)
                throw new IllegalArgumentException("out array must have length >= " + n);

            // Binary search over monotone step intervals
            int lo = 0, hi = coeffs.length - 1, idx = hi;
            while (lo <= hi) {
                int mid = (lo + hi) >>> 1;
                double a = tBounds[mid * 2], b = tBounds[mid * 2 + 1];
                double tLo = forward ? a : b, tHi = forward ? b : a;
                if      (t < tLo) hi = mid - 1;
                else if (t > tHi) lo = mid + 1;
                else { idx = mid; break; }
            }

            pool.interpolate(t, coeffs[idx], tBounds[idx * 2], tBounds[idx * 2 + 1], out);
        }
    }

    // -----------------------------------------------------------------------
    // Trajectory fields
    // -----------------------------------------------------------------------

    /** Time-series output (always non-null). */
    public final TimeSeries timeSeries;

    /**
     * Dense-output interpolant, or {@code null} when not requested.
     * Request via {@link ODEIntegral#denseOutput(boolean) denseOutput(true)}.
     */
    public final DenseOutput denseOutput;

    /**
     * Per-detector event records: {@code events[i][j]} is the j-th occurrence of the i-th
     * {@link ODEEvent}.  {@code null} when no detectors were registered.
     */
    public final EventPoint[][] events;

    private final Status status;
    private final int functionEvaluations;
    private final int jacobianEvaluations;
    private final int luDecompositions;

    /**
     * Constructs a {@code Trajectory}.
     *
     * @param status               termination status
     * @param functionEvaluations  total RHS evaluations
     * @param jacobianEvaluations  total Jacobian evaluations (0 for explicit methods)
     * @param luDecompositions     total LU factorizations (0 for explicit methods)
     * @param timeSeries           time-series output
     * @param denseOutput          dense-output interpolant, or {@code null}
     * @param events               event records, or {@code null}
     */
    public Trajectory(Status status,
                     int functionEvaluations, int jacobianEvaluations, int luDecompositions,
                     TimeSeries timeSeries, DenseOutput denseOutput, EventPoint[][] events) {
        this.timeSeries           = timeSeries;
        this.status               = status;
        this.functionEvaluations  = functionEvaluations;
        this.jacobianEvaluations  = jacobianEvaluations;
        this.luDecompositions     = luDecompositions;
        this.denseOutput          = denseOutput;
        this.events               = events;
    }

    /** Number of equations (state dimension). Delegates to {@link TimeSeries#dim}. */
    public int getDimension()           { return timeSeries.dim; }
    /** Number of output time points. Equivalent to {@code timeSeries.t.length}. */
    public int getTimePoints()          { return timeSeries.t.length; }
    /** Total number of RHS function evaluations. */
    public int getFunctionEvaluations() { return functionEvaluations; }
    /** Total number of Jacobian evaluations (0 for explicit methods). */
    public int getJacobianEvaluations() { return jacobianEvaluations; }
    /** Total number of LU factorizations (0 for explicit methods). */
    public int getLuDecompositions()    { return luDecompositions; }
    /** Termination status. */
    public Status getStatus()           { return status; }
    /** Returns {@code true} unless the integration failed ({@link Status#FAILED}). */
    public boolean isSuccessful()       { return status != Status.FAILED; }
}
