/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.ode;

import com.curioloop.numopt4j.optim.root.BrentqSolver;
import com.curioloop.numopt4j.quad.Trajectory;

/**
 * ODE event detection handler, strictly following the event detection logic in scipy {@code ivp.py}.
 *
 * <h2>Algorithm</h2>
 * <ol>
 *   <li>After each accepted step, evaluate {@code g(t, y)} for every registered {@link ODEEvent}.</li>
 *   <li>Detect sign changes between {@code gOld} (start of step) and {@code gNew} (end of step),
 *       filtered by each event's {@link ODEEvent.Trigger} direction.</li>
 *   <li>For each active event, locate the zero precisely using Brent's method on the interval
 *       {@code [tOld, tCur]}, with tolerance {@code 4 * eps}.</li>
 *   <li>If any terminal event is reached, sort active events by time and stop at the first
 *       terminal occurrence; record all events up to and including that point.</li>
 * </ol>
 *
 * <p>Call {@link #process} after each step; it returns non-null only when a terminal event fires.</p>
 */
final class EventHandler {

    /** Brent's method tolerance: 4 * machine epsilon. */
    private static final double EPS       = Math.ulp(1.0);
    private static final double BRENTQ_TOL = 4.0 * EPS;

    private final ODEEvent[] events;
    private final int[]      triggers;  // how many times each event has fired
    private final double[]   gOld;             // g values at step start
    private final double[]   gNew;             // g values at step end
    private final int[]      activeTmp;        // reused active-event index buffer
    private final double[]   rootsTmp;         // reused root-location buffer
    private final double[]   yTmp;             // reused interpolation scratch buffer

    /** Flat-array storage for all event records. */
    private final EventBuffer records;

    /**
     * Creates an event handler and evaluates the initial indicator values.
     *
     * @param events event detectors to monitor
     * @param t0        initial time
     * @param y0        initial state, length n
     */
    public EventHandler(ODEEvent[] events, double t0, double[] y0) {
        int nd = events.length;
        this.events = events;
        this.triggers = new int[nd];
        this.gOld            = new double[nd];
        this.gNew            = new double[nd];
        this.activeTmp       = new int[nd];
        this.rootsTmp        = new double[nd];
        this.yTmp            = new double[y0.length];
        this.records         = new EventBuffer(nd, y0.length);
        for (int i = 0; i < nd; i++) {
            this.gOld[i] = events[i].equation.evaluate(t0, y0);
        }
    }

    /**
     * Processes event detection after a completed step {@code [tOld, t]}.
     *
     * <p>Returns non-null only when a terminal event is reached; the returned
     * {@link EventRecord} carries the precise termination time and interpolated state.</p>
     *
     * @param solver solver instance (used for interpolation)
     * @param tOld   step start time
     * @param t      step end time
     * @param y      state at step end
     * @return {@link Trajectory.EventPoint} if a terminal event fired, {@code null} otherwise
     */
    public Trajectory.EventPoint process(IVPCore<?> solver, double tOld, double t, double[] y) {
        int nd = events.length;
        for (int i = 0; i < nd; i++) gNew[i] = events[i].equation.evaluate(t, y);

        int activeCount = findActiveEvents();
        if (activeCount == 0) {
            System.arraycopy(gNew, 0, gOld, 0, nd);
            return null;
        }

        // Locate zeros for all active events via Brent's method
        for (int k = 0; k < activeCount; k++) {
            rootsTmp[k] = solveEventEquation(events[activeTmp[k]], solver, tOld, t);
        }

        // Update occurrence counts and find first terminal event
        int terminateIdx = -1;
        for (int k = 0; k < activeCount; k++) {
            int idx = activeTmp[k];
            triggers[idx]++;
            if (terminateIdx < 0 && events[idx].terminal > 0
                    && triggers[idx] >= events[idx].terminal) {
                terminateIdx = k;
            }
        }

        // If any terminal event, sort by time and re-find the earliest terminal
        int recordCount = activeCount;
        if (terminateIdx >= 0) {
            sortByTime(activeTmp, rootsTmp, activeCount, tOld, t);
            terminateIdx = -1;
            for (int k = 0; k < activeCount; k++) {
                int idx = activeTmp[k];
                if (events[idx].terminal > 0 && triggers[idx] >= events[idx].terminal) {
                    terminateIdx = k;
                    break;
                }
            }
            recordCount = terminateIdx + 1;  // record up to and including the terminal event
        }

        // Record triggered events
        int nSys = y.length;
        for (int k = 0; k < recordCount; k++) {
            solver.interpolate(rootsTmp[k], yTmp);
            records.add(activeTmp[k], rootsTmp[k], yTmp);
        }

        System.arraycopy(gNew, 0, gOld, 0, nd);

        if (terminateIdx < 0) return null;

        double tTerm = rootsTmp[terminateIdx];
        double[] yTerm = new double[nSys];
        solver.interpolate(tTerm, yTerm);
        return new Trajectory.EventPoint(tTerm, yTerm);
    }

    /**
     * Returns all collected event records as a 2-D array.
     * {@code result[i][j]} is the j-th occurrence of the i-th detector.
     */
    public Trajectory.EventPoint[][] getRecords() {
        return records.toArray();
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /**
     * Fills {@code activeTmp} with indices of events whose sign changed in the expected direction.
     *
     * @return number of active events
     */
    private int findActiveEvents() {
        int count = 0;
        for (int i = 0; i < events.length; i++) {
            boolean up   = gOld[i] <= 0 && gNew[i] >= 0;
            boolean down = gOld[i] >= 0 && gNew[i] <= 0;
            boolean active;
            switch (events[i].trigger) {
                case RISING:  active = up;         break;
                case FALLING: active = down;       break;
                default:      active = up || down; break;
            }
            if (active) activeTmp[count++] = i;
        }
        return count;
    }

    /**
     * Locates the zero of {@code g(t, y(t))} within {@code [tOld, t]} using Brent's method.
     * Interpolates the state at each trial point via the solver's dense output.
     */
    private double solveEventEquation(ODEEvent event, IVPCore<?> solver,
                                      double tOld, double t) {
        double lo = Math.min(tOld, t), hi = Math.max(tOld, t);
        return BrentqSolver.findRoot(
            tt -> { solver.interpolate(tt, yTmp); return event.equation.evaluate(tt, yTmp); },
            lo, hi, BRENTQ_TOL, BRENTQ_TOL, 100
        );
    }

    /**
     * In-place insertion sort of {@code (events, roots)} by ascending (forward) or
     * descending (backward) root time.
     */
    private static void sortByTime(int[] events, double[] roots, int count, double tOld, double t) {
        boolean forward = t > tOld;
        for (int i = 1; i < count; i++) {
            int ei = events[i]; double ri = roots[i]; int j = i - 1;
            while (j >= 0 && (forward ? roots[j] > ri : roots[j] < ri)) {
                events[j + 1] = events[j]; roots[j + 1] = roots[j]; j--;
            }
            events[j + 1] = ei; roots[j + 1] = ri;
        }
    }

    // -----------------------------------------------------------------------
    // EventBuffer — flat-array storage for per-detector event records
    // -----------------------------------------------------------------------

    /**
     * Flat-array storage for per-detector event records, avoiding per-record object allocation
     * in the hot path.
     *
     * <p>Layout: records are stored globally in arrival order.
     * {@code tBuf[g]} and {@code yBuf[g*nSys .. (g+1)*nSys-1]} hold the g-th global record.
     * {@code start[i]} and {@code count[i]} track each detector's contiguous slice.</p>
     *
     * <p>Detectors must be added in index order within each step (which {@link #process} guarantees),
     * so that each detector's slice remains contiguous.</p>
     */
    static final class EventBuffer {
        private final int   nd;    // number of detectors
        private final int   nSys;  // state dimension
        private final int[] start; // start[i] = global index of detector i's first record
        private final int[] count; // count[i] = number of records for detector i
        private double[]    tBuf;  // flat time buffer
        private double[]    yBuf;  // flat state buffer, row-major: yBuf[g*nSys .. (g+1)*nSys-1]

        EventBuffer(int nd, int nSys) {
            this.nd    = nd;
            this.nSys  = nSys;
            this.start = new int[nd];
            this.count = new int[nd];
            int cap    = nd * 4;
            this.tBuf  = new double[cap];
            this.yBuf  = new double[cap * nSys];
        }

        /**
         * Appends a record for detector {@code detIdx}.
         * {@code y} is copied from the provided array.
         */
        void add(int detIdx, double t, double[] y) {
            int total = totalCount();
            if (total == tBuf.length) grow();
            tBuf[total] = t;
            System.arraycopy(y, 0, yBuf, total * nSys, nSys);
            count[detIdx]++;
            // Shift start indices for all detectors after detIdx
            for (int i = detIdx + 1; i < nd; i++) start[i]++;
        }

        /**
         * Builds {@code Trajectory.EventPoint[][]} from the flat buffers.
         * Called once at the end of integration.
         */
        Trajectory.EventPoint[][] toArray() {
            Trajectory.EventPoint[][] result = new Trajectory.EventPoint[nd][];
            for (int i = 0; i < nd; i++) {
                result[i] = new Trajectory.EventPoint[count[i]];
                for (int j = 0; j < count[i]; j++) {
                    int g = start[i] + j;
                    double[] yj = new double[nSys];
                    System.arraycopy(yBuf, g * nSys, yj, 0, nSys);
                    result[i][j] = new Trajectory.EventPoint(tBuf[g], yj);
                }
            }
            return result;
        }

        private int totalCount() {
            int s = 0;
            for (int c : count) s += c;
            return s;
        }

        private void grow() {
            int newCap = tBuf.length * 2;
            tBuf = java.util.Arrays.copyOf(tBuf, newCap);
            yBuf = java.util.Arrays.copyOf(yBuf, newCap * nSys);
        }
    }
}
