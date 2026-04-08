/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.ode;

/**
 * Defines an event to be detected during ODE integration.
 *
 * <p>An event is a zero crossing of a scalar indicator function {@code g(t, y)}.
 * The solver monitors sign changes of {@code g} after each step and uses Brent's method
 * to locate the exact crossing time within the step interval.</p>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Detect when y[0] crosses zero from above (falling edge), stop after first occurrence
 * ODEEvent landing = new ODEEvent(
 *     (t, y) -> y[0],           // g(t, y) = y[0]
 *     ODEEvent.Trigger.FALLING,  // triggered when g goes positive → negative
 *     1                          // terminal: stop after 1 occurrence
 * );
 *
 * // Detect all zero crossings of y[0] (non-terminal)
 * ODEEvent crossings = new ODEEvent(
 *     (t, y) -> y[0],
 *     ODEEvent.Trigger.EITHER,
 *     0                          // 0 = unlimited, non-terminal
 * );
 * }</pre>
 *
 * @see EventHandler
 * @see EventRecord
 */
public class ODEEvent {

    /**
     * Scalar indicator function {@code g(t, y)}.
     * An event is triggered when {@code g} changes sign.
     */
    @FunctionalInterface
    public interface Equation {
        /**
         * Evaluates the event indicator.
         *
         * @param t current time
         * @param y current state vector, length n (read-only)
         * @return scalar indicator value; event fires on sign change
         */
        double evaluate(double t, double[] y);
    }

    /**
     * Direction of the sign change that triggers the event.
     */
    public enum Trigger {
        /** Triggered when {@code g} transitions from negative to positive (rising edge). */
        RISING,
        /** Triggered when {@code g} transitions from positive to negative (falling edge). */
        FALLING,
        /** Triggered on either direction of sign change (default). */
        EITHER
    }

    /** The indicator function whose zero crossings are detected. */
    public final Equation equation;

    /** Which direction of sign change triggers this event. */
    public final Trigger trigger;

    /**
     * Terminal count: stop integration after this many occurrences.
     * {@code 0} means non-terminal (unlimited occurrences, integration continues).
     * {@code N > 0} means stop after the N-th occurrence.
     */
    public final int terminal;

    /**
     * Creates an event detector.
     *
     * @param equation indicator function {@code g(t, y)}
     * @param trigger  sign-change direction that fires the event
     * @param terminal stop after this many occurrences ({@code 0} = non-terminal)
     */
    public ODEEvent(Equation equation, Trigger trigger, int terminal) {
        this.equation = equation;
        this.trigger  = trigger;
        this.terminal = terminal;
    }
}
