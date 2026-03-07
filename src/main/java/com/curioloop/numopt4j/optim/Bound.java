/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

/**
 * Represents bounds for an optimization variable.
 *
 * <p>Use the factory methods to create bounds for each variable in your optimization problem.
 * Pass a {@code Bound[]} array to the {@code bounds()} method of any Problem class.</p>
 */
public final class Bound {
    
    static final Bound UNBOUNDED = new Bound(Double.NaN, Double.NaN);

    private final double lower;
    private final double upper;
    private final boolean hasUpper;
    private final boolean hasLower;

    /**
     * Creates a bound with specified lower and upper limits.
     *
     * <p>Use {@code Double.NaN} to indicate no bound on that side.
     * Prefer the static factory methods ({@link #between}, {@link #atLeast}, etc.)
     * for clarity; this constructor is provided for frameworks that require a
     * public no-arg or full-arg constructor (e.g. serialization, reflection).</p>
     *
     * @param lower Lower bound (use {@code Double.NaN} for no lower bound)
     * @param upper Upper bound (use {@code Double.NaN} for no upper bound)
     */
    public Bound(double lower, double upper) {
        if (!Double.isNaN(lower) && !Double.isNaN(upper) && lower > upper) {
            throw new IllegalArgumentException("Lower bound must not exceed upper bound");
        }
        this.hasLower = !Double.isNaN(lower);
        this.hasUpper = !Double.isNaN(upper);
        this.lower = hasLower ? lower : Double.NEGATIVE_INFINITY;
        this.upper = hasUpper ? upper : Double.POSITIVE_INFINITY;
    }
    
    /**
     * Creates an unbounded variable: x ∈ (-∞, +∞).
     *
     * <p>Use when no constraint is needed on a variable. This is the default
     * behavior when no bounds are specified.</p>
     *
     * @return Unbounded bound
     */
    public static Bound unbounded() {
        return UNBOUNDED;
    }

    /**
     * Creates a bound with both lower and upper limits: lower ≤ x ≤ upper.
     *
     * <p>Use when a variable must stay within a specific interval, e.g.,
     * a probability value constrained to [0, 1], or a physical parameter
     * with known min/max values.</p>
     *
     * @param lower Lower bound (inclusive)
     * @param upper Upper bound (inclusive)
     * @return Bound with both limits
     */
    public static Bound between(double lower, double upper) {
        return new Bound(lower, upper);
    }

    /**
     * Creates a lower-bounded variable: x ≥ value.
     *
     * <p>Use when a variable must be at least a certain value, e.g.,
     * a non-negative quantity like mass, length, or probability.</p>
     *
     * @param value Minimum value (inclusive)
     * @return Bound with lower limit only
     */
    public static Bound atLeast(double value) {
        return new Bound(value, Double.NaN);
    }

    /**
     * Creates an upper-bounded variable: x ≤ value.
     *
     * <p>Use when a variable must not exceed a certain value, e.g.,
     * a maximum capacity, concentration limit, or safety threshold.</p>
     *
     * @param value Maximum value (inclusive)
     * @return Bound with upper limit only
     */
    public static Bound atMost(double value) {
        return new Bound(Double.NaN, value);
    }

    /**
     * Creates a fixed variable: x = value.
     *
     * <p>Use when a variable is held constant during optimization, e.g.,
     * fixing a known parameter while fitting others.</p>
     *
     * @param value Exact value the variable must equal
     * @return Fixed bound
     */
    public static Bound exactly(double value) {
        return new Bound(value, value);
    }

    /**
     * Creates a non-negative variable: x ≥ 0.
     *
     * @return Bound with lower limit 0
     */
    public static Bound nonNegative() {
        return new Bound(0.0, Double.NaN);
    }

    /**
     * Creates a non-positive variable: x ≤ 0.
     *
     * @return Bound with upper limit 0
     */
    public static Bound nonPositive() {
        return new Bound(Double.NaN, 0.0);
    }

    /**
     * Get specific bound from array
     */
    public static Bound of(Bound[] bounds, int index, Bound defBnd) {
        Bound bound = bounds == null ? null: bounds[index];
        return bound != null ? bound : defBnd;
    }

    /**
     * Gets the lower bound.
     * @return Lower bound value (NaN if unbounded)
     */
    public double getLower() {
        return lower;
    }
    
    /**
     * Gets the upper bound.
     * @return Upper bound value (NaN if unbounded)
     */
    public double getUpper() {
        return upper;
    }
    
    /**
     * Checks if this bound has a lower limit.
     * @return true if lower bound exists
     */
    public boolean hasLower() {
        return hasLower;
    }

    /**
     * Checks if this bound has an upper limit.
     * @return true if upper bound exists
     */
    public boolean hasUpper() {
        return hasUpper;
    }

    /**
     * Checks if this bound has both limit.
     * @return true if lower and upper both exists
     */
    public boolean hasBoth() {
        return hasLower & hasUpper;
    }

    /**
     * Checks if this is a fixed bound (lower == upper).
     * @return true if fixed
     */
    public boolean isFixed() {
        return lower == upper;
    }
    
    /**
     * Checks if this bound is completely unbounded.
     * @return true if no bounds
     */
    public boolean isUnbounded() {
        return !(hasLower | hasUpper);
    }

    @Override
    public String toString() {
        if (isUnbounded()) return "(-∞, +∞)";
        if (isFixed()) return "[" + lower + "]";
        String l = hasLower() ? String.valueOf(lower) : "-∞";
        String u = hasUpper() ? String.valueOf(upper) : "+∞";
        return "[" + l + ", " + u + "]";
    }
}
