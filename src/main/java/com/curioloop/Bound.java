/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

/**
 * Represents bounds for an optimization variable.
 */
public final class Bound {
    
    /** Constant representing an unbounded value */
    public static final double UNBOUNDED = Double.NaN;
    
    private final double lower;
    private final double upper;
    
    /**
     * Creates a bound with specified lower and upper limits.
     * @param lower Lower bound (use UNBOUNDED for no lower bound)
     * @param upper Upper bound (use UNBOUNDED for no upper bound)
     */
    public Bound(double lower, double upper) {
        if (!Double.isNaN(lower) && !Double.isNaN(upper) && lower > upper) {
            throw new IllegalArgumentException("Lower bound must not exceed upper bound");
        }
        this.lower = lower;
        this.upper = upper;
    }
    
    /**
     * Creates an unbounded variable (no constraints).
     * @return Unbounded bound
     */
    public static Bound unbounded() {
        return new Bound(UNBOUNDED, UNBOUNDED);
    }
    
    /**
     * Creates a bound with only a lower limit.
     * @param lower Lower bound
     * @return Bound with lower limit only
     */
    public static Bound lowerBound(double lower) {
        return new Bound(lower, UNBOUNDED);
    }
    
    /**
     * Creates a bound with only an upper limit.
     * @param upper Upper bound
     * @return Bound with upper limit only
     */
    public static Bound upperBound(double upper) {
        return new Bound(UNBOUNDED, upper);
    }
    
    /**
     * Creates a bound with both lower and upper limits.
     * @param lower Lower bound
     * @param upper Upper bound
     * @return Bound with both limits
     */
    public static Bound between(double lower, double upper) {
        return new Bound(lower, upper);
    }
    
    /**
     * Creates a fixed bound where lower equals upper.
     * @param value Fixed value
     * @return Fixed bound
     */
    public static Bound fixed(double value) {
        return new Bound(value, value);
    }
    
    // ==================== Alias methods for better readability ====================
    
    /**
     * Alias for {@link #between(double, double)}.
     * Creates a bound with both lower and upper limits.
     * @param lower Lower bound
     * @param upper Upper bound
     * @return Bound with both limits
     */
    public static Bound range(double lower, double upper) {
        return between(lower, upper);
    }
    
    /**
     * Alias for {@link #lowerBound(double)}.
     * Creates a bound with only a lower limit (x >= value).
     * @param value Minimum value
     * @return Bound with lower limit only
     */
    public static Bound atLeast(double value) {
        return lowerBound(value);
    }
    
    /**
     * Alias for {@link #upperBound(double)}.
     * Creates a bound with only an upper limit (x <= value).
     * @param value Maximum value
     * @return Bound with upper limit only
     */
    public static Bound atMost(double value) {
        return upperBound(value);
    }
    
    /**
     * Alias for {@link #fixed(double)}.
     * Creates a fixed bound where the variable must equal the given value.
     * @param value Exact value
     * @return Fixed bound
     */
    public static Bound exactly(double value) {
        return fixed(value);
    }
    
    /**
     * Creates a non-negative bound (x >= 0).
     * @return Bound with lower limit of 0
     */
    public static Bound nonNegative() {
        return lowerBound(0.0);
    }
    
    /**
     * Creates a non-positive bound (x <= 0).
     * @return Bound with upper limit of 0
     */
    public static Bound nonPositive() {
        return upperBound(0.0);
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
        return !Double.isNaN(lower);
    }
    
    /**
     * Checks if this bound has an upper limit.
     * @return true if upper bound exists
     */
    public boolean hasUpper() {
        return !Double.isNaN(upper);
    }
    
    /**
     * Checks if this is a fixed bound (lower == upper).
     * @return true if fixed
     */
    public boolean isFixed() {
        return hasLower() && hasUpper() && lower == upper;
    }
    
    /**
     * Checks if this bound is completely unbounded.
     * @return true if no bounds
     */
    public boolean isUnbounded() {
        return !hasLower() && !hasUpper();
    }
    
    /**
     * Gets the bound type code for native code.
     * @return 0=none, 1=lower, 2=both, 3=upper
     */
    public int getBoundType() {
        boolean hasL = hasLower();
        boolean hasU = hasUpper();
        if (hasL && hasU) return 2;
        if (hasL) return 1;
        if (hasU) return 3;
        return 0;
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
