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
     * Creates a bound with both lower and upper limits.
     * @param lower Lower bound
     * @param upper Upper bound
     * @return Bound with both limits
     */
    public static Bound between(double lower, double upper) {
        return new Bound(lower, upper);
    }
    
    /**
     * Creates a bound with only a lower limit (x >= value).
     * @param value Minimum value
     * @return Bound with lower limit only
     */
    public static Bound atLeast(double value) {
        return new Bound(value, UNBOUNDED);
    }
    
    /**
     * Creates a bound with only an upper limit (x <= value).
     * @param value Maximum value
     * @return Bound with upper limit only
     */
    public static Bound atMost(double value) {
        return new Bound(UNBOUNDED, value);
    }
    
    /**
     * Creates a fixed bound where the variable must equal the given value.
     * @param value Exact value
     * @return Fixed bound
     */
    public static Bound exactly(double value) {
        return new Bound(value, value);
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
