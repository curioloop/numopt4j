/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

/**
 * Functional interface for constraint functions.
 * <p>
 * Constraint functions are used in SLSQP for equality and inequality constraints.
 * </p>
 * <ul>
 *   <li>Equality constraints: c(x) = 0</li>
 *   <li>Inequality constraints: c(x) >= 0</li>
 * </ul>
 */
@FunctionalInterface
public interface ConstraintFunction {
    
    /**
     * Evaluates the constraint function and optionally its gradient.
     * <p>
     * When gradient is not null, the implementation should compute and store
     * the partial derivatives in the gradient array. When gradient is null,
     * only the constraint value needs to be computed.
     * </p>
     *
     * @param x Current point (read-only)
     * @param gradient Output array for gradient (may be null)
     * @return Constraint value at x
     */
    double evaluate(double[] x, double[] gradient);
}
