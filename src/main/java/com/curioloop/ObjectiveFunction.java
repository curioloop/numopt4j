/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

/**
 * Functional interface for objective functions.
 * <p>
 * The objective function computes both the function value and optionally
 * the gradient at a given point.
 * </p>
 */
@FunctionalInterface
public interface ObjectiveFunction {
    
    /**
     * Evaluates the objective function and optionally its gradient.
     * <p>
     * When gradient is not null, the implementation should compute and store
     * the partial derivatives in the gradient array. When gradient is null,
     * only the function value needs to be computed.
     * </p>
     *
     * @param x Current point (read-only)
     * @param gradient Output array for gradient (may be null)
     * @return Function value at x
     */
    double evaluate(double[] x, double[] gradient);
}
