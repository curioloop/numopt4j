/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

/**
 * Common functional interface for function evaluation with optional gradient.
 * <p>
 * This interface is shared by both objective functions and constraint functions,
 * as they have the same signature: compute a scalar value and optionally its gradient.
 * </p>
 * 
 * <p>If analytical gradients are not available, use {@link NumericalGradient} to wrap
 * a function-only implementation:</p>
 * <pre>{@code
 * // Wrap a function without gradient using numerical differentiation
 * Evaluation objective = NumericalGradient.CENTRAL.wrap(x -> x[0]*x[0] + x[1]*x[1]);
 * }</pre>
 * 
 * @see NumericalGradient
 */
@FunctionalInterface
public interface Evaluation {
    
    /**
     * Evaluates the function and optionally its gradient.
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
