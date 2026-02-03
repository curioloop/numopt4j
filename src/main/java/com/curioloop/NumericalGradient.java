/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import java.util.function.ToDoubleFunction;

/**
 * Numerical gradient computation methods.
 * <p>
 * Provides forward, backward, central, and five-point difference methods for
 * approximating gradients when analytical gradients are not available.
 * </p>
 * 
 * <h2>Usage</h2>
 * <pre>{@code
 * // Using five-point stencil (most accurate, O(h⁴))
 * Evaluation accurate = NumericalGradient.FIVE_POINT.wrap(x -> x[0]*x[0] + x[1]*x[1]);
 * 
 * // Using central difference (accurate, O(h²))
 * Evaluation balanced = NumericalGradient.CENTRAL.wrap(x -> x[0]*x[0] + x[1]*x[1]);
 * 
 * // Using forward difference (faster, O(h))
 * Evaluation fast = NumericalGradient.FORWARD.wrap(x -> x[0]*x[0] + x[1]*x[1]);
 * 
 * // Using backward difference (O(h))
 * Evaluation backward = NumericalGradient.BACKWARD.wrap(x -> x[0]*x[0] + x[1]*x[1]);
 * }</pre>
 * 
 * @see Evaluation
 */
public enum NumericalGradient {
    
    /**
     * Forward difference method.
     * <p>
     * g[i] ≈ (f(x + h*e_i) - f(x)) / h
     * </p>
     * Faster but less accurate than central difference.
     */
    FORWARD {
        @Override
        public Evaluation wrap(ToDoubleFunction<double[]> func) {
            return (x, g) -> {
                double f = func.applyAsDouble(x);
                if (g != null) {
                    forwardDifference(func, x, f, g);
                }
                return f;
            };
        }
    },
    
    /**
     * Backward difference method.
     * <p>
     * g[i] ≈ (f(x) - f(x - h*e_i)) / h
     * </p>
     * Same accuracy as forward difference, useful when function
     * behaves better for smaller x values.
     */
    BACKWARD {
        @Override
        public Evaluation wrap(ToDoubleFunction<double[]> func) {
            return (x, g) -> {
                double f = func.applyAsDouble(x);
                if (g != null) {
                    backwardDifference(func, x, f, g);
                }
                return f;
            };
        }
    },
    
    /**
     * Central difference method.
     * <p>
     * g[i] ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2*h)
     * </p>
     * More accurate but slower than forward difference. O(h²) accuracy.
     */
    CENTRAL {
        @Override
        public Evaluation wrap(ToDoubleFunction<double[]> func) {
            return (x, g) -> {
                double f = func.applyAsDouble(x);
                if (g != null) {
                    centralDifference(func, x, g);
                }
                return f;
            };
        }
    },
    
    /**
     * Five-point stencil method.
     * <p>
     * g[i] ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
     * </p>
     * Highest accuracy O(h⁴), but requires 4 function evaluations per dimension.
     */
    FIVE_POINT {
        @Override
        public Evaluation wrap(ToDoubleFunction<double[]> func) {
            return (x, g) -> {
                double f = func.applyAsDouble(x);
                if (g != null) {
                    fivePointDifference(func, x, g);
                }
                return f;
            };
        }
    };
    
    /** Machine epsilon */
    private static final double EPSILON = Math.ulp(1.0);
    
    /** Default step size for forward difference */
    private static final double SQRT_EPSILON = Math.sqrt(EPSILON);
    
    /** Default step size for central difference */
    private static final double CBRT_EPSILON = Math.cbrt(EPSILON);
    
    /** Default step size for five-point stencil (optimal for O(h⁴) method) */
    private static final double FOURTH_ROOT_EPSILON = Math.pow(EPSILON, 0.25);
    
    /**
     * Wraps a function-only objective to include numerical gradient.
     * @param func Function that computes only the objective value
     * @return Evaluation that computes both value and gradient
     */
    public abstract Evaluation wrap(ToDoubleFunction<double[]> func);
    
    /**
     * Computes gradient using forward difference.
     */
    private static void forwardDifference(ToDoubleFunction<double[]> func, double[] x, double f0, double[] g) {
        int n = x.length;
        
        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double h = SQRT_EPSILON * Math.max(1.0, Math.abs(xi));
            if (xi < 0) h = -h;
            
            // Ensure h is representable
            double temp = xi + h;
            h = temp - xi;
            
            x[i] = xi + h;
            double f1 = func.applyAsDouble(x);
            x[i] = xi;
            
            g[i] = (f1 - f0) / h;
        }
    }
    
    /**
     * Computes gradient using backward difference.
     */
    private static void backwardDifference(ToDoubleFunction<double[]> func, double[] x, double f0, double[] g) {
        int n = x.length;
        
        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double h = SQRT_EPSILON * Math.max(1.0, Math.abs(xi));
            if (xi < 0) h = -h;
            
            // Ensure h is representable
            double temp = xi - h;
            h = xi - temp;
            
            x[i] = xi - h;
            double f1 = func.applyAsDouble(x);
            x[i] = xi;
            
            g[i] = (f0 - f1) / h;
        }
    }
    
    /**
     * Computes gradient using central difference.
     */
    private static void centralDifference(ToDoubleFunction<double[]> func, double[] x, double[] g) {
        int n = x.length;
        
        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double h = CBRT_EPSILON * Math.max(1.0, Math.abs(xi));
            
            x[i] = xi + h;
            double f1 = func.applyAsDouble(x);
            
            x[i] = xi - h;
            double f2 = func.applyAsDouble(x);
            
            x[i] = xi;
            
            g[i] = (f1 - f2) / (2.0 * h);
        }
    }
    
    /**
     * Computes gradient using five-point stencil.
     */
    private static void fivePointDifference(ToDoubleFunction<double[]> func, double[] x, double[] g) {
        int n = x.length;
        
        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double h = FOURTH_ROOT_EPSILON * Math.max(1.0, Math.abs(xi));
            
            x[i] = xi + 2*h;
            double f1 = func.applyAsDouble(x);
            
            x[i] = xi + h;
            double f2 = func.applyAsDouble(x);
            
            x[i] = xi - h;
            double f3 = func.applyAsDouble(x);
            
            x[i] = xi - 2*h;
            double f4 = func.applyAsDouble(x);
            
            x[i] = xi;
            
            g[i] = (-f1 + 8*f2 - 8*f3 + f4) / (12.0 * h);
        }
    }
    
}
