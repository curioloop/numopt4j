/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import java.util.function.ToDoubleFunction;

/**
 * Numerical gradient computation methods.
 * <p>
 * Provides forward difference and central difference methods for
 * approximating gradients when analytical gradients are not available.
 * </p>
 * 
 * <h2>Usage</h2>
 * <pre>{@code
 * // Using central difference (more accurate)
 * builder.objective(x -> x[0]*x[0] + x[1]*x[1], NumericalGradient.CENTRAL);
 * 
 * // Using forward difference (faster)
 * builder.objective(x -> x[0]*x[0] + x[1]*x[1], NumericalGradient.FORWARD);
 * }</pre>
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
        public ObjectiveFunction wrap(ToDoubleFunction<double[]> func) {
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
     * Central difference method.
     * <p>
     * g[i] ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2*h)
     * </p>
     * More accurate but slower than forward difference.
     */
    CENTRAL {
        @Override
        public ObjectiveFunction wrap(ToDoubleFunction<double[]> func) {
            return (x, g) -> {
                double f = func.applyAsDouble(x);
                if (g != null) {
                    centralDifference(func, x, g);
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
    
    /**
     * Wraps a function-only objective to include numerical gradient.
     * @param func Function that computes only the objective value
     * @return ObjectiveFunction that computes both value and gradient
     */
    public abstract ObjectiveFunction wrap(ToDoubleFunction<double[]> func);
    
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
    
}
