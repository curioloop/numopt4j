/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

/**
 * Termination criteria for optimization algorithms.
 * <p>
 * Different parameters apply to different optimizers:
 * <ul>
 *   <li>L-BFGS-B: maxIterations, maxEvaluations, gradientTolerance, functionTolerance</li>
 *   <li>SLSQP: maxIterations, accuracy</li>
 * </ul>
 */
public final class Termination {
    
    private final int maxIterations;
    private final int maxEvaluations;
    private final int nnlsIterations;
    private final long maxComputations;
    private final double accuracy;
    private final double gradientTolerance;
    private final double functionTolerance;
    
    private Termination(Builder builder) {
        this.maxIterations = builder.maxIterations;
        this.maxEvaluations = builder.maxEvaluations;
        this.nnlsIterations = builder.nnlsIterations;
        this.maxComputations = builder.maxComputations;
        this.accuracy = builder.accuracy;
        this.gradientTolerance = builder.gradientTolerance;
        this.functionTolerance = builder.functionTolerance;
    }
    
    /**
     * Creates a new builder for termination criteria.
     * @return New builder
     */
    public static Builder builder() {
        return new Builder();
    }
    
    /**
     * Creates default termination criteria.
     * @return Default termination
     */
    public static Termination defaults() {
        return builder().build();
    }
    
    /**
     * Gets the maximum number of iterations.
     * <p>Used by: L-BFGS-B, SLSQP</p>
     * @return Maximum iterations
     */
    public int getMaxIterations() {
        return maxIterations;
    }
    
    /**
     * Gets the maximum number of function evaluations.
     * <p>Used by: L-BFGS-B</p>
     * @return Maximum evaluations
     */
    public int getMaxEvaluations() {
        return maxEvaluations;
    }
    
    /**
     * Gets the maximum NNLS iterations.
     * <p>Used by: SLSQP</p>
     * @return Maximum NNLS iterations (0 = use default 3*n)
     */
    public int getNnlsIterations() {
        return nnlsIterations;
    }
    
    /**
     * Gets the maximum wall-clock time in microseconds.
     * <p>Used by: L-BFGS-B, SLSQP</p>
     * @return Maximum wall-clock time in microseconds (0 = disabled)
     */
    public long getMaxComputations() {
        return maxComputations;
    }
    
    /**
     * Gets the solution accuracy.
     * <p>Used by: SLSQP</p>
     * @return Accuracy
     */
    public double getAccuracy() {
        return accuracy;
    }
    
    /**
     * Gets the gradient tolerance.
     * <p>Used by: L-BFGS-B</p>
     * @return Gradient tolerance
     */
    public double getGradientTolerance() {
        return gradientTolerance;
    }
    
    /**
     * Gets the function tolerance factor.
     * <p>Used by: L-BFGS-B</p>
     * @return Function tolerance
     */
    public double getFunctionTolerance() {
        return functionTolerance;
    }
    
    /**
     * Builder for Termination criteria.
     */
    public static final class Builder {
        private int maxIterations = 100;
        private int maxEvaluations = 1000;
        private int nnlsIterations = 0;
        private long maxComputations = 0;
        private double accuracy = 1e-6;
        private double gradientTolerance = 1e-5;
        private double functionTolerance = 1e7;
        
        private Builder() {}
        
        /**
         * Sets the maximum number of iterations.
         * <p>Used by: L-BFGS-B, SLSQP</p>
         * @param value Maximum iterations (must be positive)
         * @return This builder
         */
        public Builder maxIterations(int value) {
            if (value <= 0) {
                throw new IllegalArgumentException("Max iterations must be positive");
            }
            this.maxIterations = value;
            return this;
        }
        
        /**
         * Sets the maximum number of function evaluations.
         * <p>Used by: L-BFGS-B</p>
         * @param value Maximum evaluations (must be positive)
         * @return This builder
         */
        public Builder maxEvaluations(int value) {
            if (value <= 0) {
                throw new IllegalArgumentException("Max evaluations must be positive");
            }
            this.maxEvaluations = value;
            return this;
        }
        
        /**
         * Sets the maximum NNLS iterations.
         * <p>Used by: SLSQP</p>
         * <p>
         * The NNLS (Non-Negative Least Squares) subproblem is solved during
         * each SLSQP iteration. This parameter limits the iterations for that
         * subproblem. Default is 0, which means use 3 * n.
         * </p>
         * @param value Maximum NNLS iterations (0 for default, must be non-negative)
         * @return This builder
         */
        public Builder nnlsIterations(int value) {
            if (value < 0) {
                throw new IllegalArgumentException("NNLS iterations must be non-negative");
            }
            this.nnlsIterations = value;
            return this;
        }
        
        /**
         * Sets the maximum wall-clock time in microseconds.
         * <p>Used by: L-BFGS-B, SLSQP</p>
         * <p>
         * When set to a positive value, optimization terminates when the
         * wall-clock time exceeds this limit. Default is 0 (disabled).
         * </p>
         * @param value Maximum wall-clock time in microseconds (0 to disable, must be non-negative)
         * @return This builder
         */
        public Builder maxComputations(long value) {
            if (value < 0) {
                throw new IllegalArgumentException("Max computations must be non-negative");
            }
            this.maxComputations = value;
            return this;
        }
        
        /**
         * Sets the solution accuracy.
         * <p>Used by: SLSQP</p>
         * @param value Accuracy (must be positive)
         * @return This builder
         */
        public Builder accuracy(double value) {
            if (value <= 0 || Double.isNaN(value)) {
                throw new IllegalArgumentException("Accuracy must be positive");
            }
            this.accuracy = value;
            return this;
        }
        
        /**
         * Sets the gradient tolerance.
         * <p>Used by: L-BFGS-B</p>
         * @param value Gradient tolerance (must be non-negative)
         * @return This builder
         */
        public Builder gradientTolerance(double value) {
            if (value < 0 || Double.isNaN(value)) {
                throw new IllegalArgumentException("Gradient tolerance must be non-negative");
            }
            this.gradientTolerance = value;
            return this;
        }
        
        /**
         * Sets the function tolerance factor.
         * <p>Used by: L-BFGS-B</p>
         * @param value Function tolerance factor (must be >= 1)
         * @return This builder
         */
        public Builder functionTolerance(double value) {
            if (value < 1 || Double.isNaN(value)) {
                throw new IllegalArgumentException("Function tolerance must be >= 1");
            }
            this.functionTolerance = value;
            return this;
        }
        
        /**
         * Builds the termination criteria.
         * @return Termination criteria
         */
        public Termination build() {
            return new Termination(this);
        }
    }
    
    @Override
    public String toString() {
        return "Termination{" +
                "maxIterations=" + maxIterations +
                ", maxEvaluations=" + maxEvaluations +
                ", nnlsIterations=" + nnlsIterations +
                ", maxComputations=" + maxComputations +
                ", accuracy=" + accuracy +
                ", gradientTolerance=" + gradientTolerance +
                ", functionTolerance=" + functionTolerance +
                '}';
    }
}
