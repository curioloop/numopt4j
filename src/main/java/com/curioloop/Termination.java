/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

/**
 * Termination criteria for optimization algorithms.
 */
public final class Termination {
    
    private final int maxIterations;
    private final int maxEvaluations;
    private final double accuracy;
    private final double gradientTolerance;
    private final double functionTolerance;
    
    private Termination(Builder builder) {
        this.maxIterations = builder.maxIterations;
        this.maxEvaluations = builder.maxEvaluations;
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
     * @return Maximum iterations
     */
    public int getMaxIterations() {
        return maxIterations;
    }
    
    /**
     * Gets the maximum number of function evaluations.
     * @return Maximum evaluations
     */
    public int getMaxEvaluations() {
        return maxEvaluations;
    }
    
    /**
     * Gets the solution accuracy.
     * @return Accuracy
     */
    public double getAccuracy() {
        return accuracy;
    }
    
    /**
     * Gets the gradient tolerance.
     * @return Gradient tolerance
     */
    public double getGradientTolerance() {
        return gradientTolerance;
    }
    
    /**
     * Gets the function tolerance factor.
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
        private double accuracy = 1e-6;
        private double gradientTolerance = 1e-5;
        private double functionTolerance = 1e7;
        
        private Builder() {}
        
        /**
         * Sets the maximum number of iterations.
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
         * Sets the solution accuracy.
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
                ", accuracy=" + accuracy +
                ", gradientTolerance=" + gradientTolerance +
                ", functionTolerance=" + functionTolerance +
                '}';
    }
}
