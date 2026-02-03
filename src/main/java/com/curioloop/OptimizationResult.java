/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import java.util.Arrays;

/**
 * Result of an optimization run.
 */
public final class OptimizationResult {
    
    private final double[] solution;
    private final double functionValue;
    private final OptimizationStatus status;
    private final int iterations;
    private final int evaluations;
    
    /**
     * Creates an optimization result.
     * @param solution Solution vector
     * @param functionValue Optimal function value
     * @param status Optimization status
     * @param iterations Number of iterations
     * @param evaluations Number of function evaluations
     */
    public OptimizationResult(double[] solution, double functionValue,
                              OptimizationStatus status, int iterations, int evaluations) {
        this.solution = solution != null ? solution.clone() : new double[0];
        this.functionValue = functionValue;
        this.status = status;
        this.iterations = iterations;
        this.evaluations = evaluations;
    }
    
    /**
     * Gets the solution vector.
     * @return Copy of solution vector
     */
    public double[] getSolution() {
        return solution.clone();
    }
    
    /**
     * Gets the optimal function value.
     * @return Function value at solution
     */
    public double getFunctionValue() {
        return functionValue;
    }
    
    /**
     * Gets the optimization status.
     * @return Status
     */
    public OptimizationStatus getStatus() {
        return status;
    }
    
    /**
     * Checks if the optimization converged successfully.
     * @return true if converged
     */
    public boolean isConverged() {
        return status.isConverged();
    }
    
    /**
     * Gets the number of iterations performed.
     * @return Iteration count
     */
    public int getIterations() {
        return iterations;
    }
    
    /**
     * Gets the number of function evaluations.
     * @return Evaluation count
     */
    public int getEvaluations() {
        return evaluations;
    }
    
    /**
     * Gets the dimension of the solution.
     * @return Solution dimension
     */
    public int getDimension() {
        return solution.length;
    }
    
    @Override
    public String toString() {
        return "OptimizationResult{" +
                "status=" + status +
                ", functionValue=" + functionValue +
                ", iterations=" + iterations +
                ", evaluations=" + evaluations +
                ", solution=" + Arrays.toString(solution) +
                '}';
    }
}
