/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

/**
 * Result of an optimization run.
 * <p>
 * When using {@code XxxOptimizer.optimize(x0)}, the solution is written back
 * into {@code x0} in-place and {@link #getSolution()} returns {@code null}.
 * When using {@code XxxProblem.solve()}, the solution is stored in
 * {@link #getSolution()} and the original initial-point array is not modified.
 * </p>
 */
public class OptimizationResult {
    
    private final double objectiveValue;
    private final OptimizationStatus status;
    private final int iterations;
    private final int evaluations;
    /** Solution vector; non-null only when produced by {@code XxxProblem.solve()}. */
    private final double[] solution;
    /** Scalar root value for 1-D root-finding results; NaN otherwise. */
    private final double root;

    /** Constructor used by core algorithms (solution lives in the caller's x array). */
    public OptimizationResult(double objectiveValue, OptimizationStatus status,
                              int iterations, int evaluations) {
        this(objectiveValue, status, iterations, evaluations, null);
    }

    /** Constructor used by {@code XxxProblem.solve()} to carry the solution. */
    public OptimizationResult(double objectiveValue, OptimizationStatus status,
                              int iterations, int evaluations, double[] solution) {
        this.objectiveValue = objectiveValue;
        this.status = status != null ? status : OptimizationStatus.ABNORMAL_TERMINATION;
        this.iterations = iterations;
        this.evaluations = evaluations;
        this.solution = solution;
        this.root = Double.NaN;
    }

    /** Constructor used by root-finding solvers. */
    public OptimizationResult(double root, double[] solution, double objectiveValue,
                               OptimizationStatus status, int evaluations) {
        this.root = root;
        this.solution = solution;
        this.objectiveValue = objectiveValue;
        this.status = status != null ? status : OptimizationStatus.ABNORMAL_TERMINATION;
        this.evaluations = evaluations;
        this.iterations = evaluations;
    }

    /**
     * Returns the scalar root for 1-D root-finding results.
     *
     * @return the root value, or {@code NaN} if this is not a scalar root result
     */
    public double getRoot() {
        return root;
    }

    public double getObjectiveValue() {
        return objectiveValue;
    }
    
    public OptimizationStatus getStatus() {
        return status;
    }
    
    public boolean isConverged() {
        return status.isConverged();
    }
    
    public int getIterations() {
        return iterations;
    }
    
    public int getEvaluations() {
        return evaluations;
    }

    /**
     * Returns the solution vector, or an empty array if the optimizer wrote the
     * solution back into the caller-supplied initial-point array instead.
     *
     * @return defensive copy of the solution, or empty array if not available
     */
    public double[] getSolution() {
        return solution != null ? solution.clone() : new double[0];
    }

    /**
     * Returns a formatted single-line summary of this result.
     *
     * @return summary string containing status, objective value, iterations and evaluations
     */
    public String getSummary() {
        return String.format("Status: %s | Objective: %.6e | Iterations: %d | Evaluations: %d",
                status.getDescription(), objectiveValue, iterations, evaluations);
    }

    /**
     * Returns whether the optimization converged successfully.
     *
     * @return true if status is converged
     */
    public boolean isSuccessful() {
        return status.isConverged();
    }

    /**
     * Returns an error description when the status is an error state, otherwise null.
     *
     * @return error message with suggestion, or null if not an error
     */
    public String getErrorMessage() {
        if (status.isError()) {
            return status.getDescription() + ". " + status.getSuggestion();
        }
        return null;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("OptimizationResult {\n");
        sb.append("  status: ").append(status.getDescription()).append('\n');
        sb.append("  objectiveValue: ").append(objectiveValue).append('\n');
        sb.append("  iterations: ").append(iterations).append('\n');
        sb.append("  evaluations: ").append(evaluations).append('\n');
        if (!status.isConverged() && status.getSuggestion() != null) {
            sb.append("  suggestion: ").append(status.getSuggestion()).append('\n');
        }
        sb.append('}');
        return sb.toString();
    }
}
