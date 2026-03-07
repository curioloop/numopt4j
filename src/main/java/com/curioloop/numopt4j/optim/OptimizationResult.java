/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

/**
 * Immutable result of an optimization or root-finding run.
 *
 * <h2>Solution access</h2>
 * <ul>
 *   <li>When using {@code XxxOptimizer.optimize(x0)}, the solution is written back
 *       into {@code x0} in-place and {@link #getSolution()} returns {@code null}.</li>
 *   <li>When using {@code XxxProblem.solve()}, the solution is stored in
 *       {@link #getSolution()} and the original initial-point array is not modified.</li>
 *   <li>For 1-D root-finding ({@code BrentqProblem}), the scalar root is available
 *       via {@link #getRoot()}; {@link #getSolution()} is {@code null}.</li>
 *   <li>For N-D root-finding ({@code HYBRProblem}, {@code BroydenProblem}),
 *       the solution vector is in {@link #getSolution()} and {@link #getRoot()}
 *       returns {@code NaN}.</li>
 * </ul>
 *
 * <h2>Cost field semantics</h2>
 * <ul>
 *   <li>Minimizers (L-BFGS-B, SLSQP): value of the objective function F(x).</li>
 *   <li>TRF: residual sum of squares ‖f(x)‖² (or robust cost when a loss is set).</li>
 *   <li>HYBR / Broyden: residual norm ‖F(x)‖.</li>
 *   <li>Brentq: |f(x)| at the returned root.</li>
 * </ul>
 */
public class OptimizationResult {

    private final double cost;
    private final OptimizationStatus status;
    private final int iterations;
    private final int evaluations;
    /** Solution vector; non-null only when produced by {@code XxxProblem.solve()}. */
    private final double[] solution;
    /** Scalar root value for 1-D root-finding results; NaN otherwise. */
    private final double root;

    /**
     * Full-args constructor.
     *
     * <p>All fields are set explicitly. Pass {@code Double.NaN} for {@code root}
     * when there is no scalar root (minimizers, N-D root finders).
     * Pass {@code null} for {@code solution} when the solution lives in the
     * caller-supplied initial-point array.</p>
     *
     * @param root       scalar root (1-D root-finding only; {@code NaN} otherwise)
     * @param solution   solution vector, or {@code null}
     * @param cost       objective / residual-norm value at termination
     * @param status     termination status (defaults to {@link OptimizationStatus#ABNORMAL_TERMINATION} if null)
     * @param iterations iteration count
     * @param evaluations function-evaluation count
     */
    public OptimizationResult(double root, double[] solution, double cost,
                              OptimizationStatus status, int iterations, int evaluations) {
        this.root       = root;
        this.solution   = solution;
        this.cost       = cost;
        this.status     = status != null ? status : OptimizationStatus.ABNORMAL_TERMINATION;
        this.iterations = iterations;
        this.evaluations = evaluations;
    }

    /**
     * Returns the scalar root for 1-D root-finding results.
     *
     * @return the root value, or {@code NaN} if this is not a scalar root result
     */
    public double getRoot() { return root; }

    /** Returns the objective / residual-norm value at termination. */
    public double getCost() { return cost; }

    /** Returns the termination status. */
    public OptimizationStatus getStatus() { return status; }

    /** Returns the number of iterations performed. */
    public int getIterations() { return iterations; }

    /** Returns the number of function evaluations performed. */
    public int getEvaluations() { return evaluations; }

    /**
     * Returns the solution vector, or {@code null} if the optimizer wrote the
     * solution back into the caller-supplied initial-point array instead.
     *
     * <p><b>Caller owns the buffer:</b> the returned array is the same reference
     * stored in this result — modifying it will affect this result.</p>
     */
    public double[] getSolution() { return solution; }

    /**
     * Returns whether the optimization converged successfully.
     *
     * @return {@code true} if {@link #getStatus()} is a converged status
     */
    public boolean isSuccessful() { return status.isConverged(); }

    /**
     * Returns an error description when the status is an error state, otherwise {@code null}.
     *
     * @return error message with suggestion, or {@code null} if not an error
     */
    public String getErrorMessage() {
        if (status.isError()) {
            return status.getDescription() + ". " + status.getSuggestion();
        }
        return null;
    }

    /** Returns a formatted single-line summary of this result. */
    public String getSummary() {
        return String.format("Status: %s | Cost: %.6e | Iterations: %d | Evaluations: %d",
                status.getDescription(), cost, iterations, evaluations);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("OptimizationResult {\n");
        sb.append("  status: ").append(status.getDescription()).append('\n');
        sb.append("  cost: ").append(cost).append('\n');
        sb.append("  iterations: ").append(iterations).append('\n');
        sb.append("  evaluations: ").append(evaluations).append('\n');
        if (!status.isConverged() && status.getSuggestion() != null) {
            sb.append("  suggestion: ").append(status.getSuggestion()).append('\n');
        }
        sb.append('}');
        return sb.toString();
    }
}
