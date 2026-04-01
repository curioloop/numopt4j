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
public class Optimization {

    // -----------------------------------------------------------------------
    // Status enum (inner class)
    // -----------------------------------------------------------------------

    /**
     * Termination status of an optimization or root-finding run.
     *
     * <p>Status codes are organized into three categories:</p>
     * <ul>
     *   <li>Convergence (3, 4, 6, 7): satisfied a convergence criterion</li>
     *   <li>Limit reached (1, 2, 5): stopped due to a resource limit</li>
     *   <li>Error (negative): failed due to a numerical or input error</li>
     * </ul>
     */
    public enum Status {

        /** Maximum iterations reached */
        MAX_ITERATIONS_REACHED(1, false,
            "Maximum iterations reached without convergence",
            "Consider increasing maxIterations or relaxing tolerances"),

        /** Maximum function evaluations reached */
        MAX_EVALUATIONS_REACHED(2, false,
            "Maximum function evaluations reached",
            "Consider increasing maxEvaluations"),

        /** Gradient tolerance satisfied */
        GRADIENT_TOLERANCE_REACHED(3, true,
            "Converged: gradient norm below tolerance", null),

        /** Function tolerance satisfied */
        FUNCTION_TOLERANCE_REACHED(4, true,
            "Converged: function value change below tolerance", null),

        /** Maximum CPU time reached */
        MAX_COMPUTATIONS_REACHED(5, false,
            "Maximum computation time reached",
            "Consider increasing maxComputations or simplifying the objective"),

        /** Coefficient/variable tolerance satisfied */
        COEFFICIENT_TOLERANCE_REACHED(6, true,
            "Converged: variable change below tolerance", null),

        /** Chi-squared tolerance satisfied (LM algorithm) */
        CHI_SQUARED_TOLERANCE_REACHED(7, true,
            "Converged: chi-squared reduction below tolerance", null),

        /** Abnormal termination */
        ABNORMAL_TERMINATION(-1, false,
            "Abnormal termination due to internal error",
            "Check objective function for numerical issues (NaN, Infinity)"),

        /** Invalid argument provided */
        INVALID_ARGUMENT(-2, false,
            "Invalid argument provided to optimizer",
            "Verify all parameters satisfy documented constraints"),

        /** Constraints are incompatible */
        CONSTRAINT_INCOMPATIBLE(-3, false,
            "Constraints are incompatible or infeasible",
            "Check that constraints do not contradict each other"),

        /** Line search failed */
        LINE_SEARCH_FAILED(-4, false,
            "Line search failed to find acceptable step",
            "Check objective function continuity and gradient accuracy"),

        /** Callback function error */
        CALLBACK_ERROR(-5, false,
            "Error in callback function evaluation",
            "Check objective/constraint functions for exceptions"),

        /** Bracket condition f(a)*f(b) <= 0 not satisfied */
        INVALID_BRACKET(-6, false,
            "Bracket condition f(a)*f(b) <= 0 not satisfied",
            "Verify the bracket [a,b] contains a root"),

        /** Initial point or function output contains NaN or Infinity */
        INVALID_INPUT(-7, false,
            "Initial point or function output contains NaN or Infinity",
            "Provide finite initial values and check function for numerical issues");

        private final int code;
        private final boolean converged;
        private final String description;
        private final String suggestion;

        Status(int code, boolean converged, String description, String suggestion) {
            this.code = code;
            this.converged = converged;
            this.description = description;
            this.suggestion = suggestion;
        }

        public int getCode()           { return code; }
        public String getDescription() { return description; }
        public String getSuggestion()  { return suggestion; }
        public boolean isConverged()   { return converged; }

        /** Returns true if a resource limit was hit. */
        public boolean isLimitReached() { return code == 1 || code == 2 || code == 5; }

        /** Returns true if the status represents a hard error ({@code code < 0}). */
        public boolean isError()       { return code < 0; }

        /**
         * Returns the status whose {@link #getCode()} matches {@code code},
         * or {@link #ABNORMAL_TERMINATION} if no match is found.
         */
        public static Status fromCode(int code) {
            for (Status s : values()) {
                if (s.code == code) return s;
            }
            return ABNORMAL_TERMINATION;
        }

        @Override
        public String toString() { return name() + "(" + code + ")"; }
    }

    // -----------------------------------------------------------------------
    // Result fields
    // -----------------------------------------------------------------------

    private final double cost;
    private final Status status;
    private final int iterations;
    private final int evaluations;
    /** Solution vector; non-null only when produced by {@code XxxProblem.solve()}. */
    private final double[] solution;
    /** Scalar root value for 1-D root-finding results; NaN otherwise. */
    private final double root;

    /**
     * Constructs an optimization result.
     *
     * @param root        scalar root (1-D root-finding only; {@code NaN} otherwise)
     * @param solution    solution vector, or {@code null}
     * @param cost        objective / residual-norm value at termination
     * @param status      termination status; defaults to {@link Status#ABNORMAL_TERMINATION} if null
     * @param iterations  iteration count
     * @param evaluations function-evaluation count
     */
    public Optimization(double root, double[] solution, double cost,
                        Status status, int iterations, int evaluations) {
        this.root        = root;
        this.solution    = solution;
        this.cost        = cost;
        this.status      = status != null ? status : Status.ABNORMAL_TERMINATION;
        this.iterations  = iterations;
        this.evaluations = evaluations;
    }

    /** Returns the scalar root for 1-D root-finding results, or {@code NaN} otherwise. */
    public double getRoot()          { return root; }
    /** Returns the objective / residual-norm value at termination. */
    public double getCost()          { return cost; }
    /** Returns the termination status. */
    public Status getStatus()        { return status; }
    /** Returns the number of iterations performed. */
    public int getIterations()       { return iterations; }
    /** Returns the number of function evaluations performed. */
    public int getEvaluations()      { return evaluations; }
    /** Returns the solution vector, or {@code null} if written back into the caller's array. */
    public double[] getSolution()    { return solution; }
    /** Returns true if {@link #getStatus()} is a converged status. */
    public boolean isSuccessful()    { return status.isConverged(); }

    /**
     * Returns an error description when the status is an error state, otherwise {@code null}.
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
        sb.append("Optimization {\n");
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
