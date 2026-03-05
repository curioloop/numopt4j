/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

/**
 * Enumeration of optimization status codes.
 * 
 * <p>Status codes are organized into three categories:</p>
 * <ul>
 *   <li>Convergence (3, 4, 6, 7): Optimization satisfied a convergence criterion</li>
 *   <li>Limit reached (1, 2, 5): Optimization stopped due to resource limit</li>
 *   <li>Error (negative): Optimization failed due to an error</li>
 * </ul>
 */
public enum OptimizationStatus {
    
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
    
    OptimizationStatus(int code, boolean converged, String description, String suggestion) {
        this.code = code;
        this.converged = converged;
        this.description = description;
        this.suggestion = suggestion;
    }
    
    /**
     * Gets the numeric status code.
     * @return Status code
     */
    public int getCode() {
        return code;
    }
    
    /**
     * Gets a human-readable description of this status.
     * @return Non-null, non-empty description string
     */
    public String getDescription() {
        return description;
    }
    
    /**
     * Gets a suggestion for resolving non-converged states.
     * @return Suggestion string, or null for converged states
     */
    public String getSuggestion() {
        return suggestion;
    }
    
    /**
     * Checks if this status indicates successful convergence.
     * @return true if converged
     */
    public boolean isConverged() {
        return converged;
    }
    
    /**
     * Checks if this status indicates a limit was reached.
     * @return true if limit reached
     */
    public boolean isLimitReached() {
        return code == 1 || code == 2 || code == 5;
    }
    
    /**
     * Checks if this status indicates an error.
     * @return true if error
     */
    public boolean isError() {
        return code < 0;
    }
    
    /**
     * Gets the status from a numeric code.
     * @param code Numeric status code
     * @return Corresponding status enum
     */
    public static OptimizationStatus fromCode(int code) {
        for (OptimizationStatus status : values()) {
            if (status.code == code) {
                return status;
            }
        }
        return ABNORMAL_TERMINATION;
    }
    
    @Override
    public String toString() {
        return name() + "(" + code + ")";
    }
}
