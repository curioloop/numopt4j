/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

/**
 * Enumeration of optimization status codes.
 */
public enum OptimizationStatus {
    
    /** Optimization converged successfully */
    CONVERGED(0, "Optimization converged successfully"),
    
    /** Maximum iterations reached */
    MAX_ITERATIONS_REACHED(1, "Maximum iterations reached"),
    
    /** Maximum function evaluations reached */
    MAX_EVALUATIONS_REACHED(2, "Maximum function evaluations reached"),
    
    /** Gradient tolerance satisfied */
    GRADIENT_TOLERANCE_REACHED(3, "Gradient tolerance satisfied"),
    
    /** Function tolerance satisfied */
    FUNCTION_TOLERANCE_REACHED(4, "Function tolerance satisfied"),
    
    /** Maximum CPU time reached */
    MAX_COMPUTATIONS_REACHED(5, "Maximum CPU time reached"),
    
    /** Abnormal termination */
    ABNORMAL_TERMINATION(-1, "Abnormal termination"),
    
    /** Invalid argument provided */
    INVALID_ARGUMENT(-2, "Invalid argument"),
    
    /** Constraints are incompatible */
    CONSTRAINT_INCOMPATIBLE(-3, "Constraints are incompatible"),
    
    /** Line search failed */
    LINE_SEARCH_FAILED(-4, "Line search failed"),
    
    /** Callback function error */
    CALLBACK_ERROR(-5, "Callback function error");
    
    private final int code;
    private final String message;
    
    OptimizationStatus(int code, String message) {
        this.code = code;
        this.message = message;
    }
    
    /**
     * Gets the numeric status code.
     * @return Status code
     */
    public int getCode() {
        return code;
    }
    
    /**
     * Gets the status message.
     * @return Status message
     */
    public String getMessage() {
        return message;
    }
    
    /**
     * Checks if this status indicates successful convergence.
     * @return true if converged
     */
    public boolean isConverged() {
        return code >= 0 && code <= 5;
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
        return name() + "(" + code + "): " + message;
    }
}
