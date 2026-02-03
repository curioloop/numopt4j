/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

/**
 * Exception thrown when optimization fails.
 */
public class OptimizationException extends RuntimeException {
    
    private static final long serialVersionUID = 1L;
    
    private final OptimizationStatus status;
    
    /**
     * Creates an optimization exception.
     * @param message Error message
     */
    public OptimizationException(String message) {
        super(message);
        this.status = OptimizationStatus.ABNORMAL_TERMINATION;
    }
    
    /**
     * Creates an optimization exception with status.
     * @param message Error message
     * @param status Optimization status
     */
    public OptimizationException(String message, OptimizationStatus status) {
        super(message);
        this.status = status;
    }
    
    /**
     * Creates an optimization exception with cause.
     * @param message Error message
     * @param cause Underlying cause
     */
    public OptimizationException(String message, Throwable cause) {
        super(message, cause);
        this.status = OptimizationStatus.CALLBACK_ERROR;
    }
    
    /**
     * Creates an optimization exception with status and cause.
     * @param message Error message
     * @param status Optimization status
     * @param cause Underlying cause
     */
    public OptimizationException(String message, OptimizationStatus status, Throwable cause) {
        super(message, cause);
        this.status = status;
    }
    
    /**
     * Gets the optimization status associated with this exception.
     * @return Optimization status
     */
    public OptimizationStatus getStatus() {
        return status;
    }
}
