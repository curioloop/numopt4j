/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad;

/**
 * Result of a one-dimensional quadrature computation.
 *
 * <p>Carries the integral estimate, an error bound, a {@link Status} code, and
 * iteration/evaluation counters.  Use {@link #isSuccessful()} to check whether
 * the requested tolerance was satisfied.</p>
 */
public class Quadrature {

    // -----------------------------------------------------------------------
    // Status enum (inner class)
    // -----------------------------------------------------------------------

    /**
     * Termination status of a quadrature computation.
     *
     * <p>Statuses with {@link #isConverged()} == true indicate the requested
     * tolerance was satisfied.  Statuses with {@link #isLimitReached()} == true
     * indicate a resource limit was hit but the best available estimate is still
     * returned.  Statuses with {@link #isError()} == true indicate a hard
     * numerical failure.</p>
     */
    public enum Status {

        CONVERGED(0, true,
                "Converged: requested tolerance satisfied",
                null),

        /** Adaptive subdivision limit reached; best estimate is returned. */
        MAX_SUBDIVISIONS_REACHED(1, false,
                "Maximum subdivisions reached without convergence",
                "Consider increasing maxSubdivisions or providing breakpoints"),

        /** Function evaluation limit reached; best estimate is returned. */
        MAX_EVALUATIONS_REACHED(2, false,
                "Maximum function evaluations reached without convergence",
                "Consider increasing maxEvaluations or relaxing tolerances"),

        /** Oscillation cycle limit reached; best estimate is returned. */
        MAX_CYCLES_REACHED(4, false,
                "Maximum oscillation cycles reached without convergence",
                "Consider increasing maxCycles or relaxing tolerances"),

        /** Endpoint refinement level limit reached; best estimate is returned. */
        MAX_REFINEMENTS_REACHED(5, false,
                "Maximum endpoint refinements reached without convergence",
                "Consider increasing maxRefinements or relaxing tolerances"),

        /** Floating-point roundoff prevented further subdivision. */
        ROUND_OFF_DETECTED(3, false,
                "Roundoff prevented further interval subdivision",
                "Consider relaxing tolerances or rescaling the integrand"),

        /** Hard numerical failure (NaN/Inf in integrand or internal computation). */
        ABNORMAL_TERMINATION(-1, false,
                "Abnormal termination due to internal numerical failure",
                "Check the integrand for NaN, Infinity, or severe singular behavior"),

        /** A required parameter was invalid (e.g. non-finite bounds, negative tolerance). */
        INVALID_ARGUMENT(-2, false,
                "Invalid argument provided to quadrature",
                "Verify bounds, tolerances, and limit parameters satisfy the documented constraints");

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
        /** Returns true if the requested tolerance was satisfied. */
        public boolean isConverged()   { return converged; }

        /** Returns true if a resource limit was hit (but a best estimate is still available). */
        public boolean isLimitReached() {
            return this == MAX_SUBDIVISIONS_REACHED
                    || this == MAX_EVALUATIONS_REACHED
                    || this == MAX_CYCLES_REACHED
                    || this == MAX_REFINEMENTS_REACHED;
        }

        /** Returns true if the status represents a hard error ({@code code < 0}). */
        public boolean isError() { return code < 0; }

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

    private final double value;           // integral estimate
    private final double estimatedError;  // absolute error bound
    private final Status status;
    private final int iterations;         // number of adaptive subdivisions or refinement levels
    private final int evaluations;        // total function evaluations

    /**
     * Constructs a quadrature result.
     *
     * @param value          integral estimate
     * @param estimatedError absolute error bound (may be 0 for fixed-point methods)
     * @param status         termination status; defaults to {@link Status#ABNORMAL_TERMINATION} if null
     * @param iterations     number of adaptive subdivisions or refinement levels performed
     * @param evaluations    total number of function evaluations
     */
    public Quadrature(double value, double estimatedError, Status status,
                      int iterations, int evaluations) {
        this.value = value;
        this.estimatedError = estimatedError;
        this.status = status != null ? status : Status.ABNORMAL_TERMINATION;
        this.iterations = iterations;
        this.evaluations = evaluations;
    }

    public double getValue()           { return value; }
    public double getEstimatedError()  { return estimatedError; }
    public Status getStatus()          { return status; }
    /** Returns the number of adaptive subdivisions or refinement levels performed. */
    public int getIterations()         { return iterations; }
    /** Returns the total number of function evaluations. */
    public int getEvaluations()        { return evaluations; }
    /** Returns true if {@link #getStatus()} is {@link Status#CONVERGED}. */
    public boolean isSuccessful()      { return status.isConverged(); }

    public String getSummary() {
        return String.format("Status: %s | Value: %.6e | Error: %.6e | Iterations: %d | Evaluations: %d",
                status.getDescription(), value, estimatedError, iterations, evaluations);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("QuadratureResult {\n");
        sb.append("  status: ").append(status.getDescription()).append('\n');
        sb.append("  value: ").append(value).append('\n');
        sb.append("  estimatedError: ").append(estimatedError).append('\n');
        sb.append("  iterations: ").append(iterations).append('\n');
        sb.append("  evaluations: ").append(evaluations).append('\n');
        if (!status.isConverged() && status.getSuggestion() != null) {
            sb.append("  suggestion: ").append(status.getSuggestion()).append('\n');
        }
        sb.append('}');
        return sb.toString();
    }
}
