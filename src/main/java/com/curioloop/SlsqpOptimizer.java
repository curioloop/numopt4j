/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import java.util.ArrayList;
import java.util.List;
import java.util.function.ToDoubleFunction;

/**
 * SLSQP optimizer for constrained nonlinear optimization.
 * <p>
 * SLSQP (Sequential Least Squares Programming) solves general constrained
 * nonlinear optimization problems:
 * </p>
 * <pre>
 *   minimize f(x)
 *   subject to c_eq(x) = 0
 *              c_ineq(x) >= 0
 *              l <= x <= u
 * </pre>
 * 
 * <h2>Thread Safety</h2>
 * <p>
 * This class is <b>not thread-safe</b>. Each optimizer instance should only be
 * used by one thread at a time. For concurrent optimization, create separate
 * optimizer instances for each thread.
 * </p>
 * <p>
 * When using workspace reuse with {@link #optimize(double[], SlsqpWorkspace)},
 * each workspace instance should also be used by only one thread at a time.
 * </p>
 * 
 * <h2>Example usage</h2>
 * <pre>{@code
 * ObjectiveFunction objective = (x, g) -> {
 *     double f = x[0]*x[0] + x[1]*x[1];
 *     if (g != null) {
 *         g[0] = 2*x[0];
 *         g[1] = 2*x[1];
 *     }
 *     return f;
 * };
 *
 * // Equality constraint: x[0] + x[1] = 1
 * ConstraintFunction eqConstraint = (x, g) -> {
 *     if (g != null) {
 *         g[0] = 1;
 *         g[1] = 1;
 *     }
 *     return x[0] + x[1] - 1;
 * };
 *
 * SlsqpOptimizer optimizer = SlsqpOptimizer.builder()
 *     .dimension(2)
 *     .objective(objective)
 *     .equalityConstraints(eqConstraint)
 *     .build();
 *
 * OptimizationResult result = optimizer.optimize(new double[]{0, 0});
 * }</pre>
 * 
 * @see SlsqpWorkspace
 */
public final class SlsqpOptimizer {
    
    /**
     * Internal batch constraint wrapper for efficient JNI calls.
     * Evaluates all constraints at once with a shared gradient array.
     */
    private static final class BatchConstraint {
        private final ConstraintFunction[] constraints;
        private final int m;  // number of constraints
        private final double[] grad;  // shared gradient array (from optimizer)
        
        BatchConstraint(ConstraintFunction[] constraints, double[] grad) {
            this.constraints = constraints;
            this.m = constraints.length;
            this.grad = grad;
        }
        
        @SuppressWarnings("unused")
        public void evaluate(double[] x, double[] c, double[] jac) {
            for (int i = 0; i < m; i++) {
                c[i] = constraints[i].evaluate(x, jac != null ? grad : null);
                if (jac != null) {
                    // Copy gradient to Jacobian row (column-major: jac[i + m*j] = dc_i/dx_j)
                    for (int j = 0; j < grad.length; j++) {
                        jac[i + m * j] = grad[j];
                    }
                }
            }
        }
    }
    
    static {
        NativeLibraryLoader.load();
    }
    
    private final int dimension;
    private final ObjectiveFunction objective;
    private final int numEqualityConstraints;
    private final int numInequalityConstraints;
    private final BatchConstraint equalityConstraint;
    private final BatchConstraint inequalityConstraint;
    private final Bound[] bounds;
    private final Termination termination;
    
    // Extended configuration options
    private final boolean exactLineSearch;
    private final double functionEvaluationTolerance;
    private final double functionDifferenceTolerance;
    private final double variableDifferenceTolerance;
    
    private SlsqpOptimizer(Builder builder, int numEq, int numIneq, 
                           BatchConstraint eqConstraint, BatchConstraint ineqConstraint) {
        this.dimension = builder.dimension;
        this.objective = builder.objective;
        this.numEqualityConstraints = numEq;
        this.numInequalityConstraints = numIneq;
        this.equalityConstraint = eqConstraint;
        this.inequalityConstraint = ineqConstraint;
        this.bounds = builder.bounds;
        this.termination = builder.termination;
        this.exactLineSearch = builder.exactLineSearch;
        this.functionEvaluationTolerance = builder.functionEvaluationTolerance;
        this.functionDifferenceTolerance = builder.functionDifferenceTolerance;
        this.variableDifferenceTolerance = builder.variableDifferenceTolerance;
    }
    
    /**
     * Creates a new builder for SLSQP optimizer.
     * @return New builder
     */
    public static Builder builder() {
        return new Builder();
    }
    
    // ==================== Convenience static methods ====================
    
    /**
     * Minimizes a function starting from the given initial point.
     * Uses default termination criteria.
     * 
     * <p>Example:</p>
     * <pre>{@code
     * ObjectiveFunction objective = (x, g) -> {
     *     double f = x[0]*x[0] + x[1]*x[1];
     *     if (g != null) { g[0] = 2*x[0]; g[1] = 2*x[1]; }
     *     return f;
     * };
     * OptimizationResult result = SlsqpOptimizer.minimize(objective, new double[]{1, 1});
     * }</pre>
     * 
     * @param objective Objective function with gradient
     * @param initialPoint Initial guess
     * @return Optimization result
     */
    public static OptimizationResult minimize(ObjectiveFunction objective, double[] initialPoint) {
        if (initialPoint == null || initialPoint.length == 0) {
            throw new IllegalArgumentException("Initial point cannot be null or empty");
        }
        return builder()
                .dimension(initialPoint.length)
                .objective(objective)
                .build()
                .optimize(initialPoint);
    }
    
    /**
     * Runs the optimization starting from the given initial point.
     * @param initialPoint Initial guess (will be modified to contain the solution)
     * @return Optimization result
     * @throws OptimizationException if optimization fails
     */
    public OptimizationResult optimize(double[] initialPoint) {
        if (initialPoint == null || initialPoint.length != dimension) {
            throw new IllegalArgumentException("Initial point must have dimension " + dimension);
        }
        
        // Create temporary workspace and delegate to workspace version
        try (SlsqpWorkspace workspace = SlsqpWorkspace.allocate(dimension, numEqualityConstraints, numInequalityConstraints)) {
            return optimize(initialPoint, workspace);
        }
    }
    
    /**
     * Runs the optimization using an external workspace.
     * <p>
     * The workspace can be reused across multiple optimization calls to reduce
     * memory allocation overhead. This is particularly useful for high-frequency
     * optimization scenarios.
     * </p>
     * 
     * @param x Initial guess, will be modified in-place to contain the solution
     * @param workspace Pre-allocated workspace
     * @return Optimization result
     * @throws IllegalArgumentException if x dimension doesn't match
     * @throws IllegalStateException if workspace is closed or incompatible
     */
    public OptimizationResult optimize(double[] x, SlsqpWorkspace workspace) {
        if (x == null || x.length != dimension) {
            throw new IllegalArgumentException("x must have dimension " + dimension);
        }
        if (workspace == null) {
            throw new IllegalArgumentException("Workspace cannot be null");
        }
        if (workspace.isClosed()) {
            throw new IllegalStateException("Workspace has been closed");
        }
        if (!workspace.isCompatible(dimension, numEqualityConstraints, numInequalityConstraints)) {
            throw new IllegalArgumentException(
                "Workspace dimensions (" + workspace.getDimension() + ", " + 
                workspace.getEqualityConstraints() + ", " + workspace.getInequalityConstraints() + 
                ") do not match problem dimensions (" + dimension + ", " + 
                numEqualityConstraints + ", " + numInequalityConstraints + ")");
        }
        
        // Set up bounds in workspace
        setupBounds(workspace);
        
        // Call native optimizer with workspace (x is modified in-place)
        // Note: equality and inequality constraints share the same arrays since they are evaluated sequentially
        int status = nativeOptimize(
            dimension, numEqualityConstraints, numInequalityConstraints, x,
            objective, workspace.getGradient(),
            equalityConstraint, inequalityConstraint,
            workspace.getConstraintValues(), workspace.getConstraintJacobian(),
            termination.getAccuracy(), termination.getMaxIterations(),
            exactLineSearch, functionEvaluationTolerance, functionDifferenceTolerance, variableDifferenceTolerance,
            workspace.getBuffer(),
            workspace.getLowerOffset(), workspace.getUpperOffset(), workspace.getResultOffset()
        );
        
        OptimizationStatus optStatus = OptimizationStatus.fromCode(status);
        
        return new OptimizationResult(
            workspace.getResultF(),
            optStatus,
            workspace.getResultIterations(),
            0  // SLSQP doesn't track evaluations separately
        );
    }
    
    /**
     * Gets the problem dimension.
     * @return Dimension
     */
    public int getDimension() {
        return dimension;
    }
    
    /**
     * Gets the number of equality constraints.
     * @return Equality constraint count
     */
    public int getEqualityConstraintCount() {
        return numEqualityConstraints;
    }
    
    /**
     * Gets the number of inequality constraints.
     * @return Inequality constraint count
     */
    public int getInequalityConstraintCount() {
        return numInequalityConstraints;
    }
    
    /**
     * Gets the termination criteria.
     * @return Termination criteria
     */
    public Termination getTermination() {
        return termination;
    }
    
    /**
     * Checks if exact line search is enabled.
     * @return true if exact line search is enabled
     */
    public boolean isExactLineSearchEnabled() {
        return exactLineSearch;
    }
    
    /**
     * Gets the function evaluation tolerance.
     * @return Function evaluation tolerance, or -1.0 if disabled
     */
    public double getFunctionEvaluationTolerance() {
        return functionEvaluationTolerance;
    }
    
    /**
     * Gets the function difference tolerance.
     * @return Function difference tolerance, or -1.0 if disabled
     */
    public double getFunctionDifferenceTolerance() {
        return functionDifferenceTolerance;
    }
    
    /**
     * Gets the variable difference tolerance.
     * @return Variable difference tolerance, or -1.0 if disabled
     */
    public double getVariableDifferenceTolerance() {
        return variableDifferenceTolerance;
    }
    
    private void setupBounds(SlsqpWorkspace workspace) {
        for (int i = 0; i < dimension; i++) {
            if (bounds != null && i < bounds.length && bounds[i] != null) {
                Bound b = bounds[i];
                workspace.setLower(i, b.hasLower() ? b.getLower() : Double.NEGATIVE_INFINITY);
                workspace.setUpper(i, b.hasUpper() ? b.getUpper() : Double.POSITIVE_INFINITY);
            } else {
                workspace.setLower(i, Double.NEGATIVE_INFINITY);
                workspace.setUpper(i, Double.POSITIVE_INFINITY);
            }
        }
    }
    
    // Native method with external workspace
    private native int nativeOptimize(
        // Problem definition
        int n, int meq, int mineq, double[] x,
        // Objective callback
        ObjectiveFunction objective, double[] gradient,
        // Constraint callbacks (share arrays since evaluated sequentially)
        BatchConstraint eqConstraint, BatchConstraint ineqConstraint,
        double[] constraintValues, double[] constraintJacobian,
        // Termination criteria
        double accuracy, int maxIter,
        boolean exactLineSearch, double fEvalTol, double fDiffTol, double xDiffTol,
        // Workspace
        java.nio.ByteBuffer workspace,
        int lowerOffset, int upperOffset, int resultOffset
    );
    
    /**
     * Builder for SLSQP optimizer.
     */
    public static final class Builder {
        private int dimension;
        private ObjectiveFunction objective;
        private Bound[] bounds;
        private Termination termination = Termination.defaults();
        
        // Lists to accumulate individual constraints (lazy initialization)
        private List<ConstraintFunction> equalityConstraints;
        private List<ConstraintFunction> inequalityConstraints;
        
        // Extended configuration options with default values
        private boolean exactLineSearch = false;
        private double functionEvaluationTolerance = -1.0;  // Disabled by default
        private double functionDifferenceTolerance = -1.0;  // Disabled by default
        private double variableDifferenceTolerance = -1.0;  // Disabled by default
        
        private Builder() {}
        
        /**
         * Sets the problem dimension.
         * @param n Dimension (must be positive)
         * @return This builder
         */
        public Builder dimension(int n) {
            if (n <= 0) {
                throw new IllegalArgumentException("Dimension must be positive");
            }
            this.dimension = n;
            return this;
        }
        
        /**
         * Sets the objective function.
         * @param func Objective function
         * @return This builder
         */
        public Builder objective(ObjectiveFunction func) {
            this.objective = func;
            return this;
        }
        
        /**
         * Sets the objective function without gradient, using numerical differentiation.
         * 
         * <p>Example:</p>
         * <pre>{@code
         * // Using central difference (more accurate)
         * builder.objective(x -> x[0]*x[0] + x[1]*x[1], NumericalGradient.CENTRAL);
         * 
         * // Using forward difference (faster)
         * builder.objective(x -> x[0]*x[0] + x[1]*x[1], NumericalGradient.FORWARD);
         * }</pre>
         * 
         * @param func Function that computes only the objective value
         * @param diff Numerical differentiation method
         * @return This builder
         */
        public Builder objective(ToDoubleFunction<double[]> func, NumericalGradient diff) {
            this.objective = diff.wrap(func);
            return this;
        }
        
        /**
         * Adds multiple equality constraints (c(x) = 0).
         * @param constraints Constraint functions
         * @return This builder
         */
        public Builder equalityConstraints(ConstraintFunction... constraints) {
            if (constraints != null) {
                for (ConstraintFunction c : constraints) {
                    if (c != null) {
                        if (equalityConstraints == null) {
                            equalityConstraints = new ArrayList<>();
                        }
                        equalityConstraints.add(c);
                    }
                }
            }
            return this;
        }
        
        /**
         * Adds multiple inequality constraints (c(x) >= 0).
         * @param constraints Constraint functions
         * @return This builder
         */
        public Builder inequalityConstraints(ConstraintFunction... constraints) {
            if (constraints != null) {
                for (ConstraintFunction c : constraints) {
                    if (c != null) {
                        if (inequalityConstraints == null) {
                            inequalityConstraints = new ArrayList<>();
                        }
                        inequalityConstraints.add(c);
                    }
                }
            }
            return this;
        }
        
        /**
         * Sets the bounds for all variables.
         * @param bounds Array of bounds (may be null for unbounded)
         * @return This builder
         * @throws IllegalStateException if dimension is not set
         * @throws IllegalArgumentException if bounds length doesn't match dimension
         */
        public Builder bounds(Bound[] bounds) {
            if (bounds != null) {
                if (dimension <= 0) {
                    throw new IllegalStateException("Set dimension before bounds");
                }
                if (bounds.length != dimension) {
                    throw new IllegalArgumentException("Bounds array length must match dimension");
                }
            }
            this.bounds = bounds;
            return this;
        }
        
        /**
         * Sets a single bound that applies to all variables.
         * @param bound Bound to apply to all variables
         * @return This builder
         */
        public Builder bounds(Bound bound) {
            if (dimension <= 0) {
                throw new IllegalStateException("Set dimension before bounds");
            }
            this.bounds = new Bound[dimension];
            for (int i = 0; i < dimension; i++) {
                this.bounds[i] = bound;
            }
            return this;
        }
        
        /**
         * Sets the termination criteria.
         * @param term Termination criteria
         * @return This builder
         */
        public Builder termination(Termination term) {
            this.termination = term != null ? term : Termination.defaults();
            return this;
        }
        
        /**
         * Enables or disables exact line search.
         * <p>
         * When enabled, uses golden section search with quadratic interpolation
         * for more precise step length determination. This may improve convergence
         * for some problems but increases computational cost.
         * </p>
         * @param enable true to enable exact line search, false to disable
         * @return This builder
         */
        public Builder exactLineSearch(boolean enable) {
            this.exactLineSearch = enable;
            return this;
        }
        
        /**
         * Sets the function evaluation tolerance.
         * <p>
         * When set to a positive value, optimization terminates when |f(x)| &lt; tolerance.
         * This is useful for problems where the optimal value is known to be near zero.
         * </p>
         * @param tol Tolerance value (positive to enable, negative to disable)
         * @return This builder
         */
        public Builder functionEvaluationTolerance(double tol) {
            this.functionEvaluationTolerance = tol;
            return this;
        }
        
        /**
         * Sets the function difference tolerance.
         * <p>
         * When set to a positive value, optimization terminates when 
         * |f(x_new) - f(x_old)| &lt; tolerance. This detects when the objective
         * function value has stopped changing significantly.
         * </p>
         * @param tol Tolerance value (positive to enable, negative to disable)
         * @return This builder
         */
        public Builder functionDifferenceTolerance(double tol) {
            this.functionDifferenceTolerance = tol;
            return this;
        }
        
        /**
         * Sets the variable difference tolerance.
         * <p>
         * When set to a positive value, optimization terminates when 
         * ||x_new - x_old||₂ &lt; tolerance. This detects when the solution
         * has stopped changing significantly.
         * </p>
         * @param tol Tolerance value (positive to enable, negative to disable)
         * @return This builder
         */
        public Builder variableDifferenceTolerance(double tol) {
            this.variableDifferenceTolerance = tol;
            return this;
        }
        
        /**
         * Builds the optimizer.
         * @return SLSQP optimizer
         * @throws IllegalArgumentException if configuration is invalid
         */
        public SlsqpOptimizer build() {
            if (dimension <= 0) {
                throw new IllegalArgumentException("Dimension must be set");
            }
            if (objective == null) {
                throw new IllegalArgumentException("Objective function is required");
            }
            
            // Create batch constraints from individual constraint lists
            // Equality and inequality constraints share the same gradient array
            int numEq = equalityConstraints != null ? equalityConstraints.size() : 0;
            int numIneq = inequalityConstraints != null ? inequalityConstraints.size() : 0;
            double[] constraintGrad = (numEq > 0 || numIneq > 0) ? new double[dimension] : null;
            BatchConstraint eqConstraint = numEq > 0 
                ? new BatchConstraint(equalityConstraints.toArray(new ConstraintFunction[0]), constraintGrad) 
                : null;
            BatchConstraint ineqConstraint = numIneq > 0
                ? new BatchConstraint(inequalityConstraints.toArray(new ConstraintFunction[0]), constraintGrad) 
                : null;
            
            return new SlsqpOptimizer(this, numEq, numIneq, eqConstraint, ineqConstraint);
        }
    }
}
