/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import java.util.function.ToDoubleFunction;

/**
 * L-BFGS-B optimizer for bound-constrained optimization.
 * <p>
 * L-BFGS-B is a limited-memory quasi-Newton algorithm for solving
 * bound-constrained optimization problems:
 * </p>
 * <pre>
 *   minimize f(x)
 *   subject to l <= x <= u
 * </pre>
 * 
 * <h2>Thread Safety</h2>
 * <p>
 * This class is <b>not thread-safe</b>. Each optimizer instance should only be
 * used by one thread at a time. For concurrent optimization, create separate
 * optimizer instances for each thread.
 * </p>
 * <p>
 * When using workspace reuse with {@link #optimize(double[], LbfgsbWorkspace)},
 * each workspace instance should also be used by only one thread at a time.
 * </p>
 * 
 * <h2>Example usage</h2>
 * <pre>{@code
 * // Simple usage with convenience method
 * OptimizationResult result = LbfgsbOptimizer.minimize(
 *     x -> x[0]*x[0] + x[1]*x[1],  // function only, gradient computed numerically
 *     new double[]{1, 1}
 * );
 * 
 * // Full control with builder
 * ObjectiveFunction rosenbrock = (x, g) -> {
 *     double f = 100 * Math.pow(x[1] - x[0]*x[0], 2) + Math.pow(1 - x[0], 2);
 *     if (g != null) {
 *         g[0] = -400 * x[0] * (x[1] - x[0]*x[0]) - 2 * (1 - x[0]);
 *         g[1] = 200 * (x[1] - x[0]*x[0]);
 *     }
 *     return f;
 * };
 *
 * LbfgsbOptimizer optimizer = LbfgsbOptimizer.builder()
 *     .dimension(2)
 *     .objective(rosenbrock)
 *     .build();
 *
 * OptimizationResult result = optimizer.optimize(new double[]{-1, 1});
 * }</pre>
 * 
 * @see LbfgsbWorkspace
 */
public final class LbfgsbOptimizer {
    
    static {
        NativeLibraryLoader.load();
    }
    
    private final int dimension;
    private final int corrections;
    private final ObjectiveFunction objective;
    private final Bound[] bounds;
    private final Termination termination;
    
    private LbfgsbOptimizer(Builder builder) {
        this.dimension = builder.dimension;
        this.corrections = builder.corrections;
        this.objective = builder.objective;
        this.bounds = builder.bounds;
        this.termination = builder.termination;
    }
    
    /**
     * Creates a new builder for L-BFGS-B optimizer.
     * @return New builder
     */
    public static Builder builder() {
        return new Builder();
    }
    
    // ==================== Convenience static methods ====================
    
    /**
     * Minimizes a function with analytical gradient starting from the given initial point.
     * Uses default termination criteria.
     * 
     * <p>Example:</p>
     * <pre>{@code
     * ObjectiveFunction rosenbrock = (x, g) -> {
     *     double f = 100 * Math.pow(x[1] - x[0]*x[0], 2) + Math.pow(1 - x[0], 2);
     *     if (g != null) {
     *         g[0] = -400 * x[0] * (x[1] - x[0]*x[0]) - 2 * (1 - x[0]);
     *         g[1] = 200 * (x[1] - x[0]*x[0]);
     *     }
     *     return f;
     * };
     * OptimizationResult result = LbfgsbOptimizer.minimize(rosenbrock, new double[]{-1, 1});
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
        try (LbfgsbWorkspace workspace = LbfgsbWorkspace.allocate(dimension, corrections)) {
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
    public OptimizationResult optimize(double[] x, LbfgsbWorkspace workspace) {
        if (x == null || x.length != dimension) {
            throw new IllegalArgumentException("x must have dimension " + dimension);
        }
        if (workspace == null) {
            throw new IllegalArgumentException("Workspace cannot be null");
        }
        if (workspace.isClosed()) {
            throw new IllegalStateException("Workspace has been closed");
        }
        if (!workspace.isCompatible(dimension, corrections)) {
            throw new IllegalArgumentException(
                "Workspace dimensions (" + workspace.getDimension() + ", " + workspace.getCorrections() + 
                ") do not match problem dimensions (" + dimension + ", " + corrections + ")");
        }
        
        // Set up bounds in workspace
        setupBounds(workspace);
        
        // Call native optimizer with workspace (x is modified in-place)
        int status = nativeOptimize(
            dimension, corrections, x,
            objective, workspace.getGradient(),
            termination.getFunctionTolerance(),
            termination.getGradientTolerance(),
            termination.getMaxIterations(),
            termination.getMaxEvaluations(),
            termination.getMaxComputations(),
            workspace.getBuffer(),
            workspace.getLowerOffset(),
            workspace.getUpperOffset(),
            workspace.getBoundTypeOffset(),
            workspace.getResultOffset()
        );
        
        OptimizationStatus optStatus = OptimizationStatus.fromCode(status);
        
        return new OptimizationResult(
            workspace.getResultF(),
            optStatus,
            workspace.getResultIterations(),
            workspace.getResultEvaluations()
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
     * Gets the number of L-BFGS corrections.
     * @return Corrections
     */
    public int getCorrections() {
        return corrections;
    }
    
    /**
     * Gets the termination criteria.
     * @return Termination criteria
     */
    public Termination getTermination() {
        return termination;
    }
    
    private void setupBounds(LbfgsbWorkspace workspace) {
        for (int i = 0; i < dimension; i++) {
            if (bounds != null && i < bounds.length && bounds[i] != null) {
                Bound b = bounds[i];
                workspace.setLower(i, b.hasLower() ? b.getLower() : Double.NEGATIVE_INFINITY);
                workspace.setUpper(i, b.hasUpper() ? b.getUpper() : Double.POSITIVE_INFINITY);
                workspace.setBoundType(i, b.getBoundType());
            } else {
                workspace.setLower(i, Double.NEGATIVE_INFINITY);
                workspace.setUpper(i, Double.POSITIVE_INFINITY);
                workspace.setBoundType(i, 0);
            }
        }
    }
    
    // Native method with external workspace
    private native int nativeOptimize(
        // Problem definition
        int n, int m, double[] x,
        // Callbacks
        ObjectiveFunction objective, double[] gradient,
        // Termination criteria
        double factr, double pgtol, int maxIter, int maxEval, long maxTime,
        // Workspace
        java.nio.ByteBuffer workspace,
        int lowerOffset, int upperOffset, int boundTypeOffset, int resultOffset
    );
    
    /**
     * Builder for L-BFGS-B optimizer.
     */
    public static final class Builder {
        private int dimension;
        private int corrections = 5;
        private ObjectiveFunction objective;
        private Bound[] bounds;
        private Termination termination = Termination.defaults();
        
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
         * Sets the number of L-BFGS corrections.
         * @param m Corrections (typically 3-20, default 5)
         * @return This builder
         */
        public Builder corrections(int m) {
            if (m <= 0) {
                throw new IllegalArgumentException("Corrections must be positive");
            }
            this.corrections = m;
            return this;
        }
        
        /**
         * Sets the objective function with analytical gradient.
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
         * Builds the optimizer.
         * @return L-BFGS-B optimizer
         * @throws IllegalArgumentException if configuration is invalid
         */
        public LbfgsbOptimizer build() {
            if (dimension <= 0) {
                throw new IllegalArgumentException("Dimension must be set");
            }
            if (objective == null) {
                throw new IllegalArgumentException("Objective function is required");
            }
            
            return new LbfgsbOptimizer(this);
        }
    }
}
