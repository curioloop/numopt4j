/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Workspace for SLSQP (Sequential Least Squares Programming) optimizer.
 * <p>
 * The workspace contains all working arrays needed by the algorithm, including
 * support for exact line search using golden section and quadratic interpolation.
 * It can be reused across multiple optimization runs with the same dimensions.
 * </p>
 * 
 * <h2>Memory Layout</h2>
 * <p>
 * The native workspace structure (SlsqpWorkspace) contains the following components:
 * </p>
 * <pre>
 * +------------------+
 * | l (n*(n+1)/2+1)  |  LDL^T factor
 * +------------------+
 * | x0 (n)           |  Initial position
 * +------------------+
 * | g (n+1)          |  Gradient
 * +------------------+
 * | c (m)            |  Constraint values
 * +------------------+
 * | a (m × (n+1))    |  Constraint Jacobian
 * +------------------+
 * | mu (m)           |  Lagrange multipliers
 * +------------------+
 * | s (n+1)          |  Search direction
 * +------------------+
 * | u (n+1)          |  Temporary vector
 * +------------------+
 * | v (n+1)          |  Temporary vector
 * +------------------+
 * | r (m+2n)         |  Residual vector
 * +------------------+
 * | w                |  General workspace
 * +------------------+
 * | jw               |  Integer workspace
 * +------------------+
 * </pre>
 * 
 * <h2>Exact Line Search Support</h2>
 * <p>
 * The workspace includes a {@code FindWork} structure for exact line search,
 * which combines golden section search with quadratic interpolation for
 * finding optimal step lengths. The FindWork structure contains:
 * </p>
 * <ul>
 *   <li>{@code a, b} - Search interval bounds</li>
 *   <li>{@code d, e} - Step information for interpolation</li>
 *   <li>{@code p, q, r} - Quadratic interpolation parameters</li>
 *   <li>{@code u, v, w, x} - Key points in the search interval</li>
 *   <li>{@code m} - Midpoint of current interval</li>
 *   <li>{@code fu, fv, fw, fx} - Function values at key points</li>
 *   <li>{@code tol1, tol2} - Convergence tolerances</li>
 * </ul>
 * 
 * <h2>Line Search Modes</h2>
 * <p>
 * The exact line search operates in the following modes:
 * </p>
 * <ul>
 *   <li>{@code FIND_NOOP (0)} - No operation</li>
 *   <li>{@code FIND_INIT (1)} - Initialize search</li>
 *   <li>{@code FIND_NEXT (2)} - Continue search iteration</li>
 *   <li>{@code FIND_CONV (3)} - Search converged</li>
 * </ul>
 * 
 * <h2>Thread Safety</h2>
 * <p>
 * Each workspace instance should be used by only one thread at a time.
 * For concurrent optimization, create separate workspace instances.
 * </p>
 * 
 * <h2>Usage Example</h2>
 * <pre>{@code
 * try (SlsqpWorkspace workspace = SlsqpWorkspace.allocate(n, meq, mineq)) {
 *     // Use workspace for optimization
 *     optimizer.optimize(workspace, ...);
 * }
 * }</pre>
 * 
 * @see SlsqpOptimizer
 * @see <a href="https://en.wikipedia.org/wiki/Sequential_quadratic_programming">Sequential Quadratic Programming</a>
 */
public final class SlsqpWorkspace implements AutoCloseable {
    
    private final int n;
    private final int meq;
    private final int mineq;
    private final ByteBuffer buffer;
    private volatile boolean closed = false;
    
    // Offsets for auxiliary arrays within the buffer
    private final int lowerOffset;
    private final int upperOffset;
    private final int resultOffset;
    
    // Reusable arrays for JNI callbacks
    private final double[] gradient;           // length n
    private final double[] constraintValues;   // length max(meq, mineq), shared by eq/ineq
    private final double[] constraintJacobian; // length max(meq, mineq) * n, shared by eq/ineq
    
    private SlsqpWorkspace(int n, int meq, int mineq, ByteBuffer buffer, int baseSize) {
        this.n = n;
        this.meq = meq;
        this.mineq = mineq;
        this.buffer = buffer;
        
        // Calculate offsets for auxiliary arrays (after native workspace)
        this.lowerOffset = baseSize;
        this.upperOffset = lowerOffset + n * Double.BYTES;
        this.resultOffset = upperOffset + n * Double.BYTES;
        
        // Pre-allocate reusable arrays for callbacks
        // Equality and inequality constraints are evaluated sequentially, so they can share arrays
        this.gradient = new double[n];
        int maxConstraints = Math.max(meq, mineq);
        this.constraintValues = maxConstraints > 0 ? new double[maxConstraints] : null;
        this.constraintJacobian = maxConstraints > 0 ? new double[maxConstraints * n] : null;
    }
    
    /**
     * Allocates a workspace using off-heap (direct) memory.
     * 
     * @param n Problem dimension (number of variables, must be positive)
     * @param meq Number of equality constraints (must be non-negative)
     * @param mineq Number of inequality constraints (must be non-negative)
     * @return New workspace instance using direct memory
     * @throws IllegalArgumentException if any parameter is invalid
     */
    public static SlsqpWorkspace allocate(int n, int meq, int mineq) {
        if (n <= 0) {
            throw new IllegalArgumentException("Dimension must be positive");
        }
        if (meq < 0) {
            throw new IllegalArgumentException("Equality constraints must be non-negative");
        }
        if (mineq < 0) {
            throw new IllegalArgumentException("Inequality constraints must be non-negative");
        }
        
        NativeLibraryLoader.load();
        int baseSize = (int) nativeWorkspaceSize(n, meq, mineq);
        int totalSize = baseSize
                      + n * Double.BYTES      // lower
                      + n * Double.BYTES      // upper
                      + 3 * Double.BYTES;     // result [f, iterations, status]
        
        ByteBuffer buffer = ByteBuffer.allocateDirect(totalSize);
        buffer.order(ByteOrder.nativeOrder());
        return new SlsqpWorkspace(n, meq, mineq, buffer, baseSize);
    }
    
    /**
     * Gets the problem dimension.
     * @return Dimension
     */
    public int getDimension() {
        return n;
    }
    
    /**
     * Gets the number of equality constraints.
     * @return Equality constraint count
     */
    public int getEqualityConstraints() {
        return meq;
    }
    
    /**
     * Gets the number of inequality constraints.
     * @return Inequality constraint count
     */
    public int getInequalityConstraints() {
        return mineq;
    }
    
    /**
     * Checks if this workspace has been closed.
     * @return true if closed
     */
    public boolean isClosed() {
        return closed;
    }
    
    /**
     * Checks if this workspace is compatible with the given problem dimensions.
     * @param dimension Problem dimension
     * @param equalityConstraints Number of equality constraints
     * @param inequalityConstraints Number of inequality constraints
     * @return true if compatible
     */
    public boolean isCompatible(int dimension, int equalityConstraints, int inequalityConstraints) {
        return this.n == dimension && this.meq == equalityConstraints && this.mineq == inequalityConstraints;
    }
    
    // Package-private accessors for SlsqpOptimizer
    
    ByteBuffer getBuffer() {
        checkNotClosed();
        return buffer;
    }
    
    int getLowerOffset() {
        return lowerOffset;
    }
    
    int getUpperOffset() {
        return upperOffset;
    }
    
    int getResultOffset() {
        return resultOffset;
    }
    
    void setLower(int i, double value) {
        buffer.putDouble(lowerOffset + i * Double.BYTES, value);
    }
    
    void setUpper(int i, double value) {
        buffer.putDouble(upperOffset + i * Double.BYTES, value);
    }
    
    double getResultF() {
        return buffer.getDouble(resultOffset);
    }
    
    int getResultIterations() {
        return (int) buffer.getDouble(resultOffset + Double.BYTES);
    }
    
    double[] getGradient() {
        return gradient;
    }
    
    double[] getConstraintValues() {
        return constraintValues;
    }
    
    double[] getConstraintJacobian() {
        return constraintJacobian;
    }
    
    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("Workspace has been closed");
        }
    }
    
    @Override
    public void close() {
        closed = true;
    }
    
    // Native methods
    private static native long nativeWorkspaceSize(int n, int meq, int mineq);
}
