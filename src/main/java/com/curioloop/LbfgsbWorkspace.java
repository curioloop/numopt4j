/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Workspace for L-BFGS-B optimizer.
 * <p>
 * The workspace contains all working arrays needed by the L-BFGS-B algorithm,
 * including memory for:
 * </p>
 * <ul>
 *   <li><b>BFGS correction matrices</b>: S and Y matrices (n × m each), plus S'Y, S'S, 
 *       and Cholesky factor matrices (m × m each)</li>
 *   <li><b>Subspace minimization</b>: K matrix and related work arrays (4m × m)</li>
 *   <li><b>Cauchy point search</b>: Arrays for breakpoint management and direction computation</li>
 *   <li><b>Line search state</b>: Moré-Thuente line search context including interval 
 *       bracketing and interpolation state</li>
 *   <li><b>Variable tracking</b>: Index arrays for free/active variable management</li>
 * </ul>
 * <p>
 * The workspace can be reused across multiple optimization runs with the same dimensions.
 * Memory is allocated on the Java side and passed to native code, which handles the
 * internal layout and initialization.
 * </p>
 * 
 * <h2>Thread Safety</h2>
 * <p>
 * This class is <b>not thread-safe</b>. Each workspace instance should only be used
 * by one thread at a time. For concurrent optimization, create separate workspace
 * instances for each thread.
 * </p>
 * 
 * <p>
 * This class implements AutoCloseable for use with try-with-resources.
 * </p>
 * 
 * @see LbfgsbOptimizer
 */
public final class LbfgsbWorkspace implements AutoCloseable {
    
    private final int n;
    private final int m;
    private final ByteBuffer buffer;
    private volatile boolean closed = false;
    
    // Offsets for auxiliary arrays within the buffer
    private final int lowerOffset;
    private final int upperOffset;
    private final int boundTypeOffset;
    private final int resultOffset;
    
    // Reusable array for JNI callback
    private final double[] gradient;
    
    private LbfgsbWorkspace(int n, int m, ByteBuffer buffer, int baseSize) {
        this.n = n;
        this.m = m;
        this.buffer = buffer;
        
        // Calculate offsets for auxiliary arrays (after native workspace)
        this.lowerOffset = baseSize;
        this.upperOffset = lowerOffset + n * Double.BYTES;
        this.boundTypeOffset = upperOffset + n * Double.BYTES;
        this.resultOffset = boundTypeOffset + n * Integer.BYTES;
        
        // Pre-allocate reusable array for gradient callback
        this.gradient = new double[n];
    }
    
    /**
     * Allocates a workspace using off-heap (direct) memory.
     * <p>
     * Off-heap memory is not subject to garbage collection and can provide
     * better performance for large workspaces or high-frequency optimization.
     * The memory will be freed when the ByteBuffer is garbage collected.
     * </p>
     * 
     * @param n Problem dimension (must be positive)
     * @param m Number of L-BFGS corrections (must be positive, typically 3-20)
     * @return New workspace
     * @throws IllegalArgumentException if n or m is not positive
     */
    public static LbfgsbWorkspace allocate(int n, int m) {
        if (n <= 0) {
            throw new IllegalArgumentException("Dimension must be positive");
        }
        if (m <= 0) {
            throw new IllegalArgumentException("Corrections must be positive");
        }
        
        NativeLibraryLoader.load();
        int baseSize = (int) nativeWorkspaceSize(n, m);
        int totalSize = baseSize 
                      + n * Double.BYTES      // lower
                      + n * Double.BYTES      // upper
                      + n * Integer.BYTES     // boundType
                      + 4 * Double.BYTES;     // result
        
        ByteBuffer buffer = ByteBuffer.allocateDirect(totalSize);
        buffer.order(ByteOrder.nativeOrder());
        return new LbfgsbWorkspace(n, m, buffer, baseSize);
    }
    
    /**
     * Gets the problem dimension.
     * @return Dimension
     */
    public int getDimension() {
        return n;
    }
    
    /**
     * Gets the number of L-BFGS corrections.
     * @return Corrections
     */
    public int getCorrections() {
        return m;
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
     * @param corrections Number of L-BFGS corrections
     * @return true if compatible
     */
    public boolean isCompatible(int dimension, int corrections) {
        return this.n == dimension && this.m == corrections;
    }
    
    // Package-private accessors for LbfgsbOptimizer
    
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
    
    int getBoundTypeOffset() {
        return boundTypeOffset;
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
    
    void setBoundType(int i, int type) {
        buffer.putInt(boundTypeOffset + i * Integer.BYTES, type);
    }
    
    double getResultF() {
        return buffer.getDouble(resultOffset);
    }
    
    int getResultIterations() {
        return (int) buffer.getDouble(resultOffset + Double.BYTES);
    }
    
    int getResultEvaluations() {
        return (int) buffer.getDouble(resultOffset + 2 * Double.BYTES);
    }
    
    double[] getGradient() {
        return gradient;
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
    private static native long nativeWorkspaceSize(int n, int m);
}
