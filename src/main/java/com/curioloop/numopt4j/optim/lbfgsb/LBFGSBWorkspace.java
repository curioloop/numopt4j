/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * L-BFGS-B Workspace - Pure Java implementation.
 *
 * This class manages all working arrays and state for the L-BFGS-B optimization
 * algorithm. It uses a unified buffer approach to minimize memory allocations
 * and improve cache locality.
 *
 * Memory Layout:
 * The workspace uses a single contiguous double[] buffer for all floating-point
 * arrays, with pre-computed offsets for efficient access. This design:
 * - Pre-allocates all required arrays during initialization
 * - Reuses allocated arrays across multiple optimization iterations
 * - Avoids new array allocations during the optimization loop
 * - Uses a single contiguous buffer for all working arrays
 * - Supports reset without reallocating memory
 *
 * Reference: Go implementation in lbfgsb/base.go (iterWork, iterCtx, iterBFGS)
 */
package com.curioloop.numopt4j.optim.lbfgsb;

/**
 * Workspace for the pure Java L-BFGS-B optimizer.
 *
 * <p>This class contains all working arrays needed by the L-BFGS-B algorithm,
 * including memory for:</p>
 * <ul>
 *   <li><b>BFGS correction matrices</b>: S and Y matrices (n × m each), plus S'Y, S'S,
 *       and Cholesky factor matrices (m × m each)</li>
 *   <li><b>Subspace minimization</b>: K matrix and related work arrays (4m × m)</li>
 *   <li><b>Cauchy point search</b>: Arrays for breakpoint management and direction computation</li>
 *   <li><b>Line search state</b>: Moré-Thuente line search context</li>
 *   <li><b>Variable tracking</b>: Index arrays for free/active variable management</li>
 * </ul>
 *
 * <h2>Thread Safety</h2>
 * <p>This class is <b>not thread-safe</b>. Each workspace instance should only be used
 * by one thread at a time. For concurrent optimization, create separate workspace
 * instances for each thread.</p>
 *
 */
public final class LBFGSBWorkspace {

    // ========================================================================
    // Problem dimensions
    // ========================================================================

    /** Problem dimension (number of variables) */
    private final int n;

    /** Number of L-BFGS corrections (limited memory parameter) */
    private final int m;

    // ========================================================================
    // Unified buffers
    // ========================================================================

    /** Unified buffer for all floating-point arrays */
    private final double[] buffer;

    /** Unified buffer for all integer arrays */
    private final int[] iBuffer;

    // ========================================================================
    // Floating-point array offsets
    // ========================================================================

    /** Offset for S matrix (n × m): stores sₖ = xₖ₊₁ - xₖ */
    private final int wsOffset;

    /** Offset for Y matrix (n × m): stores yₖ = gₖ₊₁ - gₖ */
    private final int wyOffset;

    /** Offset for S'Y matrix (m × m): inner products (sᵀy)ᵢⱼ */
    private final int syOffset;

    /** Offset for S'S matrix (m × m): inner products (sᵀs)ᵢⱼ */
    private final int ssOffset;

    /**
     * Offset for Cholesky factor T (m × m).
     * Stores the Cholesky factorization of (θSᵀS + LD⁻¹Lᵀ) = JJᵀ,
     * with Jᵀ stored in the upper triangle.
     * Used in bmv to compute the middle matrix M product.
     */
    private final int wtOffset;

    /**
     * Offset for K matrix (2m × 2m = 4m²).
     * Stores the LELᵀ factorization of the indefinite matrix:
     * <pre>
     *   K = [ -D - YᵀZZᵀY/θ    Laᵀ - Rzᵀ ]   where  E = [-I  0]
     *       [ La - Rz           θSᵀAAᵀS   ]              [ 0   I]
     * </pre>
     * On exit the upper triangle stores the LELᵀ factorization of the
     * 2×col × 2×col indefinite matrix.
     * Used in optimalDirection (subsm) to compute K⁻¹v.
     */
    private final int wnOffset;

    /**
     * Offset for inner product storage (2m × 2m = 4m²).
     * Stores the lower triangular part of:
     * <pre>
     *   WN1 = [ YᵀZZᵀY    Laᵀ + Rzᵀ ]
     *         [ La + Rz   SᵀAAᵀS    ]
     * </pre>
     * where La is the strictly lower triangular part of SᵀAAᵀY
     * and Rz is the upper triangular part of SᵀZZᵀY.
     * Updated incrementally in formK to avoid recomputing from scratch.
     */
    private final int sndOffset;

    /** Offset for Cauchy point z (n): GCP / Newton point */
    private final int zOffset;

    /** Offset for reduced gradient r (n) */
    private final int rOffset;

    /** Offset for search direction d (n) */
    private final int dOffset;

    /** Offset for breakpoint times t (n) */
    private final int tOffset;

    /** Offset for safeguard point xp (n): projected Newton direction backup */
    private final int xpOffset;

    /** Offset for work array wa (8m): shared temporary workspace */
    private final int waOffset;

    /** Offset for gradient g (n) */
    private final int gOffset;

    /** Offset for diagInv array (m): 1/Dᵢᵢ for bmv */
    private final int diagInvOffset;

    /** Offset for sqrtDiagInv array (m): 1/√Dᵢᵢ for bmv */
    private final int sqrtDiagInvOffset;

    // ========================================================================
    // Integer array offsets
    // ========================================================================

    /** Offset for index array (2n): free/active variable indices */
    private final int indexOffset;

    /** Offset for where array (n): variable status flags */
    private final int whereOffset;

    // ========================================================================
    // BFGS state (matches Go iterBFGS in base.go)
    // ========================================================================

    /** Indicates if the L-BFGS matrix has been updated */
    private boolean updated;

    /** Total number of BFGS updates prior to current iteration */
    private int updates;

    /** Actual number of variable metric corrections stored (0 ≤ col ≤ m) */
    private int col;

    /** Head pointer for circular buffer (location of first s/y vector) */
    private int head;

    /** Tail pointer for circular buffer (location of last s/y vector) */
    private int tail;

    /** Scaling factor θ specifying B₀ = θI */
    private double theta;

    // ========================================================================
    // Iteration context (matches Go iterCtx in base.go)
    // ========================================================================

    /** Current iteration number */
    private int iter;

    /** Total number of function and gradient evaluations */
    private int totalEval;

    /** Number of segments explored in GCP search */
    private int seg;

    /** Solution status: 0 = within box, 1 = beyond box, -1 = unknown */
    private int word;

    /** Number of free variables */
    private int free;

    /** Number of active constraints */
    private int active;

    /** Number of variables leaving the active set */
    private int leave;

    /** Number of variables entering the active set */
    private int enter;

    /** Whether the problem is constrained */
    private boolean constrained;

    /** Whether all variables have both bounds */
    private boolean boxed;

    /** Current function value */
    private double f;

    /** Previous function value */
    private double fOld;

    /** Infinity norm of projected gradient */
    private double sbgNorm;

    /** Square of 2-norm of search direction */
    private double dSqrt;

    /** 2-norm of search direction */
    private double dNorm;

    // ========================================================================
    // Line search context (encapsulates all line search state)
    // ========================================================================

    /** Line search context containing all line search state */
    private final SearchCtx searchCtx = new SearchCtx();

    // ========================================================================
    // Error recovery state
    // ========================================================================

    /** Number of BFGS matrix resets (for error recovery) */
    private int resetCount;

    // ========================================================================
    // Constructor
    // ========================================================================

    /**
     * Creates a new workspace for L-BFGS-B optimization.
     *
     * @param n Problem dimension (must be positive)
     * @param m Number of L-BFGS corrections (must be positive, typically 3-20)
     * @throws IllegalArgumentException if n or m is not positive
     */
    public LBFGSBWorkspace(int n, int m) {
        if (n <= 0) {
            throw new IllegalArgumentException("Dimension must be positive: " + n);
        }
        if (m <= 0) {
            throw new IllegalArgumentException("Corrections must be positive: " + m);
        }

        this.n = n;
        this.m = m;

        // Calculate buffer sizes and offsets
        // Double buffer layout (matching C implementation in utils.c):
        // ws[n×m] | wy[n×m] | sy[m×m] | ss[m×m] | wt[m×m] | wn[4m²] | snd[4m²] |
        // z[n] | r[n] | d[n] | t[n] | xp[n] | wa[8m] | g[n]

        int offset = 0;

        this.wsOffset = offset;
        offset += n * m;

        this.wyOffset = offset;
        offset += n * m;

        this.syOffset = offset;
        offset += m * m;

        this.ssOffset = offset;
        offset += m * m;

        this.wtOffset = offset;
        offset += m * m;

        this.wnOffset = offset;
        offset += 4 * m * m;

        this.sndOffset = offset;
        offset += 4 * m * m;

        this.zOffset = offset;
        offset += n;

        this.rOffset = offset;
        offset += n;

        this.dOffset = offset;
        offset += n;

        this.tOffset = offset;
        offset += n;

        this.xpOffset = offset;
        offset += n;

        this.waOffset = offset;
        offset += 8 * m;

        this.gOffset = offset;
        offset += n;

        this.diagInvOffset = offset;
        offset += m;

        this.sqrtDiagInvOffset = offset;
        offset += m;

        // Note: tempCol[m] is NOT allocated separately - it reuses wa[4m:5m]
        // This is safe because formK() and CauchyPoint.compute() don't execute simultaneously.
        // wa layout during Cauchy: p[0:2m], c[2m:4m], w[4m:6m], v[6m:8m]
        // wa layout during formK:  tempCol reuses w[4m:5m] (only needs m elements)

        // Allocate unified double buffer
        this.buffer = new double[offset];

        // Integer buffer layout:
        // index[2n] | where[n]
        this.indexOffset = 0;
        this.whereOffset = 2 * n;

        // Allocate unified integer buffer
        this.iBuffer = new int[3 * n];

        // Initialize state
        reset();
    }

    // ========================================================================
    // Public accessors
    // ========================================================================

    /**
     * Gets the problem dimension.
     * @return Problem dimension n
     */
    public int getDimension() {
        return n;
    }

    /**
     * Gets the number of L-BFGS corrections.
     * @return Number of corrections m
     */
    public int getCorrections() {
        return m;
    }

    /**
     * Checks if this workspace is compatible with the given problem dimensions.
     *
     * @param dimension Problem dimension
     * @param corrections Number of L-BFGS corrections
     * @return true if compatible (same n and m)
     */
    public boolean isCompatible(int dimension, int corrections) {
        return this.n == dimension && this.m == corrections;
    }

    /**
     * Ensures this workspace has sufficient capacity for the given dimensions.
     *
     * <p>Since L-BFGS-B workspace dimensions are fixed at construction time,
     * this method returns {@code true} if the current workspace can be reused
     * (i.e., dimensions match exactly), or {@code false} if a new workspace
     * must be allocated by the caller.</p>
     *
     * @param dimension   required problem dimension
     * @param corrections required number of L-BFGS corrections
     * @return true if this workspace is sufficient; false if reallocation is needed
     */
    public boolean ensureCapacity(int dimension, int corrections) {
        return this.n == dimension && this.m == corrections;
    }

    // ========================================================================
    // Reset method
    // ========================================================================

    /**
     * Resets the workspace for a new optimization run.
     *
     * <p>Resets all state without reallocating memory, allowing the workspace
     * to be reused across multiple optimization runs.</p>
     *
     * <p>Corresponds to Go's iterBFGS.reset() and iterCtx.clear() in base.go.</p>
     */
    public void reset() {

        // Reset BFGS state (matches Go iterBFGS.reset())
        this.col = 0;
        this.head = 0;
        this.tail = 0;
        this.theta = 1.0;
        this.updates = 0;
        this.updated = false;

        // Reset iteration context (matches Go iterCtx.clear())
        this.iter = 0;
        this.totalEval = 0;
        this.seg = 0;
        this.word = -1; // SOLUTION_UNKNOWN

        this.free = 0;
        this.active = 0;
        this.leave = 0;
        this.enter = 0;

        this.constrained = false;
        this.boxed = false;

        this.f = 0.0;
        this.fOld = 0.0;
        this.sbgNorm = 0.0;
        this.dSqrt = 0.0;
        this.dNorm = 0.0;

        // Reset line search context
        this.searchCtx.reset();

        // Reset error recovery state
        this.resetCount = 0;

        // Zero out work arrays
        java.util.Arrays.fill(buffer, 0.0);
        java.util.Arrays.fill(iBuffer, 0);
    }

    // ========================================================================
    // Buffer accessors (package-private for use by algorithm classes)
    // ========================================================================

    /** Gets the unified double buffer */
    double[] getBuffer() {
        return buffer;
    }

    /** Gets the unified integer buffer */
    int[] getIntBuffer() {
        return iBuffer;
    }

    // ========================================================================
    // Array offset accessors (package-private)
    // ========================================================================

    /** Gets offset for S matrix (n × m) */
    int getWsOffset() { return wsOffset; }

    /** Gets offset for Y matrix (n × m) */
    int getWyOffset() { return wyOffset; }

    /** Gets offset for S'Y matrix (m × m) */
    int getSyOffset() { return syOffset; }

    /** Gets offset for S'S matrix (m × m) */
    int getSsOffset() { return ssOffset; }

    /** Gets offset for Cholesky factor T (m × m) */
    int getWtOffset() { return wtOffset; }

    /** Gets offset for K matrix (4m²) */
    int getWnOffset() { return wnOffset; }

    /** Gets offset for inner product storage (4m²) */
    int getSndOffset() { return sndOffset; }

    /** Gets offset for Cauchy point z (n) */
    int getZOffset() { return zOffset; }

    /** Gets offset for reduced gradient r (n) */
    int getROffset() { return rOffset; }

    /** Gets offset for search direction d (n) */
    int getDOffset() { return dOffset; }

    /** Gets offset for breakpoint times t (n) */
    int getTOffset() { return tOffset; }

    /** Gets offset for safeguard point xp (n) */
    int getXpOffset() { return xpOffset; }

    /** Gets offset for work array wa (8m) */
    int getWaOffset() { return waOffset; }

    /** Gets offset for gradient g (n) */
    int getGOffset() { return gOffset; }

    /** Gets offset for diagInv array (m) */
    int getDiagInvOffset() { return diagInvOffset; }

    /** Gets offset for sqrtDiagInv array (m) */
    int getSqrtDiagInvOffset() { return sqrtDiagInvOffset; }

    /**
     * Gets offset for tempCol array (m).
     * This reuses wa[4m:5m] since formK() and CauchyPoint.compute() don't overlap.
     */
    int getTempColOffset() { return waOffset + 4 * m; }

    /** Gets offset for index array (2n) */
    int getIndexOffset() { return indexOffset; }

    /** Gets offset for where array (n) */
    int getWhereOffset() { return whereOffset; }

    // ========================================================================
    // BFGS state accessors (package-private)
    // ========================================================================

    boolean isUpdated() { return updated; }
    void setUpdated(boolean updated) { this.updated = updated; }

    int getUpdates() { return updates; }
    void setUpdates(int updates) { this.updates = updates; }

    int getCol() { return col; }
    void setCol(int col) { this.col = col; }

    int getHead() { return head; }
    void setHead(int head) { this.head = head; }

    int getTail() { return tail; }
    void setTail(int tail) { this.tail = tail; }

    double getTheta() { return theta; }
    void setTheta(double theta) { this.theta = theta; }

    // ========================================================================
    // Iteration context accessors (package-private)
    // ========================================================================

    int getIter() { return iter; }
    void setIter(int iter) { this.iter = iter; }

    int getTotalEval() { return totalEval; }
    void setTotalEval(int totalEval) { this.totalEval = totalEval; }
    void incrementTotalEval() { this.totalEval++; }

    int getSeg() { return seg; }
    void setSeg(int seg) { this.seg = seg; }

    int getWord() { return word; }
    void setWord(int word) { this.word = word; }

    int getFree() { return free; }
    void setFree(int free) { this.free = free; }

    int getActive() { return active; }
    void setActive(int active) { this.active = active; }

    int getLeave() { return leave; }
    void setLeave(int leave) { this.leave = leave; }

    int getEnter() { return enter; }
    void setEnter(int enter) { this.enter = enter; }

    boolean isConstrained() { return constrained; }
    void setConstrained(boolean constrained) { this.constrained = constrained; }

    boolean isBoxed() { return boxed; }
    void setBoxed(boolean boxed) { this.boxed = boxed; }

    double getF() { return f; }
    void setF(double f) { this.f = f; }

    double getFOld() { return fOld; }
    void setFOld(double fOld) { this.fOld = fOld; }

    double getSbgNorm() { return sbgNorm; }
    void setSbgNorm(double sbgNorm) { this.sbgNorm = sbgNorm; }

    double getDSqrt() { return dSqrt; }
    void setDSqrt(double dSqrt) { this.dSqrt = dSqrt; }

    double getDNorm() { return dNorm; }
    void setDNorm(double dNorm) { this.dNorm = dNorm; }

    // ========================================================================
    // Line search context accessors (package-private)
    // ========================================================================

    /** Gets the line search context */
    SearchCtx getSearchCtx() { return searchCtx; }

    // ========================================================================
    // Error recovery accessors (package-private)
    // ========================================================================

    int getResetCount() { return resetCount; }
    void setResetCount(int resetCount) { this.resetCount = resetCount; }
    void incrementResetCount() { this.resetCount++; }

    // ========================================================================
    // BFGS reset method (for error recovery)
    // ========================================================================

    /**
     * Resets the BFGS approximation to identity matrix.
     *
     * <p>This is called when numerical issues occur (singular matrices,
     * non-positive definite Cholesky, etc.) to restart with steepest descent.</p>
     *
     * <p>Corresponds to Go's iterBFGS.reset() in base.go.</p>
     */
    void resetBfgs() {
        this.col = 0;
        this.head = 0;
        this.tail = 0;
        this.theta = 1.0;
        this.updates = 0;
        this.updated = false;
    }

    // ========================================================================
    // Utility methods
    // ========================================================================

    /**
     * Calculates the required buffer size for the given dimensions.
     *
     * @param n Problem dimension
     * @param m Number of corrections
     * @return Required double buffer size
     */
    public static int calculateBufferSize(int n, int m) {
        // ws[n×m] + wy[n×m] + sy[m×m] + ss[m×m] + wt[m×m] + wn[4m²] + snd[4m²]
        // + z[n] + r[n] + d[n] + t[n] + xp[n] + wa[8m] + g[n]
        // + diagInv[m] + sqrtDiagInv[m]
        // Note: tempCol[m] reuses wa[4m:5m], so not counted separately
        return 2 * n * m +      // ws, wy
               3 * m * m +      // sy, ss, wt
               8 * m * m +      // wn, snd
               6 * n +          // z, r, d, t, xp, g
               8 * m +          // wa
               2 * m;           // diagInv, sqrtDiagInv (tempCol reuses wa)
    }

    /**
     * Calculates the required integer buffer size for the given dimensions.
     *
     * @param n Problem dimension
     * @return Required integer buffer size
     */
    public static int calculateIntBufferSize(int n) {
        // index[2n] + where[n]
        return 3 * n;
    }

    @Override
    public String toString() {
        return String.format("LBFGSBWorkspace[n=%d, m=%d, col=%d, iter=%d]",
                n, m, col, iter);
    }

    // ========================================================================
    // Line Search Context (encapsulates all line search state)
    // ========================================================================

    /**
     * Line search context containing all state for the Moré-Thuente line search.
     * <p>
     * This class encapsulates all variables needed during line search, similar to
     * SLSQP's {@link com.curioloop.numopt4j.optim.slsqp.SLSQPWorkspace.FindWork}.
     * </p>
     *
     * <h2>State Categories</h2>
     * <ul>
     *   <li><b>Work counters</b>: numEval, numBack - track evaluations and backtracks</li>
     *   <li><b>Directional derivatives</b>: gd, gdOld - current and previous g'd</li>
     *   <li><b>Step information</b>: stp, stpMin, stpMax - current and bounds</li>
     *   <li><b>Bracketing state</b>: bracket, stage - interval management</li>
     *   <li><b>Function values</b>: f0, g0, fx, fy, gx, gy - at various points</li>
     *   <li><b>Step history</b>: stx, sty, width0, width1 - interval endpoints</li>
     *   <li><b>Task status</b>: searchTask - current line search operation</li>
     * </ul>
     */
    public static final class SearchCtx {

        // ====================================================================
        // Work counters
        // ====================================================================

        /** Number of function evaluations in current line search */
        int numEval;

        /** Number of backtracking steps in current line search */
        int numBack;

        // ====================================================================
        // Directional derivatives
        // ====================================================================

        /** Directional derivative g'd */
        double gd;

        /** Previous directional derivative */
        double gdOld;

        // ====================================================================
        // Step information
        // ====================================================================

        /** Current step length */
        double stp;

        /** Minimum allowed step length */
        double stpMin;

        /** Maximum allowed step length */
        double stpMax;

        // ====================================================================
        // Bracketing state
        // ====================================================================

        /** Whether a bracket has been found */
        boolean bracket;

        /** Current stage of line search (1 or 2) */
        int stage;

        // ====================================================================
        // Function values at initial point
        // ====================================================================

        /** Function value at initial point */
        double f0;

        /** Gradient at initial point */
        double g0;

        // ====================================================================
        // Function values at bracket endpoints
        // ====================================================================

        /** Function value at endpoint x (best so far) */
        double fx;

        /** Gradient at endpoint x */
        double gx;

        /** Function value at endpoint y */
        double fy;

        /** Gradient at endpoint y */
        double gy;

        // ====================================================================
        // Step history
        // ====================================================================

        /** Step at endpoint x */
        double stx;

        /** Step at endpoint y */
        double sty;

        /** Previous interval width */
        double width0;

        /** Current interval width */
        double width1;

        // ====================================================================
        // Task status
        // ====================================================================

        /** Line search task status */
        int searchTask;

        // ====================================================================
        // Constructor and reset
        // ====================================================================

        /** Creates a new SearchCtx with default values */
        SearchCtx() {
            reset();
        }

        /** Resets all fields to default values */
        void reset() {
            numEval = 0;
            numBack = 0;
            gd = 0.0;
            gdOld = 0.0;
            stp = 0.0;
            stpMin = 0.0;
            stpMax = 0.0;
            bracket = false;
            stage = 0;
            f0 = 0.0;
            g0 = 0.0;
            fx = 0.0;
            gx = 0.0;
            fy = 0.0;
            gy = 0.0;
            stx = 0.0;
            sty = 0.0;
            width0 = 0.0;
            width1 = 0.0;
            searchTask = 0;
        }
    }
}
