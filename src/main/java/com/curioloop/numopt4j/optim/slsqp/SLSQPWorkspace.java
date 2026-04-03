/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * SLSQP (Sequential Least Squares Programming) workspace management.
 *
 * This file manages all working arrays and iteration state required by the SLSQP
 * main loop. A single contiguous double[] buffer is allocated at construction time
 * and partitioned into named sub-arrays via pre-computed offsets, avoiding repeated
 * allocation and improving cache locality.
 *
 * ── Buffer Layout (double[]) ───────────────────────────────────────────────
 *
 *   l[ll]        LDLᵀ factorisation of the Hessian approximation 𝐁ᵏ = 𝐋𝐃𝐋ᵀ
 *                  ll = (n+1)(n+2)/2,  packed lower-triangular column-major
 *   x0[n]        snapshot of 𝐱ᵏ at the start of each line search
 *   g[n+1]       gradient 𝜵𝒇(𝐱ᵏ) (element n+1 used for augmented QP)
 *   c[la]        constraint values 𝒄(𝐱ᵏ),  la = max(1, m)
 *   a[la×(n+1)]  constraint Jacobian 𝜵𝒄(𝐱ᵏ), column-major, row stride = la
 *   mu[la]       penalty multipliers 𝛒ᵏⱼ for the L1 merit function
 *   s[n+1]       search direction 𝐝 (element n+1 used for slack 𝛅)
 *   u[n+1]       lower-bound differences 𝒍 - 𝐱ᵏ (temporary / BFGS scratch)
 *   v[n+1]       upper-bound differences 𝒖 - 𝐱ᵏ (temporary / BFGS scratch)
 *   r[lr]        Lagrange multipliers 𝛌 returned by LSQ,  lr = m + 2n + 2
 *   w[...]       general floating-point workspace passed to LSQ / LSEI / NNLS
 *
 * ── Buffer Layout (int[]) ─────────────────────────────────────────────────
 *
 *   jw[...]      integer workspace passed to LSQ / LSEI / NNLS
 *
 * ── Iteration State ────────────────────────────────────────────────────────
 *
 *   iter         current iteration counter
 *   mode         current SQP mode (OK, evalFunc, evalGrad, …)
 *   alpha        current line-search step length 𝛂
 *   f0           𝒇(𝐱ᵏ) saved at the start of each iteration
 *   t0           𝞥(𝐱ᵏ;𝛒) = 𝒇(𝐱ᵏ) + ∑𝛒ᵏⱼ‖𝒄ⱼ(𝐱ᵏ)‖₁  (merit function at 𝐱ᵏ)
 *   acc          accuracy tolerance (= 10 × user-supplied accuracy)
 *   resetCount   number of consecutive BFGS resets (triggers relaxed convergence at 5)
 *   badQP        true when the original QP was infeasible (augmented QP was used)
 *   lineMode     current Brent's-method mode (FIND_NOOP / FIND_INIT / FIND_NEXT / FIND_CONV)
 *   fw           FindWork state for the exact line search
 */
package com.curioloop.numopt4j.optim.slsqp;

/**
 * Workspace for the SLSQP optimization algorithm.
 * <p>
 * This class manages all working arrays and state variables required by the SLSQP
 * algorithm. It uses a unified buffer allocation strategy to minimize memory
 * allocations and improve cache locality.
 * </p>
 *
 * <h2>Memory Layout</h2>
 * <pre>
 * double[] buffer layout:
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │ l[ll] │ x0[n] │ g[n+1] │ c[la] │ a[la×(n+1)] │ mu[la] │ s[n+1] │ ...       │
 * └─────────────────────────────────────────────────────────────────────────────┘
 * │       │       │        │       │             │        │        │
 * │ LDL^T │ init  │ grad   │ cons  │ Jacobian    │ mult   │ search │
 * │ factor│ pos   │        │ vals  │             │        │ dir    │
 *
 * Continued:
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │ u[n+1] │ v[n+1] │ r[lr] │ w[...]                                            │
 * └─────────────────────────────────────────────────────────────────────────────┘
 * │        │        │       │
 * │ lower  │ upper  │ mult  │ general workspace
 * │ diff   │ diff   │       │
 *
 * int[] iBuffer layout:
 * ┌─────────────────────────────────────────┐
 * │ jw[...]                                 │
 * └─────────────────────────────────────────┘
 * │                                         │
 * │ integer workspace                       │
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Create workspace for a problem with 5 variables, 2 equality and 3 inequality constraints
 * SLSQPWorkspace ws = new SLSQPWorkspace(5, 2, 3);
 * // Use workspace for optimization
 * optimizer.optimize(x, ws);
 *
 * // Reset and reuse for another optimization with same dimensions
 * ws.reset();
 * optimizer.optimize(x2, ws);
 * }</pre>
 *
 * @see SLSQPConstants
 * @see <a href="https://github.com/curioloop/numopt4j">numopt4j</a>
 */
public final class SLSQPWorkspace {

    // ========================================================================
    // Problem Dimensions
    // ========================================================================

    /** Problem dimension (number of variables) */
    private int n;

    /** Number of equality constraints */
    private int meq;

    /** Number of inequality constraints */
    private int mineq;

    /** Total number of constraints (meq + mineq) */
    private int m;

    /** n + 1 (used frequently in calculations) */
    private int n1;

    /** max(1, m) - leading dimension for constraint arrays */
    private int la;

    // ========================================================================
    // Unified Buffers
    // ========================================================================

    /** Unified double buffer for all floating-point arrays */
    private double[] buffer;

    /** Unified integer buffer for integer workspace */
    private int[] iBuffer;

    // ========================================================================
    // Double Array Offsets
    // ========================================================================

    /** Offset for LDL^T factor array l[(n+1)*(n+2)/2] */
    private int lOffset;

    /** Offset for initial position array x0[n] */
    private int x0Offset;

    /** Offset for gradient array g[n+1] */
    private int gOffset;

    /** Offset for constraint values array c[la] */
    private int cOffset;

    /** Offset for constraint Jacobian array a[la × (n+1)] (column-major) */
    private int aOffset;

    /** Offset for penalty multipliers array mu[la] */
    private int muOffset;

    /** Offset for search direction array s[n+1] */
    private int sOffset;

    /** Offset for lower bound difference array u[n+1] */
    private int uOffset;

    /** Offset for upper bound difference array v[n+1] */
    private int vOffset;

    /** Offset for multipliers array r[m + 2n + 2] */
    private int rOffset;

    /** Offset for general workspace array w[...] */
    private int wOffset;

    // ========================================================================
    // Integer Array Offsets
    // ========================================================================

    /** Offset for integer workspace array jw[...] */
    private int jwOffset;

    // ========================================================================
    // Array Sizes (for documentation and validation)
    // ========================================================================

    /** Size of l array: (n+1)*(n+2)/2 */
    private int lSize;

    /** Size of r array: m + 2n + 2 */
    private int rSize;

    /** Size of w array (general workspace) */
    private int wSize;

    /** Size of jw array (integer workspace) */
    private int jwSize;

    // ========================================================================
    // Iteration State Variables
    // ========================================================================

    /** Current iteration counter */
    private int iter;

    /** Current SQP mode */
    private int mode;

    /** Line search step length */
    private double alpha;

    /** Previous function value */
    private double f0;

    /** Merit function initial value (for line search) */
    private double t0;

    /** Accuracy tolerance */
    private double acc;

    /** BFGS reset counter */
    private int resetCount;

    /** Bad QP flag (constraints incompatible) */
    private boolean badQP;

    // ========================================================================
    // Exact Line Search State
    // ========================================================================

    /** Line search mode (FIND_NOOP, FIND_INIT, FIND_NEXT, FIND_CONV) */
    private int lineMode;

    /** Exact line search workspace */
    private final FindWork fw = new FindWork();

    // ========================================================================
    // Constructors
    // ========================================================================

    /**
     * Initializes (or re-initializes) the workspace for the given dimensions.
     * Computes all offsets and allocates buffers.
     */
    private void init(int n, int meq, int mineq) {
        // Validate inputs
        if (n <= 0) {
            throw new IllegalArgumentException("Problem dimension n must be positive: " + n);
        }
        if (meq < 0) {
            throw new IllegalArgumentException("Number of equality constraints meq must be non-negative: " + meq);
        }
        if (mineq < 0) {
            throw new IllegalArgumentException("Number of inequality constraints mineq must be non-negative: " + mineq);
        }

        // Store dimensions
        this.n = n;
        this.meq = meq;
        this.mineq = mineq;
        this.m = meq + mineq;
        this.n1 = n + 1;
        this.la = Math.max(1, m);

        // Calculate array sizes
        // l array: (n+1)*(n+2)/2 - matches Go's ll := (n+1)*(n+2)/2 in optimize.go
        this.lSize = (n + 1) * (n + 2) / 2;

        // r array: m + 2n + 2 - matches Go's lr := n+n+m+2 in optimize.go
        this.rSize = m + 2 * n + 2;

        // Total inequality constraints including bounds: mineq + 2*(n+1)
        int mineqTotal = mineq + 2 * n1;

        // General workspace size - matches C implementation in utils.c
        this.wSize = n1 * (n1 + 1) + meq * (n1 + 1) + mineqTotal * (n1 + 1) +
                     (n1 - meq + 1) * (mineqTotal + 2) + 2 * mineqTotal +
                     (n1 + mineqTotal) * (n1 - meq) + 2 * meq + n1 +
                     n1 * n / 2 + 2 * m + 3 * n + 3 * n1 + 1;

        // Integer workspace size: max(mineqTotal, n1 - mineqTotal)
        this.jwSize = Math.max(mineqTotal, n1 - mineqTotal);

        // Calculate offsets (sequential layout)
        int offset = 0;

        this.lOffset = offset;
        offset += lSize;

        this.x0Offset = offset;
        offset += n;

        this.gOffset = offset;
        offset += n1;

        this.cOffset = offset;
        offset += la;

        this.aOffset = offset;
        offset += la * n1;

        this.muOffset = offset;
        offset += la;

        this.sOffset = offset;
        offset += n1;

        this.uOffset = offset;
        offset += n1;

        this.vOffset = offset;
        offset += n1;

        this.rOffset = offset;
        offset += rSize;

        this.wOffset = offset;
        offset += wSize;

        // Allocate unified double buffer
        this.buffer = new double[offset];

        // Integer buffer offset (always 0 since it's a separate array)
        this.jwOffset = 0;

        // Allocate integer buffer
        this.iBuffer = new int[jwSize];
    }

    /**
     * Ensures this workspace has sufficient capacity for the given dimensions.
     * Reallocates if needed (any dimension differs), then resets state.
     *
     * @param n      problem dimension (number of variables), must be positive
     * @param meq    number of equality constraints, must be non-negative
     * @param mineq  number of inequality constraints, must be non-negative
     */
    public void ensure(int n, int meq, int mineq) {
        if (n <= 0) {
            throw new IllegalArgumentException("Problem dimension n must be positive: " + n);
        }
        if (meq < 0) {
            throw new IllegalArgumentException("Number of equality constraints meq must be non-negative: " + meq);
        }
        if (mineq < 0) {
            throw new IllegalArgumentException("Number of inequality constraints mineq must be non-negative: " + mineq);
        }
        if (n > getN() || meq != getMeq() || mineq != getMineq()) {
            init(n, meq, mineq);
        }
        reset();
    }

    // ========================================================================
    // Reset Method
    // ========================================================================

    /**
     * Resets the workspace state for a new optimization run.
     * <p>
     * Resets all state variables without reallocating memory.
     * The allocated arrays are reused across multiple optimization iterations.
     * </p>
     */
    public void reset() {
        // Reset iteration state
        this.iter = 0;
        this.mode = SLSQPConstants.MODE_OK;
        this.alpha = 1.0;
        this.f0 = 0.0;
        this.t0 = 0.0;
        this.acc = 0.0;
        this.resetCount = 0;
        this.badQP = false;

        // Reset line search state
        this.lineMode = SLSQPConstants.FIND_NOOP;
        if (this.fw != null) this.fw.reset();

        // Note: We do NOT zero the arrays here for performance.
        // The algorithm initializes the arrays as needed.
    }

    /**
     * Checks if this workspace is compatible with the given problem dimensions.
     * <p>
     * A workspace is compatible if it was allocated for the same dimensions.
     * This allows workspace reuse for multiple optimizations with the same
     * problem structure.
     * </p>
     *
     * @param n      problem dimension
     * @param meq    number of equality constraints
     * @param mineq  number of inequality constraints
     * @return true if this workspace can be used for the given dimensions
     */
    public boolean isCompatible(int n, int meq, int mineq) {
        return this.n == n && this.meq == meq && this.mineq == mineq;
    }

    // ========================================================================
    // Dimension Accessors
    // ========================================================================

    /**
     * Returns the problem dimension (number of variables).
     *
     * @return problem dimension
     */
    public int getN() {
        return n;
    }

    /**
     * Returns the number of equality constraints.
     *
     * @return number of equality constraints
     */
    public int getMeq() {
        return meq;
    }

    /**
     * Returns the number of inequality constraints.
     *
     * @return number of inequality constraints
     */
    public int getMineq() {
        return mineq;
    }

    /**
     * Returns the total number of constraints (meq + mineq).
     *
     * @return total number of constraints
     */
    public int getM() {
        return m;
    }

    /**
     * Returns n + 1 (frequently used in calculations).
     *
     * @return n + 1
     */
    public int getN1() {
        return n1;
    }

    /**
     * Returns the leading dimension for constraint arrays: max(1, m).
     *
     * @return leading dimension for constraint arrays
     */
    public int getLa() {
        return la;
    }

    // ========================================================================
    // Buffer Accessors
    // ========================================================================

    /**
     * Returns the unified double buffer.
     * <p>
     * This provides direct access to the underlying buffer for efficient
     * array operations. Use the offset methods to access specific arrays.
     * </p>
     *
     * @return the unified double buffer
     */
    public double[] getBuffer() {
        return buffer;
    }

    /**
     * Returns the unified integer buffer.
     *
     * @return the unified integer buffer
     */
    public int[] getIBuffer() {
        return iBuffer;
    }

    // ========================================================================
    // Offset Accessors
    // ========================================================================

    /**
     * Returns the offset for the LDL^T factor array l.
     * <p>
     * Array size: (n+1)*(n+2)/2
     * </p>
     *
     * @return offset for l array
     */
    public int getLOffset() {
        return lOffset;
    }

    /**
     * Returns the offset for the initial position array x0.
     * <p>
     * Array size: n
     * </p>
     *
     * @return offset for x0 array
     */
    public int getX0Offset() {
        return x0Offset;
    }

    /**
     * Returns the offset for the gradient array g.
     * <p>
     * Array size: n+1
     * </p>
     *
     * @return offset for g array
     */
    public int getGOffset() {
        return gOffset;
    }

    /**
     * Returns the offset for the constraint values array c.
     * <p>
     * Array size: max(1, m)
     * </p>
     *
     * @return offset for c array
     */
    public int getCOffset() {
        return cOffset;
    }

    /**
     * Returns the offset for the constraint Jacobian array a.
     * <p>
     * Array size: max(1, m) × (n+1), stored in column-major order.
     * Element a[i,j] is at buffer[aOffset + i + la * j].
     * </p>
     *
     * @return offset for a array
     */
    public int getAOffset() {
        return aOffset;
    }

    /**
     * Returns the offset for the penalty multipliers array mu.
     * <p>
     * Array size: max(1, m)
     * </p>
     *
     * @return offset for mu array
     */
    public int getMuOffset() {
        return muOffset;
    }

    /**
     * Returns the offset for the search direction array s.
     * <p>
     * Array size: n+1
     * </p>
     *
     * @return offset for s array
     */
    public int getSOffset() {
        return sOffset;
    }

    /**
     * Returns the offset for the lower bound difference array u.
     * <p>
     * Array size: n+1
     * </p>
     *
     * @return offset for u array
     */
    public int getUOffset() {
        return uOffset;
    }

    /**
     * Returns the offset for the upper bound difference array v.
     * <p>
     * Array size: n+1
     * </p>
     *
     * @return offset for v array
     */
    public int getVOffset() {
        return vOffset;
    }

    /**
     * Returns the offset for the multipliers array r.
     * <p>
     * Array size: m + 2n + 2
     * </p>
     *
     * @return offset for r array
     */
    public int getROffset() {
        return rOffset;
    }

    /**
     * Returns the offset for the general workspace array w.
     *
     * @return offset for w array
     */
    public int getWOffset() {
        return wOffset;
    }

    /**
     * Returns the offset for the integer workspace array jw.
     *
     * @return offset for jw array
     */
    public int getJwOffset() {
        return jwOffset;
    }

    // ========================================================================
    // Array Size Accessors
    // ========================================================================

    /**
     * Returns the size of the l array.
     *
     * @return size of l array: (n+1)*(n+2)/2
     */
    public int getLSize() {
        return lSize;
    }

    /**
     * Returns the size of the r array.
     *
     * @return size of r array: m + 2n + 2
     */
    public int getRSize() {
        return rSize;
    }

    /**
     * Returns the size of the w array.
     *
     * @return size of w array
     */
    public int getWSize() {
        return wSize;
    }

    /**
     * Returns the size of the jw array.
     *
     * @return size of jw array
     */
    public int getJwSize() {
        return jwSize;
    }

    // ========================================================================
    // State Variable Accessors and Mutators
    // ========================================================================

    /**
     * Returns the current iteration counter.
     *
     * @return current iteration
     */
    public int getIter() {
        return iter;
    }

    /**
     * Sets the current iteration counter.
     *
     * @param iter iteration counter
     */
    public void setIter(int iter) {
        this.iter = iter;
    }

    /**
     * Returns the current SQP mode.
     *
     * @return current mode
     */
    public int getMode() {
        return mode;
    }

    /**
     * Sets the current SQP mode.
     *
     * @param mode SQP mode
     */
    public void setMode(int mode) {
        this.mode = mode;
    }

    /**
     * Returns the line search step length.
     *
     * @return step length alpha
     */
    public double getAlpha() {
        return alpha;
    }

    /**
     * Sets the line search step length.
     *
     * @param alpha step length
     */
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    /**
     * Returns the previous function value.
     *
     * @return previous function value f0
     */
    public double getF0() {
        return f0;
    }

    /**
     * Sets the previous function value.
     *
     * @param f0 previous function value
     */
    public void setF0(double f0) {
        this.f0 = f0;
    }

    /**
     * Returns the merit function initial value.
     *
     * @return merit function initial value t0
     */
    public double getT0() {
        return t0;
    }

    /**
     * Sets the merit function initial value.
     *
     * @param t0 merit function initial value
     */
    public void setT0(double t0) {
        this.t0 = t0;
    }

    /**
     * Returns the accuracy tolerance.
     *
     * @return accuracy tolerance
     */
    public double getAcc() {
        return acc;
    }

    /**
     * Sets the accuracy tolerance.
     *
     * @param acc accuracy tolerance
     */
    public void setAcc(double acc) {
        this.acc = acc;
    }

    /**
     * Returns the BFGS reset counter.
     *
     * @return reset counter
     */
    public int getResetCount() {
        return resetCount;
    }

    /**
     * Sets the BFGS reset counter.
     *
     * @param resetCount reset counter
     */
    public void setResetCount(int resetCount) {
        this.resetCount = resetCount;
    }

    /**
     * Increments the BFGS reset counter.
     *
     * @return the new reset count
     */
    public int incrementResetCount() {
        return ++resetCount;
    }

    /**
     * Returns whether the QP subproblem had incompatible constraints.
     *
     * @return true if constraints were incompatible
     */
    public boolean isBadQP() {
        return badQP;
    }

    /**
     * Sets the bad QP flag.
     *
     * @param badQP true if constraints are incompatible
     */
    public void setBadQP(boolean badQP) {
        this.badQP = badQP;
    }

    // ========================================================================
    // Line Search State Accessors
    // ========================================================================

    /**
     * Returns the line search mode.
     *
     * @return line search mode (FIND_NOOP, FIND_INIT, FIND_NEXT, or FIND_CONV)
     */
    public int getLineMode() {
        return lineMode;
    }

    /**
     * Sets the line search mode.
     *
     * @param lineMode line search mode
     */
    public void setLineMode(int lineMode) {
        this.lineMode = lineMode;
    }

    /**
     * Returns the exact line search workspace.
     *
     * @return FindWork instance for exact line search
     */
    public FindWork getFindWork() {
        return fw;
    }

    // ========================================================================
    // FindWork Inner Class
    // ========================================================================

    /**
     * Workspace for Brent's exact line search (golden section + parabolic interpolation).
     * <p>
     * Stores the complete state of the reverse-communication minimisation of
     * 𝞿(𝛂) = 𝞥(𝐱ᵏ + 𝛂𝐝) over the interval [a, b].
     * </p>
     *
     * <h2>Bracketing Interval</h2>
     * <p>
     * [a, b] is maintained so that the minimum lies inside it.
     * After each accepted step the interval shrinks monotonically.
     * </p>
     *
     * <h2>Key Points</h2>
     * <ul>
     *   <li>x — current best point: f(x) = fx is the lowest value seen so far</li>
     *   <li>w — second best point: f(w) = fw ≤ f(v)</li>
     *   <li>v — previous value of w: f(v) = fv</li>
     *   <li>u — next evaluation point returned to the caller</li>
     * </ul>
     *
     * <h2>Step Variables</h2>
     * <ul>
     *   <li>d — current step: u = x + d</li>
     *   <li>e — step from the iteration before last (used to decide whether
     *       the parabolic step is acceptable: |p| &lt; ½|q·r| requires |e| &gt; tol₁)</li>
     *   <li>p, q — numerator and denominator of the parabolic step d = p/q</li>
     *   <li>r — saved value of e before the parabola was fitted</li>
     * </ul>
     *
     * <h2>Convergence Test</h2>
     * <pre>
     *   tol₁ = √ε · |x| + tol
     *   tol₂ = 2 · tol₁
     *   converged when |x - m| ≤ tol₂ - ½(b - a),  m = ½(a + b)
     * </pre>
     *
     * @see LineSearch#findMin
     * @see <a href="https://en.wikipedia.org/wiki/Brent%27s_method">Brent's method</a>
     */
    public static final class FindWork {

        // ====================================================================
        // Search Interval
        // ====================================================================

        /** Lower bound of search interval */
        public double a;

        /** Upper bound of search interval */
        public double b;

        // ====================================================================
        // Step Information
        // ====================================================================

        /** Current step size */
        public double d;

        /** Previous step size (for parabolic interpolation check) */
        public double e;

        // ====================================================================
        // Parabolic Interpolation Parameters
        // ====================================================================

        /** Numerator of parabolic step */
        public double p;

        /** Denominator of parabolic step */
        public double q;

        /** Previous e value (for interpolation validity check) */
        public double r;

        // ====================================================================
        // Key Points
        // ====================================================================

        /** Next evaluation point */
        public double u;

        /** Previous value of w */
        public double v;

        /** Second best point */
        public double w;

        /** Current best point (lowest function value) */
        public double x;

        // ====================================================================
        // Midpoint
        // ====================================================================

        /** Midpoint of interval: (a + b) / 2 */
        public double m;

        // ====================================================================
        // Function Values at Key Points
        // ====================================================================

        /** Function value at u */
        public double fu;

        /** Function value at v */
        public double fv;

        /** Function value at w */
        public double fw;

        /** Function value at x (current best) */
        public double fx;

        // ====================================================================
        // Convergence Tolerances
        // ====================================================================

        /** Primary tolerance: sqrt(eps) * |x| + tol */
        public double tol1;

        /** Secondary tolerance: 2 * tol1 */
        public double tol2;

        // ====================================================================
        // Constructor
        // ====================================================================

        /**
         * Creates a new FindWork instance with default values.
         */
        public FindWork() {
            reset();
        }

        // ====================================================================
        // Reset Method
        // ====================================================================

        /**
         * Resets all fields to their initial values.
         */
        public void reset() {
            a = 0.0;
            b = 0.0;
            d = 0.0;
            e = 0.0;
            p = 0.0;
            q = 0.0;
            r = 0.0;
            u = 0.0;
            v = 0.0;
            w = 0.0;
            x = 0.0;
            m = 0.0;
            fu = 0.0;
            fv = 0.0;
            fw = 0.0;
            fx = 0.0;
            tol1 = 0.0;
            tol2 = 0.0;
        }
    }
}
