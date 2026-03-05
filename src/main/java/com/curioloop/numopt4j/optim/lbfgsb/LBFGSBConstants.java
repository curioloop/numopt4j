/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * L-BFGS-B Constants - Pure Java implementation.
 *
 * This interface defines all constants used by the L-BFGS-B optimization algorithm.
 * Constants are organized into the following categories:
 * - Variable status constants (varWhere)
 * - Bound type constants (bndHint)
 * - Solution status constants
 * - Line search constants
 * - Minpack search task constants
 * - Error codes
 *
 * Reference: Go implementation in lbfgsb/base.go
 * Reference: C implementation in lbfgsb/lbfgsb.c and optimizer.h
 *
 */
package com.curioloop.numopt4j.optim.lbfgsb;

/**
 * Constants for the L-BFGS-B optimization algorithm.
 *
 * <p>This interface provides all numerical constants used throughout the L-BFGS-B
 * implementation. Classes that need these constants can implement this interface
 * or access them directly via {@code LBFGSBConstants.CONSTANT_NAME}.</p>
 *
 * <h2>Variable Status Constants</h2>
 * <p>These constants track the status of each variable during optimization,
 * matching Go's {@code varWhere} type in base.go:</p>
 * <ul>
 *   <li>{@link #VAR_NOT_MOVE} (-3): Variable is free with bounds but did not move</li>
 *   <li>{@link #VAR_UNBOUND} (-1): Variable has no bounds (always free)</li>
 *   <li>{@link #VAR_FREE} (0): Variable is free and has bounds</li>
 *   <li>{@link #VAR_AT_LOWER} (1): Variable is fixed at lower bound (u ≠ l)</li>
 *   <li>{@link #VAR_AT_UPPER} (2): Variable is fixed at upper bound (u ≠ l)</li>
 *   <li>{@link #VAR_FIXED} (3): Variable is always fixed (u = l)</li>
 * </ul>
 *
 * <h2>Bound Type Constants</h2>
 * <p>These constants specify the type of bounds on each variable,
 * matching Go's {@code bndHint} type in base.go:</p>
 * <ul>
 *   <li>{@link #BOUND_NONE} (0): No bounds on variable</li>
 *   <li>{@link #BOUND_LOWER} (1): Only lower bound</li>
 *   <li>{@link #BOUND_BOTH} (2): Both lower and upper bounds</li>
 *   <li>{@link #BOUND_UPPER} (3): Only upper bound</li>
 * </ul>
 *
 * @see LBFGSBWorkspace
 */
public interface LBFGSBConstants {

    // ========================================================================
    // Numerical Constants
    // ========================================================================

    /** Machine epsilon (smallest x such that 1.0 + x != 1.0) */
    double EPS = Math.ulp(1.0);

    /** Square root of machine epsilon */
    double SQRT_EPS = Math.sqrt(EPS);

    // ========================================================================
    // Variable Status Constants (varWhere)
    // Matches Go varWhere type in base.go
    // Matches C VAR_* constants in lbfgsb.c
    // ========================================================================

    /**
     * Variable is free with bounds but did not move.
     * <p>Go: varNotMove = -3</p>
     * <p>C: VAR_NOT_MOVE = -3</p>
     */
    int VAR_NOT_MOVE = -3;

    /**
     * Variable has no bounds (always free).
     * <p>Go: varUnbound = -1</p>
     * <p>C: VAR_UNBOUND = -1</p>
     */
    int VAR_UNBOUND = -1;

    /**
     * Variable is free and has bounds.
     * <p>Go: varFree = 0</p>
     * <p>C: VAR_FREE = 0</p>
     */
    int VAR_FREE = 0;

    /**
     * Variable is fixed at lower bound (u ≠ l).
     * <p>Go: varAtLB = 1</p>
     * <p>C: VAR_AT_LOWER = 1</p>
     */
    int VAR_AT_LOWER = 1;

    /**
     * Variable is fixed at upper bound (u ≠ l).
     * <p>Go: varAtUB = 2</p>
     * <p>C: VAR_AT_UPPER = 2</p>
     */
    int VAR_AT_UPPER = 2;

    /**
     * Variable is always fixed (u = l).
     * <p>Go: varFixed = 3</p>
     * <p>C: VAR_FIXED = 3</p>
     */
    int VAR_FIXED = 3;

    // ========================================================================
    // Bound Type Constants (bndHint)
    // Matches Go bndHint type in base.go
    // Matches C BoundType enum in optimizer.h
    // ========================================================================

    /**
     * No bounds on variable.
     * <p>Go: bndNo = 0</p>
     * <p>C: BOUND_NONE = 0</p>
     */
    int BOUND_NONE = 0;

    /**
     * Only lower bound.
     * <p>Go: bndLow = 1</p>
     * <p>C: BOUND_LOWER = 1</p>
     */
    int BOUND_LOWER = 1;

    /**
     * Both lower and upper bounds.
     * <p>Go: bndBoth = 2</p>
     * <p>C: BOUND_BOTH = 2</p>
     */
    int BOUND_BOTH = 2;

    /**
     * Only upper bound.
     * <p>Go: bndUp = 3</p>
     * <p>C: BOUND_UPPER = 3</p>
     */
    int BOUND_UPPER = 3;

    // ========================================================================
    // Solution Status Constants
    // Matches Go solutionUnknown/solutionWithinBox/solutionBeyondBox in base.go
    // Matches C SOLUTION_* constants in lbfgsb.c
    // ========================================================================

    /**
     * Subspace minimization not performed.
     * <p>Go: solutionUnknown = -1</p>
     * <p>C: SOLUTION_UNKNOWN = -1</p>
     */
    int SOLUTION_UNKNOWN = -1;

    /**
     * Unconstrained minimizer is within bounds.
     * <p>Go: solutionWithinBox = 0</p>
     * <p>C: SOLUTION_WITHIN_BOX = 0</p>
     */
    int SOLUTION_WITHIN_BOX = 0;

    /**
     * Minimizer hit a bound constraint.
     * <p>Go: solutionBeyondBox = 1</p>
     * <p>C: SOLUTION_BEYOND_BOX = 1</p>
     */
    int SOLUTION_BEYOND_BOX = 1;

    // ========================================================================
    // Line Search Tolerance Constants
    // Matches Go linesearch.go constants
    // Matches C SEARCH_* constants in lbfgsb.c
    // ========================================================================

    /**
     * Armijo condition parameter (sufficient decrease).
     * <p>Default value for the sufficient decrease condition in line search.</p>
     * <p>Go: searchAlpha = 1.0e-3</p>
     * <p>C: SEARCH_ALPHA = 1.0e-3</p>
     */
    double SEARCH_ALPHA = 1.0e-3;

    /**
     * Curvature condition parameter.
     * <p>Default value for the curvature condition in line search.</p>
     * <p>Go: searchBeta = 0.9</p>
     * <p>C: SEARCH_BETA = 0.9</p>
     */
    double SEARCH_BETA = 0.9;

    /**
     * Relative tolerance for line search.
     * <p>Go: searchEps = 0.1</p>
     * <p>C: SEARCH_EPS = 0.1</p>
     */
    double SEARCH_EPS = 0.1;

    /**
     * Unbounded step length marker.
     * <p>Used when there is no upper bound on the step length.</p>
     * <p>Go: searchNoBnd = 1.0e+10</p>
     * <p>C: SEARCH_NO_BND = 1.0e+10</p>
     */
    double SEARCH_NO_BND = 1.0e+10;

    /**
     * Maximum backtracking iterations before exit.
     * <p>C: SEARCH_BACK_EXIT = 20</p>
     */
    int SEARCH_BACK_EXIT = 20;

    /**
     * Threshold for "too many searches" warning.
     * <p>C: SEARCH_BACK_SLOW = 10</p>
     */
    int SEARCH_BACK_SLOW = 10;

    // ========================================================================
    // Minpack Search Task Constants
    // Matches Go SearchTask type in base.go
    // Matches C MINPACK_SEARCH_* constants in lbfgsb.c
    // ========================================================================

    /**
     * Line search starting state.
     * <p>Go: searchStart = 0</p>
     * <p>C: MINPACK_SEARCH_START = 0</p>
     */
    int SEARCH_START = 0;

    /**
     * Line search converged.
     * <p>Go: searchConv = 1 << 5 = 32</p>
     * <p>C: MINPACK_SEARCH_CONV = 1 << 5</p>
     */
    int SEARCH_CONV = 1 << 5;

    /**
     * Need to compute function and gradient.
     * <p>Go: searchFG = 1 << 6 = 64</p>
     * <p>C: MINPACK_SEARCH_FG = 1 << 6</p>
     */
    int SEARCH_FG = 1 << 6;

    /**
     * Line search error.
     * <p>Go: searchError = 1 << 7 = 128</p>
     * <p>C: MINPACK_SEARCH_ERROR = 1 << 7</p>
     */
    int SEARCH_ERROR = 1 << 7;

    /**
     * Line search warning.
     * <p>Go: searchWarn = 1 << 8 = 256</p>
     * <p>C: MINPACK_SEARCH_WARN = 1 << 8</p>
     */
    int SEARCH_WARN = 1 << 8;

    // ========================================================================
    // Error Codes
    // Matches Go errInfo type in base.go
    // Matches C ERR_* constants in optimizer.h
    // ========================================================================

    /**
     * No error.
     * <p>Go: ok = 0</p>
     * <p>C: ERR_NONE = 0</p>
     */
    int ERR_NONE = 0;

    /**
     * First K matrix is not positive definite (Cholesky failed).
     * <p>Go: errNotPosDef1stK = -1</p>
     * <p>C: ERR_NOT_POS_DEF = -1</p>
     */
    int ERR_NOT_POS_DEF_1ST_K = -1;

    /**
     * Second K matrix is not positive definite.
     * <p>Go: errNotPosDef2ndK = -2</p>
     * <p>C: ERR_NOT_POS_DEF_2ND_K = -2</p>
     */
    int ERR_NOT_POS_DEF_2ND_K = -2;

    /**
     * T matrix is not positive definite.
     * <p>Go: errNotPosDefT = -3</p>
     * <p>C: ERR_NOT_POS_DEF_T = -3</p>
     */
    int ERR_NOT_POS_DEF_T = -3;

    /**
     * Invalid derivative (not a descent direction).
     * <p>Go: errDerivative = -4</p>
     * <p>C: ERR_DERIVATIVE = -4</p>
     */
    int ERR_DERIVATIVE = -4;

    /**
     * Triangular matrix is singular.
     * <p>Go: errSingularTriangular = -5</p>
     * <p>C: ERR_SINGULAR_TRIANGULAR = -5</p>
     */
    int ERR_SINGULAR_TRIANGULAR = -5;

    /**
     * Line search failed to find valid step.
     * <p>Go: errLineSearchFailed = -6</p>
     * <p>C: ERR_LINE_SEARCH_FAILED = -6</p>
     */
    int ERR_LINE_SEARCH_FAILED = -6;

    /**
     * Line search tolerance error.
     * <p>Go: errLineSearchTol = -7</p>
     * <p>C: ERR_LINE_SEARCH_TOL = -7</p>
     */
    int ERR_LINE_SEARCH_TOL = -7;

    /**
     * Too many BFGS matrix resets.
     * <p>C: ERR_TOO_MANY_RESETS = -8</p>
     */
    int ERR_TOO_MANY_RESETS = -8;

    // ========================================================================
    // Warning Codes
    // Matches Go warnInfo type in base.go
    // ========================================================================

    /**
     * Warning: restart loop detected.
     * <p>Go: warnRestartLoop = 1</p>
     */
    int WARN_RESTART_LOOP = 1;

    /**
     * Warning: too many line search iterations.
     * <p>Go: warnTooManySearch = 2</p>
     */
    int WARN_TOO_MANY_SEARCH = 2;

    // ========================================================================
    // Algorithm Limits
    // ========================================================================

    /**
     * Maximum number of BFGS resets before termination.
     * <p>Used for error recovery when numerical issues occur.</p>
     */
    int MAX_BFGS_RESETS = 5;

    // ========================================================================
    // Iteration Task Constants
    // Matches Go iterTask type in base.go
    // ========================================================================

    /**
     * Continue iteration loop.
     * <p>Go: iterLoop = 0</p>
     */
    int ITER_LOOP = 0;

    /**
     * Iteration converged.
     * <p>Go: iterConv = 1 << 4 = 16</p>
     */
    int ITER_CONV = 1 << 4;

    /**
     * Iteration stopped.
     * <p>Go: iterStop = 1 << 5 = 32</p>
     */
    int ITER_STOP = 1 << 5;

    // ========================================================================
    // Convergence Status Constants
    // Matches Go ConvGradProgNorm, ConvEnoughAccuracy, etc. in base.go
    // ========================================================================

    /**
     * Converged: projected gradient norm below tolerance.
     * <p>Go: ConvGradProgNorm = iterConv | 1 = 17</p>
     */
    int CONV_GRAD_PROG_NORM = ITER_CONV | 1;

    /**
     * Converged: function value accuracy achieved.
     * <p>Go: ConvEnoughAccuracy = iterConv | 2 = 18</p>
     */
    int CONV_ENOUGH_ACCURACY = ITER_CONV | 2;

    /**
     * Stopped: abnormal line search.
     * <p>Go: StopAbnormalSearch = iterStop | 3 = 35</p>
     */
    int STOP_ABNORMAL_SEARCH = ITER_STOP | 3;

    /**
     * Stopped: evaluation panic (NaN/Inf).
     * <p>Go: HaltEvalPanic = iterStop | 4 = 36</p>
     */
    int HALT_EVAL_PANIC = ITER_STOP | 4;

    /**
     * Stopped: iteration limit exceeded.
     * <p>Go: OverIterLimit = iterStop | 5 = 37</p>
     */
    int OVER_ITER_LIMIT = ITER_STOP | 5;

    /**
     * Stopped: evaluation limit exceeded.
     * <p>Go: OverEvalLimit = iterStop | 6 = 38</p>
     */
    int OVER_EVAL_LIMIT = ITER_STOP | 6;

    /**
     * Stopped: time limit exceeded.
     * <p>Go: OverTimeLimit = iterStop | 7 = 39</p>
     */
    int OVER_TIME_LIMIT = ITER_STOP | 7;

    /**
     * Stopped: gradient threshold exceeded.
     * <p>Go: OverGradThresh = iterStop | 8 = 40</p>
     */
    int OVER_GRAD_THRESH = ITER_STOP | 8;
}
