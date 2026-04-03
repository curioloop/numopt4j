/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * SLSQP (Sequential Least Squares Programming) core optimization loop.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Algorithm Overview
 * ════════════════════════════════════════════════════════════════════════════
 *
 * SLSQP solves general constrained nonlinear optimization problems (NLP):
 *
 *   minimize   f(x)
 *   subject to cⱼ(x) = 0    (j = 1 ··· mₑ)
 *              cⱼ(x) ≥ 0    (j = mₑ+1 ··· m)
 *              lᵢ ≤ xᵢ ≤ uᵢ (i = 1 ··· n)
 *
 * SQP decomposes NLP into a series of QP sub-problems, each of which solves
 * a descent direction d and step length α, ensuring f(x + αd) < f(x) and
 * the updated x satisfies the constraints.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Direction (QP Sub-Problem)
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The Lagrangian function is ℒ(x,λ) = f(x) - ∑λⱼcⱼ(x).
 * A quadratic approximation of ℒ at xᵏ gives the QP sub-problem:
 *
 *   minimize   ½ dᵀBᵏd + ∇f(xᵏ)d
 *   subject to ∇cⱼ(xᵏ)d + cⱼ(xᵏ) = 0    (j = 1 ··· mₑ)
 *              ∇cⱼ(xᵏ)d + cⱼ(xᵏ) ≥ 0    (j = mₑ+1 ··· m)
 *
 * where Bᵏ ≈ ∇²ℒ(xᵏ,λᵏ) is the symmetric Hessian approximation.
 * The descent direction d is determined by the above problem.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Inconsistent Constraints
 * ════════════════════════════════════════════════════════════════════════════
 *
 * When the QP constraints become inconsistent, an augmented QP relaxation
 * with slack variable δ is introduced:
 *
 *   minimize   ½ dᵀBᵏd + ∇f(xᵏ)d + ½ρδ²
 *   subject to ∇cⱼ(xᵏ)d + cⱼ(xᵏ) + δcⱼ(xᵏ) = 0    (j = 1 ··· mₑ)
 *              ∇cⱼ(xᵏ)d + cⱼ(xᵏ) + δζⱼcⱼ(xᵏ) ≥ 0   (j = mₑ+1 ··· m)
 *              0 ≤ δ ≤ 1
 *
 * where 10² ≤ ρ ≤ 10⁷ penalizes constraint violation, and
 * ζⱼ = 0 if cⱼ(xᵏ) > 0, ζⱼ = 1 if cⱼ(xᵏ) ≤ 0.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Step (Line Search with Merit Function)
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The step length α minimizes the L1 merit function:
 *
 *   φ(x;ρ) = f(x) + ∑ρⱼ‖cⱼ(x)‖₁
 *
 * where ‖cⱼ(x)‖₁ = |cⱼ(x)| for equality, max(0,-cⱼ(x)) for inequality.
 *
 * Penalty parameters are updated iteratively:
 *   ρⱼᵏ⁺¹ = max[ ½(ρⱼᵏ + |λⱼ|), |λⱼ| ]
 *
 * The directional derivative of the merit function along d is:
 *   ∇φ(d;xᵏ,ρᵏ) = ∇f(xᵏ)ᵀd - (1-δ)∑ρᵏⱼ‖cⱼ(xᵏ)‖₁
 *
 * Line search uses the Armijo condition:
 *   φ(xᵏ + αd;λ,ρ) - φ(xᵏ;λ,ρ) < η · α · ∇φ(d;xᵏ,ρᵏ)   (0 < η < 0.5)
 *
 * ════════════════════════════════════════════════════════════════════════════
 * BFGS Update (LDLᵀ Factorization)
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The Hessian approximation is stored as B = LDLᵀ where L is lower triangular
 * with unit diagonal and D is diagonal with positive elements.
 *
 * Modified BFGS update:
 *   Bᵏ⁺¹ = Bᵏ + qqᵀ/qᵀs - Bᵏssᵀ Bᵏ/sᵀBᵏs
 *   s = xᵏ⁺¹ - xᵏ
 *   q = θη + (1-θ)Bᵏs
 *   η = ∇ℒ(xᵏ⁺¹,λᵏ) - ∇ℒ(xᵏ,λᵏ)
 *   θ = 1                              if sᵀη ≥ ⅕ sᵀBᵏs
 *   θ = ⅘ sᵀBᵏs / (sᵀBᵏs - sᵀη)     otherwise
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Convergence Criteria
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Checked after solving the QP sub-problem:
 *   C𝑣𝑖𝑜 = ∑‖cⱼ(xᵏ)‖₁                              (feasibility)
 *   C𝑜𝑝𝑡 = |∇f(xᵏ)ᵀd| + |λᵏ|ᵀ × ‖c(xᵏ)‖₁          (optimality)
 *   C𝑠𝑡𝑝 = ‖d‖₂                                     (step length)
 *
 * Checked after line search:
 *   Ĉ𝑣𝑖𝑜 = ∑‖cⱼ(xᵏ + αd)‖₁
 *   Ĉ𝑜𝑝𝑡 = |f(xᵏ + αd) - f(xᵏ)|
 *   Ĉ𝑠𝑡𝑝 = ‖d‖₂
 *
 * Reference: Dieter Kraft, "A software package for sequential quadratic
 * programming". DFVLR-FB 88-28, 1988
 */
package com.curioloop.numopt4j.optim.slsqp;

import com.curioloop.numopt4j.optim.Optimization;

import com.curioloop.numopt4j.optim.Univariate;
import com.curioloop.numopt4j.optim.Bound;

import com.curioloop.numopt4j.linalg.blas.BLAS;

import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.FIND_CONV;
import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.FIND_NOOP;
import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.INF_BND;
import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.MODE_CONS_INCOMPAT;
import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.MODE_HAS_SOLUTION;
import static com.curioloop.numopt4j.optim.slsqp.SLSQPConstants.MODE_LSEI_SINGULAR;

/**
 * Core optimization loop for the SLSQP algorithm.
 *
 * <p>This class implements the main SQP (Sequential Quadratic Programming) iteration
 * loop. Each iteration solves a QP sub-problem to obtain the search direction d and
 * Lagrange multipliers λ, then performs a line search to find step length α.</p>
 *
 * <h2>QP Sub-Problem</h2>
 * <p>At each iteration k, the following QP is solved:</p>
 * <pre>
 *   minimize   ½ dᵀBᵏd + ∇f(xᵏ)d
 *   subject to ∇cⱼ(xᵏ)d + cⱼ(xᵏ) = 0    (j = 1 ··· mₑ)
 *              ∇cⱼ(xᵏ)d + cⱼ(xᵏ) ≥ 0    (j = mₑ+1 ··· m)
 * </pre>
 *
 * <h2>Merit Function</h2>
 * <p>The L1 merit function with penalty parameters ρⱼ:</p>
 * <pre>
 *   φ(x;ρ) = f(x) + ∑ρⱼ‖cⱼ(x)‖₁
 * </pre>
 *
 * <h2>BFGS Update</h2>
 * <p>The Hessian approximation B = LDLᵀ is updated by the modified BFGS formula
 * using curvature pair (s, q) where s = xᵏ⁺¹ - xᵏ and q = θη + (1-θ)Bᵏs.</p>
 *
 * @see LSQSolver
 * @see LineSearch
 * @see BfgsUpdate
 */
final class SLSQPCore {

    private SLSQPCore() {
        // Utility class - prevent instantiation
    }

    // ========================================================================
    // Convergence Check Functions
    // ========================================================================

    /**
     * Check if optimization should stop based on various criteria.
     *
     * <p>Matches the Go checkStop function in solver.go.
     * Checks multiple termination criteria in order:</p>
     * <ol>
     *   <li>Ĉ𝑣𝑖𝑜 ≥ tol or badQP or f is NaN → continue (not converged)</li>
     *   <li>Ĉ𝑜𝑝𝑡 = |f - f₀| &lt; tol → converged (function value change)</li>
     *   <li>Ĉ𝑠𝑡𝑝 = ‖d‖₂ &lt; tol → converged (step length)</li>
     *   <li>|f| &lt; fEvalTol (if fEvalTol ≥ 0) → converged</li>
     *   <li>|f - f₀| &lt; fDiffTol (if fDiffTol ≥ 0) → converged</li>
     *   <li>‖x - x₀‖₂ &lt; xDiffTol (if xDiffTol ≥ 0) → converged</li>
     * </ol>
     *
     * @param vio       constraint violation Ĉ𝑣𝑖𝑜 = ∑‖cⱼ(xᵏ + αd)‖₁
     * @param tol       relaxed tolerance (10 × accuracy)
     * @param badQP     whether QP was inconsistent (ctx.bad)
     * @param f         current function value f(xᵏ⁺¹)
     * @param f0        previous function value f(xᵏ)
     * @param s         search direction d (step vector)
     * @param sOff      offset into s array
     * @param n         problem dimension
     * @param fEvalTol  function evaluation tolerance (-1 to disable)
     * @param fDiffTol  function difference tolerance (-1 to disable)
     * @param xDiffTol  variable difference tolerance (-1 to disable)
     * @param x         current position xᵏ⁺¹
     * @param xOff      offset into x array
     * @param x0        previous position xᵏ
     * @param x0Off     offset into x0 array
     * @param u         workspace for ‖x - x₀‖₂ computation
     * @param uOff      offset into u array
     * @return Optimization.Status (null = not converged)
     */
    static Optimization.Status checkStop(
            double vio, double tol, boolean badQP,
            double f, double f0,
            double[] s, int sOff, int n,
            double fEvalTol, double fDiffTol, double xDiffTol,
            double[] x, int xOff,
            double[] x0, int x0Off,
            double[] u, int uOff) {

        // Matches C: if (vio >= tol || bad_qp || isnan(f)) { return 0; }
        if (vio >= tol || badQP || Double.isNaN(f)) {
            return null;
        }

        // Matches C: if (fabs(f - f0) < tol) { return 1; }
        if (Math.abs(f - f0) < tol) {
            return Optimization.Status.FUNCTION_TOLERANCE_REACHED;
        }

        // Matches C: if (dnrm2(n, s, 1) < tol) { return 1; }
        if (BLAS.dnrm2(n, s, sOff, 1) < tol) {
            return Optimization.Status.GRADIENT_TOLERANCE_REACHED;
        }

        // Matches C: if (config->f_eval_tol >= ZERO && fabs(f) < config->f_eval_tol) { return 1; }
        if (fEvalTol >= 0.0 && Math.abs(f) < fEvalTol) {
            return Optimization.Status.FUNCTION_TOLERANCE_REACHED;
        }

        // Matches C: if (config->f_diff_tol >= ZERO && fabs(f - f0) < config->f_diff_tol) { return 1; }
        if (fDiffTol >= 0.0 && Math.abs(f - f0) < fDiffTol) {
            return Optimization.Status.FUNCTION_TOLERANCE_REACHED;
        }

        // Matches C: if (config->x_diff_tol >= ZERO) { ... }
        if (xDiffTol >= 0.0) {
            // Compute ||x - x0||_2
            BLAS.dcopy(n, x, xOff, 1, u, uOff, 1);
            BLAS.daxpy(n, -1.0, x0, x0Off, 1, u, uOff, 1);
            if (BLAS.dnrm2(n, u, uOff, 1) < xDiffTol) {
                return Optimization.Status.COEFFICIENT_TOLERANCE_REACHED;
            }
        }

        return null;
    }

    /**
     * Compute constraint violation and check convergence.
     *
     * <p>Matches the Go checkConv function in solver.go.
     * Computes the L1 norm of constraint violations and calls checkStop:</p>
     * <pre>
     *   Ĉ𝑣𝑖𝑜 = ∑‖cⱼ(x)‖₁
     *         = ∑ max(-cⱼ(x), 0)   for inequality constraints (j ≥ mₑ)
     *         = ∑ max(-cⱼ(x), cⱼ(x)) = ∑|cⱼ(x)|  for equality constraints (j &lt; mₑ)
     * </pre>
     *
     * @param c         constraint values cⱼ(x)
     * @param cOff      offset into c array
     * @param m         total number of constraints
     * @param meq       number of equality constraints mₑ
     * @param tol       relaxed tolerance (10 × accuracy)
     * @param badQP     whether QP was inconsistent (ctx.bad)
     * @param f         current function value f(xᵏ⁺¹)
     * @param f0        previous function value f(xᵏ)
     * @param s         search direction d
     * @param sOff      offset into s array
     * @param n         problem dimension
     * @param fEvalTol  function evaluation tolerance (-1 to disable)
     * @param fDiffTol  function difference tolerance (-1 to disable)
     * @param xDiffTol  variable difference tolerance (-1 to disable)
     * @param x         current position xᵏ⁺¹
     * @param xOff      offset into x array
     * @param x0        previous position xᵏ
     * @param x0Off     offset into x0 array
     * @param u         workspace
     * @param uOff      offset into u array
     * @param vioOut    output: constraint violation Ĉ𝑣𝑖𝑜 (array of length 1, can be null)
     * @return Optimization.Status (null = not converged)
     */
    static Optimization.Status checkConv(
            double[] c, int cOff, int m, int meq,
            double tol, boolean badQP,
            double f, double f0,
            double[] s, int sOff, int n,
            double fEvalTol, double fDiffTol, double xDiffTol,
            double[] x, int xOff,
            double[] x0, int x0Off,
            double[] u, int uOff,
            double[] vioOut) {

        // Compute constraint violation: h3 = sum of ||c_j||_1
        // Matches C: for (int j = 0; j < m; j++) { double h1 = (j < meq) ? c[j] : ZERO; h3 += fmax(-c[j], h1); }
        double h3 = 0.0;
        for (int j = 0; j < m; j++) {
            double h1 = (j < meq) ? c[cOff + j] : 0.0;
            h3 += Math.max(-c[cOff + j], h1);
        }

        if (vioOut != null) {
            vioOut[0] = h3;
        }

        return checkStop(h3, tol, badQP, f, f0, s, sOff, n,
                fEvalTol, fDiffTol, xDiffTol, x, xOff, x0, x0Off, u, uOff);
    }

    // ========================================================================
    // Main Optimization Function
    // ========================================================================

    /**
     * SLSQP main optimization loop.
     *
     * <p>Implements the main SQP iteration loop matching Go's mainLoop() in solver.go.</p>
     *
     * <h3>Iteration Flow:</h3>
     * <ol>
     *   <li><b>INITIALIZATION</b>: evaluate f₀, g₀, c₀, a₀; reset BFGS (L = I, D = I)</li>
     *   <li><b>MAIN LOOP</b> (for k = 0, 1, 2, ...):
     *     <ol>
     *       <li>Transform bounds: lᵢ - xᵢᵏ ≤ dᵢ ≤ uᵢ - xᵢᵏ</li>
     *       <li>Solve QP sub-problem via LSQ → search direction d, multipliers λ</li>
     *       <li>If inconsistent: solve augmented QP with slack δ, penalty ρ ∈ [10², 10⁷]</li>
     *       <li>Update penalty parameters: ρⱼᵏ⁺¹ = max[ ½(ρⱼᵏ + |λⱼ|), |λⱼ| ]</li>
     *       <li>Check KKT convergence: C𝑜𝑝𝑡 &lt; acc and C𝑣𝑖𝑜 &lt; acc</li>
     *       <li>Compute merit function: φ(xᵏ;ρ) = f(xᵏ) + ∑ρⱼ‖cⱼ(xᵏ)‖₁</li>
     *       <li>Compute directional derivative: h₃ = ∇f(xᵏ)ᵀd - (1-δ)∑ρⱼ‖cⱼ(xᵏ)‖₁</li>
     *       <li>If h₃ ≥ 0: reset BFGS (ascent direction detected)</li>
     *       <li>Line search: find α satisfying Armijo condition on φ</li>
     *       <li>BFGS update with (s, q) = (αd, θη + (1-θ)Bᵏs)</li>
     *     </ol>
     *   </li>
     * </ol>
     *
     * @param x             initial/final position (modified in place)
     * @param objEval       objective function f(x) and gradient ∇f(x)
     * @param eqCons        equality constraint evaluators cⱼ(x) = 0 (may be null)
     * @param ineqCons      inequality constraint evaluators cⱼ(x) ≥ 0 (may be null)
     * @param bounds        variable bounds lᵢ ≤ xᵢ ≤ uᵢ (may be null)
     * @param accuracy      convergence tolerance (primary criterion)
     * @param maxIter       maximum SQP iterations
     * @param maxEval       maximum function evaluations (0 = unlimited)
     * @param nnlsIter      maximum NNLS iterations (0 = default 3n)
     * @param maxTimeNanos  maximum time in nanoseconds (0 = no limit)
     * @param exactSearch   true for exact line search (Brent's method), false for Armijo backtracking
     * @param fEvalTol      convergence if |f| &lt; fEvalTol (-1 to disable)
     * @param fDiffTol      convergence if |f - f₀| &lt; fDiffTol (-1 to disable)
     * @param xDiffTol      convergence if ‖x - x₀‖₂ &lt; xDiffTol (-1 to disable)
     * @param ws            workspace
     * @return optimization result
     */
    static Optimization optimize(
            double[] x,
            Univariate objEval,
            Univariate[] eqCons,
            Univariate[] ineqCons,
            Bound[] bounds,
            double accuracy, int maxIter, int maxEval, int nnlsIter, long maxTimeNanos,
            boolean exactSearch,
            double fEvalTol, double fDiffTol, double xDiffTol,
            SLSQPWorkspace ws) {

        // Validate inputs
        if (x == null || objEval == null) {
            return new Optimization(Double.NaN, null, 0.0, Optimization.Status.INVALID_ARGUMENT, 0, 0);
        }

        int meq = (eqCons != null) ? eqCons.length : 0;
        int mineq = (ineqCons != null) ? ineqCons.length : 0;
        int m = meq + mineq;
        int n = x.length;

        // Temporary array for gradient (to avoid overwriting workspace buffer)
        double[] g = new double[n];

        // Get workspace arrays and offsets
        double[] buffer = ws.getBuffer();
        int[] iBuffer = ws.getIBuffer();

        int lOff = ws.getLOffset();
        int x0Off = ws.getX0Offset();
        int gOff = ws.getGOffset();
        int cOff = ws.getCOffset();
        int aOff = ws.getAOffset();
        int muOff = ws.getMuOffset();
        int sOff = ws.getSOffset();
        int uOff = ws.getUOffset();
        int vOff = ws.getVOffset();
        int rOff = ws.getROffset();
        int wOff = ws.getWOffset();
        int jwOff = ws.getJwOffset();

        int n1 = n + 1;
        int n2 = n * n1 / 2;
        int la = ws.getLa();

        double tol = 10.0 * accuracy;
        int nnlsIterActual = nnlsIter > 0 ? nnlsIter : 3 * n;

        long startTimeNanos = (maxTimeNanos > 0) ? System.nanoTime() : 0;
        int nfev = 0;

        // Initialize LDL^T factorization: L = I, D = I
        // Matches C: dzero(n2 + 1, l, 1); for (int i = 0, j = 0; i < n; i++) { l[j] = ONE; j += n - i; }
        BLAS.dset(n2 + 1, 0, buffer, lOff, 1);
        for (int i = 0, j = 0; i < n; i++) {
            buffer[lOff + j] = 1.0;
            j += n - i;
        }
        ws.setResetCount(1);

        // Initialize multipliers and search direction
        // Matches C: dzero(n + 1, s, 1); dzero(m > 0 ? m : 1, mu, 1);
        BLAS.dset(n + 1, 0, buffer, sOff, 1);
        BLAS.dset(m > 0 ? m : 1, 0, buffer, muOff, 1);

        // Evaluate initial function and gradient (use temporary array to avoid overwriting buffer)
        double f = objEval.evaluate(x, n, g);
        nfev++;
        // Copy gradient to workspace g array
        System.arraycopy(g, 0, buffer, gOff, n);

        if (Double.isNaN(f) || Double.isInfinite(f)) {
            return new Optimization(Double.NaN, null, f, Optimization.Status.CALLBACK_ERROR, 0, nfev);
        }

        // Evaluate constraints and their gradients (if provided)
        if (meq > 0) {
            for (int i = 0; i < meq; i++) {
                double val = eqCons[i].evaluate(x, n, g);
                buffer[cOff + i] = val;
                // Fill Jacobian column-major: a[aOff + i + la*j] = dc_i/dx_j
                for (int j = 0; j < n; j++) {
                    buffer[aOff + i + la * j] = g[j];
                }
            }
        }
        if (mineq > 0) {
            for (int i = 0; i < mineq; i++) {
                double val = ineqCons[i].evaluate(x, n, g);
                buffer[cOff + meq + i] = val;
                for (int j = 0; j < n; j++) {
                    buffer[aOff + meq + i + la * j] = g[j];
                }
            }
        }

        // Main iteration loop
        boolean badQP;
        for (int iter = 0; iter < maxIter; iter++) {

            // Check time limit
            if (maxTimeNanos > 0) {
                long elapsedNanos = System.nanoTime() - startTimeNanos;
                if (elapsedNanos >= maxTimeNanos) {
                    return new Optimization(Double.NaN, null, f, Optimization.Status.MAX_COMPUTATIONS_REACHED, iter, nfev);
                }
            }

            ws.setIter(iter);

            // Transform bounds from l <= x <= u to l - x^k <= d <= u - x^k
            for (int i = 0; i < n; i++) {
                Bound b = Bound.of(bounds, i, Bound.unbounded());
                buffer[uOff + i] = b.hasLower() ? b.getLower() - x[i] : -INF_BND;
                buffer[vOff + i] = b.hasUpper() ? b.getUpper() - x[i] : INF_BND;
            }

            // Solve QP subproblem to get search direction s and multipliers r
            double[] norm = new double[1];
            int lsqMode = LSQSolver.lsq(m, meq, n, n2 + 1,
                    buffer, lOff,
                    buffer, gOff,
                    buffer, aOff,
                    buffer, cOff,
                    buffer, uOff,
                    buffer, vOff,
                    buffer, sOff,
                    buffer, rOff,
                    buffer, wOff,
                    iBuffer, jwOff,
                    nnlsIterActual, INF_BND, norm);

            // Handle singular C matrix case
            if (lsqMode == MODE_LSEI_SINGULAR && n == meq) {
                lsqMode = MODE_CONS_INCOMPAT;
            }

            double h4 = 1.0;
            badQP = (lsqMode == MODE_CONS_INCOMPAT);

            if (badQP) {
                // Form augmented QP relaxation
                int augAOff = aOff + n * la;
                for (int j = 0; j < m; j++) {
                    if (j < meq) {
                        buffer[augAOff + j] = -buffer[cOff + j];
                    } else {
                        buffer[augAOff + j] = (buffer[cOff + j] <= 0.0) ? -buffer[cOff + j] : 0.0;
                    }
                }
                buffer[gOff + n] = 0.0;
                buffer[lOff + n2] = 100.0;  // rho = 100
                BLAS.dset(n, 0, buffer, sOff, 1);
                buffer[sOff + n] = 1.0;   // delta = 1
                buffer[uOff + n] = 0.0;
                buffer[vOff + n] = 1.0;   // 0 <= delta <= 1

                // Try to solve augmented problem with increasing penalty
                for (int relax = 0; relax <= 5; relax++) {
                    lsqMode = LSQSolver.lsq(m, meq, n + 1, n2 + 1,
                            buffer, lOff,
                            buffer, gOff,
                            buffer, aOff,
                            buffer, cOff,
                            buffer, uOff,
                            buffer, vOff,
                            buffer, sOff,
                            buffer, rOff,
                            buffer, wOff,
                            iBuffer, jwOff,
                            nnlsIterActual, INF_BND, norm);
                    h4 = 1.0 - buffer[sOff + n];  // 1 - delta
                    if (lsqMode != MODE_CONS_INCOMPAT) {
                        break;
                    }
                    buffer[lOff + n2] *= 10.0;  // rho = rho * 10
                }
            }

            // Unable to solve LSQ even the augmented one
            if (lsqMode != MODE_HAS_SOLUTION) {
                return new Optimization(Double.NaN, null, f, Optimization.Status.CONSTRAINT_INCOMPATIBLE, iter, nfev);
            }

            // Update multipliers for L1-test: v[i] = g[i] - λᵀ∇c[i]
            for (int i = 0; i < n; i++) {
                buffer[vOff + i] = buffer[gOff + i] - BLAS.ddot(m, buffer, aOff + i * la, 1, buffer, rOff, 1);
            }

            // Save current state
            double f0 = f;
            BLAS.dcopy(n, x, 0, 1, buffer, x0Off, 1);

            // Compute optimality and feasibility measures
            double gs = BLAS.ddot(n, buffer, gOff, 1, buffer, sOff, 1);  // g'*d
            double h1 = Math.abs(gs);  // |g'*d|
            double h2 = 0.0;          // constraint violation

            for (int j = 0; j < m; j++) {
                double h3 = (j < meq) ? buffer[cOff + j] : 0.0;
                h2 += Math.max(-buffer[cOff + j], h3);  // ||c_j(x^k)||_1
                h3 = Math.abs(buffer[rOff + j]);        // |lambda_j|
                h1 += h3 * Math.abs(buffer[cOff + j]);  // |lambda_j| * ||c_j(x^k)||_1
                buffer[muOff + j] = Math.max(h3, (buffer[muOff + j] + h3) / 2);  // rho_j^{k+1}
            }

            // Check convergence
            if (h1 < accuracy && h2 < accuracy && !badQP && !Double.isNaN(f)) {
                return new Optimization(Double.NaN, null, f, Optimization.Status.GRADIENT_TOLERANCE_REACHED, iter, nfev);
            }

            // Compute directional derivative of merit function
            h1 = 0.0;
            for (int j = 0; j < m; j++) {
                double h3Tmp = (j < meq) ? buffer[cOff + j] : 0.0;
                h1 += buffer[muOff + j] * Math.max(-buffer[cOff + j], h3Tmp);
            }

            double t0 = f + h1;  // Merit function at x^k
            double h3 = gs - h1 * h4;  // Directional derivative

            if (h3 >= 0.0) {
                // Reset BFGS when ascent direction is generated
                int resetCount = ws.incrementResetCount();
                if (resetCount > 5) {
                    // Check relaxed convergence
                    double[] vio = new double[1];
                    Optimization.Status convStatus = checkConv(buffer, cOff, m, meq, tol, badQP, f, f0,
                            buffer, sOff, n, fEvalTol, fDiffTol, xDiffTol,
                            x, 0, buffer, x0Off, buffer, uOff, vio);
                    if (convStatus != null) {
                        return new Optimization(Double.NaN, null, f, convStatus, iter, nfev);
                    }
                    // Not converged even with relaxed tolerance
                    return new Optimization(Double.NaN, null, f, Optimization.Status.LINE_SEARCH_FAILED, iter, nfev);
                }
                // Reset L = I, D = I
                BLAS.dset(n2 + 1, 0, buffer, lOff, 1);
                for (int i = 0, j = 0; i < n; i++) {
                    buffer[lOff + j] = 1.0;
                    j += n - i;
                }
                continue;
            }

            // Line search with merit function
            ws.setT0(t0);
            double alphaMin = 0.1;
            double alphaMax = 1.0;
            double alpha = alphaMax;
            int lineCount = 0;

            if (exactSearch) {
                // Initialize exact line search
                ws.setLineMode(FIND_NOOP);
                LineSearch.exactSearch(ws, Double.NaN, tol, alphaMin, alphaMax,
                        n, x, 0, buffer, x0Off, buffer, sOff);
            } else {
                // Initialize inexact line search
                lineCount = 1;
                alpha = alphaMax;

                // Scale s by alpha, then x = x0 + s
                BLAS.dscal(n, alpha, buffer, sOff, 1);
                BLAS.dcopy(n, buffer, x0Off, 1, x, 0, 1);
                BLAS.daxpy(n, 1.0, buffer, sOff, 1, x, 0, 1);

                projectOntoBounds(n, x, bounds);

                h3 *= alpha;  // Update directional derivative
            }

            // Line search loop
            boolean lsDone = false;
            for (int lsIter = 0; lsIter < 20 && !lsDone; lsIter++) {
                // Evaluate function value only (no gradient needed during line search)
                // Matches Go: evalLoc(evalFunc) passes nil for gradient
                f = objEval.evaluate(x, n, null);
                nfev++;

                if (Double.isNaN(f) || Double.isInfinite(f)) {
                    return new Optimization(Double.NaN, null, f0, Optimization.Status.CALLBACK_ERROR, iter, nfev);
                }
                if (maxEval > 0 && nfev >= maxEval) {
                    return new Optimization(Double.NaN, null, f, Optimization.Status.MAX_EVALUATIONS_REACHED, iter, nfev);
                }

                // Evaluate constraint values only (no gradient)
                if (meq > 0) {
                    for (int i = 0; i < meq; i++) {
                        buffer[cOff + i] = eqCons[i].evaluate(x, n, null);
                    }
                }
                if (mineq > 0) {
                    for (int i = 0; i < mineq; i++) {
                        buffer[cOff + meq + i] = ineqCons[i].evaluate(x, n, null);
                    }
                }

                // Compute merit function at new point: t = f + sum(mu_j * ||c_j||_1)
                double t = f;
                for (int j = 0; j < m; j++) {
                    double tmp = (j < meq) ? buffer[cOff + j] : 0.0;
                    t += buffer[muOff + j] * Math.max(-buffer[cOff + j], tmp);
                }

                // Line search decision
                if (!exactSearch) {
                    // Inexact line search (Armijo backtracking)
                    double h1Ls = t - t0;

                    if (h1Ls <= h3 / 10.0 || lineCount > 10) {
                        // Armijo condition satisfied or max iterations reached
                        double[] vio = new double[1];
                        Optimization.Status convStatus = checkConv(buffer, cOff, m, meq, accuracy, badQP, f, f0,
                                buffer, sOff, n, fEvalTol, fDiffTol, xDiffTol,
                                x, 0, buffer, x0Off, buffer, uOff, vio);
                        if (convStatus != null) {
                            // Evaluate gradients at accepted point before returning
                            evalGradients(x, n, m, meq, mineq, la, objEval, eqCons, ineqCons, g, buffer, gOff, cOff, aOff);
                            return new Optimization(Double.NaN, null, f, convStatus, iter, nfev);
                        }
                        h3 = vio[0];
                        lsDone = true;
                    } else {
                        // Armijo condition not satisfied, reduce step
                        alpha = h3 / (2.0 * (h3 - h1Ls));
                        alpha = Math.max(Math.min(alpha, alphaMax), alphaMin);

                        lineCount++;
                        BLAS.dscal(n, alpha, buffer, sOff, 1);
                        BLAS.dcopy(n, buffer, x0Off, 1, x, 0, 1);
                        BLAS.daxpy(n, 1.0, buffer, sOff, 1, x, 0, 1);

                        projectOntoBounds(n, x, bounds);

                        h3 *= alpha;
                    }
                } else {
                    // Exact line search
                    int fm = LineSearch.exactSearch(ws, t, tol, alphaMin, alphaMax,
                            n, x, 0, buffer, x0Off, buffer, sOff);

                    if (fm == FIND_CONV) {
                        // Search converged
                        double[] vio = new double[1];
                        Optimization.Status convStatus = checkConv(buffer, cOff, m, meq, accuracy, badQP, f, f0,
                                buffer, sOff, n, fEvalTol, fDiffTol, xDiffTol,
                                x, 0, buffer, x0Off, buffer, uOff, vio);
                        if (convStatus != null) {
                            // Evaluate gradients at accepted point before returning
                            evalGradients(x, n, m, meq, mineq, la, objEval, eqCons, ineqCons, g, buffer, gOff, cOff, aOff);
                            return new Optimization(Double.NaN, null, f, convStatus, iter, nfev);
                        }
                        h3 = vio[0];
                        lsDone = true;
                    }
                }
            }

            // Evaluate gradients at accepted point for BFGS update
            // Matches Go: updateBFGS() → evalLoc(evalGrad)
            evalGradients(x, n, m, meq, mineq, la, objEval, eqCons, ineqCons, g, buffer, gOff, cOff, aOff);

            // BFGS update
            // Compute eta = grad_L(x^{k+1}, lambda^k) - grad_L(x^k, lambda^k)
            for (int i = 0; i < n; i++) {
                buffer[uOff + i] = buffer[gOff + i] - BLAS.ddot(m, buffer, aOff + i * la, 1, buffer, rOff, 1) - buffer[vOff + i];
            }

            // Compute L'*s
            for (int i = 0, k = 0; i < n; i++) {
                k++;
                double sm = 0.0;
                for (int j = i + 1; j < n; j++) {
                    sm += buffer[lOff + k] * buffer[sOff + j];
                    k++;
                }
                buffer[vOff + i] = buffer[sOff + i] + sm;
            }

            // Compute D*L'*s
            for (int i = 0, k = 0; i < n; i++) {
                buffer[vOff + i] = buffer[lOff + k] * buffer[vOff + i];
                k += n - i;
            }

            // Compute L*D*L'*s = B^k*s
            for (int i = n - 1; i >= 0; i--) {
                int k = i;
                double sm = 0.0;
                for (int j = 0; j < i; j++) {
                    sm += buffer[lOff + k] * buffer[vOff + j];
                    k += n - 1 - j;
                }
                buffer[vOff + i] += sm;
            }

            h1 = BLAS.ddot(n, buffer, sOff, 1, buffer, uOff, 1);  // s'*eta
            h2 = BLAS.ddot(n, buffer, sOff, 1, buffer, vOff, 1);  // s'*B^k*s
            h3 = 0.2 * h2;

            if (h1 < h3) {
                // theta = 4/5 * s'*B^k*s / (s'*B^k*s - s'*eta)
                h4 = (h2 - h3) / (h2 - h1);
                h1 = h3;
                BLAS.dscal(n, h4, buffer, uOff, 1);
                BLAS.daxpy(n, 1.0 - h4, buffer, vOff, 1, buffer, uOff, 1);
            }

            if (h1 == 0.0 || h2 == 0.0) {
                // Reset BFGS
                int resetCount = ws.incrementResetCount();
                if (resetCount > 5) {
                    // Check relaxed convergence
                    double[] vio = new double[1];
                    Optimization.Status convStatus = checkConv(buffer, cOff, m, meq, tol, badQP, f, f0,
                            buffer, sOff, n, fEvalTol, fDiffTol, xDiffTol,
                            x, 0, buffer, x0Off, buffer, uOff, vio);
                    if (convStatus != null) {
                        return new Optimization(Double.NaN, null, f, convStatus, iter, nfev);
                    }
                    // Not converged even with relaxed tolerance
                    return new Optimization(Double.NaN, null, f, Optimization.Status.LINE_SEARCH_FAILED, iter, nfev);
                }
                // Reset L = I, D = I
                BLAS.dset(n2 + 1, 0, buffer, lOff, 1);
                for (int i = 0, j = 0; i < n; i++) {
                    buffer[lOff + j] = 1.0;
                    j += n - i;
                }
            } else {
                // Update LDL^T factorization
                BfgsUpdate.compositeT(n, buffer, lOff, buffer, uOff, 1.0 / h1, null, 0);
                BfgsUpdate.compositeT(n, buffer, lOff, buffer, vOff, -1.0 / h2, buffer, uOff);
            }
        }

        // Maximum iterations reached
        return new Optimization(Double.NaN, null, f, Optimization.Status.MAX_ITERATIONS_REACHED, maxIter, nfev);
    }

    /**
     * Evaluate objective and constraint gradients at x.
     * Called once after line search converges, matching Go's evalLoc(evalGrad).
     */
    private static void evalGradients(
            double[] x, int n, int m, int meq, int mineq, int la,
            Univariate objEval, Univariate[] eqCons, Univariate[] ineqCons,
            double[] g, double[] buffer, int gOff, int cOff, int aOff) {
        if (meq > 0) {
            for (int i = 0; i < meq; i++) {
                eqCons[i].evaluate(x, n, g);
                for (int j = 0; j < n; j++) buffer[aOff + i + la * j] = g[j];
            }
        }
        if (mineq > 0) {
            for (int i = 0; i < mineq; i++) {
                ineqCons[i].evaluate(x, n, g);
                for (int j = 0; j < n; j++) buffer[aOff + meq + i + la * j] = g[j];
            }
        }
        objEval.evaluate(x, n, g);
        System.arraycopy(g, 0, buffer, gOff, n);
    }

    /**
     * Project x onto the bounds [lower, upper].
     *
     * @param n      problem dimension
     * @param x      position to project (modified)
     * @param bounds
     */
    private static void projectOntoBounds(int n, double[] x, Bound[] bounds) {
        for (int i = 0; i < n; i++) {
            Bound b = Bound.of(bounds, i, Bound.unbounded());
            double lb = b.hasLower() ? b.getLower() : -INF_BND;
            double ub = b.hasUpper() ? b.getUpper() : INF_BND;
            if (lb > -INF_BND && x[i] < lb) {
                x[i] = lb;
            } else if (ub < INF_BND && x[i] > ub) {
                x[i] = ub;
            }
        }
    }
}
