/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * LSQSolver - Consolidated least-squares solver routines for SLSQP.
 *
 * This class merges the following formerly separate classes into one:
 *   Householder, NNLS, HFTI, LDP, LSI, LSEI, LSQ
 *
 * Call chain (top-down):
 *   lsq → lsei → lsi / hfti → ldp → nnls → h1/h2/g1/g2
 *
 * References:
 *   C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
 *   Prentice Hall, 1974. (revised 1995 edition)
 */
package com.curioloop.numopt4j.optim.slsqp;

import com.curioloop.numopt4j.linalg.blas.BLAS;

/**
 * Consolidated least-squares solver for SLSQP.
 * <p>
 * Provides all routines needed by the SLSQP QP subproblem solver,
 * from low-level Householder/Givens primitives up to the full LSQ entry point.
 * </p>
 */
final class LSQSolver {

    private LSQSolver() {
        // Utility class - prevent instantiation
    }

    // =========================================================================
    // NNLS - Non-Negative Least Squares
    //
    // Solves a least-squares problem 𝚖𝚒𝚗 ‖ 𝐀𝐱 - 𝐛 ‖₂ subject to 𝐱 ≥ 0 with active-set method.
    //   - 𝐀 is m × n column-major matrix with 𝚛𝚊𝚗𝚔(𝐀) = n (the columns of 𝐀 are linearly independent)
    //   - 𝐱 ∈ ℝⁿ
    //   - 𝐛 ∈ ℝᵐ
    //
    // There are two index set ℤ(zero) and ℙ(pivot):
    //   - 𝐱ⱼ = 0, j ∈ ℤ : variable indexed in active set ℤ will be held at the value zero
    //   - 𝐱ⱼ > 0, j ∈ ℙ : variable indexed in passive set ℙ will be free to take any positive value
    //
    // When 𝐱ⱼ < 0 occurred, NNLS will change its value to a non-negative value and move its index j from ℙ to ℤ.
    //
    // The m × k matrix 𝐀ₖ is a subset columns of 𝐀 defined by indices of ℙ.
    // NNLS apply QR composition 𝐐𝐀ₖ = [𝐑ₖᵀ:O]ᵀ to solve least-squares [𝐀ₖ:O]𝐱 ≅ 𝐛
    // where 𝐐 is m × m orthogonal matrix and 𝐑ₖ is k × k upper triangular matrix.
    //
    // Once 𝐐 and 𝐑ₖ is computed, the solution is given by 𝐱߮ = [𝐑ₖ⁻¹:O]𝐐𝐛.
    //
    // Let 𝐛 = [𝐛₁:𝐛₂] (𝐛₁ ∈ ℝⁿ, 𝐛₂ ∈ ℝᵐ⁻ⁿ) and rewrite 𝚖𝚒𝚗‖ 𝐀𝐱 - 𝐛 ‖₂ to 𝚖𝚒𝚗‖ 𝐐ᵀ𝐐[𝐑ₙ:O]𝐱 - 𝐐ᵀ[𝐛₁:𝐛₂] ‖₂
    //   - the solution 𝐱 satisfied 𝐑ₙ𝐱 = 𝐐ᵀ𝐛₁ (𝐐ᵀ𝐐 = 𝐈ₘ)
    //   - the residual is given by 𝐫 = 𝐐𝐐ᵀ[𝐛₁:𝐛₂]ᵀ - 𝐐[𝐑ₙᵀ𝐱:O]ᵀ = 𝐐[O:𝐐ᵀ𝐛₂]
    //   - the norm of residual is given by ‖ 𝐫 ‖₂ = ‖ 𝐐ᵀ𝐛₂ ‖₂
    //
    // The input will be treated as a whole m × (n+1) working space 𝐐[𝐀:𝐛] where
    //   - space of matrix 𝐀 will be used to store the 𝐐𝐀 result
    //   - space of vector 𝐛 will be used to store the 𝐐𝐛 result
    //
    // Optimality Conditions:
    // Given a problem 𝚖𝚒𝚗 𝒇(𝐱) subject to 𝒉ⱼ(𝐱) = 0 (j = 1 ··· mₑ) and 𝒈ⱼ(𝐱) ≤ 0 (j = mₑ+1 ··· m),
    // its optimality at location 𝐱ᵏ are given by below KKT conditions:
    //   - 𝜵ℒ(𝐱ᵏ,𝛌ᵏ) = 𝜵𝒇(𝐱ᵏ) + ∑𝛌ᵏⱼ𝜵𝒈ⱼ(𝐱ᵏ) = 0
    //   - 𝒈ⱼ(𝐱ᵏ) = 0   (j = 1 ··· mₑ)
    //   - 𝒈ⱼ(𝐱ᵏ) ≤ 0   (j = mₑ+1 ··· m)
    //   - 𝛌ᵏⱼ ≥ 0      (j = mₑ+1 ··· m)
    //   - 𝛌ᵏⱼ𝒈ⱼ(𝐱) = 0  (j = mₑ+1 ··· m)
    //
    // and substitute NNLS to the KKT conditions:
    //   - 𝒇(𝐱) = ½𝐱ᵀ𝐀𝐱 - 2𝐛ᵀ𝐀𝐱 + ½𝐛ᵀ𝐛  →  𝜵𝒇(𝐱) = 𝐀ᵀ(𝐀𝐱 + 𝐛)
    //   - 𝒈ⱼ(𝐱) = 0  (j = 1 ··· mₑ)    →  𝜵𝒈ⱼ(𝐱) = 0
    //   - 𝒈ⱼ(𝐱) = -𝐱ⱼ (j = mₑ+1 ··· m) →  𝜵𝒈ⱼ(𝐱) = -1
    //
    // the optimality conditions for NNLS are given:
    //   - 𝜵ℒ(𝐱ᵏ,𝛌ᵏ) = 𝐀ᵀ(𝐀𝐱ᵏ + 𝐛) - ∑𝛌ᵏⱼ = 0
    //   - 𝛌ᵏⱼ ≥ 0 ∀j
    //   - 𝛌ᵏⱼ𝒈ⱼ(𝐱) = 0 ∀j
    //
    // NNLS introduce a dual m-vector 𝐰 = -𝝺 = -𝜵𝒇(𝐱) = 𝐀ᵀ(𝐛 - 𝐀𝐱) and optimality is given by:
    //   - 𝐰ⱼ = 0, ∀j ∈ ℙ
    //   - 𝐰ⱼ ≤ 0, ∀j ∈ ℤ
    //
    // Active Set Method:
    // The optimality of the activity set method is described by the KKT condition.
    //
    // Let 𝐱ᵏ be a feasible vector, the inequality constraints 𝒈ⱼ(𝐱ᵏ) (j = mₑ+1 ··· m) has to two status:
    //   - active inequality constraints : 𝒈ⱼ(𝐱ᵏ) = 0
    //   - passive inequality constraints : 𝒈ⱼ(𝐱ᵏ) < 0
    //
    // Recall the 𝝺 describes how 𝒇(𝐱) change when relaxing constraints 𝒈ⱼ(𝐱) ≤ 0 → 𝛆 with a interruption 𝛆 > 0:
    //   - for 𝛌ⱼ < 0, relax the 𝒈ⱼ(𝐱) will decrease 𝒇(𝐱)
    //   - for 𝛌ⱼ > 0, relax the 𝒈ⱼ(𝐱) will increase 𝒇(𝐱)
    //
    // When we found some active constraints with 𝛌ⱼ < 0:
    //   - relax 𝒈ⱼ(𝐱) and move it from ℤ to ℙ
    //   - form a new pure equality constrain sub-problem EQP base on new ℤ
    //   - solve EQP with variable elimination method
    //
    // Assume 𝐬 is the EQP solution, then there is 𝒇(𝐬) < 𝒇(𝐱ᵏ) and :
    //   - if 𝐬 is feasible, update ℤ and ℙ and solve new EQP until feasible solution is not change
    //   - if 𝐬 is infeasible, we just obtain a descending direction 𝐝 = 𝐬 - 𝐱ᵏ and need to find a step length α > 0 such that 𝐱ᵏ + α𝐝 is feasible.
    //
    // The α can be obtained by projecting the infeasible 𝐬 to the boundaries defined by ℙ.
    //
    // Once new location 𝐱ᵏ⁺¹ = 𝐱ᵏ + α𝐝 is determined, update ℤ and ℙ and solve new EQP again.
    //
    // In case of NNLS, the EQP is a unconstrained least-squares problem 𝚖𝚒𝚗 ½‖ 𝐀ᴾ𝐱 - 𝐛 ‖₂.
    // The matrix 𝐀ᴾ is a matrix containing only the variables currently in ℙ.
    // Thus the solution is given by 𝐬 = [(𝐀ᴾ)ᵀ𝐀ᴾ]⁻¹(𝐀ᴾ)ᵀ𝐛 which is actually computed by QR decomposition.
    //
    // Non-negative solution:
    // Consider an m × (n+1) augmented matrix [𝐀:𝐛] defined by least-squares problem 𝐀𝐱 ≅ 𝐛.
    //
    // Let 𝐐 be an m × m orthogonal matrix that zeros the sub-diagonal elements in first n-1 cols of 𝐀.
    //
    //         n     1       n-1  1   1
    //        ┌┴┐   ┌┴┐      ┌┴┐ ┌┴┐ ┌┴┐
    //   𝐐[  𝐀 ﹕  𝐛 ] = ⎡  𝐑   𝒔   𝒖 ⎤ ]╴ n-1
    //                   ⎣  ０   𝒕   𝒗 ⎦ ]╴ m-n+1
    //
    // where 𝐑 is an m × m upper triangular full-rank matrix.
    //
    // Since orthogonal transformation preserves the relationship between the columns of augmented matrix, so there is:
    //
    //   (𝐐𝐀)ᵀ𝐐𝐛 ＝ 𝐀ᵀ𝐛 ＝ ⎡ 𝑹ᵀ ０ ⎤⎡ 𝒖 ⎤ ＝ ⎡    𝑹ᵀ𝒖   ⎤
    //                     ⎣ 𝒔ᵀ  𝒕ᵀ ⎦⎣ 𝒗 ⎦   ⎣ 𝒔ᵀ𝒖 + 𝒕ᵀ𝒗 ⎦
    //
    //                               n-1    1
    //                             ┌──┴──┐ ┌┴┐
    //   Assume there is 𝐀ᵀ𝐛 = [ 0 ··· 0  ω ]ᵀ = [𝑹ᵀ𝒖 : 𝒔ᵀ𝒖 + 𝒕ᵀ𝒗]ᵀ.
    //   Since 𝐑 is non-singular, 𝑹ᵀ𝒖 has only the trivial solution 𝒖 = 0 which means 𝒕ᵀ𝒗 = ω.
    //
    // The n-th component of solution to 𝐀𝐱 ≅ 𝐛 is the least squares solution of 𝒕𝐱ₙ ≅ 𝒗 which is 𝐱ₙ = 𝒕ᵀ𝒗/𝒕ᵀ𝒕 = ω/𝒕ᵀ𝒕.
    //
    // Thus when the n-th component of 𝐀ᵀ𝐛 is positive (ω > 0), then the n-th component of solution satisfied 𝐱ₙ > 0.
    //
    // Reference:
    //   C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
    //   Prentice Hall, 1974. Chapters 23, Algorithm 23.10.
    // =========================================================================

    /** Factor for checking linear independence in NNLS */
    private static final double NNLS_FACTOR = 0.01;

    /**
     * nnls - Solve the Non-Negative Least Squares problem.
     * <p>
     * Solves min ||Ax - b||_2 subject to x >= 0
     * </p>
     *
     * @param m        Number of rows in A
     * @param n        Number of columns in A
     * @param a        Matrix A (column-major), modified on return to QA
     * @param aOff     Offset into array a
     * @param mda      Leading dimension of A
     * @param b        Vector b, modified on return to Qb
     * @param bOff     Offset into array b
     * @param x        Output: solution vector x of primal problem
     * @param xOff     Offset into array x
     * @param w        Output: dual vector w describing the weight of constraint
     * @param wOff     Offset into array w
     * @param z        Working array of length m
     * @param zOff     Offset into array z
     * @param index    Working array of length n, stores P ∪ Z = {0,...,n-1}
     *                 P = index[:np] defines the subset columns of A
     *                 Z = index[z1:]
     * @param indexOff Offset into array index
     * @param maxIter  Maximum iterations (0 = 3*n)
     * @param rnorm    Output: residual norm ||Q^T b_2||_2 (array of length 1)
     * @return         Status code (0 = success, 1 = exceeded max iterations, negative = error)
     */
    static int nnls(
            int m, int n,
            double[] a, int aOff, int mda,
            double[] b, int bOff,
            double[] x, int xOff,
            double[] w, int wOff,
            double[] z, int zOff,
            int[] index, int indexOff,
            int maxIter,
            double[] rnorm) {

        int i, ii, ip, iter, iz, izmax, j, jj, jz, l, np, z1;
        double alpha, asave, cc, ss, t, unorm, up, wmax, ztest;

        if (m <= 0 || n <= 0 || mda < m) {
            return -1;  // Bad argument
        }

        if (maxIter <= 0) {
            maxIter = 3 * n;
        }

        np = 0;   // Number of elements in set P
        z1 = 0;   // Start index of set Z

        // Initialize index = P ∪ Z = {0,...,n-1}
        for (i = 0; i < n; i++) {
            index[indexOff + i] = i;
        }

        // Start from x = O and all indices are initially in Z
        for (i = 0; i < n; i++) {
            x[xOff + i] = 0.0;
        }

        iter = 0;

        // Main loop: continued until no more active constraints can be set free
        mainLoop:
        for (;;) {
            // Quit if all coefficients are positive: Z = ∅ (x >= 0),
            // or if m columns of A have been triangularized
            if (z1 >= n || np >= m) {
                break mainLoop;
            }

            // Compute components of the dual vector w = A^T(b - Ax) (negative gradient).
            // Since w_j = 0 for j ∈ P, we only compute w_j for j ∈ Z.
            // Given x_j = 0 for j ∈ Z, the update simplifies to w = A^T b.
            for (iz = z1; iz < n; iz++) {
                j = index[indexOff + iz];
                w[wOff + j] = BLAS.ddot(m - np, a, aOff + np + mda * j, 1, b, bOff + np, 1);
            }

            for (;;) {
                // Find index t ∈ Z such that w_t = arg max { w_j: j ∈ Z }
                wmax = 0.0;
                izmax = 0;
                for (iz = z1; iz < n; iz++) {
                    j = index[indexOff + iz];
                    if (w[wOff + j] > wmax) {
                        wmax = w[wOff + j];
                        izmax = iz;
                    }
                }

                // Quit when w_j <= 0, ∀j ∈ Z (no more constraint could be relaxed)
                // This indicates satisfaction of the Kuhn-Tucker conditions
                if (wmax <= 0.0) {
                    break mainLoop;
                }

                // Move index t from Z to P
                iz = izmax;
                j = index[indexOff + iz];
                int ajOff = aOff + mda * j;

                // Given j-th column of A, compute corresponding Householder vector u.
                // Save the pivot-th component of j-th column A_pj.
                asave = a[ajOff + np];
                up = Householder.h1(np, np + 1, m, a, ajOff, 1);
                // Now the pivot-th component of j-th column is (QA)_pj.
                // The pivot-th component of u is returned as u_p.

                // Check new diagonal element to avoid near linear dependence
                unorm = BLAS.dnrm2(np, a, ajOff, 1);  // ||u||_2
                if (Math.abs(a[ajOff + np]) * NNLS_FACTOR >= unorm * SLSQPConstants.EPS) {
                    // Column j is sufficiently independent.
                    // Compute Householder transformation z = Qb = [-σ||b||_2 0 ... 0]^T
                    System.arraycopy(b, bOff, z, zOff, m);
                    Householder.h2(np, np + 1, m, a, ajOff, 1, up, z, zOff, 1, 1, 1);

                    // Solve Q(Ax)_j ≅ Qb_j for proposed new value for x_j
                    // x = (QA)^+ Qb
                    ztest = z[zOff + np] / a[ajOff + np];

                    if (ztest > 0.0) {
                        // Accept j: x_j > 0

                        // Update b = Qb
                        System.arraycopy(z, zOff, b, bOff, m);

                        // Move j from Z to P
                        index[indexOff + iz] = index[indexOff + z1];
                        index[indexOff + z1] = j;
                        z1++;
                        np++;

                        // Apply Householder transformations to cols in new Z
                        if (z1 < n) {
                            for (jz = z1; jz < n; jz++) {
                                jj = index[indexOff + jz];
                                Householder.h2(np - 1, np, m, a, ajOff, 1, up, a, aOff + mda * jj, 1, mda, 1);
                            }
                        }

                        // Zero sub-diagonal elements in col j
                        for (i = np; i < m; i++) {
                            a[ajOff + i] = 0.0;
                        }

                        // Set w_j = 0 for j ∈ P
                        w[wOff + j] = 0.0;
                        break;
                    }
                }

                // Reject j as a candidate to be moved from Z to P,
                // restore A_pj and test dual coefficients again
                a[ajOff + np] = asave;
                w[wOff + j] = 0.0;
            }

            // Inner loop: When new j joins P, the coefficients of the free variables
            // in the unconstrained solution s may turn negative.
            // The inner loop continues until all violating variables have been moved to Z.
            for (;;) {
                // Compute EQP solution s by solving triangular system x̂ = [R_k^(-1):O]Qb
                for (ip = np - 1; ip >= 0; ip--) {
                    if (ip < np - 1) {
                        jj = index[indexOff + ip + 1];
                        BLAS.daxpy(ip + 1, -z[zOff + ip + 1], a, aOff + mda * jj, 1, z, zOff, 1);
                    }
                    jj = index[indexOff + ip];
                    z[zOff + ip] /= a[aOff + ip + mda * jj];
                }

                // Check iteration count
                if (++iter > maxIter) {
                    rnorm[0] = (np < m) ? BLAS.dnrm2(m - np, b, bOff + np, 1) : 0.0;
                    return 1;  // Exceeded max iterations
                }

                // See if all new constrained coefficients are feasible.
                // Find index t ∈ P such that x_t/(x_t-z_t) = arg min { x_j/(x_j-z_j) : z_j <= 0, j ∈ P }
                alpha = 2.0;
                jj = -1;
                for (ip = 0; ip < np; ip++) {
                    l = index[indexOff + ip];
                    if (z[zOff + ip] <= 0.0) {
                        // Found unfeasible coefficient, compute alpha.
                        // α = x_t/(x_t-z_t)
                        t = -x[xOff + l] / (z[zOff + ip] - x[xOff + l]);
                        if (alpha > t) {
                            alpha = t;
                            jj = ip;
                        }
                    }
                }

                // If all coefficients are feasible, exit inner loop to main loop
                if (jj < 0) {
                    for (ip = 0; ip < np; ip++) {
                        l = index[indexOff + ip];
                        x[xOff + l] = z[zOff + ip];
                    }
                    break;
                }

                // Interpolate between x and z: x = x + α(s - x)
                for (ip = 0; ip < np; ip++) {
                    l = index[indexOff + ip];
                    x[xOff + l] += alpha * (z[zOff + ip] - x[xOff + l]);
                }

                // Move coefficient i from P to Z
                i = index[indexOff + jj];
                double[] g1Result = new double[3];
                for (;;) {
                    x[xOff + i] = 0.0;
                    if (++jj < np) {
                        for (j = jj; j < np; j++) {
                            ii = index[indexOff + j];
                            int ciOff = aOff + mda * ii;
                            index[indexOff + j - 1] = ii;
                            Householder.g1(a[ciOff + j - 1], a[ciOff + j], g1Result);
                            cc = g1Result[0];
                            ss = g1Result[1];
                            a[ciOff + j - 1] = g1Result[2];
                            a[ciOff + j] = 0.0;
                            for (l = 0; l < n; l++) {
                                if (l != ii) {
                                    int clOff = aOff + mda * l;
                                    Householder.g2(cc, ss, a, clOff + j - 1, clOff + j);
                                }
                            }
                            Householder.g2(cc, ss, b, bOff + j - 1, bOff + j);
                        }
                    }

                    np--;
                    z1--;
                    index[indexOff + z1] = i;

                    // See if the remaining coefficients in P are feasible.
                    // They should be because of the way α was determined.
                    // If any are infeasible, it is due to round-off error.
                    // Any that are non-positive will be set to zero and moved from P to Z.
                    break;
                }

                // Copy b into z, then solve again and loop back
                System.arraycopy(b, bOff, z, zOff, m);
            }
        }

        // Calculate norm-2 of the residual vector: ||Q^T b_2||_2
        if (np < m) {
            rnorm[0] = BLAS.dnrm2(m - np, b, bOff + np, 1);
        } else {
            rnorm[0] = 0.0;
            for (i = 0; i < n; i++) {
                w[wOff + i] = 0.0;
            }
        }
        return 0;  // Success
    }


    // =========================================================================
    // HFTI - Householder Forward Triangulation with column Interchanges
    //
    // Solves a least-squares problem 𝐀𝐗 ≅ 𝐁.
    //   - 𝐀 is m × n matrix with 𝚙𝚜𝚎𝚞𝚍𝚘-𝚛𝚊𝚗𝚔(𝐀) = k
    //   - 𝐗 is n × nb matrix having column vectors 𝐱ⱼ
    //   - 𝐁 is m × nb matrix
    //
    // Basics:
    // Recall the least-squares problem 𝐀𝐱 ≅ 𝐛 where 𝚛𝚊𝚗𝚔(𝐀) = k with below orthogonal transformation.
    //
    //   𝐀ₘₓₙ = 𝐇ₘₓₘ[𝐑ₖₓₖ ೦]𝐊ᵀₙₓₙ   𝐊ᵀ𝐱 = [𝐲₁ 𝐲₂]ᵀ   𝐇ᵀ𝐛 = [𝐠₁ 𝐠₂]ᵀ
    //
    // where 𝐇 and 𝐊 are orthogonal, 𝐑 is full-rank, 𝐲₁, 𝐠₁ is k-vector and 𝐲₂, 𝐠₂ is (n-k)-vector, such that:
    //   - ‖ 𝐀𝐱 - 𝐛 ‖₂ = ‖ 𝐑𝐲₁ - 𝐠₁ ‖₂ + ‖𝐠₂‖₂ (since orthogonal transformation preserve the norm)
    //   - 𝚖𝚒𝚗‖ 𝐀𝐱 - 𝐛 ‖₂ = 𝚖𝚒𝚗‖ 𝐑𝐲₁ - 𝐠₁ ‖₂    (since ‖𝐠₂‖₂ is constant)
    //   - 𝐲₁ = 𝐑⁻¹𝐠₁                          (since 𝐑 is invertible)
    //   - 𝐲₂ is arbitrary                     (usually set 𝐲₂ = O)
    //
    // The unique solution of minimum length is given by 𝐱 = 𝐊[𝐲₁ 𝐲₂]ᵀ = 𝐊[𝐑⁻¹𝐠₁ ೦]ᵀ and the norm of residual satisfies ‖𝐫‖ = ‖𝐠₂‖.
    //
    // When 𝚛𝚊𝚗𝚔(𝐀) = k < 𝚖𝚒𝚗(m,n), there exist orthogonal matrix 𝐐 and permutation matrix 𝐏 such that 𝐐𝐀𝐏 = 𝐑
    //
    //   ⎡𝐑₁₁ 𝐑₁₂⎤  where 𝐑₁₁ is k × k matrix, 𝐑₁₂ is k × (n-k) matrix
    //   ⎣ ೦  𝐑₂₂⎦    and 𝐑₂₂ is (n-k) × (n-k) matrix
    //
    //   - permutation matrix 𝐏 interchange column of 𝐀 resulting first k columns of 𝐀𝐏 is linearly independent
    //   - orthogonal matrix 𝐐 interchange column of 𝐀 resulting 𝐐𝐀𝐏 is zero below the main diagonal
    //
    // HFTI assume 𝐀 is rank-deficient that make problem very ill-conditioned.
    //
    // To stabilizing such problem, HFTI first figure out a 𝚙𝚜𝚎𝚞𝚍𝚘-𝚛𝚊𝚗𝚔(𝐀) = k < 𝛍 where 𝛍 = 𝚖𝚒𝚗(m,n) by computing 𝐑.
    // By setting 𝐑₂₂ = ೦ and replace the 𝐀 with 𝐀߬ = 𝐐ᵀ[𝐑₁₁ 𝐑₁₂]ₙₓₙ𝐏ᵀ and 𝐛 with 𝐜 = 𝐐ᵀ𝐛 = [𝐜₁ 𝐜₂]ᵀ the problem become 𝐀߬ 𝐱 ≅ 𝐜.
    //
    // Since [𝐑₁₁:𝐑₁₂]ₖₓₙ is full-row-rank, its triangulation can be obtained by orthogonal transformation 𝐊
    // such that [𝐑₁₁:𝐑₁₂]𝐊ₙₓₙ = [𝐖ₖₓₖ:೦] and 𝐊ᵀ𝐱 = [𝐲₁ 𝐲₂]ᵀ.
    //   - For forward triangulation, 𝐖 is a non-singular upper triangular matrix
    //   - For backward triangulation, 𝐖 is a non-singular lower triangular matrix
    //
    // The minimum length solution of 𝐀߬ 𝐱 ≅ 𝐜 is given by 𝐱 = 𝐏𝐊[𝐲₁ 𝐲₂]ᵀ = 𝐏𝐊[𝐖⁻¹𝐜₁ ೦]ᵀ.
    // Note that 𝐖 is triangular, computation of 𝐖⁻¹𝐜₁ is simple.
    //
    // Pseudo Rank:
    // The pseudo-rank is not a nature of 𝐀 but determined by a user-specified tolerance 𝛕 > 0.
    // All sub-diagonal elements in 𝐑 = 𝐐𝐀𝐏 are zero and its diagonal elements satisfy |rᵢ₊₁| < |rᵢ| where i = 1, ..., 𝛍-1.
    // The pseudo-rank k equal to the number of diagonal elements of 𝐑 exceeding 𝛕 in magnitude.
    //
    // Column Pivoting:
    //   𝐏 is constructed as product of 𝛍 transposition matrix 𝐏₁ × ··· × 𝐏ᵤ
    //   where 𝐏ⱼ = (j, pⱼ) denotes the interchange between column j and pⱼ.
    //
    //   𝐐 is constructed as product of 𝛍 Householder matrix 𝐐ᵤ × ··· × 𝐐₁
    //   where 𝐐ⱼ corresponding to the j column after interchange interchange.
    //
    // This column is the best candidate for numerical stability.
    // For the construction of j-th Householder transformation, we consider remaining columns j,...,n
    // and select the 𝝺-th column whose sum of squares of components in rows j,...,m is greatest.
    //
    // Algorithm Outline:
    // HFTI first transforms the augmented matrix [𝐀:𝐁] ≡ [𝐑:𝐂] = [𝐐𝐀𝐏:𝐐𝐁] using
    // pre-multiplying Householder transformation 𝐐 with column interchange 𝐏
    // where 𝐀𝐏 is linearly independent and 𝐐 resulting all sub-diagonal elements in 𝐀𝐏 are zero.
    //
    // After determining the pseudo-rank k by diagonal element of 𝐑, apply forward triangulation
    // to 𝐑𝐊 = [𝐖:೦] using Householder transformation 𝐊.
    //
    // Then solve triangular system 𝐖𝐲₁ = 𝐜₁ and apply 𝐊 to 𝐲₁.
    // Finally the solution 𝐱 is obtained by rearranging the 𝐊𝐲₁ = 𝐊𝐖⁻¹𝐜₁ by 𝐏.
    //
    // Memory Layout:
    // The space of input data 𝐀 is will be modified to store the intermediate results:
    //
    //          k        n-k
    //      ┌───┴───┐  ┌──┴──┐
    //   ⎡ w₁₁ w₁₂ w₁₃ k₁₄ k₁₅ ⎤┐          the data that define 𝐐 occupy the lower triangular part of 𝐀
    //   ⎥ u₁₂ w₂₂ w₂₃ k₂₄ k₂₅ ⎥├ k        the data that define 𝐊 occupy the rectangular portion of 𝐀
    //   ⎥ u₁₃ u₂₃ w₃₃ k₃₄ k₃₅ ⎥┘          the data that define 𝐖 occupy the rectangular portion of 𝐀
    //   ⎥ u₁₄ u₂₄ u₃₄  †   †  ⎥┐
    //   ⎥ u₁₅ u₂₅ u₃₅ u₄₅  †  ⎥├ n-k
    //   ⎣ u₁₆ u₂₆ u₃₆ u₄₆ u₅₆ ⎦┘
    //
    // And 3 × 𝚖𝚒𝚗(m,n) additional working space required:
    //
    //   g: [ u₁₁ u₂₂ u₃₃ u₄₄ u₅₅ ]    the pivot scalars for 𝐐
    //   h: [ k₁₁ k₂₂ k₃₃  †   †  ]    the pivot scalars for 𝐊
    //   p: [ p₁  p₂  p₃  p₄  p₅  ]    interchange record define 𝐏
    //
    // Reference:
    //   C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
    //   Prentice Hall, 1974. (revised 1995 edition) Chapters 14, Algorithm 14.9.
    // =========================================================================

    /** Factor for recomputing column norms in HFTI */
    private static final double HFTI_FACTOR = 0.001;

    /**
     * hfti - Solve a least-squares problem AX ≅ B using Householder transformations
     * with column pivoting.
     *
     * @param m        Number of rows in A (either m ≥ n or m < n is permitted)
     * @param n        Number of columns in A (no restriction on rank(A))
     * @param a        Matrix A (column-major), modified on return to store intermediate results
     * @param aOff     Offset into a array
     * @param mda      Leading dimension of A
     * @param b        Matrix B (column-major), contains solution X in first n rows on return
     * @param bOff     Offset into b array
     * @param mdb      Leading dimension of B
     * @param nb       Number of columns in B (right-hand sides), if nb = 0 no reference to b
     * @param tau      Absolute tolerance τ for pseudo-rank determination
     * @param rnorm    Output: residual norms ‖g₂‖ for each column of B
     * @param rnormOff Offset into rnorm array
     * @param h        Working array of length n (column norms and K pivots)
     * @param hOff     Offset into h array
     * @param g        Working array of length min(m,n) (Q pivots)
     * @param gOff     Offset into g array
     * @param ip       Working array of length min(m,n) (permutation P)
     * @param ipOff    Offset into ip array
     * @return         Pseudo-rank k
     */
    static int hfti(
            int m, int n,
            double[] a, int aOff, int mda,
            double[] b, int bOff, int mdb, int nb,
            double tau,
            double[] rnorm, int rnormOff,
            double[] h, int hOff,
            double[] g, int gOff,
            int[] ip, int ipOff) {

        int diag, i, j, jb, k, l, lmax;
        double hmax, sm, t, up, v;

        diag = Math.min(m, n);
        if (diag <= 0) {
            return 0;
        }

        hmax = 0.0;

        for (j = 0; j < diag; j++) {
            /* Update the squared column lengths and find lmax. */
            lmax = j;

            if (j > 0) {
                v = -1e308;  /* Use large negative value for NaN-safe comparison */
                for (l = j; l < n; l++) {
                    t = a[aOff + (j - 1) + mda * l];
                    h[hOff + l] -= t * t;
                    if (!(h[hOff + l] <= v)) {  /* handles NaN correctly */
                        lmax = l;
                        v = h[hOff + l];
                    }
                }
            }

            /* Compute squared column lengths and find lmax. */
            if (j == 0 || HFTI_FACTOR * h[hOff + lmax] < hmax * SLSQPConstants.EPS) {
                v = -1e308;
                for (l = j; l < n; l++) {
                    sm = 0.0;
                    for (i = j; i < m; i++) {
                        t = a[aOff + i + mda * l];
                        sm += t * t;
                    }
                    h[hOff + l] = sm;
                    if (!(h[hOff + l] <= v)) {
                        lmax = l;
                        v = h[hOff + l];
                    }
                }
                hmax = h[hOff + lmax];
            }

            /* Perform column interchange P if needed. */
            ip[ipOff + j] = lmax;
            if (ip[ipOff + j] != j) {
                /* Swap columns j and lmax */
                for (i = 0; i < m; i++) {
                    t = a[aOff + i + mda * j];
                    a[aOff + i + mda * j] = a[aOff + i + mda * lmax];
                    a[aOff + i + mda * lmax] = t;
                }
                h[hOff + lmax] = h[hOff + j];
            }

            /* Compute the j-th transformation and apply it to A and B. */
            int nextCol = Math.min(j + 1, n - 1);
            up = Householder.h1(j, j + 1, m, a, aOff + mda * j, 1);                                      /* Q */
            Householder.h2(j, j + 1, m, a, aOff + mda * j, 1, up, a, aOff + mda * nextCol, 1, mda, n - j - 1); /* R = QAP */
            Householder.h2(j, j + 1, m, a, aOff + mda * j, 1, up, b, bOff, 1, mdb, nb);                        /* C = QB */
        }

        /* Determine the pseudo-rank
         * k = max_j |R_jj| > τ */
        k = diag;
        for (j = 0; j < diag; j++) {
            if (Math.abs(a[aOff + j + mda * j]) <= tau) {
                k = j;
                break;
            }
        }

        /* Compute the norms of the residual vectors ‖g₂‖ ≡ ‖c₂‖ */
        for (jb = 0; jb < nb; jb++) {
            sm = 0.0;
            if (k < m) {
                for (i = k; i < m; i++) {
                    t = b[bOff + i + mdb * jb];
                    sm += t * t;
                }
            }
            rnorm[rnormOff + jb] = Math.sqrt(sm);
        }

        if (k > 0) {
            /* If the pseudo-rank is less than n,
             * compute Householder decomposition of first k rows. */
            if (k < n) {
                for (i = k - 1; i >= 0; i--) {
                    g[gOff + i] = Householder.h1(i, k, n, a, aOff + i, mda);                    /* K */
                    Householder.h2(i, k, n, a, aOff + i, mda, g[gOff + i], a, aOff, mda, 1, i); /* R₁₁K = W */
                }
            }

            /* If B is provided, compute X */
            for (jb = 0; jb < nb; jb++) {
                int cbOff = bOff + mdb * jb;

                /* Solve k × k triangular system Wy₁ = c₁ */
                for (i = k - 1; i >= 0; i--) {
                    sm = 0.0;
                    for (j = i + 1; j < k; j++) {
                        sm += a[aOff + i + mda * j] * b[cbOff + j];
                    }
                    b[cbOff + i] = (b[cbOff + i] - sm) / a[aOff + i + mda * i];
                }

                /* Complete computation of solution vector. */
                if (k < n) {
                    /* Ky₂ = 0 */
                    for (i = k; i < n; i++) {
                        b[cbOff + i] = 0.0;
                    }
                    /* Ky₁ = KW⁻¹c₁ */
                    for (i = 0; i < k; i++) {
                        Householder.h2(i, k, n, a, aOff + i, mda, g[gOff + i], b, cbOff, 1, mdb, 1);
                    }
                }

                /* Re-order solution vector Ky by P to obtain x. */
                for (j = diag - 1; j >= 0; j--) {
                    l = ip[ipOff + j];
                    if (l != j) {
                        t = b[cbOff + l];
                        b[cbOff + l] = b[cbOff + j];
                        b[cbOff + j] = t;
                    }
                }
            }
        } else if (nb > 0) {
            for (jb = 0; jb < nb; jb++) {
                for (i = 0; i < n; i++) {
                    b[bOff + i + mdb * jb] = 0.0;
                }
            }
        }

        /* The solution vectors X are now in the first n rows of B. */
        return k;
    }


    // =========================================================================
    // LDP - Least Distance Programming
    //
    // Solves the problem min ||x||_2 subject to Gx >= h.
    //   - G is m × n matrix (no assumption need to be made for its rank)
    //   - x ∈ ℝⁿ
    //   - h ∈ ℝᵐ
    //
    // NNLS could solve LDP by given:
    //   - an (n+1) × m matrix A = [G : h]^T
    //   - an (n+1)-vector b = [O_n : 1]
    //
    // Assume m-vector u is optimal solution to NNLS solution:
    //   - the residual is an (n+1)-vector r = Au - b = [G^T u : h^T u - 1]^T = [r_1 ... r_n : r_{n+1}]^T
    //   - The dual vector is an m-vector w = A^T(b - Au) = A^T r
    //
    // The w^T u = 0 which is given by:
    //   - w_i >= 0 → u_i = 0
    //   - w_i = 0 → u_i > 0
    //
    // Thus the norm-2 of NNLS residual satisfied: ||r||_2 = r^T r = r^T(Au - b) = (A^T r)u - r^T b = w^T u - r_{n+1} = -r_{n+1}
    //   - ||r||_2 > 0 → r_{n+1} < 0
    //   - ||r||_2 = 0 → r_{n+1} = 0
    //
    // Constraints Gx >= h is satisfied when ||r||_2 > 0 since:
    //
    //   (Gx - h)||r||_2 = [G:h][x:-1]^T(-r_{n+1}) = A^T r = w >= 0
    //
    // Substitute LDP to the KKT conditions:
    //   - f(x) = ½||x||_2                   →  ∇f(x) = x
    //   - g_j(x) = 0  (j = 1 ... m_e)       →  ∇g_j(x) = 0
    //   - g_j(x) = h_j - G_j x (j = m_e+1 ... m) →  ∇g_j(x) = -G
    //
    // the optimality conditions for LDP are given:
    //   - ∇L(x^k, λ^k) = x^k - G^T λ^k = 0
    //   - λ^k_j >= 0 ∀j
    //   - λ^k_j(h_j - G_j x) = 0 ∀j
    //
    // Solution of LDP is given by x = [r_1 ... r_n]^T/(-r_{n+1}) = G^T u / ||r||_2.
    // The Lagrange multiplier of LDP inequality constraint λ = G^(-1) x = u / ||r||_2.
    //
    // References
    // ----------
    // C.L. Lawson, R.J. Hanson, 'Solving least squares problems' Prentice Hall, 1974. (revised 1995 edition)
    // Chapters 23, Algorithm 23.27.
    // =========================================================================

    /**
     * ldp - Solve the Least Distance Programming problem.
     * <p>
     * Solves min ||x||_2 subject to Gx >= h
     * </p>
     * <p>
     * The algorithm transforms this to NNLS form:
     * <ol>
     *   <li>Form matrix A = [G^T; h^T] (augmented matrix)</li>
     *   <li>Form vector b = [0; ...; 0; 1] (unit vector)</li>
     *   <li>Solve NNLS: minimize ||Au - b||_2 subject to u >= 0</li>
     *   <li>Recover solution: x = G^T u / ||r|| where r = b - Au</li>
     *   <li>Compute Lagrange multipliers: λ = u / ||r||</li>
     * </ol>
     * </p>
     * <p>
     * Strictly follows the implementation in ldp.c.
     * </p>
     *
     * @param m       Number of constraints (rows in G)
     * @param n       Number of variables (columns in G)
     * @param g       Constraint matrix G (column-major, m × n)
     * @param gOff    Offset into array g
     * @param mdg     Leading dimension of G
     * @param h       Constraint vector h (m-vector)
     * @param hOff    Offset into array h
     * @param x       Output: solution vector x (n-vector)
     * @param xOff    Offset into array x
     * @param w       Working array of length (n+1)×(m+2)+2m
     *                On return, w[wOff:wOff+m] contains Lagrange multipliers λ
     * @param wOff    Offset into array w
     * @param jw      Working array of length m
     * @param jwOff   Offset into array jw
     * @param maxIter Maximum iterations for NNLS
     * @param xnorm   Output: ||x||_2 (array of length 1)
     * @return        Status code (0 = success, -1 = bad argument, -2 = constraints incompatible)
     */
    static int ldp(
            int m, int n,
            double[] g, int gOff, int mdg,
            double[] h, int hOff,
            double[] x, int xOff,
            double[] w, int wOff,
            int[] jw, int jwOff,
            int maxIter,
            double[] xnorm) {

        int i, j, iw, status;
        double fac, rnorm;

        // Pointers into working array
        int aOff, bOff, zOff, uOff, dvOff;

        if (n <= 0) {
            return -1;  // Bad argument
        }

        if (m <= 0) {
            xnorm[0] = 0.0;
            return 0;  // OK
        }

        /*
         * Working space layout:
         * w[:(n+1)×m]                     =  (n+1)×m matrix A
         * w[(n+1)×m:(n+1)×(m+1)]          =  (n+1)-vector b
         * w[(n+1)×(m+1):(n+1)×(m+2)]      =  (n+1)-vector z (working space)
         * w[(n+1)×(m+2):(n+1)×(m+2)+m]    =  m-vector u
         * w[(n+1)×(m+2)+m:(n+1)×(m+2)+2m] =  m-vector dv (dual)
         */

        iw = wOff;
        aOff = iw;
        iw += m * (n + 1);
        bOff = iw;
        iw += (n + 1);
        zOff = iw;
        iw += (n + 1);
        uOff = iw;
        iw += m;
        dvOff = iw;

        for (j = 0; j < m; j++) {
            // Copy G^T into first n rows and m columns of A
            BLAS.dcopy(n, g, gOff + j, mdg, w, aOff + j * (n + 1), 1);
            // Copy h^T into row n+1 of A
            w[aOff + j * (n + 1) + n] = h[hOff + j];
        }

        // Initialize b = [O_n : 1]
        for (i = 0; i < n; i++) {
            w[bOff + i] = 0.0;
        }
        w[bOff + n] = 1.0;

        // Solve NNLS problem: min ||Au - b||_2 subject to u >= 0
        double[] rnormArr = new double[1];
        status = nnls(n + 1, m, w, aOff, n + 1, w, bOff, w, uOff, w, dvOff, w, zOff, jw, jwOff, maxIter, rnormArr);
        rnorm = rnormArr[0];

        if (status == 0) {
            if (rnorm <= 0.0) {
                // ||r||_2 = 0 → constraints incompatible
                return SLSQPConstants.ERR_CONS_INCOMPAT;
            }

            // fac = -r_{n+1} = 1 - h^T u
            fac = 1.0 - BLAS.ddot(m, h, hOff, 1, w, uOff, 1);

            if (Double.isNaN(fac) || fac < SLSQPConstants.EPS) {
                // Constraints incompatible
                return SLSQPConstants.ERR_CONS_INCOMPAT;
            }

            fac = 1.0 / fac;

            // x = G^T u / ||r||_2
            for (j = 0; j < n; j++) {
                x[xOff + j] = BLAS.ddot(m, g, gOff + mdg * j, 1, w, uOff, 1) * fac;
            }

            // Store Lagrange multipliers: λ = u / ||r||_2
            for (j = 0; j < m; j++) {
                w[wOff + j] = w[uOff + j] * fac;
            }

            xnorm[0] = BLAS.dnrm2(n, x, xOff, 1);  // ||x||_2
            return 0;  // Success
        }

        return status;  // NNLS error
    }


    // =========================================================================
    // LSI - Least Squares with Inequality constraints
    //
    // Solves the problem 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ subject to 𝐆𝐱 ≥ 𝐡.
    //   - 𝐄 is m × n matrix with 𝚛𝚊𝚗𝚔(𝐄) = n
    //   - 𝐟 ∈ ℝⁿ
    //   - 𝐛 ∈ ℝᵐ
    //   - 𝐆 is mg × n matrix
    //   - 𝐡 ∈ ℝᵐᵍ
    //
    // Consider below orthogonal decomposition of 𝐄
    //
    //                   n    m-n
    //                  ┌┴┐   ┌┴┐
    //   𝐄 = 𝐐⎡𝐑 ೦⎤𝐊ᵀ ≡ [ 𝐐₁ : 𝐐₂ ]⎡𝐑⎤ 𝐊ᵀ
    //        ⎣೦ ೦⎦                 ⎣೦⎦
    //
    // where:
    //   - 𝐐 is m × m orthogonal
    //   - 𝐊 is n × n orthogonal
    //   - 𝐑 is n × n non-singular
    //
    // By introducing orthogonal change of variable 𝐱 = 𝐊ᵀ𝐲 one can obtain:
    //
    //   ⎡𝐐₁ᵀ⎤(𝐄𝐱 - 𝐟) = ⎡𝐑𝐲 - 𝐐₁ᵀ𝐟⎤
    //   ⎣𝐐₂ᵀ⎦          ⎣   𝐐₂ᵀ𝐟  ⎦
    //
    // Since orthogonal transformation does not change matrix norm and ‖ 𝐐₂ᵀ𝐟 ‖₂ is constant,
    // the LSI objective could be rewritten as 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ = 𝚖𝚒𝚗‖ 𝐑𝐲 - 𝐐₁ᵀ𝐟 ‖₂.
    //
    // By following definitions:
    //   - 𝐟߫₁ = 𝐐₁ᵀ𝐟
    //   - 𝐟߫₂ = 𝐐₂ᵀ𝐟
    //   - 𝐳 = 𝐑𝐲 - 𝐟߫₁
    //   - 𝐱 = 𝐊𝐑⁻¹(𝐳 + 𝐟߫₁)
    //
    // the LSI problem is equivalent to LDP problem 𝚖𝚒𝚗 ‖ 𝐳 ‖₂ subject to 𝐆𝐊𝐑⁻¹𝐳 ≥ 𝐡 - 𝐆𝐊𝐑⁻¹𝐟߫₁
    // and the residual vector norm of LSI problem can be computed from (‖ 𝐳 ‖₂ + ‖ 𝐟߫₂ ‖₂)¹ᐟ².
    //
    // References:
    //   C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
    //   Prentice Hall, 1974. (revised 1995 edition)
    //   Chapters 23, Section 5.
    // =========================================================================
    // =========================================================================

    /**
     * lsi - Solve the Least Squares with Inequality constraints problem.
     * <p>
     * Solves min ||Ex - f||_2 subject to Gx >= h
     * </p>
     * <p>
     * The algorithm:
     * <ol>
     *   <li>Compute QR factorization of E using Householder transformations: QE = [R; 0]</li>
     *   <li>Apply Q to f: Qf = [f1; f2]</li>
     *   <li>Transform to LDP: minimize ||y||_2 subject to GR^(-1)y >= h - GR^(-1)f1</li>
     *   <li>Call LDP to solve the transformed problem</li>
     *   <li>Recover solution: x = R^(-1)(y + f1)</li>
     *   <li>Compute residual norm: ||f2||_2</li>
     * </ol>
     * </p>
     * <p>
     * Strictly follows the implementation in lsei.c (lsi function).
     * </p>
     *
     * @param e       Matrix E (column-major, me × n), modified on return
     * @param eOff    Offset into array e
     * @param f       Vector f (me-vector), modified on return
     * @param fOff    Offset into array f
     * @param g       Matrix G (column-major, mg × n), modified on return
     * @param gOff    Offset into array g
     * @param h       Vector h (mg-vector), modified on return
     * @param hOff    Offset into array h
     * @param le      Leading dimension of E
     * @param me      Number of rows in E
     * @param lg      Leading dimension of G
     * @param mg      Number of rows in G (constraints)
     * @param n       Number of variables
     * @param x       Output: solution vector x (n-vector)
     * @param xOff    Offset into array x
     * @param w       Working array of length (n+1)×(mg+2)+2×mg
     * @param wOff    Offset into array w
     * @param jw      Working array of length mg
     * @param jwOff   Offset into array jw
     * @param maxIter Maximum iterations for LDP
     * @param xnorm   Output: residual norm (array of length 1)
     * @return        Status code (0 = success, -1 = bad argument, -3 = E singular)
     */
    static int lsi(
            double[] e, int eOff, double[] f, int fOff,
            double[] g, int gOff, double[] h, int hOff,
            int le, int me, int lg, int mg, int n,
            double[] x, int xOff,
            double[] w, int wOff,
            int[] jw, int jwOff,
            int maxIter,
            double[] xnorm) {

        int i, j, status;
        double t, diag;

        if (n < 1) {
            return -1;  // Bad argument
        }

        // QR-factors of E and application to f
        for (i = 0; i < n; i++) {
            j = (i + 1 < n - 1) ? i + 1 : n - 1;
            // Compute Householder transformation for column i
            t = Householder.h1(i, i + 1, me, e, eOff + i * le, 1);
            // Apply to remaining columns of E: QE = R (triangular)
            Householder.h2(i, i + 1, me, e, eOff + i * le, 1, t, e, eOff + j * le, 1, le, n - i - 1);
            // Apply to f: Qf = [ f̃₁ : f̃₂ ]
            Householder.h2(i, i + 1, me, e, eOff + i * le, 1, t, f, fOff, 1, 1, 1);
        }

        // Transform G and h to get LDP
        for (i = 0; i < mg; i++) {
            for (j = 0; j < n; j++) {
                diag = e[eOff + j + le * j];
                if (Math.abs(diag) < SLSQPConstants.EPS || Double.isNaN(diag)) {
                    return SLSQPConstants.ERR_LSI_SINGULAR_E;  // rank(E) < n (E is singular)
                }
                // GKR⁻¹ (K = Iₙ)
                g[gOff + i + lg * j] = (g[gOff + i + lg * j] - BLAS.ddot(j, g, gOff + i, lg, e, eOff + j * le, 1)) / diag;
            }
            // h - GKR⁻¹f̃₁
            h[hOff + i] -= BLAS.ddot(n, g, gOff + i, lg, f, fOff, 1);
        }

        // Solve LDP: min || z ||₂  subject to  GKR⁻¹z >= h - GKR⁻¹f̃₁
        status = ldp(mg, n, g, gOff, lg, h, hOff, x, xOff, w, wOff, jw, jwOff, maxIter, xnorm);

        if (status == 0) {
            // z + f̃₁
            BLAS.daxpy(n, 1.0, f, fOff, 1, x, xOff, 1);

            // KR⁻¹(z + f̃₁)
            for (i = n - 1; i >= 0; i--) {
                j = (i + 1 < n - 1) ? i + 1 : n - 1;
                x[xOff + i] = (x[xOff + i] - BLAS.ddot(n - i - 1, e, eOff + i + le * j, le, x, xOff + j, 1)) / e[eOff + i + le * i];
            }

            // Compute residual norm: (|| z ||₂ + || f̃₂ ||₂)^(1/2)
            j = (n < me - 1) ? n : me - 1;
            t = BLAS.dnrm2(me - n, f, fOff + j, 1);  // || f̃₂ ||₂
            xnorm[0] = Math.hypot(xnorm[0], t);
        }

        return status;
    }


    // =========================================================================
    // LSEI - Least Squares with Equality and Inequality constraints
    //
    // Solves the problem 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ subject to 𝐂𝐱 = 𝐝 and 𝐆𝐱 ≥ 𝐡.
    //   - 𝐄 is me × n matrix (no assumption need to be made for its rank)
    //   - 𝐱 ∈ ℝⁿ
    //   - 𝐟 ∈ ℝᵐ
    //   - 𝐂 is mc × n matrix with 𝚛𝚊𝚗𝚔(𝐂) = k = mc < n
    //   - 𝐝 ∈ ℝᵐᶜ
    //   - 𝐆 is mg × n matrix
    //   - 𝐡 ∈ ℝᵐᵍ
    //
    // LSE Problem:
    // -----------
    // Consider a LSE (Least-Squares with linear Equality) problem 𝚖𝚒𝚗‖ 𝐀𝐱 - 𝐛 ‖₂ subject to 𝐂𝐱 = 𝐝.
    //
    // Given an orthogonal transformation of matrix 𝐂 where 𝐇 and 𝐊 are orthogonal, 𝐑 is full-rank:
    //   𝐂ₘ₁ₓₙ = 𝐇ₘ₁ₓₘ₁[𝐑ₖₓₖ ೦]𝐊ᵀₘ₁ₓₙ
    //
    // Its pseudo-inverse is defined by 𝐂⁺ = 𝐊𝐑⁺𝐇ᵀ where 𝐑⁺ = [𝐑⁻¹ ೦].
    //
    // Define partition 𝐊 = [𝐊₁ 𝐊₂] and [𝐊₁ 𝐊₂]ᵀ𝐱 = [𝐲₁ 𝐲₂]ᵀ where
    // 𝐊₁ is an n × k matrix, 𝐊₂ is an n × (n-k) matrix.
    //
    // Assume k = mc such that 𝐇 = 𝐈 and let 𝐊 satisfied that 𝐂𝐊 is lower triangular:
    //
    //   ⎡ 𝐂 ⎤ 𝐊 = ⎡ 𝐂߬₁  ೦  ⎤
    //   ⎣ 𝐀 ⎦     ⎣ 𝐀߬₁  𝐀߬₂ ⎦
    //
    // The solution of LSE problem is given by 𝐱߮ = 𝐊[𝐲߮₁ 𝐲߮₂]ᵀ where:
    //   - 𝐲߮₁ is obtained by solving triangular system 𝐂߬₁𝐲₁ = 𝐝
    //   - 𝐲߮₂ is obtained by solving least-squares 𝐀߬₂𝐲₂ ≅ 𝐛 - 𝐀߬₁𝐲߮₁
    //
    // Reduce to LSI:
    // -------------
    // Using the conclusion of LSE, the equality constraints can be eliminated by introducing
    // orthogonal basis 𝐊 = [𝐊₁:𝐊₂] of null space 𝐂𝐊₂ = 0 and let 𝐊ᵀ𝐱 = [𝐲₁ 𝐲₂]ᵀ such that:
    //
    //              mc  n-mc
    //             ┌┴┐  ┌┴┐
    //   ⎡ 𝐂 ⎤ 𝐊 = ⎡ 𝐂߬₁   ೦  ⎤ ]╴mc       𝐱 = 𝐊⎡ 𝐲₁ ⎤ ]╴ mc
    //   ⎥ 𝐄 ⎥     ⎥ 𝐄߬₁   𝐄߬₂ ⎥ ]╴me            ⎣ 𝐲₂ ⎦ ]╴ n-mc
    //   ⎣ 𝐆 ⎦     ⎣ 𝐆߬₁   𝐆߬₂ ⎦ ]╴mg
    //
    // The 𝐲߮₁ is determined as solution of triangular system 𝐂߬₁𝐲₁ = 𝐝,
    // and 𝐲߮₂ is the solution of LSI problem 𝚖𝚒𝚗‖ 𝐄߬₂𝐲₂ - (𝐟 - 𝐄߬₂𝐲߮₁) ‖₂ subject to 𝐆߬₂𝐲₂ ≥ 𝐡 - 𝐆߬₁𝐲߮₁.
    //
    // Finally the solution of LSEI problem is given by 𝐱߮ = 𝐊[𝐲߮₁ 𝐲߮₂]ᵀ.
    //
    // Lagrange Multiplier:
    // -------------------
    // 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ subject to 𝐂𝐱 = 𝐝 and 𝐆𝐱 ≥ 𝐡.
    // Substitute LSEI to the KKT conditions:
    //   - 𝒇(𝐱) = ½‖ 𝐄𝐱 - 𝐟 ‖₂               →  𝜵𝒇(𝐱) = 𝐄ᵀ(𝐄𝐱 - 𝐟)
    //   - 𝒈ⱼ(𝐱) = 𝐝ⱼ - 𝐂ⱼ𝐱 (j = 1 ··· mₑ)   →  𝜵𝒈ⱼ(𝐱) = -𝐂
    //   - 𝒈ⱼ(𝐱) = 𝐡ⱼ - 𝐆ⱼ𝐱 (j = mₑ+1 ··· m) →  𝜵𝒈ⱼ(𝐱) = -𝐆
    //
    // The optimality conditions for LSEI are given:
    //   - 𝜵ℒ(𝐱ᵏ,𝛍ᵏ,𝛌ᵏ) = 𝐄ᵀ(𝐄𝐱 - 𝐟) - 𝐂ᵀ𝛍ᵏ - 𝐆ᵀ𝛌ᵏ = 0
    //   - 𝛌ᵏⱼ ≥ 0 (j = mₑ+1 ··· m)
    //   - 𝛌ᵏⱼ(𝐡ⱼ - 𝐆ⱼ𝐱) = 0 (j = mₑ+1 ··· m)
    //
    // Multiplier of inequality constraints:
    //   - when mᵍ = 0, the ine multiplier 𝛌 = 0
    //   - when mᵍ > 0, 𝛌 is solving by LDP
    //
    // Multiplier of equality constraints is given by 𝛍ᵏ = (𝐂ᵀ)⁻¹[𝐄ᵀ(𝐄𝐱 - 𝐟) - 𝐆ᵀ𝛌ᵏ].
    //
    // References:
    //   C.L. Lawson, R.J. Hanson, 'Solving least squares problems'
    //   Prentice Hall, 1974. (revised 1995 edition)
    //   Chapters 20, Algorithm 20.24.
    //   Chapters 23, Section 6.
    // =========================================================================

    /**
     * lsei - Solve the Least Squares with Equality and Inequality constraints problem.
     * <p>
     * Solves min ||Ex - f||_2 subject to Cx = d (equality) and Gx >= h (inequality)
     * </p>
     * <p>
     * The algorithm:
     * <ol>
     *   <li>Triangularize C using Householder transformations: CK = [C̃₁ 0]</li>
     *   <li>Apply K to E and G: EK = [Ẽ₁ Ẽ₂], GK = [G̃₁ G̃₂]</li>
     *   <li>Solve triangular system C̃₁y₁ = d for ŷ₁</li>
     *   <li>If inequality constraints exist, call LSI to solve the reduced problem</li>
     *   <li>If no inequality constraints, call HFTI to solve the unconstrained least squares</li>
     *   <li>Compute Lagrange multipliers for both equality and inequality constraints</li>
     * </ol>
     * </p>
     * <p>
     * Strictly follows the implementation in lsei.c (lsei function).
     * </p>
     *
     * @param c       Matrix C (column-major, mc × n), modified on return
     * @param cOff    Offset into array c
     * @param d       Vector d (mc-vector), modified on return
     * @param dOff    Offset into array d
     * @param e       Matrix E (column-major, me × n), modified on return
     * @param eOff    Offset into array e
     * @param f       Vector f (me-vector), modified on return
     * @param fOff    Offset into array f
     * @param g       Matrix G (column-major, mg × n), modified on return
     * @param gOff    Offset into array g
     * @param h       Vector h (mg-vector), modified on return
     * @param hOff    Offset into array h
     * @param lc      Leading dimension of C
     * @param mc      Number of equality constraints (rows in C)
     * @param le      Leading dimension of E
     * @param me      Number of rows in E
     * @param lg      Leading dimension of G
     * @param mg      Number of inequality constraints (rows in G)
     * @param n       Number of variables
     * @param x       Output: solution vector x (n-vector)
     * @param xOff    Offset into array x
     * @param w       Working array: 2×mc+me+(me+mg)×(n-mc) + (n-mc+1)×(mg+2)+2×mg
     *                On return: Lagrange multipliers μ = w[0:mc] and λ = w[mc:mc+mg]
     * @param wOff    Offset into array w
     * @param jw      Working array: max(mg, min(me, n-mc))
     * @param jwOff   Offset into array jw
     * @param maxIter Maximum iterations for LDP
     * @param norm    Output: residual norm (array of length 1)
     * @return        Status code (0 = success, -1 = bad argument, -4 = C singular, -5 = HFTI rank defect)
     */
    static int lsei(
            double[] c, int cOff, double[] d, int dOff,
            double[] e, int eOff, double[] f, int fOff,
            double[] g, int gOff, double[] h, int hOff,
            int lc, int mc, int le, int me, int lg, int mg, int n,
            double[] x, int xOff,
            double[] w, int wOff,
            int[] jw, int jwOff,
            int maxIter,
            double[] norm) {

        int i, j, l, iw, status, rank;
        double t, diag;
        int wsOff, wpOff, weOff, wfOff, wgOff;
        int k;

        if (n < 1 || mc > n) {
            return -1;  // Bad argument
        }

        l = n - mc;

        /* Working space layout (matching C implementation):
         * w[0:mc]                          = Lagrange multipliers for equality constraints (μ)
         * w[mc:mc+(l+1)*(mg+2)+2*mg]       = workspace for LSI (ws)
         * w[...+mc]                        = Householder pivots for K (wp)
         * w[...+me*l]                      = Ẽ₂ (we)
         * w[...+me]                        = f - Ẽ₁ŷ₁ (wf)
         * w[...+mg*l]                      = G̃₂ (wg)
         */

        iw = wOff + mc;
        wsOff = iw;
        iw += (l + 1) * (mg + 2) + 2 * mg;
        wpOff = iw;
        iw += mc;
        weOff = iw;
        iw += me * l;
        wfOff = iw;
        iw += me;
        wgOff = iw;

        /* Triangularize C and apply factors to E and G */
        for (i = 0; i < mc; i++) {
            j = (i + 1 < lc - 1) ? i + 1 : lc - 1;
            // Compute Householder transformation for row i of C
            w[wpOff + i] = Householder.h1(i, i + 1, n, c, cOff + i, lc);
            // Apply to remaining rows of C: CK = [C̃₁ 0]
            Householder.h2(i, i + 1, n, c, cOff + i, lc, w[wpOff + i], c, cOff + j, lc, 1, mc - i - 1);
            // Apply to E: EK = [Ẽ₁ Ẽ₂]
            Householder.h2(i, i + 1, n, c, cOff + i, lc, w[wpOff + i], e, eOff, le, 1, me);
            // Apply to G: GK = [G̃₁ G̃₂]
            Householder.h2(i, i + 1, n, c, cOff + i, lc, w[wpOff + i], g, gOff, lg, 1, mg);
        }

        /* Solve triangular system C̃₁y₁ = d */
        for (i = 0; i < mc; i++) {
            diag = c[cOff + i + lc * i];
            if (Math.abs(diag) < SLSQPConstants.EPS) {
                return SLSQPConstants.ERR_LSEI_SINGULAR_C;  // rank(C) < mc (C is singular)
            }
            // ŷ₁ = C̃₁⁻¹d
            x[xOff + i] = (d[dOff + i] - BLAS.ddot(i, c, cOff + i, lc, x, xOff, 1)) / diag;
        }

        /* First [mg] of working space store the multiplier returned by LDP */
        for (i = 0; i < mg; i++) {
            w[wsOff + i] = 0.0;
        }

        if (mc < n) {  // rank(C) < n
            /* f - Ẽ₁ŷ₁ */
            for (i = 0; i < me; i++) {
                w[wfOff + i] = f[fOff + i] - BLAS.ddot(mc, e, eOff + i, le, x, xOff, 1);
            }

            if (l > 0) {
                /* Copy Ẽ₂ */
                for (i = 0; i < me; i++) {
                    BLAS.dcopy(l, e, eOff + i + le * mc, le, w, weOff + i, me);
                }
                /* Copy G̃₂ */
                for (i = 0; i < mg; i++) {
                    BLAS.dcopy(l, g, gOff + i + lg * mc, lg, w, wgOff + i, mg);
                }
            }

            if (mg > 0) {
                /* h - G̃₁ŷ₁ */
                for (i = 0; i < mg; i++) {
                    h[hOff + i] -= BLAS.ddot(mc, g, gOff + i, lg, x, xOff, 1);
                }

                /* Compute ŷ₂ by solving LSI: min|| Ẽ₂y₂ - (f - Ẽ₂ŷ₁) ||₂  s.t  G̃₂y₂ >= h - G̃₁ŷ₁ */
                status = lsi(w, weOff, w, wfOff, w, wgOff, h, hOff,
                        me, me, mg, mg, l,
                        x, xOff + mc,
                        w, wsOff,
                        jw, jwOff,
                        maxIter, norm);

                if (mc == 0) {
                    /* Multipliers returned as λ = w[0:mg] */
                    return status;
                }

                if (status != 0) {
                    return status;
                }

                t = BLAS.dnrm2(mc, x, xOff, 1);
                norm[0] = Math.hypot(norm[0], t);
            } else {
                /* Solve unconstrained: min|| Ẽ₂y₂ - (f - Ẽ₂ŷ₁) ||₂ */
                k = (le > n) ? le : n;
                double[] nrm = new double[1];

                rank = hfti(me, l, w, weOff, me, w, wfOff, k, 1,
                        SLSQPConstants.SQRT_EPS, nrm, 0, w, wOff, w, wOff + l, jw, jwOff);
                norm[0] = nrm[0];
                BLAS.dcopy(l, w, wfOff, 1, x, xOff + mc, 1);

                if (rank != l) {
                    return SLSQPConstants.ERR_HFTI_RANK_DEFECT;  // HFTI rank defect
                }
            }
        }

        /* Eᵀ(Ex - f) */
        for (i = 0; i < me; i++) {
            f[fOff + i] = BLAS.ddot(n, e, eOff + i, le, x, xOff, 1) - f[fOff + i];
        }

        /* Eᵀ(Ex - f) - Gᵀλ */
        for (i = 0; i < mc; i++) {
            d[dOff + i] = BLAS.ddot(me, e, eOff + i * le, 1, f, fOff, 1) -
                    BLAS.ddot(mg, g, gOff + i * lg, 1, w, wsOff, 1);
        }

        /* x̂ = K[ŷ₁ ŷ₂]ᵀ */
        for (i = mc - 1; i >= 0; i--) {
            Householder.h2(i, i + 1, n, c, cOff + i, lc, w[wpOff + i], x, xOff, 1, 1, 1);
        }

        /* μ = (Cᵀ)⁻¹[Eᵀ(Ex - f) - Gᵀλ] */
        for (i = mc - 1; i >= 0; i--) {
            j = (i + 1 < lc - 1) ? i + 1 : lc - 1;
            w[wOff + i] = (d[dOff + i] - BLAS.ddot(mc - i - 1, c, cOff + j + lc * i, 1, w, wOff + j, 1)) / c[cOff + i + lc * i];
        }

        /* Copy λ multipliers from ws to w[mc:mc+mg] */
        for (i = 0; i < mg; i++) {
            w[wOff + mc + i] = w[wsOff + i];
        }

        /* Multipliers returned as μ = w[0:mc] and λ = w[mc:mc+mg] */
        return 0;  // Success
    }



    // =========================================================================
    // LSQ - Least Squares Quadratic Programming
    //
    // LSQ (Least Squares Quadratic programming) solves the problem
    //
    //   minimize ‖ 𝐃¹ᐟ²𝐋ᵀ𝐱 + 𝐃⁻¹ᐟ²𝐋⁻¹𝐠 ‖₂ subject to
    //     - 𝐀ⱼ𝐱 - 𝐛ⱼ = 0  (j = 1 ··· mₑ)
    //     - 𝐀ⱼ𝐱 - 𝐛ⱼ ≥ 0  (j = mₑ+1 ··· m)
    //     - 𝒍ᵢ ≤ 𝐱ᵢ ≤ 𝒖ᵢ (i = 1 ··· n)
    //
    // where
    //   - 𝐋 is an n × n lower triangular matrix with unit diagonal elements
    //   - 𝐃 is an n × n diagonal matrix
    //   - 𝐠 is an n-vector
    //   - 𝐀 is an m × n matrix
    //   - 𝐛 is an m-vector
    //
    // LSQ can be solved as LSEI problem 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ subject to 𝐂𝐱 = 𝐝 and 𝐆𝐱 ≥ 𝐡 with:
    //   - 𝐄 = 𝐃¹ᐟ²𝐋ᵀ
    //   - 𝐟 = -𝐃⁻¹ᐟ²𝐋⁻¹𝐠
    //   - 𝐂 = { 𝐀ⱼ: j = 1 ··· mₑ }
    //   - 𝐝 = { -𝐛ⱼ: j = 1 ··· mₑ }
    //   - 𝐆ⱼ = { 𝐀ⱼ: j = mₑ+1 ··· m }
    //   - 𝐡ⱼ = { -𝐛ⱼ: j = mₑ+1 ··· m }
    //
    // and the bounds is equivalent to inequality constraints 𝐈𝐱 ≥ 𝒍 and -𝐈𝐱 ≥ -𝒖 such that:
    //   - 𝐆ⱼ = { 𝐈ⱼ: j = m+1 ··· m+n }
    //   - 𝐡ⱼ = { 𝒍ⱼ: j = m+1 ··· m+n }
    //   - 𝐆ⱼ = { -𝐈ⱼ: j = m+n ··· m+2n }
    //   - 𝐡ⱼ = { -𝒖ⱼ: j = m+n ··· m+2n }
    //
    // where
    //   - 𝐄 is an n × n upper triangular matrix
    //   - 𝐟 is an n-vector
    //   - 𝐂 is an mₑ × n matrix
    //   - 𝐝 is an mₑ-vector
    //   - 𝐆 is an (m-mₑ+2n) × n matrix
    //   - 𝐡 is an (m-mₑ+2n)-vector
    // =========================================================================

    /**
     * lsq - Solve the QP subproblem using LSEI.
     * <p>
     * Solves the QP subproblem:
     * </p>
     * <pre>
     *   minimize ‖ D^(1/2)L^T x + D^(-1/2)L^(-1)g ‖₂
     *
     *   subject to:
     *     - A_j x - b_j = 0  (j = 1 ··· m_eq)   [equality constraints]
     *     - A_j x - b_j >= 0  (j = m_eq+1 ··· m) [inequality constraints]
     *     - l <= x <= u                          [bound constraints]
     * </pre>
     * <p>
     * The LDL^T factorization B = LDL^T is used where:
     * </p>
     * <ul>
     *   <li>L is lower triangular with unit diagonal</li>
     *   <li>D is diagonal</li>
     * </ul>
     * <p>
     * This is transformed to LSEI problem min‖ Ex - f ‖₂ s.t. Cx = d, Gx >= h:
     * </p>
     * <ul>
     *   <li>E = D^(1/2)L^T</li>
     *   <li>f = -D^(-1/2)L^(-1)g</li>
     *   <li>C = { A_j: j = 1 ··· m_eq }</li>
     *   <li>d = { -b_j: j = 1 ··· m_eq }</li>
     *   <li>G = { A_j: j = m_eq+1 ··· m } ∪ { ±I for bounds }</li>
     *   <li>h = { -b_j: j = m_eq+1 ··· m } ∪ { l, -u for bounds }</li>
     * </ul>
     * <p>
     * Strictly follows the implementation in lsq.c.
     * </p>
     *
     * @param m       Total number of constraints (m = m_eq + m_ineq)
     * @param meq     Number of equality constraints (m_eq)
     * @param n       Number of variables
     * @param nl      Length of l array: n(n+1)/2 + 1 for normal, +1 for augmented
     * @param l       L + D in packed form (LDL^T factorization of Hessian)
     * @param lOff    Offset into array l
     * @param g       Gradient vector g = ∇f(x^k)
     * @param gOff    Offset into array g
     * @param a       Constraint Jacobian A (column-major, leading dimension = max(m,1))
     * @param aOff    Offset into array a
     * @param b       Constraint values b = c(x^k)
     * @param bOff    Offset into array b
     * @param xl      Lower bounds l
     * @param xlOff   Offset into array xl
     * @param xu      Upper bounds u
     * @param xuOff   Offset into array xu
     * @param x       Output: solution vector x
     * @param xOff    Offset into array x
     * @param y       Output: Lagrange multipliers λ (m + 2n elements)
     * @param yOff    Offset into array y
     * @param w       Working array
     * @param wOff    Offset into array w
     * @param jw      Working array (integer)
     * @param jwOff   Offset into array jw
     * @param maxIter Maximum iterations for NNLS solver
     * @param infBnd  Infinity bound value (bounds beyond this are ignored)
     * @param norm    Output: residual norm ‖Ex - f‖₂ (array of length 1)
     * @return        Status code:
     *                 0 = HasSolution (success)
     *                -2 = ConsIncompatible (constraints incompatible)
     *                -3 = LSISingularE (singular E matrix in LSI)
     *                -4 = LSEISingularC (singular C matrix in LSEI)
     *                -5 = HFTIRankDefect (rank defect in HFTI)
     */
    static int lsq(
            int m, int meq, int n, int nl,
            double[] l, int lOff,
            double[] g, int gOff,
            double[] a, int aOff,
            double[] b, int bOff,
            double[] xl, int xlOff,
            double[] xu, int xuOff,
            double[] x, int xOff,
            double[] y, int yOff,
            double[] w, int wOff,
            int[] jw, int jwOff,
            int maxIter, double infBnd,
            double[] norm) {

        int i, j, i2, i3, i4, la, mineq, m1, n1, n2, n3, bnd, status;
        int e0, f0, c0, d0, g0, h0, w0;
        double diag;

        mineq = m - meq;
        m1 = mineq + n + n;  // Total inequality constraints including bounds
        la = (m > 1) ? m : 1;  // Leading dimension of A, matches C: max(m, 1)

        // Determine problem type
        n1 = n + 1;
        if ((n + 1) * n / 2 + 1 == nl) {
            // Solve the original problem m × n
            n2 = 0;
            n3 = n;
        } else {
            // Solve the augmented problem m × (n+1)
            n2 = 1;
            n3 = n - 1;
        }

        /* Working space indices - matches C implementation
         * Layout: [E(n×n) | f(n) | C(meq×n) | d(meq) | G(m1×n) | h(m1) | workspace]
         */
        e0 = wOff;                      // Start index of E: n×n upper triangular
        f0 = e0 + n * n;                // Start index of f: n-vector
        c0 = f0 + n;                    // Start index of C: meq×n matrix
        d0 = c0 + meq * n;              // Start index of d: meq-vector
        g0 = d0 + meq;                  // Start index of G: m1×n matrix
        h0 = g0 + m1 * n;               // Start index of h: m1-vector
        w0 = h0 + m1;                   // Start index of workspace

        /* =========================================================================
         * Recover matrix E and vector f from L, D, and g
         *
         * LDL^T Factorization Recovery:
         *   𝐄 = 𝐃¹ᐟ²𝐋ᵀ  (upper triangular)
         *   𝐟 = -𝐃⁻¹ᐟ²𝐋⁻¹𝐠
         * ========================================================================= */
        i2 = 0;
        i3 = 0;
        i4 = 0;

        for (j = 0; j < n3; j++) {
            i = n - j;
            diag = Math.sqrt(l[lOff + i2]);  // 𝐃¹ᐟ²

            // Zero out column: dzero(w[i3 : i3+i])
            for (int k = 0; k < i; k++) {
                w[e0 + i3 + k] = 0.0;
            }

            // 𝐄ⱼ = 𝐋ⱼᵀ : dcopy(i-n2, l[i2:], 1, w[i3:], n)
            BLAS.dcopy(i - n2, l, lOff + i2, 1, w, e0 + i3, n);

            // 𝐄ⱼ = 𝐃¹ᐟ²𝐋ⱼᵀ : dscal(i-n2, diag, w[i3:], n)
            BLAS.dscal(i - n2, diag, w, e0 + i3, n);

            // 𝐄ⱼⱼ = 𝐃¹ᐟ²ⱼⱼ : w[i3] = diag
            w[e0 + i3] = diag;

            /* 𝐲 = 𝐋⁻¹𝐠  →  𝐲ⱼ = (𝐠ⱼ - ∑ᵢ𝐋ⱼᵢ𝐲ᵢ) / 𝐋ⱼⱼ
             * 𝐋ⱼⱼ = 1   →  (𝐋⁻¹𝐠)ⱼ = (𝐠ⱼ - ∑ᵢ𝐋ⱼᵢ𝐲ᵢ)
             * 𝐟ⱼ = 𝐃⁻¹ᐟ²ⱼⱼ(𝐋⁻¹𝐠)ⱼ */
            w[f0 + j] = (g[gOff + j] - BLAS.ddot(j, w, e0 + i4, 1, w, f0, 1)) / diag;

            i2 += i - n2;
            i3 += n1;
            i4 += n;
        }

        /* Handle augmented problem case (for inconsistent constraints relaxation)
         * In augmented problem, an extra variable δ is added with 𝐄ⱼⱼ = ρ (penalty) */
        if (n2 == 1) {
            w[e0 + i3] = l[lOff + nl - 1];  // 𝐄ⱼⱼ = ρ (penalty parameter)
            // dzero(w[i4 : i4+n3])
            for (int k = 0; k < n3; k++) {
                w[e0 + i4 + k] = 0.0;
            }
            w[f0 + n3] = 0.0;   // 𝐟ⱼ = 0
        }

        // 𝐟 = -𝐃⁻¹ᐟ²𝐋⁻¹𝐠 : negate the computed values
        BLAS.dscal(n, -1.0, w, f0, 1);

        /* =========================================================================
         * Recover matrix C and vector d from equality constraints
         *
         * 𝐂 = { 𝐀ⱼ: j = 1 ··· mₑ }  (equality constraint Jacobian)
         * 𝐝 = { -𝐛ⱼ: j = 1 ··· mₑ } (negated equality constraint values)
         * ========================================================================= */
        if (meq > 0) {
            // Recover matrix C from upper part of A
            for (i = 0; i < meq; i++) {
                BLAS.dcopy(n, a, aOff + i, la, w, c0 + i, meq);  // 𝐂ⱼ = 𝐀ⱼ = -𝒄ⱼ(𝐱ᵏ)
            }
            // Recover vector d from upper part of b
            BLAS.dcopy(meq, b, bOff, 1, w, d0, 1);
            BLAS.dscal(meq, -1.0, w, d0, 1);  // 𝐝ⱼ = -𝐛ⱼ = -𝒄ⱼ(𝐱ᵏ)
        }

        /* =========================================================================
         * Recover matrix G and vector h from inequality constraints
         *
         * 𝐆 = { 𝐀ⱼ: j = mₑ+1 ··· m }  (inequality constraint Jacobian)
         * 𝐡 = { -𝐛ⱼ: j = mₑ+1 ··· m } (negated inequality constraint values)
         * ========================================================================= */
        if (mineq > 0) {
            // Recover matrix G from lower part of A
            for (i = 0; i < mineq; i++) {
                BLAS.dcopy(n, a, aOff + meq + i, la, w, g0 + i, m1);  // 𝐆ⱼ = 𝐀ⱼ = -𝒄ⱼ(𝐱ᵏ)
            }
            // Recover vector h from lower part of b
            BLAS.dcopy(mineq, b, bOff + meq, 1, w, h0, 1);
            BLAS.dscal(mineq, -1.0, w, h0, 1);  // 𝐡ⱼ = -𝐛ⱼ = -𝒄ⱼ(𝐱ᵏ)
        }

        /* =========================================================================
         * Bound Constraint Transformation
         *
         * Transform bounds 𝒍 ≤ 𝐱 ≤ 𝒖 to inequality constraints:
         *
         * Lower bounds (𝐱 ≥ 𝒍):
         *   𝐆ⱼ = 𝐈ⱼ (j-th row of identity matrix)
         *   𝐡ⱼ = 𝒍ⱼ
         *
         * Upper bounds (𝐱 ≤ 𝒖 ⟺ -𝐱 ≥ -𝒖):
         *   𝐆ⱼ = -𝐈ⱼ (negated j-th row of identity)
         *   𝐡ⱼ = -𝒖ⱼ
         * ========================================================================= */
        bnd = mineq;

        // Lower bounds: 𝐆ⱼ = 𝐈ⱼ, 𝐡ⱼ = 𝒍ⱼ (constraint: xᵢ ≥ lᵢ)
        for (i = 0; i < n; i++) {
            if (!Double.isNaN(xl[xlOff + i]) && xl[xlOff + i] > -infBnd) {
                int ip = g0 + bnd;
                int il = h0 + bnd;
                w[il] = xl[xlOff + i];  // 𝐡ⱼ = 𝒍ⱼ
                // Zero out row, then set diagonal element to 1
                for (int k = 0; k < n; k++) {
                    w[ip + m1 * k] = 0.0;
                }
                w[ip + m1 * i] = 1.0;  // 𝐆ⱼ = 𝐈ⱼ
                bnd++;
            }
        }

        // Upper bounds: 𝐆ⱼ = -𝐈ⱼ, 𝐡ⱼ = -𝒖ⱼ (constraint: -xᵢ ≥ -uᵢ ⟺ xᵢ ≤ uᵢ)
        for (i = 0; i < n; i++) {
            if (!Double.isNaN(xu[xuOff + i]) && xu[xuOff + i] < infBnd) {
                int ip = g0 + bnd;
                int il = h0 + bnd;
                w[il] = -xu[xuOff + i];  // 𝐡ⱼ = -𝒖ⱼ
                // Zero out row, then set diagonal element to -1
                for (int k = 0; k < n; k++) {
                    w[ip + m1 * k] = 0.0;
                }
                w[ip + m1 * i] = -1.0;  // 𝐆ⱼ = -𝐈ⱼ
                bnd++;
            }
        }

        /* Calculate number of NaN bounds (unused bound constraints)
         * nan = total possible bounds - actual bounds used
         * Matches C: nan := (n + n) - (bnd - mineq) */
        int nanCount = (n + n) - (bnd - mineq);

        /* =========================================================================
         * Call lsei solver
         *
         * Solve: 𝚖𝚒𝚗‖ 𝐄𝐱 - 𝐟 ‖₂ subject to 𝐂𝐱 = 𝐝 and 𝐆𝐱 ≥ 𝐡
         * ========================================================================= */
        int meqMax = (meq > 1) ? meq : 1;

        status = lsei(w, c0, w, d0,
                w, e0, w, f0,
                w, g0, w, h0,
                meqMax, meq, n, n, m1, m1 - nanCount, n,
                x, xOff,
                w, w0,
                jw, jwOff,
                maxIter, norm);

        /* =========================================================================
         * Process results
         *
         * If solution found:
         *   1. Restore Lagrange multipliers λ from workspace
         *   2. Set unused multipliers to NaN
         *   3. Enforce bounds on solution (project onto feasible region)
         * ========================================================================= */
        if (status == 0) {
            // Restore Lagrange multipliers λ
            BLAS.dcopy(m, w, w0, 1, y, yOff, 1);

            /* Set unused multipliers to NaN (for bounds that weren't active) */
            if (n3 > 0) {
                y[yOff + m] = Double.NaN;
                for (i = 1; i < n3 + n3; i++) {
                    y[yOff + m + i] = Double.NaN;
                }
            }

            /* Enforce lower bounds on solution: xᵢ = max(xᵢ, lᵢ) */
            for (i = 0; i < n; i++) {
                if (!Double.isNaN(xl[xlOff + i]) && xl[xlOff + i] > -infBnd && x[xOff + i] < xl[xlOff + i]) {
                    x[xOff + i] = xl[xlOff + i];
                }
            }
            /* Enforce upper bounds on solution: xᵢ = min(xᵢ, uᵢ) */
            for (i = 0; i < n; i++) {
                if (!Double.isNaN(xu[xuOff + i]) && xu[xuOff + i] < infBnd && x[xOff + i] > xu[xuOff + i]) {
                    x[xOff + i] = xu[xuOff + i];
                }
            }
        }

        return status;
    }

}
