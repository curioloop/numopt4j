/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * L-BFGS-B BFGS Update - Pure Java implementation.
 *
 * This module implements the L-BFGS-B matrix update routines:
 *   - updateCorrection (matupd): Updates correction matrices Sₖ and Yₖ
 *   - formT (formt): Forms and factorizes T = θSᵀS + LD⁻¹Lᵀ
 *   - formK (formk): Forms the LELᵀ factorization of the indefinite K matrix
 *
 * The L-BFGS-B algorithm maintains a limited-memory approximation to the
 * inverse Hessian using the most recent m correction pairs:
 *   Sₖ = [sₖ₋ₘ, ..., sₖ₋₁]  where sᵢ = xᵢ₊₁ - xᵢ
 *   Yₖ = [yₖ₋ₘ, ..., yₖ₋₁]  where yᵢ = gᵢ₊₁ - gᵢ
 *
 * Reference: C implementation in lbfgsb/update.c
 * Reference: Go implementation in lbfgsb/update.go
 * Reference: Byrd, Lu, Nocedal, Zhu, "A Limited Memory Algorithm for Bound
 *            Constrained Optimization", SIAM J. Scientific Computing, 1995.
 *
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import static com.curioloop.numopt4j.linalg.blas.BLAS.*;

/**
 * BFGS matrix update for the L-BFGS-B algorithm.
 *
 * <p>This class maintains the limited-memory BFGS approximation by:</p>
 * <ol>
 *   <li>Updating correction matrices Sₖ and Yₖ with new step/gradient pairs</li>
 *   <li>Computing the Cholesky factor of T = θSᵀS + LD⁻¹Lᵀ</li>
 *   <li>Forming the LELᵀ factorization of the indefinite K matrix</li>
 * </ol>
 *
 * @see LBFGSBWorkspace
 * @see CauchyPoint
 */
final class BfgsUpdater implements LBFGSBConstants {

    private BfgsUpdater() {}

    // ========================================================================
    // Cholesky Factorization (moved from Linpack for better locality)
    // ========================================================================

    /**
     * Cholesky factorization of a symmetric positive definite matrix.
     *
     * <p>Factors a double precision symmetric positive definite matrix A = Rᵀ * R.</p>
     *
     * <p>Optimized version with inlined ddot for better performance on small matrices
     * typical in L-BFGS-B (n = 5-20).</p>
     *
     * @param a   the symmetric matrix to be factored (modified in place)
     * @param aOff offset into array a
     * @param lda the leading dimension of the array a
     * @param n   the order of the matrix a
     * @return 0 for normal return, k signals an error condition (the leading minor
     *         of order k is not positive definite)
     */
    private static int dpofa(double[] a, int aOff, int lda, int n) {
        for (int j = 0; j < n; j++) {
            int jCol = aOff + j * lda;  // Precompute column j offset
            double s = 0.0;

            for (int k = 0; k < j; k++) {
                int kCol = aOff + k * lda;  // Precompute column k offset

                // Inline ddot with stride=lda: compute dot product of column k and column j
                double dot = 0.0;
                for (int i = 0; i < k; i++) {
                    dot += a[aOff + k + i * lda] * a[aOff + j + i * lda];
                }

                double t = (a[kCol + j] - dot) / a[kCol + k];
                a[kCol + j] = t;
                s += t * t;
            }

            s = a[jCol + j] - s;
            if (s <= 0.0) {
                return j + 1;  // Not positive definite
            }
            a[jCol + j] = Math.sqrt(s);
        }
        return 0;
    }

    /**
     * Updates correction matrices Sₖ and Yₖ.
     *
     * <p>Given the new correction pair:</p>
     * <pre>
     *   sₖ = xₖ₊₁ - xₖ  (step vector)
     *   yₖ = gₖ₊₁ - gₖ  (gradient difference)
     * </pre>
     *
     * <p>The subroutine:</p>
     * <ol>
     *   <li>Checks the curvature condition: sₖᵀyₖ > ε‖yₖ‖²</li>
     *   <li>Updates the circular buffer storing Sₖ and Yₖ</li>
     *   <li>Computes θ = yₖᵀyₖ / sₖᵀyₖ (scaling factor)</li>
     *   <li>Updates SᵀS (upper triangular) and SᵀY (lower triangular)</li>
     * </ol>
     *
     * <p>Reference: C function update_correction() in update.c</p>
     * <p>Reference: Go function updateCorrection() in update.go</p>
     *
     * @param n Problem dimension
     * @param m Maximum number of correction pairs
     * @param ws Workspace containing correction matrices
     *
     */
    static void updateCorrection(int n, int m, double[] g, LBFGSBWorkspace ws) {
        LBFGSBWorkspace.SearchCtx ctx = ws.getSearchCtx();
        double[] buffer = ws.getBuffer();

        int wsOffset = ws.getWsOffset();  // S matrix (n × m)
        int wyOffset = ws.getWyOffset();  // Y matrix (n × m)
        int syOffset = ws.getSyOffset();  // S'Y matrix (m × m)
        int ssOffset = ws.getSsOffset();  // S'S matrix (m × m)

        int col = ws.getCol();
        int head = ws.getHead();
        int tail = ws.getTail();
        int updates = ws.getUpdates();

        int sOffset = ws.getDOffset(); // sₖ = xₖ₊₁ - xₖ
        int yOffset = ws.getROffset(); // yₖ = gₖ₊₁ - gₖ
        for (int i = 0; i < n; i++) {
            buffer[yOffset + i] = g[i] - buffer[yOffset + i];
        }

        double rr = ddot(n, buffer, yOffset, 1, buffer, yOffset, 1);  // yₖᵀyₖ
        double dr = ctx.gd - ctx.gdOld;  // sₖᵀyₖ
        double gd = -ctx.gdOld;          // gₖᵀdₖ

        double stp = ctx.stp;
        if (stp != 1) {
            dr *= stp;
            gd *= stp;
            dscal(n, stp, buffer, sOffset, 1);
        }

        // Skip update when curvature condition sₖᵀyₖ ≤ ε‖yₖ‖² is not satisfied
        if (dr <= EPS * gd) {
            ws.setUpdated(false);
            return;
        }

        ws.setUpdated(true);
        ws.setUpdates(updates + 1);
        updates = ws.getUpdates();

        // Update pointers for matrices S and Y
        if (updates <= m) {
            col = updates;
            tail = (head + updates - 1) % m;
        } else {
            tail = (tail + 1) % m;
            head = (head + 1) % m;
        }

        ws.setCol(col);
        ws.setHead(head);
        ws.setTail(tail);

        // Update matrices Sₖ and Yₖ
        // Store sₖ in column 'tail' of Sₖ matrix
        // Store yₖ in column 'tail' of Yₖ matrix
        for (int i = 0; i < n; i++) {
            buffer[wsOffset + i * m + tail] = buffer[sOffset + i];
            buffer[wyOffset + i * m + tail] = buffer[yOffset + i];
        }

        // Update θ = yₖᵀyₖ / sₖᵀyₖ
        ws.setTheta(rr / dr);

        // Move old information when buffer is full
        if (updates > m) {
            for (int j = 0; j < col - 1; j++) {
                // S'S upper triangle: shift rows
                dcopy(col - (j + 1), buffer, ssOffset + (j + 1) * m + (j + 1), 1,
                        buffer, ssOffset + j * m + j, 1);
                // S'Y lower triangle: shift rows
                dcopy(j + 1, buffer, syOffset + (j + 1) * m + 1, 1,
                        buffer, syOffset + j * m, 1);
            }
        }

        // Add new information
        int ptr = head;
        for (int j = 0; j < col - 1; j++) {
            // Last row of S'Y: (S'Y)[(col-1), j] = sₖᵀyⱼ
            buffer[syOffset + (col - 1) * m + j] = ddot(n, buffer, sOffset, 1, buffer, wyOffset + ptr, m);
            // Last column of S'S: (S'S)[j, (col-1)] = sⱼᵀsₖ
            buffer[ssOffset + j * m + (col - 1)] = ddot(n, buffer, wsOffset + ptr, m, buffer, sOffset, 1);
            ptr = (ptr + 1) % m;
        }

        // Update diagonal elements
        buffer[syOffset + (col - 1) * m + (col - 1)] = dr;  // sₖᵀyₖ
        buffer[ssOffset + (col - 1) * m + (col - 1)] = ws.getDSqrt();  // sₖᵀsₖ
        if (stp != 1) {
            buffer[ssOffset + (col - 1) * m + (col - 1)] *= stp * stp;
        }
    }


    /**
     * Forms the matrix T = θSᵀS + LD⁻¹Lᵀ and performs Cholesky factorization.
     *
     * <p>The matrix T appears in the compact representation of the L-BFGS
     * inverse Hessian approximation. It is used in the BMV (B matrix-vector)
     * multiplication during the Cauchy point computation.</p>
     *
     * <p>Components:</p>
     * <ul>
     *   <li>θ: scaling factor = yₖᵀyₖ / sₖᵀyₖ</li>
     *   <li>SᵀS: inner products of step vectors</li>
     *   <li>D = diag{sᵢᵀyᵢ}: diagonal matrix from SᵀY</li>
     *   <li>L = {sᵢᵀyⱼ}ᵢ>ⱼ: strictly lower triangular part of SᵀY</li>
     * </ul>
     *
     * <p>Reference: C function form_t() in update.c</p>
     * <p>Reference: Go function formT() in update.go</p>
     *
     * @param m Maximum number of correction pairs
     * @param ws Workspace containing S'S, S'Y, and output wt
     * @return 0 on success, ERR_NOT_POS_DEF_T if T is not positive definite
     *
     */
    static int formT(int m, LBFGSBWorkspace ws) {
        double[] buffer = ws.getBuffer();

        int col = ws.getCol();
        double theta = ws.getTheta();

        int wtOffset = ws.getWtOffset();  // Cholesky factor output
        int ssOffset = ws.getSsOffset();  // S'S matrix
        int syOffset = ws.getSyOffset();  // S'Y matrix

        if (col == 0) {
            return ERR_NONE;
        }

        // Form the upper half of T = θS'S + LD⁻¹L'
        // First row: T[0,j] = θ × (S'S)[0,j]
        for (int j = 0; j < col; j++) {
            buffer[wtOffset + j] = theta * buffer[ssOffset + j];
        }

        // Remaining rows: T[i,j] = θ(S'S)[i,j] + (LD⁻¹L')ᵢⱼ
        for (int i = 1; i < col; i++) {
            for (int j = i; j < col; j++) {
                double ldl = 0.0;
                int kk = Math.min(i, j);
                for (int k = 0; k < kk; k++) {
                    ldl += buffer[syOffset + i * m + k] * buffer[syOffset + j * m + k]
                         / buffer[syOffset + k * m + k];
                }
                buffer[wtOffset + i * m + j] = ldl + theta * buffer[ssOffset + i * m + j];
            }
        }

        // Cholesky factorize T = JJ' with J' stored in upper triangle of wt
        if (dpofa(buffer, wtOffset, m, col) != 0) {
            return ERR_NOT_POS_DEF_T;
        }

        return ERR_NONE;
    }

    /**
     * Forms the LEL' factorization of the indefinite K matrix.
     *
     * <p>The matrix K is:</p>
     * <pre>
     *   K = [-D - Y'ZZ'Y/θ    La' - Rz']   where  E = [-I  0]
     *       [La - Rz          θS'AA'S  ]              [ 0  I]
     * </pre>
     *
     * <p>Notation:</p>
     * <ul>
     *   <li>Z: indices of free variables (not at bounds)</li>
     *   <li>A: indices of active variables (at bounds)</li>
     *   <li>D = diag{sᵢᵀyᵢ}: diagonal matrix</li>
     *   <li>La: strictly lower triangular part of S'AA'Y</li>
     *   <li>Rz: upper triangular part of S'ZZ'Y</li>
     * </ul>
     *
     * <p>Reference: C function form_k() in update.c</p>
     * <p>Reference: Go function formK() in update.go</p>
     *
     * @param n Problem dimension
     * @param m Maximum number of correction pairs
     * @param ws Workspace containing matrices and index arrays
     * @return 0 on success, ERR_NOT_POS_DEF_1ST_K or ERR_NOT_POS_DEF_2ND_K on failure
     *
     */
    static int formK(int n, int m, LBFGSBWorkspace ws) {
        double[] buffer = ws.getBuffer();
        int[] iBuffer = ws.getIntBuffer();

        int col = ws.getCol();
        int head = ws.getHead();
        int m2 = 2 * m;
        int col2 = 2 * col;

        int wnOffset = ws.getWnOffset();   // K matrix and LEL' factorization
        int sndOffset = ws.getSndOffset(); // Inner products storage (wn1)
        int wsOffset = ws.getWsOffset();   // S matrix
        int wyOffset = ws.getWyOffset();   // Y matrix
        int syOffset = ws.getSyOffset();   // S'Y matrix

        int indexOffset = ws.getIndexOffset();

        int free = ws.getFree();
        int enter = ws.getEnter();
        int leave = ws.getLeave();
        boolean updated = ws.isUpdated();
        int updates = ws.getUpdates();
        double theta = ws.getTheta();

        if (col == 0) {
            return ERR_NONE;
        }

        // Form the lower triangular part of WN1
        if (updated) {
            if (updates > m) {
                // Shift old parts of WN1
                for (int jy = 0; jy < m - 1; jy++) {
                    int js = m + jy;
                    // Y'ZZ'Y: shift rows
                    dcopy(jy + 1, buffer, sndOffset + (jy + 1) * m2 + 1, 1,
                          buffer, sndOffset + jy * m2, 1);
                    // S'AA'S: shift rows
                    dcopy(jy + 1, buffer, sndOffset + (js + 1) * m2 + 1 + m, 1,
                          buffer, sndOffset + js * m2 + m, 1);
                    // La + Rz: shift rows
                    dcopy(m - 1, buffer, sndOffset + (js + 1) * m2 + 1, 1,
                          buffer, sndOffset + js * m2, 1);
                }
            }

            int pBeg = 0, pEnd = free;      // free variables indices (Z)
            int dBeg = free, dEnd = n;      // active bounds indices (A)

            // Add new rows to blocks (1,1), (2,1), and (2,2)
            int iptr = (head + col - 1) % m;
            int jptr = head;

            int iyRow = sndOffset + (col - 1) * m2;  // last row of Y'ZZ'Y
            int isRow = sndOffset + (m + col - 1) * m2;  // last row of S'AA'S and La + Rz

            for (int jy = 0; jy < col; jy++) {
                int js = m + jy;

                double temp1 = 0.0;
                double temp2 = 0.0;
                double temp3 = 0.0;

                // Sum over free variables (Z)
                for (int k = pBeg; k < pEnd; k++) {
                    int k1 = iBuffer[indexOffset + k];
                    temp1 += buffer[wyOffset + k1 * m + iptr] * buffer[wyOffset + k1 * m + jptr];
                }

                // Sum over active bound variables (A)
                for (int k = dBeg; k < dEnd; k++) {
                    int k1 = iBuffer[indexOffset + k];
                    temp2 += buffer[wsOffset + k1 * m + iptr] * buffer[wsOffset + k1 * m + jptr];
                    temp3 += buffer[wsOffset + k1 * m + iptr] * buffer[wyOffset + k1 * m + jptr];
                }

                buffer[iyRow + jy] = temp1;  // Y'ZZ'Y
                buffer[isRow + js] = temp2;  // S'AA'S
                buffer[isRow + jy] = temp3;  // La

                jptr = (jptr + 1) % m;
            }

            // Add new column to block (2,1) - Rz part
            jptr = (head + col - 1) % m;
            iptr = head;

            int jyCol = sndOffset + m * m2 + (col - 1);  // last column of La + Rz

            for (int i = 0; i < col; i++) {
                double temp3 = 0.0;
                for (int k = pBeg; k < pEnd; k++) {
                    int k1 = iBuffer[indexOffset + k];
                    temp3 += buffer[wsOffset + k1 * m + iptr] * buffer[wyOffset + k1 * m + jptr];
                }
                buffer[jyCol + i * m2] = temp3;  // Rz
                iptr = (iptr + 1) % m;
            }
        }

        // Modify the old parts in blocks (1,1) and (2,2)
        int nUpdate = col;
        if (updated) {
            nUpdate--;
        }

        int iptr = head;
        for (int iy = 0; iy < nUpdate; iy++) {
            int is = m + iy;

            int jptr = head;
            for (int jy = 0; jy <= iy; jy++) {
                int js = m + jy;

                double temp1 = 0.0;
                double temp2 = 0.0;
                double temp3 = 0.0;
                double temp4 = 0.0;

                // Variables entering free set (from Z to A)
                for (int k = 0; k < enter; k++) {
                    int k1 = iBuffer[indexOffset + n + k];
                    temp1 += buffer[wyOffset + k1 * m + iptr] * buffer[wyOffset + k1 * m + jptr];
                    temp2 += buffer[wsOffset + k1 * m + iptr] * buffer[wsOffset + k1 * m + jptr];
                }

                // Variables leaving free set (from A to Z)
                for (int k = leave; k < n; k++) {
                    int k1 = iBuffer[indexOffset + n + k];
                    temp3 += buffer[wyOffset + k1 * m + iptr] * buffer[wyOffset + k1 * m + jptr];
                    temp4 += buffer[wsOffset + k1 * m + iptr] * buffer[wsOffset + k1 * m + jptr];
                }

                buffer[sndOffset + iy * m2 + jy] += temp1 - temp3;  // Y'ZZ'Y
                buffer[sndOffset + is * m2 + js] += temp4 - temp2;  // S'AA'S

                jptr = (jptr + 1) % m;
            }
            iptr = (iptr + 1) % m;
        }

        // Modify the old parts in block (2,1)
        iptr = head;
        for (int is = m; is < m + nUpdate; is++) {
            int jptr2 = head;
            for (int jy = 0; jy < nUpdate; jy++) {
                double temp1 = 0.0;
                double temp3 = 0.0;

                for (int k = 0; k < enter; k++) {
                    int k1 = iBuffer[indexOffset + n + k];
                    temp1 += buffer[wsOffset + k1 * m + iptr] * buffer[wyOffset + k1 * m + jptr2];
                }

                for (int k = leave; k < n; k++) {
                    int k1 = iBuffer[indexOffset + n + k];
                    temp3 += buffer[wsOffset + k1 * m + iptr] * buffer[wyOffset + k1 * m + jptr2];
                }

                if (is - m <= jy) {
                    buffer[sndOffset + is * m2 + jy] += temp1 - temp3;  // Rz
                } else {
                    buffer[sndOffset + is * m2 + jy] -= temp1 - temp3;  // La
                }

                jptr2 = (jptr2 + 1) % m;
            }
            iptr = (iptr + 1) % m;
        }

        // Form the upper triangle of 2*col x 2*col indefinite matrix
        for (int iy = 0; iy < col; iy++) {
            int is = col + iy;
            int is1 = m + iy;

            for (int jy = 0; jy <= iy; jy++) {
                int js = col + jy;
                int js1 = m + jy;
                buffer[wnOffset + jy * m2 + iy] = buffer[sndOffset + iy * m2 + jy] / theta;
                buffer[wnOffset + js * m2 + is] = buffer[sndOffset + is1 * m2 + js1] * theta;
            }

            for (int jy = 0; jy < iy; jy++) {
                buffer[wnOffset + jy * m2 + is] = -buffer[sndOffset + is1 * m2 + jy];
            }
            for (int jy = iy; jy < col; jy++) {
                buffer[wnOffset + jy * m2 + is] = buffer[sndOffset + is1 * m2 + jy];
            }

            buffer[wnOffset + iy * m2 + iy] += buffer[syOffset + iy * m + iy];
        }

        // First Cholesky factor (1,1) block
        if (dpofa(buffer, wnOffset, m2, col) != 0) {
            return ERR_NOT_POS_DEF_1ST_K;
        }

        // Solve Lx = (-La'+Rz') to form L⁻¹(-La'+Rz')
        // Use pre-allocated tempCol from workspace instead of allocating new array
        int tempColOffset = ws.getTempColOffset();
        for (int js = col; js < col2; js++) {
            for (int i = 0; i < col; i++) {
                buffer[tempColOffset + i] = buffer[wnOffset + i * m2 + js];
            }
            dtrsl(buffer, wnOffset, m2, col, buffer, tempColOffset, Uplo.Upper, Transpose.Trans);
            for (int i = 0; i < col; i++) {
                buffer[wnOffset + i * m2 + js] = buffer[tempColOffset + i];
            }
        }

        // Form S'AA'Sθ + [L⁻¹(-La'+Rz')]'[L⁻¹(-La'+Rz')]
        for (int is = col; is < col2; is++) {
            for (int js = is; js < col2; js++) {
                double dot = 0.0;
                for (int k = 0; k < col; k++) {
                    dot += buffer[wnOffset + k * m2 + is] * buffer[wnOffset + k * m2 + js];
                }
                buffer[wnOffset + is * m2 + js] += dot;
            }
        }

        // Cholesky factorization of (2,2) block
        if (dpofa(buffer, wnOffset + col * m2 + col, m2, col) != 0) {
            return ERR_NOT_POS_DEF_2ND_K;
        }

        return ERR_NONE;
    }
}
