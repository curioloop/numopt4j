/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * LAPACK DGGEV: Computes eigenvalues and optionally eigenvectors of a real
 * generalized non-symmetric eigenvalue problem A*x = lambda*B*x.
 *
 * <p>The eigenvalues are returned as (alphar[i] + i*alphai[i]) / beta[i].
 * Real eigenvalues have alphai[i] = 0; complex eigenvalues come in conjugate pairs.
 *
 * <p>Algorithm:
 * <ol>
 *   <li>Balance A via {@code dgebal}</li>
 *   <li>Reduce A to upper Hessenberg: {@code dgehrd} + {@code dorghr}</li>
 *   <li>QR-factorize B; apply Q_B^T to H and to the accumulated Q</li>
 *   <li>Reduce (H, T) to generalized Hessenberg form (DGGHRD, inlined)</li>
 *   <li>QZ iteration: {@code dhgeqz}</li>
 *   <li>Compute eigenvectors via {@code dtgevc}</li>
 *   <li>Back-transform eigenvectors via {@code dgebak}</li>
 * </ol>
 *
 * <p>Workspace layout (all sliced from the caller-supplied {@code work} array):
 * <pre>
 *   [0      .. n-1  ]  scaleA   (n)
 *   [n      .. 2n-1 ]  tau_A    (n)   Hessenberg reflectors
 *   [2n     .. 3n-1 ]  tau_B    (n)   QR reflectors for B
 *   [3n     .. 3n+nn-1] Q       (n*n) accumulated left Schur vectors
 *   [3n+nn  .. 3n+2nn-1] Z      (n*n) accumulated right Schur vectors
 *   [3n+2nn .. +nn? ]  tmpVL    (n*n, only if wantVL)
 *   [+nn?   .. +nn? ]  tmpVR    (n*n, only if wantVR)
 *   [end    ..      ]  scratch  (max of dgehrd/dorghr/dormqr/dhgeqz/dtgevc needs)
 * </pre>
 * where nn = n*n.  Total = 3n + 2n² + (wantVL?n²:0) + (wantVR?n²:0) + scratch.
 *
 * <p>Row-major storage: element (i,j) is at array[i*ld + j].
 */
interface Dggev {

    // Workspace slot sizes
    static int wsScaleA(int n) { return n; }
    static int wsTauA(int n)   { return n; }
    static int wsTauB(int n)   { return n; }
    static int wsQ(int n)      { return n * n; }
    static int wsZ(int n)      { return n * n; }

    /** Minimum scratch needed beyond the fixed slots above. */
    static int wsScratch(int n) {
        // dgehrd/dorghr: query; dormqr: n; dhgeqz: n; dtgevc: 6n
        return max(6 * n, n);
    }

    /** Total workspace required. tmpVL/tmpVR slots only allocated when needed. */
    static int workSize(int n, boolean wantVL, boolean wantVR) {
        int tmpSlots = (wantVL ? n * n : 0) + (wantVR ? n * n : 0);
        return 3 * n + 2 * n * n + tmpSlots + wsScratch(n);
    }

    /**
     * Workspace query: writes required size to work[workOff] and returns 0.
     * Also performs the actual computation when lwork >= required size.
     */
    static int dggev(boolean wantVL, boolean wantVR, int n,
                     double[] A, int lda, double[] B, int ldb,
                     double[] alphar, double[] alphai, double[] beta,
                     double[] VL, int ldvl, double[] VR, int ldvr,
                     double[] work, int workOff, int lwork) {

        if (n == 0) return 0;

        // --- Workspace query: determine dgehrd optimal size ---
        // dgehrd supports lwork=-1 query; dorghr does not (accesses A), so use static bound.
        double[] qry = new double[1];
        BLAS.dgehrd(n, 1, n, null, lda, null, 0, qry, 0, -1);
        int gehrdOpt = (int) qry[0];
        int orghrOpt = 64 * n; // dorghr optimal ≈ n*nb, nb typically 64
        int dormqrOpt = 64 * n; // dormqr optimal ≈ n*nb
        int scratchNeeded = max(max(gehrdOpt, orghrOpt), max(dormqrOpt, 6 * n));
        // tmpVL/tmpVR slots only allocated when the caller actually wants eigenvectors
        int tmpSlots = (wantVL ? n * n : 0) + (wantVR ? n * n : 0);
        int required = 3 * n + 2 * n * n + tmpSlots + scratchNeeded;

        if (lwork == -1) {
            work[workOff] = required;
            return 0;
        }

        // Slice workspace
        int off = workOff;
        int scaleOff = off;                    off += n;          // scaleA[n]
        int tauAOff  = off;                    off += n;          // tau_A[n]
        int tauBOff  = off;                    off += n;          // tau_B[n]
        int qOff     = off;                    off += n * n;      // Q[n*n]
        int zOff     = off;                    off += n * n;      // Z[n*n]
        int tmpVLOff = off; if (wantVL)        off += n * n;      // tmpVL[n*n] only if needed
        int tmpVROff = off; if (wantVR)        off += n * n;      // tmpVR[n*n] only if needed
        int scrOff   = off;                                       // scratch[scratchNeeded]
        int scrLen   = lwork - (scrOff - workOff);
        if (scrLen < 1) scrLen = 1; // guard

        // --- Step 1: Balance A ---
        double[] balOut = new double[2];
        BLAS.dgebal('B', n, A, lda, work, scaleOff, balOut, 0);
        int ilo = (int) balOut[0]; // 0-based
        int ihi = (int) balOut[1]; // 0-based inclusive

        // --- Step 2: Reduce A to upper Hessenberg form ---
        // dgehrd uses 0-based ilo/ihi (same convention as dgebal output)
        BLAS.dgehrd(n, ilo, ihi, A, lda, work, tauAOff, work, scrOff, scrLen);

        // --- Step 3: Build Q from Hessenberg reduction ---
        // Copy A (which contains H + Householder reflectors) to work[qOff],
        // then run dorghr in-place on work[qOff] to build Q there.
        // A retains the upper Hessenberg form H for subsequent steps.
        BLAS.dlacpy('A', n, n, A, 0, lda, work, qOff, n);
        BLAS.dorghr(n, ilo, ihi, work, qOff, n, work, tauAOff, work, scrOff, scrLen);
        // Q is now in work[qOff..qOff+n*n-1]

        // --- Step 4: Initialize Z = I ---
        BLAS.dlaset('A', n, n, 0.0, 1.0, work, zOff, n);

        // --- Step 5: QR-factorize B to get upper triangular T ---
        BLAS.dgeqrf(n, n, B, 0, ldb, work, tauBOff, work, scrOff, scrLen);

        // Apply Q_B^T to H (A) from the left: A <- Q_B^T * A
        BLAS.dormqr(BLAS.Side.Left, BLAS.Transpose.Trans, n, n, n,
                    B, 0, ldb, work, tauBOff, A, 0, lda, work, scrOff, scrLen);

        // Apply Q_B^T to Q from the left: Q <- Q_B^T * Q
        BLAS.dormqr(BLAS.Side.Left, BLAS.Transpose.Trans, n, n, n,
                    B, 0, ldb, work, tauBOff, work, qOff, n, work, scrOff, scrLen);

        // Zero lower triangle of B (now upper triangular T)
        for (int i = 1; i < n; i++)
            for (int j = 0; j < i; j++)
                B[i * ldb + j] = 0.0;

        // --- Step 6: Reduce (H, T) to generalized Hessenberg form (DGGHRD inlined) ---
        // Givens rotations: zero A[jrow, jcol] for jrow > jcol, then restore T upper triangular
        double[] rot = work; // reuse scratch as rotation buffer (rot[scrOff..scrOff+2])
        for (int jcol = ilo; jcol <= ihi - 1; jcol++) {
            for (int jrow = ihi; jrow >= jcol + 1; jrow--) {
                // Left rotation to zero A[jrow, jcol]
                Dlartg.dlartg(A[(jrow-1)*lda + jcol], A[jrow*lda + jcol], rot, scrOff);
                double c = rot[scrOff], s = rot[scrOff+1];
                A[(jrow-1)*lda + jcol] = rot[scrOff+2];
                A[jrow*lda + jcol] = 0.0;
                BLAS.drot(n - jcol - 1, A, (jrow-1)*lda + jcol+1, 1, A, jrow*lda + jcol+1, 1, c, s);
                BLAS.drot(n - jrow + 1, B, (jrow-1)*ldb + jrow-1, 1, B, jrow*ldb + jrow-1, 1, c, s);
                BLAS.drot(n, work, qOff + (jrow-1)*n, 1, work, qOff + jrow*n, 1, c, s);

                // Right rotation to restore T upper triangular
                Dlartg.dlartg(B[jrow*ldb + jrow], B[jrow*ldb + jrow-1], rot, scrOff);
                c = rot[scrOff]; s = rot[scrOff+1];
                B[jrow*ldb + jrow]   = rot[scrOff+2];
                B[jrow*ldb + jrow-1] = 0.0;
                BLAS.drot(ihi - jcol + 1, A, jcol*lda + jrow, lda, A, jcol*lda + jrow-1, lda, c, s);
                BLAS.drot(jrow, B, jrow, ldb, B, jrow-1, ldb, c, s);
                BLAS.drot(n, work, zOff + jrow, n, work, zOff + jrow-1, n, c, s);
            }
        }

        // --- Step 7: QZ iteration ---
        int info = Dhgeqz.dhgeqz('S', 'V', 'V', n, ilo, ihi,
                                  A, 0, lda, B, 0, ldb,
                                  alphar, alphai, beta,
                                  work, qOff, n, work, zOff, n,
                                  work, scrOff, scrLen);
        if (info != 0) return info;

        // --- Step 8: Compute eigenvectors via dtgevc ---
        if (wantVL || wantVR) {
            // tmpVL/tmpVR are carved from work — no heap allocation
            Dtgevc.dtgevc('B', 'A', null, n,
                          A, 0, lda, B, 0, ldb,
                          wantVL ? work : null, tmpVLOff, n,
                          wantVR ? work : null, tmpVROff, n,
                          n, work, scrOff);

            // --- Step 9: Back-transform: multiply by Q (left) and Z (right) ---
            if (wantVL) {
                BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, n, n, n,
                           1.0, work, qOff, n, work, tmpVLOff, n, 0.0, VL, 0, ldvl);
                BLAS.dgebak('B', BLAS.Side.Left, n, ilo, ihi, work, scaleOff, n, VL, ldvl);
                normalizeEigvecs(VL, ldvl, n, alphai);
            }
            if (wantVR) {
                BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, n, n, n,
                           1.0, work, zOff, n, work, tmpVROff, n, 0.0, VR, 0, ldvr);
                BLAS.dgebak('B', BLAS.Side.Right, n, ilo, ihi, work, scaleOff, n, VR, ldvr);
                normalizeEigvecs(VR, ldvr, n, alphai);
            }
        }

        return 0;
    }

    /** Normalize eigenvectors column by column (infinity norm). */
    static void normalizeEigvecs(double[] V, int ldv, int n, double[] alphai) {
        int j = 0;
        while (j < n) {
            if (alphai[j] == 0.0) {
                double norm = 0.0;
                for (int i = 0; i < n; i++) norm = max(norm, abs(V[i * ldv + j]));
                if (norm > 0.0) for (int i = 0; i < n; i++) V[i * ldv + j] /= norm;
                j++;
            } else {
                double norm = 0.0;
                for (int i = 0; i < n; i++)
                    norm = max(norm, abs(V[i * ldv + j]) + abs(V[i * ldv + j + 1]));
                if (norm > 0.0) {
                    for (int i = 0; i < n; i++) {
                        V[i * ldv + j]     /= norm;
                        V[i * ldv + j + 1] /= norm;
                    }
                }
                j += 2;
            }
        }
    }
}
