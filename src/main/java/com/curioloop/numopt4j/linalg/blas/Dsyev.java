/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * LAPACK DSYEV: Computes all eigenvalues and optionally eigenvectors of a symmetric matrix.
 * 
 * Based on LAPACK.
 * 
 * <p>Optimizations:
 * <ul>
 *   <li>BLAS for eigenvalue scaling</li>
 *   <li>BLAS for matrix operations</li>
 * </ul>
 * </p>
 */
public interface Dsyev {

    /**
     * DSYEV computes all eigenvalues and, optionally, eigenvectors of a real
     * symmetric matrix A. Supports workOff for workspace reuse.
     *
     * @param jobz  'N' for eigenvalues only, 'V' for eigenvalues and eigenvectors
     * @param uplo  'U' for upper triangle, 'L' for lower triangle
     * @param n     order of matrix A
     * @param A     symmetric matrix (n x n, row-major), overwritten on exit
     * @param lda   leading dimension of A
     * @param w     eigenvalues (output, length n)
     * @param wOff  offset into w
     * @param work  workspace (length lwork)
     * @param workOff offset into work
     * @param lwork length of workspace (>= 3*n-1, or -1 for query)
     * @return 0 on success, non-zero on failure
     */
    static int dsyev(char jobz, char uplo, int n, double[] A, int lda,
                     double[] w, int wOff, double[] work, int workOff, int lwork) {
        if (n == 0) {
            return 0;
        }

        boolean wantz = Character.toUpperCase(jobz) == 'V';
        boolean upper = Character.toUpperCase(uplo) == 'U';

        // Compute optimal workspace size
        int nb = Ilaenv.ilaenv(1, "DSYTRD", String.valueOf(uplo), n, 0, 0, -1);
        int lworkopt = Math.max(1, (nb + 2) * n);

        // Query mode: return optimal size
        if (lwork == -1) {
            work[workOff] = lworkopt;
            return 0;
        }

        // Validate dimensions
        if (lda < Math.max(1, n)) {
            return -5;
        }
        if (lwork < Math.max(1, 3 * n - 1)) {
            return -9;
        }
        if (w.length < wOff + n) {
            return -6;
        }

        // Handle n = 1
        if (n == 1) {
            w[wOff] = A[0];
            work[workOff] = 2;
            if (wantz) {
                A[0] = 1;
            }
            return 0;
        }

        // Get machine constants
        double safmin = Dlamch.safmin();
        double eps = Dlamch.eps();
        double smlnum = safmin / eps;
        double bignum = 1 / smlnum;
        double rmin = Math.sqrt(smlnum);
        double rmax = Math.sqrt(bignum);

        // Scale matrix if necessary - reuse work array for DLANSY when possible
        double anrm = Dlansy.dlansy('M', upper ? BLAS.Uplo.Upper : BLAS.Uplo.Lower, n, A, lda, work);
        boolean scaled = false;
        double sigma = 1.0;

        if (anrm > 0 && anrm < rmin) {
            scaled = true;
            sigma = rmin / anrm;
        } else if (anrm > rmax) {
            scaled = true;
            sigma = rmax / anrm;
        }

        if (scaled) {
            // Scale matrix by sigma using BLAS
            char kind = upper ? 'U' : 'L';
            Dlamv.dlascl(kind, 0, 0, 1.0, sigma, n, n, A, 0, lda);
        }

        // work[0:n] = e (off-diagonal for Dsterf/Dsteqr) - n elements for Dsytrd
        // work[n:2n] = tau (Householder scalars)  
        // work[2n:] = work for Dsytrd
        int inde = 0;
        int indtau = inde + n;
        int indwork = indtau + n;
        int llwork = lwork - indwork;
        
        // Reduce to tridiagonal form
        Dsytrd.dsytrd(upper ? BLAS.Uplo.Upper : BLAS.Uplo.Lower, n, A, lda, w, wOff, work, inde, work, indtau, work, indwork, llwork);
                
        // Compute eigenvalues
        boolean ok;
        if (!wantz) {
            // Eigenvalues only: use DSTERF
            ok = Dsterf.dsterf(n, w, wOff, work, inde);
            // w already contains eigenvalues from Dsytrd
        } else {
            // Eigenvalues and eigenvectors
            // Generate orthogonal matrix Q from tridiagonalization
            Dorgtr.dorgtr(upper ? BLAS.Uplo.Upper : BLAS.Uplo.Lower, n, A, lda, work, indtau, work, indwork, llwork);
            
            // Compute eigenvectors from tridiagonal form
            ok = Dsteqr.dsteqr('V', n, w, wOff, work, inde, A, 0, lda, work, indtau) == 0;
            // w already contains eigenvalues from Dsteqr
        }

        if (!ok) {
            return 1;
        }

        // If matrix was scaled, rescale eigenvalues using BLAS
        if (scaled) {
            // w = w / sigma using BLAS dscal
            BLAS.dscal(n, 1.0 / sigma, w, wOff, 1);
        }

        work[workOff] = lworkopt;
        return 0;
    }
}
