/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import static java.lang.Math.*;

/**
 * LAPACK DSYCON: Estimates the reciprocal of the condition number (in the
 * 1-norm) of a real symmetric matrix A using the factorization
 * A = U * D * U^T or A = L * D * L^T computed by {@link Dsytrf}.
 *
 * <p>An estimate is obtained for ||A^{-1}||, and the reciprocal of the
 * condition number is computed as rcond = 1 / (||A|| * ||A^{-1}||).</p>
 *
 * <p>Reference: Fortran netlib LAPACK DSYCON</p>
 */
interface Dsycon {

    static double dsycon(BLAS.Uplo uplo, int n, double[] A, int aOff, int lda, int[] ipiv, int ipivOff,
                         double anorm, double[] work, int[] iwork) {
        boolean upper = uplo == BLAS.Uplo.Upper;

        if (n == 0) return 1.0;
        if (anorm <= 0) return 0.0;

        if (upper) {
            for (int i = n - 1; i >= 0; i--) {
                if (ipiv[ipivOff + i] > 0 && A[aOff + i * lda + i] == 0) {
                    return 0.0;
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                if (ipiv[ipivOff + i] > 0 && A[aOff + i * lda + i] == 0) {
                    return 0.0;
                }
            }
        }

        double ainvnm = 0.0;
        int[] isave = new int[3];
        int kase = 0;

        while (true) {
            ainvnm = Dlacn2.dlacn2(n, work, n, work, 0, iwork, 0, ainvnm, isave, 2, isave, 0);
            kase = isave[2];

            if (kase == 0) {
                break;
            }

            Dsytrs.dsytrs(uplo, n, 1, A, aOff, lda, ipiv, ipivOff, work, 0, n);
        }

        if (ainvnm != 0) {
            return (1.0 / ainvnm) / anorm;
        }
        return 0.0;
    }
}
