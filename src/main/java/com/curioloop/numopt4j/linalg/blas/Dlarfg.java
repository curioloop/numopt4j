/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * Generates Householder reflections.
 * LAPACK DLARFG algorithm.
 *
 * <p>Given a vector (alpha, x)^T, generates a Householder transformation H:
 * H * (alpha, x)^T = (beta, 0)^T</p>
 *
 * <p>Where H = I - tau * v * v^T, with v = (1, x)^T</p>
 *
 * <p>The Householder reflection is represented as:</p>
 * <ul>
 *   <li>beta: returned directly</li>
 *   <li>tau: returned directly</li>
 *   <li>v: stored in x (output), with implicit v[0] = 1</li>
 * </ul>
 */
interface Dlarfg {

    /** Machine epsilon constant. */
    // dlamchE = 2^-53 (epsilon)
    static final double DLAMCH_E = 0x1p-53;
    // dlamchS = 2^-1022 (underflow threshold)
    static final double DLAMCH_S = 0x1p-1022;
    // safmin = dlamchS / dlamchE = 2^-969
    static final double SAFMIN = DLAMCH_S / DLAMCH_E;

    /**
     * Generates a Householder reflection.
     *
     * <p>DLARFG generates a elementary reflection H of order n such that:</p>
     * <pre>
     * H * [alpha] = [beta]
     *     [ x   ]   [ 0  ]
     * </pre>
     * <p>where H = I - tau * v * v^T, with v = (1, x)^T.</p>
     *
     * @param n       the order of the elementary reflection (n >= 1)
     * @param alpha   the scalar alpha
     * @param x       the vector x (length n-1), on exit: the vector v
     * @param xOff    offset into x
     * @param incx    the increment between elements of x (incx != 0)
     * @param tau     on exit: the scalar tau
     * @param tauOff  offset into tau
     * @return beta    the scalar beta
     */
    static double dlarfg(int n, double alpha,
                         double[] x, int xOff, int incx,
                         double[] tau, int tauOff) {
        if (n <= 1) {
            tau[tauOff] = 0.0;
            return alpha;
        }

        // Compute xnorm = ||x||
        double xnorm = BLAS.dnrm2(n - 1, x, xOff, incx);
        if (xnorm == 0.0) {
            // x is zero, H = I
            tau[tauOff] = 0.0;
            return alpha;
        }

        // Compute beta using stable hypot
        double beta = -Math.copySign(Math.hypot(alpha, xnorm), alpha);

        // Scale x if necessary to avoid underflow/overflow
        int knt = 0;
        if (Math.abs(beta) < SAFMIN) {
            // xnorm and beta may be inaccurate, scale x and recompute.
            double rsafmn = 1.0 / SAFMIN;
            do {
                knt++;
                BLAS.dscal(n - 1, rsafmn, x, xOff, incx);
                beta *= rsafmn;
                alpha *= rsafmn;
            } while (Math.abs(beta) < SAFMIN);
            
            // Recompute xnorm and beta
            xnorm = BLAS.dnrm2(n - 1, x, xOff, incx);
            beta = -Math.copySign(Math.hypot(alpha, xnorm), alpha);
        }

        // Compute tau
        tau[tauOff] = (beta - alpha) / beta;

        // Scale x and store in place: x = x / (alpha - beta)
        double invAlphaBeta = 1.0 / (alpha - beta);
        BLAS.dscal(n - 1, invAlphaBeta, x, xOff, incx);

        // Undo scaling on beta
        for (int j = 0; j < knt; j++) {
            beta *= SAFMIN;
        }
        
        return beta;
    }
}
