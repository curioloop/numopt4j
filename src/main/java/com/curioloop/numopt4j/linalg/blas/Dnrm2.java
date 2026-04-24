/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DNRM2 computes the Euclidean norm of a vector.
 * 
 * <p>Mathematical operation: ‖x‖₂ = √(Σxᵢ²)</p>
 * 
 * <p>Uses LAPACK 3.x three-interval algorithm for robust overflow/underflow protection:</p>
 * <ul>
 *   <li>Small values (|x| &lt; tsml): accumulated in asml with scaling by ssml</li>
 *   <li>Medium values (tsml ≤ |x| ≤ tbig): accumulated directly in amed</li>
 *   <li>Large values (|x| &gt; tbig): accumulated in abig with scaling by sbig</li>
 * </ul>
 * 
 * <p>Reference: LAPACK 3.x DNRM2 (Blue's algorithm variant)</p>
 */
interface Dnrm2 {

    /** Threshold for small values: 2^(-500) ≈ 3.27e-151 */
    double TSML = Math.pow(2.0, Math.ceil((-1021 - 1) * 0.5));
    
    /** Threshold for large values: 2^(500) ≈ 3.27e+150 */
    double TBIG = Math.pow(2.0, Math.floor((1024 - 53 + 1) * 0.5));
    
    /** Scale factor for small values: 2^(537) */
    double SSML = Math.pow(2.0, -Math.floor((-1021 - 53) * 0.5));
    
    /** Scale factor for large values: 2^(-538) */
    double SBIG = Math.pow(2.0, -Math.ceil((1024 + 53 - 1) * 0.5));

    /**
     * Computes the Euclidean norm of vector x using LAPACK 3.x algorithm.
     * 
     * @param n    number of elements
     * @param x    input vector
     * @param xOff offset in x
     * @param incX storage spacing
     * @return ‖x‖₂
     */
    static double dnrm2(int n, double[] x, int xOff, int incX) {
        if (n <= 0) return 0.0;
        if (n == 1) return Math.abs(x[xOff]);

        double asml = 0.0;
        double amed = 0.0;
        double abig = 0.0;
        boolean notbig = true;

        int ix = xOff + ((incX > 0) ? 0 : (n - 1) * (-incX));

        for (int i = 0; i < n; i++) {
            double ax = Math.abs(x[ix]);
            
            if (ax > TBIG) {
                double scaled = ax * SBIG;
                abig = Math.fma(scaled, scaled, abig);
                notbig = false;
            } else if (ax < TSML) {
                if (notbig) {
                    double scaled = ax * SSML;
                    asml = Math.fma(scaled, scaled, asml);
                }
            } else {
                amed = Math.fma(ax, ax, amed);
            }
            ix += incX;
        }

        double scl, sumsq;
        if (abig > 0.0) {
            if (amed > 0.0 || Double.isNaN(amed)) {
                abig += (amed * SBIG) * SBIG;
            }
            scl = 1.0 / SBIG;
            sumsq = abig;
        } else if (asml > 0.0) {
            if (amed > 0.0 || Double.isNaN(amed)) {
                double ymin, ymax;
                amed = Math.sqrt(amed);
                asml = Math.sqrt(asml) / SSML;
                if (asml > amed) {
                    ymin = amed;
                    ymax = asml;
                } else {
                    ymin = asml;
                    ymax = amed;
                }
                scl = 1.0;
                double ratio = ymin / ymax;
                sumsq = ymax * ymax * (1.0 + ratio * ratio);
            } else {
                scl = 1.0 / SSML;
                sumsq = asml;
            }
        } else {
            scl = 1.0;
            sumsq = amed;
        }

        return scl * Math.sqrt(sumsq);
    }
}
