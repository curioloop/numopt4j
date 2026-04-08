package com.curioloop.numopt4j.quad.ode.rk;

/** Bogacki-Shampine RK23 coefficients, strictly following scipy rk.py RK23 class. */
final class RK23Coeffs {
    static final int ORDER = 3;
    static final int ERROR_ORDER = 2;
    static final int N_STAGES = 3;
    static final int INTERP_ORDER = 3; // cubic Hermite interpolation

    // C: time nodes
    static final double[] C = {0.0, 0.5, 0.75};

    // A: Butcher tableau lower triangle, row-major expansion; row s offset = s*(s-1)/2, length = s
    // A[1][0]=0.5, A[2][0]=0, A[2][1]=0.75
    static final double[] A = {
        // row 1
        0.5,
        // row 2
        0.0, 0.75
    };

    // B: high-order solution coefficients (3rd order)
    static final double[] B = {2.0/9, 1.0/3, 4.0/9};

    // E: error estimation coefficients (E = B_high - B_low, length N_STAGES+1)
    static final double[] E = {5.0/72, -1.0/12, -1.0/9, 1.0/8};

    // P: dense output coefficients, shape (N_STAGES+1) × INTERP_ORDER, row-major expansion
    // corresponding to scipy RK23.P
    static final double[] P = {
        1,    -4.0/3,   5.0/9,
        0,     1.0,    -2.0/3,
        0,     4.0/3,  -8.0/9,
        0,    -1.0,     1.0
    };

}
