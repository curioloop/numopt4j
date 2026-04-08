package com.curioloop.numopt4j.quad.ode.rk;

/** Dormand-Prince RK45 coefficients, strictly following scipy rk.py RK45 class. */
final class RK45Coeffs {
    static final int ORDER = 5;
    static final int ERROR_ORDER = 4;
    static final int N_STAGES = 6;
    static final int INTERP_ORDER = 4; // 4th-order interpolation

    static final double[] C = {0.0, 1.0/5, 3.0/10, 4.0/5, 8.0/9, 1.0};

    // A: Butcher tableau lower triangle, row-major expansion; row s offset = s*(s-1)/2, length = s
    static final double[] A = {
        // row 1
        1.0/5,
        // row 2
        3.0/40,        9.0/40,
        // row 3
        44.0/45,      -56.0/15,       32.0/9,
        // row 4
        19372.0/6561, -25360.0/2187,  64448.0/6561, -212.0/729,
        // row 5
        9017.0/3168,  -355.0/33,      46732.0/5247,   49.0/176,  -5103.0/18656
    };

    static final double[] B = {35.0/384, 0.0, 500.0/1113, 125.0/192, -2187.0/6784, 11.0/84};

    // E: error coefficients, length N_STAGES+1
    static final double[] E = {
        -71.0/57600, 0.0, 71.0/16695, -71.0/1920, 17253.0/339200, -22.0/525, 1.0/40
    };

    // P: dense output coefficients, shape (N_STAGES+1) × INTERP_ORDER, row-major expansion
    // corresponding to scipy RK45.P (Shampine 1986)
    static final double[] P = {
        1.0, -8048581381.0/2820520608.0,   8663915743.0/2820520608.0,  -12715105075.0/11282082432.0,
        0.0,  0.0,                          0.0,                          0.0,
        0.0,  131558114200.0/32700410799.0, -68118460800.0/10900136933.0, 87487479700.0/32700410799.0,
        0.0, -1754552775.0/470086768.0,     14199869525.0/1410260304.0,  -10690763975.0/1880347072.0,
        0.0,  127303824393.0/49829197408.0, -318862633887.0/49829197408.0, 701980252875.0/199316789632.0,
        0.0, -282668133.0/205662961.0,       2019193451.0/616988883.0,    -1453857185.0/822651844.0,
        0.0,  40617522.0/29380423.0,        -110615467.0/29380423.0,       69997945.0/29380423.0
    };

}
