/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DRSCL multiplies a vector by the reciprocal of a scalar, avoiding overflow or underflow.
 *
 * <p>Computes {@code x = x / a} safely by iteratively scaling with {@code smlnum} or
 * {@code bignum} until the ratio {@code 1/a} can be represented without overflow.
 *
 * <p>Corresponds to LAPACK auxiliary routine {@code DRSCL}.
 *
 * @see <a href="https://netlib.org/lapack/explore-html/d4/d7a/drscl_8f.html">LAPACK DRSCL reference (Netlib)</a>
 */
interface Drscl {

    static void drscl(int n, double a, double[] x, int xOff, int incX) {
        if (n <= 0) return;

        double cden = a;
        double cnum = 1.0;
        double smlnum = Dlamch.dlamch('S');
        double bignum = 1.0 / smlnum;

        while (true) {
            double cden1 = cden * smlnum;
            double cnum1 = cnum / bignum;
            double mul;
            boolean done;

            if (cnum != 0 && abs(cden1) > abs(cnum)) {
                mul = smlnum;
                done = false;
                cden = cden1;
            } else if (abs(cnum1) > abs(cden)) {
                mul = bignum;
                done = false;
                cnum = cnum1;
            } else {
                mul = cnum / cden;
                done = true;
            }

            BLAS.dscal(n, mul, x, xOff, incX);
            if (done) break;
        }
    }

    static double abs(double x) {
        return x < 0 ? -x : x;
    }
}
