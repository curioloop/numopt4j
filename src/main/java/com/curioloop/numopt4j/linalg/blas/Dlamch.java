/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * LAPACK DLAMCH: Returns machine-dependent constants.
 * 
 * This is a simple implementation returning common double precision constants.
 */
interface Dlamch {

    // Machine constants (IEEE 754 double precision)
    // Epsilon: 2^-52 = b^(1-t) where b=2, t=53
    double EPSILON = 0x1.0p-52;  // dlamchE: epsilon
    // Safe minimum: 2^-1022 (smallest normalized number)
    double SAFE_MIN = 0x1.0p-1022;  // dlamchS: safe minimum
    double BASE = 2.0;  // dlamchB
    // Precision: eps * base = 2^-51
    double PREC = 0x1.0p-51;  // dlamchP: base * epsilon

    /**
     * DLAMCH returns machine-dependent constants.
     *
     * @param cmach Specifies which constant to return:
     *        'E' or 'e' - epsilon (relative machine precision)
     *        'S' or 's' - safe minimum (smallest normalized number)
     *        'B' or 'b' - base
     *        'P' or 'p' - base * epsilon
     *        'N' or 'n' - number of digits in mantissa
     *        'M' or 'm' - minimum exponent
     *        'U' or 'u' - underflow threshold
     *        'L' or 'l' - largest exponent
     *        'O' or 'o' - overflow threshold
     *        'R' or 'r' - 1 / safe minimum
     * @return the requested constant
     */
    static double dlamch(char cmach) {
        switch (Character.toUpperCase(cmach)) {
            case 'E':
                // epsilon: relative machine precision
                return EPSILON;
            case 'S':
                // safe minimum: smallest normalized number
                return SAFE_MIN;
            case 'B':
                // base
                return BASE;
            case 'P':
                // base * epsilon
                return PREC;
            case 'N':
                // number of digits in mantissa
                return 53.0;
            case 'M':
                // minimum exponent (denormalized)
                return -1021.0;
            case 'U':
                // underflow threshold
                return Math.pow(2, -1022);
            case 'L':
                // largest exponent
                return 1024.0;
            case 'O':
                // overflow threshold
                return Math.pow(2, 1023);
            case 'R':
                // reciprocal of safe minimum
                return 1.0 / SAFE_MIN;
            default:
                throw new IllegalArgumentException("Invalid DLAMCH parameter: " + cmach);
        }
    }

    /**
     * Returns the safe minimum (smallest normalized number).
     */
    static double safmin() {
        return SAFE_MIN;
    }

    /**
     * Returns epsilon (relative machine precision).
     */
    static double eps() {
        return EPSILON;
    }
}
