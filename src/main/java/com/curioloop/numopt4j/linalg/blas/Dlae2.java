/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DLAE2: Computes the eigenvalues of a 2×2 symmetric matrix [a b; b c].
 * Returns the eigenvalue with larger absolute value as rt1, smaller as rt2.
 * Also provides DLAEV2 which additionally computes the eigenvectors.
 *
 * <p>Reference: gonum/lapack/gonum/dlae2.go</p>
 */
interface Dlae2 {

    static void dlae2(double a, double b, double c, double[] out, int outOff) {
        double sm = a + c;
        double df = a - c;
        double adf = Math.abs(df);
        double tb = b + b;
        double ab = Math.abs(tb);

        double acmx = c;
        double acmn = a;
        if (Math.abs(a) > Math.abs(c)) {
            acmx = a;
            acmn = c;
        }

        double rt;
        if (adf > ab) {
            rt = adf * Math.hypot(1.0, ab / adf);
        } else if (adf < ab) {
            rt = ab * Math.hypot(1.0, adf / ab);
        } else {
            rt = ab * Math.sqrt(2.0);
        }

        double rt1, rt2;
        if (sm < 0.0) {
            rt1 = 0.5 * (sm - rt);
            rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
        } else if (sm > 0.0) {
            rt1 = 0.5 * (sm + rt);
            rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
        } else {
            rt1 = 0.5 * rt;
            rt2 = -0.5 * rt;
        }
        out[outOff] = rt1;
        out[outOff + 1] = rt2;
    }

    static void dlaev2(double a, double b, double c, double[] out, int off) {
        double sm = a + c;
        double df = a - c;
        double adf = Math.abs(df);
        double tb = b + b;
        double ab = Math.abs(tb);

        double acmx = c;
        double acmn = a;
        if (Math.abs(a) > Math.abs(c)) {
            acmx = a;
            acmn = c;
        }

        double rt;
        if (adf > ab) {
            rt = adf * Math.sqrt(1.0 + (ab / adf) * (ab / adf));
        } else if (adf < ab) {
            rt = ab * Math.sqrt(1.0 + (adf / ab) * (adf / ab));
        } else {
            rt = ab * Math.sqrt(2.0);
        }

        double rt1, rt2, cs, sn;
        double sgn1;
        if (sm < 0.0) {
            rt1 = 0.5 * (sm - rt);
            sgn1 = -1.0;
            rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
        } else if (sm > 0.0) {
            rt1 = 0.5 * (sm + rt);
            sgn1 = 1.0;
            rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
        } else {
            rt1 = 0.5 * rt;
            rt2 = -0.5 * rt;
            sgn1 = 1.0;
        }

        double cs2;
        double sgn2;
        if (df >= 0.0) {
            cs2 = df + rt;
            sgn2 = 1.0;
        } else {
            cs2 = df - rt;
            sgn2 = -1.0;
        }

        double acs = Math.abs(cs2);
        if (acs > ab) {
            double ct = -tb / cs2;
            sn = 1.0 / Math.sqrt(1.0 + ct * ct);
            cs = ct * sn;
        } else {
            if (ab == 0.0) {
                cs = 1.0;
                sn = 0.0;
            } else {
                double tn = -cs2 / tb;
                cs = 1.0 / Math.sqrt(1.0 + tn * tn);
                sn = tn * cs;
            }
        }

        if (sgn1 == sgn2) {
            double tn = cs;
            cs = -sn;
            sn = tn;
        }

        out[off] = rt1;
        out[off + 1] = rt2;
        out[off + 2] = cs;
        out[off + 3] = sn;
    }
}
