/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

class DlartgTest {

    private static final double SAFMIN = 0x1p-1022;
    private static final double SAFMAX = 1.0 / SAFMIN;
    private static final double ULP = Math.ulp(1.0);
    private static final double TOL = 20 * ULP;

    @Test
    @DisplayName("Dlartg: basic cases")
    void testBasicCases() {
        testDlartg(1.0, 0.0);
        testDlartg(0.0, 1.0);
        testDlartg(1.0, 1.0);
        testDlartg(1.0, -1.0);
        testDlartg(-1.0, 1.0);
        testDlartg(-1.0, -1.0);
    }

    @Test
    @DisplayName("Dlartg: extreme values")
    void testExtremeValues() {
        testDlartg(-SAFMAX, -SAFMAX);
        testDlartg(-1 / ULP, -1 / ULP);
        testDlartg(-1.0, -1.0);
        testDlartg(-1.0 / 3, -1.0 / 3);
        testDlartg(-ULP, -ULP);
        testDlartg(-SAFMIN, -SAFMIN);
        testDlartg(0, 0);
        testDlartg(SAFMIN, SAFMIN);
        testDlartg(ULP, ULP);
        testDlartg(1.0 / 3, 1.0 / 3);
        testDlartg(1.0, 1.0);
        testDlartg(1 / ULP, 1 / ULP);
        testDlartg(SAFMAX, SAFMAX);
    }

    @Test
    @DisplayName("Dlartg: mixed values")
    void testMixedValues() {
        double[] values = {-SAFMAX, -1 / ULP, -1.0, -1.0 / 3, -ULP, -SAFMIN, 0, SAFMIN, ULP, 1.0 / 3, 1.0, 1 / ULP, SAFMAX};
        
        for (double f : values) {
            for (double g : values) {
                testDlartg(f, g);
            }
        }
    }

    @Test
    @DisplayName("Dlartg: infinity and NaN")
    void testInfinityAndNaN() {
        double[] specialValues = {Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, Double.NaN};
        double[] normalValues = {0, 1.0, -1.0, SAFMIN, SAFMAX};
        
        for (double f : specialValues) {
            for (double g : normalValues) {
                testDlartgSpecial(f, g);
            }
        }
        
        for (double f : normalValues) {
            for (double g : specialValues) {
                testDlartgSpecial(f, g);
            }
        }
        
        for (double f : specialValues) {
            for (double g : specialValues) {
                testDlartgSpecial(f, g);
            }
        }
    }

    private void testDlartg(double f, double g) {
        double[] out = new double[3];
        Dlartg.dlartg(f, g, out, 0);
        double cs = out[0];
        double sn = out[1];
        double r = out[2];

        double d = Math.max(Math.abs(f), Math.abs(g));
        d = Math.min(Math.max(SAFMIN, d), SAFMAX);
        double fs = f / d;
        double gs = g / d;
        double rs = r / d;

        double rnorm = Math.abs(rs);
        if (rnorm == 0) {
            rnorm = Math.max(Math.abs(fs), Math.abs(gs));
            if (rnorm == 0) {
                rnorm = 1;
            }
        }

        double resid = Math.abs(rs - (cs * fs + sn * gs)) / rnorm;
        assertTrue(resid <= TOL, 
            String.format("cs*f + sn*g != r for f=%g, g=%g; resid=%g", f, g, resid));

        resid = Math.abs(-sn * fs + cs * gs);
        assertTrue(resid <= TOL, 
            String.format("-sn*f + cs*g != 0 for f=%g, g=%g; resid=%g", f, g, resid));

        resid = Math.abs(1 - (cs * cs + sn * sn));
        assertTrue(resid <= TOL, 
            String.format("cs*cs + sn*sn != 1 for f=%g, g=%g; resid=%g", f, g, resid));

        if (Math.abs(f) > Math.abs(g) && cs < 0) {
            fail(String.format("cs is negative for f=%g, g=%g; cs=%g", f, g, cs));
        }
    }

    private void testDlartgSpecial(double f, double g) {
        double[] out = new double[3];
        Dlartg.dlartg(f, g, out, 0);
        double r = out[2];

        if (Double.isNaN(f) || Double.isNaN(g)) {
            assertTrue(Double.isNaN(r), 
                String.format("unexpected r=%g for f=%g, g=%g; want NaN", r, f, g));
        } else if (Double.isInfinite(f) || Double.isInfinite(g)) {
            assertTrue(Double.isNaN(r) || Double.isInfinite(r), 
                String.format("unexpected r=%g for f=%g, g=%g; want NaN or Inf", r, f, g));
        }
    }
}
