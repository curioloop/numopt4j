/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * DLARTG: Generates Givens rotation.
 * Based on Algorithm 978 (2017) - Safe Scaling in the Level 1 BLAS.
 */
interface Dlartg {
    
    static final double SAFMIN = 0x1p-1022;
    
    /**
     * Generate Givens rotation such that:
     * [ cs  sn] * [f] = [r]
     * [-sn  cs]   [g]   [0]
     * 
     * @param f first element
     * @param g second element
     * @param out output array, out[0]=cs, out[1]=sn, out[2]=r
     * @param off offset into out array
     */
    static void dlartg(double f, double g, double[] out, int off) {
        if (g == 0.0) {
            out[off] = 1.0;
            out[off + 1] = 0.0;
            out[off + 2] = f;
            return;
        }
        
        double g1 = Math.abs(g);
        
        if (f == 0.0) {
            out[off] = 0.0;
            out[off + 1] = Math.copySign(1.0, g);
            out[off + 2] = g1;
            return;
        }
        
        double safmin = SAFMIN;
        double safmax = 1.0 / safmin;
        double rtmin = Math.sqrt(safmin);
        double rtmax = Math.sqrt(safmax / 2.0);
        
        double f1 = Math.abs(f);
        double cs, r, sn;
        if (rtmin < f1 && f1 < rtmax && rtmin < g1 && g1 < rtmax) {
            double d = Math.hypot(f, g);
            cs = f1 / d;
            r = Math.copySign(d, f);
            sn = g / r;
        } else {
            double u = Math.min(Math.max(safmin, Math.max(f1, g1)), safmax);
            double fs = f / u;
            double gs = g / u;
            double d = Math.hypot(fs, gs);
            cs = Math.abs(fs) / d;
            r = Math.copySign(d, f);
            sn = gs / r;
            r *= u;
        }
        out[off] = cs;
        out[off + 1] = sn;
        out[off + 2] = r;
    }
}
