/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

final class SIMD {

    private static final String SIMD_PROPERTY = "numopt4j.vector";
    private static final boolean SIMD_ENABLED =
        !"false".equalsIgnoreCase(System.getProperty(SIMD_PROPERTY, "true"));

    private static final boolean DAXPY_SUPPORTED_AT_INIT = detectDaxpy();
    private static final boolean DSCAL_SUPPORTED_AT_INIT = detectDscal();
    private static final boolean DDOT_SUPPORTED_AT_INIT = detectDdot();
    private static final boolean DGEMV_SUPPORTED_AT_INIT = detectDgemv();
    private static final boolean DTRMV_SUPPORTED_AT_INIT = detectDtrmv();
    private static final boolean DGER_SUPPORTED_AT_INIT = detectDger();
    private static final boolean DSYR2_SUPPORTED_AT_INIT = detectDsyr2();
    private static final boolean DSYMV_SUPPORTED_AT_INIT = detectDsymv();

    private SIMD() {
    }

    static boolean supportDaxpy() {
        return DAXPY_SUPPORTED_AT_INIT;
    }

    static boolean supportDscal() {
        return DSCAL_SUPPORTED_AT_INIT;
    }

    static boolean supportDdot() {
        return DDOT_SUPPORTED_AT_INIT;
    }

    static boolean supportDgemv() {
        return DGEMV_SUPPORTED_AT_INIT;
    }

    static boolean supportDtrmv() {
        return DTRMV_SUPPORTED_AT_INIT;
    }

    static boolean supportDger() {
        return DGER_SUPPORTED_AT_INIT;
    }

    static boolean supportDsyr2() {
        return DSYR2_SUPPORTED_AT_INIT;
    }

    static boolean supportDsymv() {
        return DSYMV_SUPPORTED_AT_INIT;
    }

    private static boolean detectDaxpy() {
        if (!SIMD_ENABLED) {
            return false;
        }
        try {
            return DaxpySIMD.probe();
        } catch (LinkageError e) {
            return false;
        }
    }

    private static boolean detectDscal() {
        if (!SIMD_ENABLED) {
            return false;
        }
        try {
            return DscalSIMD.probe();
        } catch (LinkageError e) {
            return false;
        }
    }

    private static boolean detectDdot() {
        if (!SIMD_ENABLED) {
            return false;
        }
        try {
            return DdotSIMD.probe();
        } catch (LinkageError e) {
            return false;
        }
    }

    private static boolean detectDgemv() {
        if (!SIMD_ENABLED) {
            return false;
        }
        try {
            return DgemvSIMD.probe();
        } catch (LinkageError e) {
            return false;
        }
    }

    private static boolean detectDtrmv() {
        if (!SIMD_ENABLED) {
            return false;
        }
        try {
            return DtrmvSIMD.probe();
        } catch (LinkageError e) {
            return false;
        }
    }

    private static boolean detectDger() {
        if (!SIMD_ENABLED) {
            return false;
        }
        try {
            return DgerSIMD.probe();
        } catch (LinkageError e) {
            return false;
        }
    }

    private static boolean detectDsyr2() {
        if (!SIMD_ENABLED) {
            return false;
        }
        try {
            return Dsyr2SIMD.probe();
        } catch (LinkageError e) {
            return false;
        }
    }

    private static boolean detectDsymv() {
        if (!SIMD_ENABLED) {
            return false;
        }
        try {
            return DsymvSIMD.probe();
        } catch (LinkageError e) {
            return false;
        }
    }
}

final class Gate {

    static final VectorSpecies<Double> SPECIES = preferredSpecies256Cap();
    static final int LANES = SPECIES.length();

    private Gate() {
    }

    private static VectorSpecies<Double> preferredSpecies256Cap() {
        VectorSpecies<Double> preferred = DoubleVector.SPECIES_PREFERRED;
        if (preferred.length() >= DoubleVector.SPECIES_256.length()) {
            return DoubleVector.SPECIES_256;
        }
        return preferred;
    }
}