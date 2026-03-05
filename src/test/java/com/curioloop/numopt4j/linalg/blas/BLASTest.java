/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;
import static com.curioloop.numopt4j.linalg.blas.Dlamv.*;

/**
 * Unit tests for BLAS Level 1 routines.
 */
class BLASTest {

    private static final double EPSILON = 1e-14;

    // ==================== DCOPY Tests ====================

    @Test
    @DisplayName("dcopy: contiguous copy")
    void testDcopyContiguous() {
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] y = new double[5];
        
        dcopy(5, x, 0, 1, y, 0, 1);
        
        assertArrayEquals(x, y, EPSILON);
    }

    @Test
    @DisplayName("dcopy: strided copy")
    void testDcopyStrided() {
        double[] x = {1.0, 0.0, 2.0, 0.0, 3.0};
        double[] y = new double[5];
        
        dcopy(3, x, 0, 2, y, 0, 2);
        
        assertEquals(1.0, y[0], EPSILON);
        assertEquals(2.0, y[2], EPSILON);
        assertEquals(3.0, y[4], EPSILON);
    }

    // ==================== DNRM2 Tests ====================

    @Test
    @DisplayName("dnrm2: basic norm")
    void testDnrm2Basic() {
        double[] x = {3.0, 4.0};
        assertEquals(5.0, Dnrm2.dnrm2(2, x, 0, 1), EPSILON);
    }

    @Test
    @DisplayName("dnrm2: overflow protection")
    void testDnrm2Overflow() {
        double[] x = {1e154, 1e154};
        double result = Dnrm2.dnrm2(2, x, 0, 1);
        double expected = Math.sqrt(2) * 1e154;
        assertEquals(expected, result, expected * 1e-14);
    }

    @Test
    @DisplayName("dnrm2: underflow protection")
    void testDnrm2Underflow() {
        double[] x = {1e-154, 1e-154};
        double result = Dnrm2.dnrm2(2, x, 0, 1);
        double expected = Math.sqrt(2) * 1e-154;
        assertEquals(expected, result, expected * 1e-14);
    }

    @Test
    @DisplayName("dnrm2: mixed magnitudes")
    void testDnrm2MixedMagnitudes() {
        double[] x = {1e100, 1.0, 1e-100};
        double result = Dnrm2.dnrm2(3, x, 0, 1);
        assertEquals(1e100, result, 1e86);
    }

    // ==================== DAMAX Tests ====================

    @Test
    @DisplayName("damax: basic infinity norm")
    void testDamaxBasic() {
        double[] x = {1.0, -5.0, 3.0, -2.0};
        assertEquals(5.0, Damax.damax(4, x, 0, 1), EPSILON);
    }

    @Test
    @DisplayName("damax: all negative")
    void testDamaxAllNegative() {
        double[] x = {-1.0, -5.0, -3.0};
        assertEquals(5.0, Damax.damax(3, x, 0, 1), EPSILON);
    }

    // ==================== IDAMAX Tests ====================

    @Test
    @DisplayName("idamax: basic index")
    void testIdamaxBasic() {
        double[] x = {1.0, -5.0, 3.0, -2.0};
        assertEquals(1, Damax.idamax(4, x, 0, 1));
    }

    @Test
    @DisplayName("idamax: first element max")
    void testIdamaxFirst() {
        double[] x = {10.0, 5.0, 3.0};
        assertEquals(0, Damax.idamax(3, x, 0, 1));
    }

    @Test
    @DisplayName("idamax: last element max")
    void testIdamaxLast() {
        double[] x = {1.0, 2.0, 10.0};
        assertEquals(2, Damax.idamax(3, x, 0, 1));
    }

    // ==================== DSWAP Tests ====================

    @Test
    @DisplayName("dswap: contiguous swap")
    void testDswapContiguous() {
        double[] x = {1.0, 2.0, 3.0};
        double[] y = {4.0, 5.0, 6.0};
        
        dswap(3, x, 0, 1, y, 0, 1);
        
        assertArrayEquals(new double[]{4.0, 5.0, 6.0}, x, EPSILON);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, y, EPSILON);
    }

    @Test
    @DisplayName("dswap: strided swap")
    void testDswapStrided() {
        double[] x = {1.0, 0.0, 2.0, 0.0, 3.0};
        double[] y = {4.0, 0.0, 5.0, 0.0, 6.0};
        
        dswap(3, x, 0, 2, y, 0, 2);
        
        assertEquals(4.0, x[0], EPSILON);
        assertEquals(5.0, x[2], EPSILON);
        assertEquals(6.0, x[4], EPSILON);
        assertEquals(1.0, y[0], EPSILON);
        assertEquals(2.0, y[2], EPSILON);
        assertEquals(3.0, y[4], EPSILON);
    }

    // ==================== DROT Tests ====================

    @Test
    @DisplayName("drot: 90 degree rotation")
    void testDrot90() {
        double[] x = {1.0, 0.0};
        double[] y = {0.0, 1.0};
        double c = 0.0;
        double s = 1.0;
        
        Drot.drot(2, x, 0, 1, y, 0, 1, c, s);
        
        assertEquals(0.0, x[0], EPSILON);
        assertEquals(1.0, x[1], EPSILON);
        assertEquals(-1.0, y[0], EPSILON);
        assertEquals(0.0, y[1], EPSILON);
    }

    @Test
    @DisplayName("drot: 45 degree rotation")
    void testDrot45() {
        double[] x = {1.0};
        double[] y = {0.0};
        double c = Math.sqrt(2) / 2;
        double s = Math.sqrt(2) / 2;
        
        Drot.drot(1, x, 0, 1, y, 0, 1, c, s);
        
        assertEquals(c, x[0], EPSILON);
        assertEquals(-s, y[0], EPSILON);
    }

    // ==================== DSET Tests ====================

    @Test
    @DisplayName("dset: contiguous fill")
    void testDsetContiguous() {
        double[] x = new double[5];
        
        dset(5, 3.14, x, 0, 1);
        
        for (double v : x) {
            assertEquals(3.14, v, EPSILON);
        }
    }

    @Test
    @DisplayName("dset: strided fill")
    void testDsetStrided() {
        double[] x = new double[5];
        
        dset(3, 7.0, x, 0, 2);
        
        assertEquals(7.0, x[0], EPSILON);
        assertEquals(0.0, x[1], EPSILON);
        assertEquals(7.0, x[2], EPSILON);
        assertEquals(0.0, x[3], EPSILON);
        assertEquals(7.0, x[4], EPSILON);
    }

    // ==================== DZERO Tests ====================

    @Test
    @DisplayName("dzero: contiguous zero")
    void testDzeroContiguous() {
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0};
        
        dzero(5, x, 0, 1);
        
        for (double v : x) {
            assertEquals(0.0, v, EPSILON);
        }
    }

    @Test
    @DisplayName("dzero: strided zero")
    void testDzeroStrided() {
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0};
        
        dzero(3, x, 0, 2);
        
        assertEquals(0.0, x[0], EPSILON);
        assertEquals(2.0, x[1], EPSILON);  // unchanged
        assertEquals(0.0, x[2], EPSILON);
        assertEquals(4.0, x[3], EPSILON);  // unchanged
        assertEquals(0.0, x[4], EPSILON);
    }

    @Test
    @DisplayName("dzero: with offset")
    void testDzeroWithOffset() {
        double[] x = {1.0, 2.0, 3.0, 4.0, 5.0};
        
        dzero(3, x, 1, 1);
        
        assertEquals(1.0, x[0], EPSILON);  // unchanged
        assertEquals(0.0, x[1], EPSILON);
        assertEquals(0.0, x[2], EPSILON);
        assertEquals(0.0, x[3], EPSILON);
        assertEquals(5.0, x[4], EPSILON);  // unchanged
    }

    @Test
    @DisplayName("dzero: n=0 does nothing")
    void testDzeroZeroN() {
        double[] x = {1.0, 2.0, 3.0};
        
        dzero(0, x, 0, 1);
        
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, x, EPSILON);
    }

    // ==================== Edge Cases ====================

    @Test
    @DisplayName("edge case: n=0")
    void testEdgeCaseZeroN() {
        double[] x = {1.0, 2.0};
        double[] y = {3.0, 4.0};
        
        dcopy(0, x, 0, 1, y, 0, 1);
        dswap(0, x, 0, 1, y, 0, 1);
        Drot.drot(0, x, 0, 1, y, 0, 1, 0.5, 0.5);
        dset(0, 5.0, x, 0, 1);
        
        assertArrayEquals(new double[]{1.0, 2.0}, x, EPSILON);
        assertArrayEquals(new double[]{3.0, 4.0}, y, EPSILON);
        
        assertEquals(0.0, Dnrm2.dnrm2(0, x, 0, 1));
        assertEquals(0.0, Damax.damax(0, x, 0, 1));
        assertEquals(-1, Damax.idamax(0, x, 0, 1));
    }

    @Test
    @DisplayName("edge case: n=1")
    void testEdgeCaseOneN() {
        double[] x = {-5.0};
        
        assertEquals(5.0, Dnrm2.dnrm2(1, x, 0, 1));
        assertEquals(5.0, Damax.damax(1, x, 0, 1));
        assertEquals(0, Damax.idamax(1, x, 0, 1));
    }
}
