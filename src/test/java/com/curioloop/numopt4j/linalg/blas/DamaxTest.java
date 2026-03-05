/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DamaxTest {

    @Test
    void testBasic() {
        double[] x = {1, -5, 3, -2, 4};
        int idx = Damax.idamax(5, x, 0, 1);
        assertEquals(1, idx);
    }

    @Test
    void testAllPositive() {
        double[] x = {1, 2, 3, 4, 5};
        int idx = Damax.idamax(5, x, 0, 1);
        assertEquals(4, idx);
    }

    @Test
    void testAllNegative() {
        double[] x = {-1, -2, -3, -4, -5};
        int idx = Damax.idamax(5, x, 0, 1);
        assertEquals(4, idx);
    }

    @Test
    void testEmpty() {
        int idx = Damax.idamax(0, new double[0], 0, 1);
        assertEquals(-1, idx);
    }

    @Test
    void testSingleElement() {
        double[] x = {5};
        int idx = Damax.idamax(1, x, 0, 1);
        assertEquals(0, idx);
    }
}
