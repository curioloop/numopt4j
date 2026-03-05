/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DgebalTest {

    @Test
    void testBasic() {
        double[] A = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        };
        int n = 3;
        double[] scale = new double[n];
        double[] out = new double[2];

        Dgebal.dgebal('B', n, A, n, scale, 0, out, 0);

        assertTrue(out[0] >= 0);
    }

    @Test
    void testEmpty() {
        double[] out = new double[2];
        Dgebal.dgebal('N', 0, new double[0], 0, new double[0], 0, out, 0);
        assertEquals(0, out[0], 1e-10);
        assertEquals(0, out[1], 1e-10);
    }
}
