/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Dtrevc3Test {

    @Test
    void testBasic() {
        double[] T = {
            1, 2, 3,
            0, 4, 5,
            0, 0, 6
        };
        int n = 3;
        double[] VL = new double[n * n];
        double[] VR = new double[n * n];
        double[] work = new double[100];

        int m = Dtrevc3.dtrevc3('B', 'A', null, n, T, 0, n, VL, 0, n, VR, 0, n, n, work, 0, work.length);

        assertEquals(n, m);
    }

    @Test
    void testEmpty() {
        int info = Dtrevc3.dtrevc3('R', 'A', null, 0, new double[0], 0, 0, null, 0, 0, new double[0], 0, 0, 0, new double[1], 0, 0);
        assertEquals(0, info);
    }
}
