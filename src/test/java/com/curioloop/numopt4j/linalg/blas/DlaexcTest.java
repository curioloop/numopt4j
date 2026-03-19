/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class DlaexcTest {

    @Test
    void testEmpty() {
        double[] work = new double[10];
        boolean ok = Dlaexc.dlaexc(false, 0, new double[0], 0, 0, new double[0], 0, 0, 0, 0, 0, work, 0);
        assertTrue(ok);
    }
}
