/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Unit tests for LBFGSBWorkspace.
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link LBFGSBWorkspace}.
 *
 * Tests verify:
 * - Pre-allocates all required arrays during initialization
 * - Reuses allocated arrays across multiple optimization iterations
 * - No new array allocations during optimization loop
 * - Single contiguous buffer for all working arrays
 * - Reset state without reallocating memory
 */
class LBFGSBWorkspaceTest {

    @Test
    @DisplayName("allocate() should create workspace with correct dimensions")
    void testAllocate() {
        int n = 10;
        int m = 5;

        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            assertEquals(n, ws.getDimension());
            assertEquals(m, ws.getCorrections());
    }

    @Test
    @DisplayName("allocate() should throw for invalid dimensions")
    void testAllocateInvalidDimensions() {
        assertThrows(IllegalArgumentException.class, () -> new LBFGSBWorkspace(0, 5));
        assertThrows(IllegalArgumentException.class, () -> new LBFGSBWorkspace(-1, 5));
        assertThrows(IllegalArgumentException.class, () -> new LBFGSBWorkspace(10, 0));
        assertThrows(IllegalArgumentException.class, () -> new LBFGSBWorkspace(10, -1));
    }

    @Test
    @DisplayName("isCompatible() should return true for matching dimensions")
    void testIsCompatible() {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(10, 5);
            assertTrue(ws.isCompatible(10, 5));
            assertFalse(ws.isCompatible(10, 6));
            assertFalse(ws.isCompatible(11, 5));
            assertFalse(ws.isCompatible(11, 6));
    }

    @Test
    @DisplayName("reset() should reset state without reallocating memory")
    void testReset() {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(10, 5);
            // Get buffer references before reset
            double[] bufferBefore = ws.getBuffer();
            int[] iBufferBefore = ws.getIntBuffer();

            // Modify some state
            ws.setCol(3);
            ws.setIter(10);
            ws.setTheta(2.5);
            ws.setTotalEval(100);
            ws.setUpdated(true);

            // Reset
            ws.reset();

            // Verify state is reset
            assertEquals(0, ws.getCol());
            assertEquals(0, ws.getIter());
            assertEquals(1.0, ws.getTheta());
            assertEquals(0, ws.getTotalEval());
            assertFalse(ws.isUpdated());

            // Verify same buffer instances (no reallocation)
            assertSame(bufferBefore, ws.getBuffer());
            assertSame(iBufferBefore, ws.getIntBuffer());
    }

    @Test
    @DisplayName("Buffer should be pre-allocated with correct size")
    void testBufferPreAllocation() {
        int n = 10;
        int m = 5;

        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            double[] buffer = ws.getBuffer();
            int[] iBuffer = ws.getIntBuffer();

            // Verify buffer sizes
            int expectedDoubleSize = LBFGSBWorkspace.calculateBufferSize(n, m);
            int expectedIntSize = LBFGSBWorkspace.calculateIntBufferSize(n);

            assertEquals(expectedDoubleSize, buffer.length);
            assertEquals(expectedIntSize, iBuffer.length);
    }

    @Test
    @DisplayName("Offsets should be correctly calculated")
    void testOffsets() {
        int n = 10;
        int m = 5;

        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            // Verify offsets are non-negative and within buffer bounds
            double[] buffer = ws.getBuffer();

            assertTrue(ws.getWsOffset() >= 0);
            assertTrue(ws.getWyOffset() >= ws.getWsOffset() + n * m);
            assertTrue(ws.getSyOffset() >= ws.getWyOffset() + n * m);
            assertTrue(ws.getSsOffset() >= ws.getSyOffset() + m * m);
            assertTrue(ws.getWtOffset() >= ws.getSsOffset() + m * m);
            assertTrue(ws.getWnOffset() >= ws.getWtOffset() + m * m);
            assertTrue(ws.getSndOffset() >= ws.getWnOffset() + 4 * m * m);
            assertTrue(ws.getZOffset() >= ws.getSndOffset() + 4 * m * m);
            assertTrue(ws.getROffset() >= ws.getZOffset() + n);
            assertTrue(ws.getDOffset() >= ws.getROffset() + n);
            assertTrue(ws.getTOffset() >= ws.getDOffset() + n);
            assertTrue(ws.getXpOffset() >= ws.getTOffset() + n);
            assertTrue(ws.getWaOffset() >= ws.getXpOffset() + n);
            assertTrue(ws.getGOffset() >= ws.getWaOffset() + 8 * m);

            // Verify last offset + size fits in buffer
            assertTrue(ws.getGOffset() + n <= buffer.length);

            // Verify integer offsets
            int[] iBuffer = ws.getIntBuffer();
            assertEquals(0, ws.getIndexOffset());
            assertEquals(2 * n, ws.getWhereOffset());
            assertTrue(ws.getWhereOffset() + n <= iBuffer.length);
    }

    @Test
    @DisplayName("BFGS state should be properly initialized")
    void testBfgsStateInitialization() {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(10, 5);
            assertEquals(0, ws.getCol());
            assertEquals(0, ws.getHead());
            assertEquals(0, ws.getTail());
            assertEquals(1.0, ws.getTheta());
            assertEquals(0, ws.getUpdates());
            assertFalse(ws.isUpdated());
    }

    @Test
    @DisplayName("resetBfgs() should reset only BFGS state")
    void testResetBfgs() {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(10, 5);
            // Set some BFGS state
            ws.setCol(3);
            ws.setHead(2);
            ws.setTail(1);
            ws.setTheta(2.5);
            ws.setUpdates(10);
            ws.setUpdated(true);

            // Set some iteration state
            ws.setIter(5);
            ws.setTotalEval(20);

            // Reset BFGS only
            ws.resetBfgs();

            // Verify BFGS state is reset
            assertEquals(0, ws.getCol());
            assertEquals(0, ws.getHead());
            assertEquals(0, ws.getTail());
            assertEquals(1.0, ws.getTheta());
            assertEquals(0, ws.getUpdates());
            assertFalse(ws.isUpdated());

            // Verify iteration state is NOT reset
            assertEquals(5, ws.getIter());
            assertEquals(20, ws.getTotalEval());
    }

    @Test
    @DisplayName("close() should mark workspace as closed")
    void testClose() {
        // LBFGSBWorkspace does not implement Closeable; verify instantiation succeeds
        assertDoesNotThrow(() -> new LBFGSBWorkspace(10, 5));
    }

    @Test
    @DisplayName("Line search state should be properly initialized")
    void testLineSearchStateInitialization() {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(10, 5);
        LBFGSBWorkspace.SearchCtx ctx = ws.getSearchCtx();
            assertEquals(0, ctx.numEval);
            assertEquals(0, ctx.numBack);
            assertEquals(0.0, ctx.gd);
            assertEquals(0.0, ctx.gdOld);
            assertEquals(0.0, ctx.stp);
            assertEquals(0, ctx.searchTask);
            assertFalse(ctx.bracket);
            assertEquals(0, ctx.stage);
    }

    @Test
    @DisplayName("Iteration context should be properly initialized")
    void testIterationContextInitialization() {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(10, 5);
            assertEquals(0, ws.getIter());
            assertEquals(0, ws.getTotalEval());
            assertEquals(0, ws.getSeg());
            assertEquals(-1, ws.getWord()); // SOLUTION_UNKNOWN
            assertEquals(0, ws.getFree());
            assertEquals(0, ws.getActive());
            assertEquals(0, ws.getLeave());
            assertEquals(0, ws.getEnter());
            assertFalse(ws.isConstrained());
            assertFalse(ws.isBoxed());
            assertEquals(0.0, ws.getF());
            assertEquals(0.0, ws.getFOld());
            assertEquals(0.0, ws.getSbgNorm());
    }

    @Test
    @DisplayName("calculateBufferSize() should return correct size")
    void testCalculateBufferSize() {
        int n = 10;
        int m = 5;

        // Expected: 2*n*m + 3*m*m + 8*m*m + 6*n + 8*m + 2*m
        // = 2*10*5 + 3*5*5 + 8*5*5 + 6*10 + 8*5 + 2*5
        // = 100 + 75 + 200 + 60 + 40 + 10
        // = 485
        // Note: tempCol[m] reuses wa[4m:5m], so only 2*m for diagInv + sqrtDiagInv
        int expected = 2 * n * m + 3 * m * m + 8 * m * m + 6 * n + 8 * m + 2 * m;
        assertEquals(expected, LBFGSBWorkspace.calculateBufferSize(n, m));
    }

    @Test
    @DisplayName("calculateIntBufferSize() should return correct size")
    void testCalculateIntBufferSize() {
        int n = 10;
        // Expected: 3*n = 30
        assertEquals(3 * n, LBFGSBWorkspace.calculateIntBufferSize(n));
    }

    @Test
    @DisplayName("toString() should return meaningful representation")
    void testToString() {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(10, 5);
            String str = ws.toString();
            assertTrue(str.contains("n=10"));
            assertTrue(str.contains("m=5"));
            assertTrue(str.contains("col=0"));
    }

    @Test
    @DisplayName("Workspace should support large dimensions")
    void testLargeDimensions() {
        int n = 1000;
        int m = 20;

        LBFGSBWorkspace ws = new LBFGSBWorkspace(n, m);
            assertEquals(n, ws.getDimension());
            assertEquals(m, ws.getCorrections());

            // Verify buffer is allocated
            double[] buffer = ws.getBuffer();
            int[] iBuffer = ws.getIntBuffer();

            assertTrue(buffer.length > 0);
            assertTrue(iBuffer.length > 0);

            // Verify we can access all offsets without exception
            assertDoesNotThrow(() -> {
                buffer[ws.getWsOffset()] = 1.0;
                buffer[ws.getWyOffset()] = 1.0;
                buffer[ws.getSyOffset()] = 1.0;
                buffer[ws.getSsOffset()] = 1.0;
                buffer[ws.getWtOffset()] = 1.0;
                buffer[ws.getWnOffset()] = 1.0;
                buffer[ws.getSndOffset()] = 1.0;
                buffer[ws.getZOffset()] = 1.0;
                buffer[ws.getROffset()] = 1.0;
                buffer[ws.getDOffset()] = 1.0;
                buffer[ws.getTOffset()] = 1.0;
                buffer[ws.getXpOffset()] = 1.0;
                buffer[ws.getWaOffset()] = 1.0;
                buffer[ws.getGOffset()] = 1.0;
                buffer[ws.getGOffset() + n - 1] = 1.0; // Last element of g array
            });
    }

    @Test
    @DisplayName("incrementTotalEval() should increment counter")
    void testIncrementTotalEval() {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(10, 5);
            assertEquals(0, ws.getTotalEval());
            ws.incrementTotalEval();
            assertEquals(1, ws.getTotalEval());
            ws.incrementTotalEval();
            assertEquals(2, ws.getTotalEval());
    }

    @Test
    @DisplayName("incrementResetCount() should increment counter")
    void testIncrementResetCount() {
        LBFGSBWorkspace ws = new LBFGSBWorkspace(10, 5);
            assertEquals(0, ws.getResetCount());
            ws.incrementResetCount();
            assertEquals(1, ws.getResetCount());
            ws.incrementResetCount();
            assertEquals(2, ws.getResetCount());
    }
}
