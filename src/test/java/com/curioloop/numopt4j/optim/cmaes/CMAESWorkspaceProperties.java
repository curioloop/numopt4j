/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.cmaes;

import net.jqwik.api.*;
import net.jqwik.api.constraints.IntRange;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Property-based tests for CMAESWorkspace.
 */
public class CMAESWorkspaceProperties {

    // Feature: cmaes-optimizer, Property 16: 工作空间数组尺寸正确性
    @Property(tries = 200)
    @Label("Feature: cmaes-optimizer, Property 16: 工作空间数组尺寸正确性")
    void workspaceArraySizesCorrect(
            @ForAll @IntRange(min = 1, max = 30) int n,
            @ForAll @IntRange(min = 4, max = 50) int lambda) {

        CMAESWorkspace ws = new CMAESWorkspace();
        ws.ensure(n, lambda, false);

        // Arena sizes
        assertThat(ws.nVec.length).isEqualTo(8 * n);
        assertThat(ws.lVec.length).isEqualTo(3 * lambda);
        assertThat(ws.mat.length).isEqualTo(2 * n * n);

        // Independent arrays
        assertThat(ws.arz.length).isEqualTo(n * lambda);
        assertThat(ws.arx.length).isEqualTo(n * lambda);
        assertThat(ws.arindex.length).isEqualTo(lambda);

        // History
        assertThat(ws.histSize).isEqualTo(10 + (int)(3.0 * 10 * n / lambda));
        assertThat(ws.fitnessHistory.length).isEqualTo(ws.histSize);

        // Dimensions stored correctly
        assertThat(ws.n).isEqualTo(n);
        assertThat(ws.lambda).isEqualTo(lambda);

        // Offset constants
        assertThat(ws.XMEAN).isEqualTo(0);
        assertThat(ws.XOLD).isEqualTo(n);
        assertThat(ws.PS).isEqualTo(2 * n);
        assertThat(ws.PC).isEqualTo(3 * n);
        assertThat(ws.D_OFF).isEqualTo(4 * n);
        assertThat(ws.DIAGC).isEqualTo(5 * n);
        assertThat(ws.DIAGD).isEqualTo(6 * n);
        assertThat(ws.EVALS).isEqualTo(7 * n);

        assertThat(ws.FITNESS).isEqualTo(0);
        assertThat(ws.WEIGHTS).isEqualTo(lambda);
        assertThat(ws.PENALTY).isEqualTo(2 * lambda);

        assertThat(ws.C_OFF).isEqualTo(0);
        assertThat(ws.B_OFF).isEqualTo(n * n);
    }

    // Feature: cmaes-optimizer, Property 17: reset() 等价于重新分配
    @Property(tries = 100)
    @Label("Feature: cmaes-optimizer, Property 17: reset() 等价于重新分配")
    void resetEquivalentToFreshAllocation(
            @ForAll @IntRange(min = 2, max = 15) int n,
            @ForAll @IntRange(min = 4, max = 20) int lambda) {

        CMAESWorkspace ws = new CMAESWorkspace();
        ws.ensure(n, lambda, false);

        // Dirty the workspace
        for (int i = 0; i < n; i++) ws.nVec[ws.XMEAN + i] = i + 1.0;
        for (int i = 0; i < n * n; i++) ws.mat[ws.C_OFF + i] = i * 0.1;
        ws.sigma = 5.0;
        ws.iterations = 100;
        ws.evaluations = 500;

        // Reset
        ws.reset();

        // Check key fields match fresh state
        assertThat(ws.sigma).isEqualTo(0.0);
        assertThat(ws.iterations).isEqualTo(0);
        assertThat(ws.evaluations).isEqualTo(0);
        assertThat(ws.normps).isEqualTo(0.0);

        // xmean should be zeroed
        for (int i = 0; i < n; i++) {
            assertThat(ws.nVec[ws.XMEAN + i]).isEqualTo(0.0);
        }

        // C should be identity
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(ws.mat[ws.C_OFF + i * n + j]).isEqualTo(expected);
            }
        }

        // B should be identity
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertThat(ws.mat[ws.B_OFF + i * n + j]).isEqualTo(expected);
            }
        }

        // D should be ones
        for (int i = 0; i < n; i++) {
            assertThat(ws.nVec[ws.D_OFF + i]).isEqualTo(1.0);
        }

        // fitnessHistory should be MAX_VALUE
        for (int i = 0; i < ws.histSize; i++) {
            assertThat(ws.fitnessHistory[i]).isEqualTo(Double.MAX_VALUE);
        }
    }
}
