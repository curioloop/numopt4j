/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.cmaes;

import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.Optimization;
import net.jqwik.api.*;
import net.jqwik.api.constraints.*;

import java.util.Arrays;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Property-based tests for the rewritten CMAESCore algorithm.
 * Validates properties 1-10 from the design document.
 */
public class CMAESCoreRewriteProperties {

    // ── Property 2: 正权重归一化 ──────────────────────────────────────────

    /**
     * Validates: Requirements 1.1, 1.2
     *
     * For any valid (n, lambda), after initParams:
     *   |Σᵢ₌₀^{mu-1} weights[i] - 1.0| < 1e-15
     */
    @Property(tries = 200)
    @Label("Feature: cmaes-core-rewrite, Property 2: 正权重归一化")
    void prop_positiveWeightsNormalized(
            @ForAll @IntRange(min = 1, max = 20) int n,
            @ForAll @IntRange(min = 4, max = 50) int lambda) {

        CMAESWorkspace ws = new CMAESWorkspace(n, lambda);
        CMAESCore.initParams(ws, 1000);

        double sumPos = 0;
        for (int i = 0; i < ws.mu; i++) sumPos += ws.lVec[ws.WEIGHTS + i];
        assertThat(Math.abs(sumPos - 1.0))
            .as("Positive weights sum should be 1.0, got %f", sumPos)
            .isLessThan(1e-15);

        for (int i = 0; i < ws.mu; i++) {
            assertThat(ws.lVec[ws.WEIGHTS + i]).as("weights[%d] should be positive", i).isGreaterThan(0.0);
        }

        for (int i = 0; i < ws.mu - 1; i++) {
            assertThat(ws.lVec[ws.WEIGHTS + i]).as("weights[%d] > weights[%d]", i, i+1)
                .isGreaterThan(ws.lVec[ws.WEIGHTS + i + 1]);
        }
    }

    // ── Property 3: damps 无 maxIterations 项 ────────────────────────────

    /**
     * Validates: Requirements 2.5
     *
     * For same (n, lambda) but different maxIterations, damps must be identical.
     */
    @Property(tries = 100)
    @Label("Feature: cmaes-core-rewrite, Property 3: damps 无 maxIterations 项")
    void prop_dampsIndependentOfMaxIter(
            @ForAll @IntRange(min = 1, max = 20) int n,
            @ForAll @IntRange(min = 4, max = 50) int lambda) {

        CMAESWorkspace ws1 = new CMAESWorkspace(n, lambda);
        CMAESWorkspace ws2 = new CMAESWorkspace(n, lambda);
        CMAESCore.initParams(ws1, 100);
        CMAESCore.initParams(ws2, 100000);

        assertThat(ws1.damps)
            .as("damps should be independent of maxIterations")
            .isEqualTo(ws2.damps);
    }

    // ── Property 1: 零衰减条件 ────────────────────────────────────────────

    /**
     * Validates: Requirements 1.4, 1.7
     */
    @Property(tries = 200)
    @Label("Feature: cmaes-core-rewrite, Property 1: 零衰减条件")
    void prop_zeroDrift(
            @ForAll @IntRange(min = 5, max = 20) int n) {

        int lambda = 4 + (int) Math.floor(3.0 * Math.log(n));
        CMAESWorkspace ws = new CMAESWorkspace(n, lambda);
        CMAESCore.initParams(ws, 1000);

        double sumW = 0;
        for (int i = 0; i < lambda; i++) sumW += ws.lVec[ws.WEIGHTS + i];

        double zeroDrift = Math.abs(ws.ccov1 + ws.ccovmu * sumW);
        assertThat(zeroDrift)
            .as("|c1 + cmu * sumW| should be < 1e-10, got %e (n=%d, lambda=%d)", zeroDrift, n, lambda)
            .isLessThan(1e-10);
    }

    // ── Property 4: Active CMA 协方差正定性 ──────────────────────────────

    /**
     * Validates: Requirements 5.4, 5.6, 11.3
     */
    @Property(tries = 50)
    @Label("Feature: cmaes-core-rewrite, Property 4: Active CMA 协方差正定性")
    void prop_activeCMAPositiveDefinite(
            @ForAll @IntRange(min = 2, max = 8) int n) {

        int lambda = 4 + (int) Math.floor(3.0 * Math.log(n));
        CMAESWorkspace ws = new CMAESWorkspace(n, lambda);
        CMAESCore.initParams(ws, 1000);

        // Initialize state via arena offsets
        Arrays.fill(ws.nVec, ws.XMEAN, ws.XMEAN + n, 1.0);
        ws.sigma = 0.3;
        ws.sigma0 = 0.3;
        for (int i = 0; i < n; i++) {
            ws.mat[ws.C_OFF + i * n + i] = 1.0;
            ws.mat[ws.B_OFF + i * n + i] = 1.0;
            ws.nVec[ws.D_OFF + i] = 1.0;
        }

        Random rng = new Random(42);

        for (int iter = 1; iter <= 20; iter++) {
            ws.iterations = iter;

            CMAESCore.sampleOffspring(ws, false, rng, null, 0);

            for (int k = 0; k < lambda; k++) {
                double f = 0;
                for (int d = 0; d < n; d++) {
                    double xi = ws.arx[d * lambda + k];
                    f += xi * xi;
                }
                ws.lVec[ws.FITNESS + k] = f;
            }
            ws.evaluations += lambda;

            CMAESCore.sortIndices(ws.lVec, ws.FITNESS, ws.arindex, lambda);
            CMAESCore.updateMean(ws);
            boolean hsig = CMAESCore.updateEvolutionPaths(ws, false);
            double sumW4 = 0; for (int i = 0; i < lambda; i++) sumW4 += ws.lVec[ws.WEIGHTS + i];
            CMAESCore.updateCovariance(ws, hsig, true, sumW4);
            CMAESCore.updateSigma(ws);
            CMAESCore.eigenDecompose(ws);

            double maxD = ws.nVec[ws.D_OFF], minD = ws.nVec[ws.D_OFF];
            for (int i = 1; i < n; i++) {
                if (ws.nVec[ws.D_OFF + i] > maxD) maxD = ws.nVec[ws.D_OFF + i];
                if (ws.nVec[ws.D_OFF + i] < minD) minD = ws.nVec[ws.D_OFF + i];
            }

            assertThat(minD)
                .as("All eigenvalues should be positive (iter=%d)", iter)
                .isGreaterThan(0.0);

            if (minD > 0) {
                double condNum = (maxD * maxD) / (minD * minD);
                assertThat(condNum)
                    .as("Condition number should be <= 1e14 (iter=%d)", iter)
                    .isLessThanOrEqualTo(1e14);
            }
        }
    }

    // ── Property 5: 采样分布统计正确性 ───────────────────────────────────

    /**
     * Validates: Requirements 3.1
     */
    @Property(tries = 50)
    @Label("Feature: cmaes-core-rewrite, Property 5: 采样分布统计正确性")
    void prop_samplingDistribution(
            @ForAll @IntRange(min = 2, max = 5) int n) {

        int lambda = 10000;
        CMAESWorkspace ws = new CMAESWorkspace(n, lambda);
        CMAESCore.initParams(ws, 1000);

        for (int i = 0; i < n; i++) ws.nVec[ws.XMEAN + i] = i * 0.5;
        ws.sigma = 0.3;
        ws.sigma0 = 0.3;
        for (int i = 0; i < n; i++) {
            ws.mat[ws.B_OFF + i * n + i] = 1.0;
            ws.nVec[ws.D_OFF + i] = 1.0;
        }

        Random rng = new Random(42);
        CMAESCore.sampleOffspring(ws, false, rng, null, 0);

        double[] sampleMean = new double[n];
        for (int k = 0; k < lambda; k++) {
            for (int i = 0; i < n; i++) {
                sampleMean[i] += ws.arx[i * lambda + k];
            }
        }
        for (int i = 0; i < n; i++) sampleMean[i] /= lambda;

        double tolerance = 3.0 * ws.sigma / Math.sqrt(lambda);
        for (int i = 0; i < n; i++) {
            assertThat(Math.abs(sampleMean[i] - ws.nVec[ws.XMEAN + i]))
                .as("Sample mean[%d] should be close to xmean[%d]", i, i)
                .isLessThan(tolerance);
        }
    }

    // ── Property 6: 边界惩罚单调性 ───────────────────────────────────────

    /**
     * Validates: Requirements 9.1
     */
    @Property(tries = 200)
    @Label("Feature: cmaes-core-rewrite, Property 6: 边界惩罚单调性")
    void prop_penaltyMonotonicity(
            @ForAll @IntRange(min = 1, max = 5) int n) {

        com.curioloop.numopt4j.optim.Bound[] bounds = new com.curioloop.numopt4j.optim.Bound[n];
        for (int i = 0; i < n; i++) bounds[i] = com.curioloop.numopt4j.optim.Bound.between(-1.0, 1.0);

        int lambda = 3;
        double[] arx = new double[n * lambda];
        for (int i = 0; i < n; i++) arx[i * lambda + 0] = 0.0;
        for (int i = 0; i < n; i++) arx[i * lambda + 1] = 1.5;
        for (int i = 0; i < n; i++) arx[i * lambda + 2] = 3.0;

        double p0 = CMAESCore.computeRawPenalty(arx, 0, lambda, n, bounds);
        double p1 = CMAESCore.computeRawPenalty(arx, 1, lambda, n, bounds);
        double p2 = CMAESCore.computeRawPenalty(arx, 2, lambda, n, bounds);

        assertThat(p0).as("Feasible individual should have zero penalty").isEqualTo(0.0);
        assertThat(p1).as("Slightly infeasible should have positive penalty").isGreaterThan(0.0);
        assertThat(p2).as("More infeasible should have larger penalty").isGreaterThan(p1);
    }

    // ── Property 7: 停止条件优先级 ───────────────────────────────────────

    /**
     * Validates: Requirements 10.1, 10.10
     */
    @Property(tries = 100)
    @Label("Feature: cmaes-core-rewrite, Property 7: 停止条件优先级")
    void prop_stopConditionPriority(
            @ForAll @IntRange(min = 2, max = 10) int n) {

        int lambda = 4 + (int) Math.floor(3.0 * Math.log(n));
        Optimization r = Minimizer.cmaes()
            .objective(x -> { double s = 0; for (double v : x) s += v*v; return s; })
            .initialPoint(new double[n])
            .populationSize(lambda)
            .maxEvaluations(lambda)
            .maxIterations(lambda)
            .random(new Random(42))
            .solve();

        assertThat(r.getStatus())
            .as("Should return MAX_EVALUATIONS_REACHED with maxEval=lambda")
            .isEqualTo(Optimization.Status.MAX_EVALUATIONS_REACHED);
    }

    // ── Property 8: 特征值非负性 ─────────────────────────────────────────

    /**
     * Validates: Requirements 8.3
     */
    @Property(tries = 100)
    @Label("Feature: cmaes-core-rewrite, Property 8: 特征值非负性")
    void prop_eigenvaluesNonNegative(
            @ForAll @IntRange(min = 2, max = 10) int n) {

        int lambda = 4 + (int) Math.floor(3.0 * Math.log(n));
        CMAESWorkspace ws = new CMAESWorkspace(n, lambda);
        CMAESCore.initParams(ws, 1000);

        for (int i = 0; i < n; i++) {
            ws.mat[ws.C_OFF + i * n + i] = 1.0;
            ws.mat[ws.B_OFF + i * n + i] = 1.0;
            ws.nVec[ws.D_OFF + i] = 1.0;
        }
        ws.sigma = 0.3;
        ws.sigma0 = 0.3;
        Arrays.fill(ws.nVec, ws.XMEAN, ws.XMEAN + n, 1.0);

        Random rng = new Random(42);

        for (int iter = 1; iter <= 10; iter++) {
            ws.iterations = iter;
            CMAESCore.sampleOffspring(ws, false, rng, null, 0);
            for (int k = 0; k < lambda; k++) {
                double f = 0;
                for (int d = 0; d < n; d++) f += ws.arx[d * lambda + k] * ws.arx[d * lambda + k];
                ws.lVec[ws.FITNESS + k] = f;
            }
            ws.evaluations += lambda;
            CMAESCore.sortIndices(ws.lVec, ws.FITNESS, ws.arindex, lambda);
            CMAESCore.updateMean(ws);
            boolean hsig = CMAESCore.updateEvolutionPaths(ws, false);
            double sumW8 = 0; for (int i = 0; i < lambda; i++) sumW8 += ws.lVec[ws.WEIGHTS + i];
            CMAESCore.updateCovariance(ws, hsig, true, sumW8);
            CMAESCore.updateSigma(ws);
            CMAESCore.eigenDecompose(ws);
        }

        double minD = Math.sqrt(1e-20);
        for (int i = 0; i < n; i++) {
            assertThat(ws.nVec[ws.D_OFF + i])
                .as("D[%d] should be >= sqrt(1e-20)", i)
                .isGreaterThanOrEqualTo(minD);
        }
    }

    // ── Property 9: sigma 更新有界性 ─────────────────────────────────────

    /**
     * Validates: Requirements 7.1, 7.2
     */
    @Property(tries = 200)
    @Label("Feature: cmaes-core-rewrite, Property 9: sigma 更新有界性")
    void prop_sigmaUpdateBounded(
            @ForAll @DoubleRange(min = 0.01, max = 100.0) double normps,
            @ForAll @DoubleRange(min = 0.1, max = 10.0) double chiN,
            @ForAll @DoubleRange(min = 0.01, max = 1.0) double cs,
            @ForAll @DoubleRange(min = 0.1, max = 5.0) double damps) {

        CMAESWorkspace ws = new CMAESWorkspace(2, 4);
        ws.normps = normps;
        ws.chiN = chiN;
        ws.cs = cs;
        ws.damps = damps;
        ws.sigma = 1.0;

        CMAESCore.updateSigma(ws);

        double ratio = ws.sigma;
        assertThat(ratio)
            .as("sigma_new / sigma_old should be <= exp(1)")
            .isLessThanOrEqualTo(Math.exp(1.0) + 1e-12);
        assertThat(ratio).isGreaterThan(0.0);
        assertThat(Double.isFinite(ratio)).isTrue();
    }

    // ── Property 10: 工作空间无 NaN/Inf ──────────────────────────────────

    /**
     * Validates: Requirements 11.5
     */
    @Property(tries = 50)
    @Label("Feature: cmaes-core-rewrite, Property 10: 工作空间无 NaN/Inf")
    void prop_noNaNInWorkspace(
            @ForAll @IntRange(min = 2, max = 8) int n) {

        int lambda = 4 + (int) Math.floor(3.0 * Math.log(n));
        CMAESProblem p = Minimizer.cmaes()
            .objective(x -> { double s = 0; for (double v : x) s += v*v; return s; })
            .initialPoint(new double[n])
            .populationSize(lambda)
            .maxEvaluations(lambda * 50)
            .random(new Random(42));

        CMAESWorkspace ws = p.alloc();
        p.solve(ws);

        for (int i = 0; i < n; i++) {
            assertThat(Double.isFinite(ws.nVec[ws.XMEAN + i]))
                .as("xmean[%d] should be finite", i).isTrue();
        }
        for (int i = 0; i < n; i++) {
            assertThat(Double.isFinite(ws.nVec[ws.PS + i]))
                .as("ps[%d] should be finite", i).isTrue();
        }
        for (int i = 0; i < n; i++) {
            assertThat(Double.isFinite(ws.nVec[ws.PC + i]))
                .as("pc[%d] should be finite", i).isTrue();
        }
        for (int i = 0; i < n; i++) {
            assertThat(Double.isFinite(ws.mat[ws.C_OFF + i * n + i]))
                .as("C[%d,%d] should be finite", i, i).isTrue();
        }
        for (int i = 0; i < n; i++) {
            assertThat(Double.isFinite(ws.nVec[ws.D_OFF + i]))
                .as("D[%d] should be finite", i).isTrue();
        }
        assertThat(Double.isFinite(ws.sigma))
            .as("sigma should be finite").isTrue();
    }

    // ── Property 1 (Arena): Arena 字段尺寸 ───────────────────────────────

    /**
     * Validates: Requirements 1.1, 2.1, 3.1
     *
     * For any valid (n, lambda):
     *   nVec.length == 8*n, lVec.length == 3*lambda, mat.length == 2*n*n
     */
    @Property(tries = 200)
    @Label("Feature: cmaes-workspace-arena, Property 1: Arena 字段尺寸")
    void arena_prop1_arenaFieldSizes(
            @ForAll @IntRange(min = 1, max = 20) int n,
            @ForAll @IntRange(min = 4, max = 50) int lambda) {

        CMAESWorkspace ws = new CMAESWorkspace(n, lambda);

        assertThat(ws.nVec.length)
            .as("nVec.length should be 8*n=%d, got %d", 8*n, ws.nVec.length)
            .isEqualTo(8 * n);
        assertThat(ws.lVec.length)
            .as("lVec.length should be 3*lambda=%d, got %d", 3*lambda, ws.lVec.length)
            .isEqualTo(3 * lambda);
        assertThat(ws.mat.length)
            .as("mat.length should be 2*n*n=%d, got %d", 2*n*n, ws.mat.length)
            .isEqualTo(2 * n * n);
    }

    // ── Property 2 (Arena): 偏移量正确性 ─────────────────────────────────

    /**
     * Validates: Requirements 1.2-1.9, 2.2-2.4, 3.2-3.4
     *
     * For any valid (n, lambda): verify offset constant values are correct and
     * write-then-read equivalence holds.
     */
    @Property(tries = 200)
    @Label("Feature: cmaes-workspace-arena, Property 2: 偏移量正确性")
    void arena_prop2_offsetCorrectness(
            @ForAll @IntRange(min = 1, max = 20) int n,
            @ForAll @IntRange(min = 4, max = 50) int lambda) {

        CMAESWorkspace ws = new CMAESWorkspace(n, lambda);

        // Verify nVec offset constants
        assertThat(ws.XMEAN).as("XMEAN").isEqualTo(0);
        assertThat(ws.XOLD).as("XOLD").isEqualTo(n);
        assertThat(ws.PS).as("PS").isEqualTo(2 * n);
        assertThat(ws.PC).as("PC").isEqualTo(3 * n);
        assertThat(ws.D_OFF).as("D_OFF").isEqualTo(4 * n);
        assertThat(ws.DIAGC).as("DIAGC").isEqualTo(5 * n);
        assertThat(ws.DIAGD).as("DIAGD").isEqualTo(6 * n);
        assertThat(ws.EVALS).as("EVALS").isEqualTo(7 * n);

        // Verify lVec offset constants
        assertThat(ws.FITNESS).as("FITNESS").isEqualTo(0);
        assertThat(ws.WEIGHTS).as("WEIGHTS").isEqualTo(lambda);
        assertThat(ws.PENALTY).as("PENALTY").isEqualTo(2 * lambda);

        // Verify mat offset constants
        assertThat(ws.C_OFF).as("C_OFF").isEqualTo(0);
        assertThat(ws.B_OFF).as("B_OFF").isEqualTo(n * n);

        // Write-then-read equivalence: write to nVec[XMEAN+i], read back
        double sentinel = 3.14159;
        for (int i = 0; i < n; i++) {
            ws.nVec[ws.XMEAN + i] = sentinel + i;
        }
        for (int i = 0; i < n; i++) {
            assertThat(ws.nVec[ws.XMEAN + i])
                .as("nVec[XMEAN+%d] write-then-read should be equal", i)
                .isEqualTo(sentinel + i);
        }
    }

    // ── Property 5 (Arena): 独立字段尺寸 ─────────────────────────────────

    /**
     * Validates: Requirements 9.1-9.4
     *
     * For any valid (n, lambda):
     *   arz.length == n*lambda, arx.length == n*lambda, arindex.length == lambda
     */
    @Property(tries = 200)
    @Label("Feature: cmaes-workspace-arena, Property 5: 独立字段尺寸")
    void arena_prop5_independentFieldSizes(
            @ForAll @IntRange(min = 1, max = 20) int n,
            @ForAll @IntRange(min = 4, max = 50) int lambda) {

        CMAESWorkspace ws = new CMAESWorkspace(n, lambda);

        assertThat(ws.arz.length)
            .as("arz.length should be n*lambda=%d, got %d", n*lambda, ws.arz.length)
            .isEqualTo(n * lambda);
        assertThat(ws.arx.length)
            .as("arx.length should be n*lambda=%d, got %d", n*lambda, ws.arx.length)
            .isEqualTo(n * lambda);
        assertThat(ws.arindex.length)
            .as("arindex.length should be lambda=%d, got %d", lambda, ws.arindex.length)
            .isEqualTo(lambda);
    }

    // ── Property 4 (Arena): reset() 后状态正确性 ─────────────────────────

    /**
     * Validates: Requirements 8.1-8.4
     *
     * For any valid (n, lambda), after reset() all arena slots are restored to
     * their specified initial values.
     */
    @Property(tries = 100)
    @Label("Feature: cmaes-workspace-arena, Property 4: reset() 后状态正确性")
    void arena_prop4_resetStateCorrectness(
            @ForAll @IntRange(min = 1, max = 20) int n,
            @ForAll @IntRange(min = 4, max = 50) int lambda) {

        CMAESWorkspace ws = new CMAESWorkspace(n, lambda);

        // Verify initial state after construction (which calls reset())
        assertResetState(ws, n, lambda);

        // Modify some values, then reset and verify again
        Arrays.fill(ws.nVec, 99.0);
        Arrays.fill(ws.lVec, 99.0);
        Arrays.fill(ws.mat, 99.0);
        ws.reset();

        assertResetState(ws, n, lambda);
    }

    // ── Property 3 (Arena): 端到端数值等价性 ─────────────────────────────

    /**
     * Validates: Requirements 3.1
     *
     * Verifies that the arena-refactored implementation produces numerically correct
     * results on the sphere function for both diagonalOnly=true and diagonalOnly=false.
     */
    @Property(tries = 50)
    @Label("Feature: cmaes-workspace-arena, Property 3: 端到端数值等价性")
    void arena_prop3_endToEndNumericalEquivalence(
            @ForAll @IntRange(min = 2, max = 8) int n) {

        int lambda = 4 + (int) Math.floor(3.0 * Math.log(n));
        double[] x0 = new double[n];
        Arrays.fill(x0, 2.0);

        // Run with diagonalOnly=false
        Optimization rFull = Minimizer.cmaes()
            .objective(x -> { double s = 0; for (double v : x) s += v * v; return s; })
            .initialPoint(x0.clone())
            .populationSize(lambda)
            .sigma(0.5)
            .diagonalOnly(false)
            .maxIterations(500)
            .maxEvaluations(500 * lambda)
            .random(new Random(42 + n))
            .solve();

        assertThat(rFull.getCost())
            .as("diagonalOnly=false: sphere should converge to < 1e-6 (n=%d, lambda=%d, fitness=%e)",
                n, lambda, rFull.getCost())
            .isLessThan(1e-6);

        // Run with diagonalOnly=true
        Optimization rDiag = Minimizer.cmaes()
            .objective(x -> { double s = 0; for (double v : x) s += v * v; return s; })
            .initialPoint(x0.clone())
            .populationSize(lambda)
            .sigma(0.5)
            .diagonalOnly(true)
            .maxIterations(500)
            .maxEvaluations(500 * lambda)
            .random(new Random(42 + n))
            .solve();

        assertThat(rDiag.getCost())
            .as("diagonalOnly=true: sphere should converge to < 1e-6 (n=%d, lambda=%d, fitness=%e)",
                n, lambda, rDiag.getCost())
            .isLessThan(1e-6);
    }

    private void assertResetState(CMAESWorkspace ws, int n, int lambda) {
        // nVec slots that should be zero
        for (int i = 0; i < n; i++) {
            assertThat(ws.nVec[ws.XMEAN + i]).as("nVec[XMEAN+%d] should be 0", i).isEqualTo(0.0);
            assertThat(ws.nVec[ws.XOLD  + i]).as("nVec[XOLD+%d] should be 0", i).isEqualTo(0.0);
            assertThat(ws.nVec[ws.PS    + i]).as("nVec[PS+%d] should be 0", i).isEqualTo(0.0);
            assertThat(ws.nVec[ws.PC    + i]).as("nVec[PC+%d] should be 0", i).isEqualTo(0.0);
            assertThat(ws.nVec[ws.EVALS + i]).as("nVec[EVALS+%d] should be 0", i).isEqualTo(0.0);
        }

        // nVec slots that should be ones
        for (int i = 0; i < n; i++) {
            assertThat(ws.nVec[ws.D_OFF + i]).as("nVec[D_OFF+%d] should be 1", i).isEqualTo(1.0);
            assertThat(ws.nVec[ws.DIAGC + i]).as("nVec[DIAGC+%d] should be 1", i).isEqualTo(1.0);
            assertThat(ws.nVec[ws.DIAGD + i]).as("nVec[DIAGD+%d] should be 1", i).isEqualTo(1.0);
        }

        // lVec slots that should be zero
        for (int i = 0; i < lambda; i++) {
            assertThat(ws.lVec[ws.FITNESS  + i]).as("lVec[FITNESS+%d] should be 0", i).isEqualTo(0.0);
            assertThat(ws.lVec[ws.WEIGHTS  + i]).as("lVec[WEIGHTS+%d] should be 0", i).isEqualTo(0.0);
            assertThat(ws.lVec[ws.PENALTY  + i]).as("lVec[PENALTY+%d] should be 0", i).isEqualTo(0.0);
        }

        // mat: C = identity
        for (int i = 0; i < n; i++) {
            assertThat(ws.mat[ws.C_OFF + i*n + i]).as("mat[C_OFF+%d*n+%d] should be 1", i, i).isEqualTo(1.0);
        }
        // mat: B = identity
        for (int i = 0; i < n; i++) {
            assertThat(ws.mat[ws.B_OFF + i*n + i]).as("mat[B_OFF+%d*n+%d] should be 1", i, i).isEqualTo(1.0);
        }
        // mat: off-diagonal of C and B should be zero
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    assertThat(ws.mat[ws.C_OFF + i*n + j]).as("mat[C_OFF+%d*n+%d] off-diag should be 0", i, j).isEqualTo(0.0);
                    assertThat(ws.mat[ws.B_OFF + i*n + j]).as("mat[B_OFF+%d*n+%d] off-diag should be 0", i, j).isEqualTo(0.0);
                }
            }
        }
    }
}
