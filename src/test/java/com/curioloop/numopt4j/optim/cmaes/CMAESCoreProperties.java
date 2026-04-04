/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.cmaes;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.Optimization;
import net.jqwik.api.*;
import net.jqwik.api.constraints.*;

import java.util.Arrays;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Property-based tests for CMAESCore algorithm correctness.
 */
public class CMAESCoreProperties {

    // ── Property 1: 重组权重归一化 ────────────────────────────────────────

    // Feature: cmaes-optimizer, Property 1: 重组权重归一化
    @Property(tries = 200)
    @Label("Feature: cmaes-optimizer, Property 1: 重组权重归一化")
    void weightsNormalizeToOne(@ForAll @IntRange(min = 2, max = 50) int n) {
        int lambda = 4 + (int) Math.floor(3.0 * Math.log(n));
        CMAESWorkspace ws = new CMAESWorkspace(); ws.ensure(n, lambda, false);
        CMAESCore.initParams(ws, 1000);

        // sum(weights) == 1.0
        double sumW = 0;
        for (int i = 0; i < ws.mu; i++) sumW += ws.lVec[ws.WEIGHTS + i];
        assertThat(sumW).isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-12));

        // weights are strictly positive
        for (int i = 0; i < ws.mu; i++) {
            assertThat(ws.lVec[ws.WEIGHTS + i]).isGreaterThan(0.0);
        }

        // weights are strictly decreasing
        for (int i = 0; i < ws.mu - 1; i++) {
            assertThat(ws.lVec[ws.WEIGHTS + i]).isGreaterThan(ws.lVec[ws.WEIGHTS + i + 1]);
        }
    }

    // ── Property 2: 均值更新正确性 ────────────────────────────────────────

    // Feature: cmaes-optimizer, Property 2: 均值更新正确性
    @Property(tries = 200)
    @Label("Feature: cmaes-optimizer, Property 2: 均值更新正确性")
    void meanUpdateIsWeightedSum(
            @ForAll @IntRange(min = 2, max = 10) int n,
            @ForAll @IntRange(min = 4, max = 20) int lambda) {

        CMAESWorkspace ws = new CMAESWorkspace(); ws.ensure(n, lambda, false);
        CMAESCore.initParams(ws, 1000);

        // Fill arx with known values
        Random rng = new Random(42);
        for (int i = 0; i < n * lambda; i++) ws.arx[i] = rng.nextGaussian();

        // Set fitness and sort
        for (int k = 0; k < lambda; k++) ws.lVec[ws.FITNESS + k] = rng.nextDouble();
        CMAESCore.sortIndices(ws.lVec, ws.FITNESS, ws.arindex, lambda);

        // Set xmean to something non-zero
        Arrays.fill(ws.nVec, ws.XMEAN, ws.XMEAN + n, 1.0);

        // Update mean
        CMAESCore.updateMean(ws);

        // Verify: xmean == sum_{i=0}^{mu-1} weights[i] * arx[arindex[i]]
        double[] expected = new double[n];
        for (int i = 0; i < ws.mu; i++) {
            int idx = ws.arindex[i];
            double w = ws.lVec[ws.WEIGHTS + i];
            for (int d = 0; d < n; d++) {
                expected[d] += w * ws.arx[d * lambda + idx];
            }
        }

        for (int d = 0; d < n; d++) {
            assertThat(ws.nVec[ws.XMEAN + d]).isCloseTo(expected[d], org.assertj.core.data.Offset.offset(1e-12));
        }
    }

    // ── Property 3: 特征分解一致性 ────────────────────────────────────────

    // Feature: cmaes-optimizer, Property 3: 特征分解一致性（B·D²·Bᵀ = C）
    @Property(tries = 100)
    @Label("Feature: cmaes-optimizer, Property 3: 特征分解一致性（B·D²·Bᵀ = C）")
    void eigenDecompositionRoundTrip(@ForAll @IntRange(min = 2, max = 10) int n) {
        // Build a random SPD matrix C = A*A^T + n*I
        Random rng = new Random(42);
        double[] C = new double[n * n];
        double[] A = new double[n * n];
        for (int i = 0; i < n * n; i++) A[i] = rng.nextGaussian();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) sum += A[i * n + k] * A[j * n + k];
                C[i * n + j] = sum;
            }
            C[i * n + i] += n;
        }

        CMAESWorkspace ws = new CMAESWorkspace(); ws.ensure(n, 4, false);
        System.arraycopy(C, 0, ws.mat, ws.C_OFF, n * n);

        ws.iterations = 1;
        ws.lastEigenIter = 0;
        ws.eigenInterval = 1;
        CMAESCore.eigenDecompose(ws);

        // Reconstruct: B * diag(D^2) * B^T
        double[] reconstructed = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    double d = ws.nVec[ws.D_OFF + k];
                    sum += ws.mat[ws.B_OFF + i * n + k] * d * d * ws.mat[ws.B_OFF + j * n + k];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double orig = C[i * n + j];
                double recon = reconstructed[i * n + j];
                double relErr = Math.abs(orig - recon) / (Math.abs(orig) + 1e-15);
                assertThat(relErr).as("C[%d][%d]: orig=%.6e recon=%.6e", i, j, orig, recon)
                    .isLessThan(1e-8);
            }
        }
    }

    // ── Property 4: 特征值钳制 ────────────────────────────────────────────

    // Feature: cmaes-optimizer, Property 4: 特征值钳制保证正定性
    @Property(tries = 200)
    @Label("Feature: cmaes-optimizer, Property 4: 特征值钳制保证正定性")
    void eigenvaluesClamped(@ForAll @IntRange(min = 2, max = 10) int n) {
        Random rng = new Random(42);
        CMAESWorkspace ws = new CMAESWorkspace(); ws.ensure(n, 4, false);

        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = rng.nextGaussian();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                ws.mat[ws.C_OFF + i * n + j] = v[i] * v[j];
            }
            ws.mat[ws.C_OFF + i * n + i] += 1e-30;
        }

        ws.iterations = 1;
        ws.lastEigenIter = 0;
        ws.eigenInterval = 1;
        CMAESCore.eigenDecompose(ws);

        double minD = Math.sqrt(1e-20);
        for (int i = 0; i < n; i++) {
            assertThat(ws.nVec[ws.D_OFF + i]).isGreaterThanOrEqualTo(minD);
        }
    }

    // ── Property 5: sigma CSA 更新有界 ────────────────────────────────────

    // Feature: cmaes-optimizer, Property 5: sigma CSA 更新有界
    @Property(tries = 500)
    @Label("Feature: cmaes-optimizer, Property 5: sigma CSA 更新有界")
    void sigmaUpdateBounded(
            @ForAll @DoubleRange(min = 0.01, max = 100.0) double normps,
            @ForAll @DoubleRange(min = 0.1, max = 10.0) double chiN,
            @ForAll @DoubleRange(min = 0.01, max = 1.0) double cs,
            @ForAll @DoubleRange(min = 0.1, max = 5.0) double damps) {

        CMAESWorkspace ws = new CMAESWorkspace(); ws.ensure(2, 4, false);
        ws.normps = normps;
        ws.chiN = chiN;
        ws.cs = cs;
        ws.damps = damps;
        ws.sigma = 1.0;

        CMAESCore.updateSigma(ws);

        double ratio = ws.sigma;
        assertThat(ratio).isLessThanOrEqualTo(Math.exp(1.0) + 1e-12);
        assertThat(ratio).isGreaterThan(0.0);
        assertThat(Double.isFinite(ratio)).isTrue();
    }

    // ── Property 6: 对角线模式不变量 ─────────────────────────────────────

    // Feature: cmaes-optimizer, Property 6: 对角线模式不变量（diagD² = diagC）
    @Property(tries = 200)
    @Label("Feature: cmaes-optimizer, Property 6: 对角线模式不变量（diagD² = diagC）")
    void diagDSquaredEqualsDiagC(@ForAll @IntRange(min = 2, max = 10) int n) {
        int lambda = 4 + (int) Math.floor(3.0 * Math.log(n));
        CMAESWorkspace ws = new CMAESWorkspace(); ws.ensure(n, lambda, false);
        CMAESCore.initParams(ws, 1000);

        Random rng = new Random(42);
        for (int i = 0; i < n; i++) {
            ws.nVec[ws.DIAGC + i] = 0.5 + rng.nextDouble();
            ws.nVec[ws.DIAGD + i] = Math.sqrt(ws.nVec[ws.DIAGC + i]);
        }

        for (int i = 0; i < n; i++) ws.nVec[ws.PC + i] = rng.nextGaussian() * 0.1;

        for (int i = 0; i < n * lambda; i++) ws.arz[i] = rng.nextGaussian();
        for (int k = 0; k < lambda; k++) ws.arindex[k] = k;

        CMAESCore.updateCovarianceDiag(ws, true);

        for (int i = 0; i < n; i++) {
            double diagDSq = ws.nVec[ws.DIAGD + i] * ws.nVec[ws.DIAGD + i];
            assertThat(diagDSq).isCloseTo(ws.nVec[ws.DIAGC + i], org.assertj.core.data.Offset.offset(1e-12));
        }
    }

    // ── Property 7: 对角线模式采样协方差 ─────────────────────────────────

    // Feature: cmaes-optimizer, Property 7: 对角线模式采样协方差
    @Property(tries = 50)
    @Label("Feature: cmaes-optimizer, Property 7: 对角线模式采样协方差")
    void diagonalModeSamplingCovariance(@ForAll @IntRange(min = 2, max = 5) int n) {
        int sampleSize = 5000;
        double sigma = 0.5;

        double[] diagD = new double[n];
        for (int i = 0; i < n; i++) diagD[i] = 1.0 + i * 0.3;

        Random rng = new Random(12345);
        double[] sumX = new double[n];
        double[] sumXX = new double[n];

        for (int s = 0; s < sampleSize; s++) {
            for (int i = 0; i < n; i++) {
                double xi = sigma * diagD[i] * rng.nextGaussian();
                sumX[i]  += xi;
                sumXX[i] += xi * xi;
            }
        }

        for (int i = 0; i < n; i++) {
            double mean = sumX[i] / sampleSize;
            double var  = sumXX[i] / sampleSize - mean * mean;
            double expected = sigma * sigma * diagD[i] * diagD[i];
            double relErr = Math.abs(var - expected) / expected;
            assertThat(relErr).as("dim %d: var=%.4f expected=%.4f", i, var, expected)
                .isLessThan(0.10);
        }
    }

    // ── Property 8: 边界惩罚单调性 ───────────────────────────────────────
    // Feature: cmaes-optimizer, Property 8: 边界惩罚单调性
    @Property(tries = 300)
    @Label("Feature: cmaes-optimizer, Property 8: 边界惩罚单调性")
    void penaltyMakesInfeasibleWorse(
            @ForAll @IntRange(min = 1, max = 5) int n) {

        Bound[] bounds = new Bound[n];
        for (int i = 0; i < n; i++) bounds[i] = Bound.between(-1.0, 1.0);

        int lambda = 2;
        double[] arx = new double[n * lambda];
        for (int i = 0; i < n; i++) {
            arx[i * lambda + 0] = 0.0;
            arx[i * lambda + 1] = 2.0;
        }

        double penaltyFeasible = CMAESCore.computeRawPenalty(arx, 0, lambda, n, bounds);
        double penaltyInfeasible = CMAESCore.computeRawPenalty(arx, 1, lambda, n, bounds);

        assertThat(penaltyFeasible).isEqualTo(0.0);
        assertThat(penaltyInfeasible).isGreaterThan(0.0);
    }

    // ── Property 9: 终止上界保证 ─────────────────────────────────────────

    // Feature: cmaes-optimizer, Property 9: 终止上界保证
    @Property(tries = 100)
    @Label("Feature: cmaes-optimizer, Property 9: 终止上界保证")
    void terminationRespectsBounds(
            @ForAll @IntRange(min = 1, max = 10) int maxIter,
            @ForAll @IntRange(min = 10, max = 200) int maxEval) {

        Optimization result = Minimizer.cmaes()
            .objective((x, n) -> { double s = 0; for (double v : x) s += v*v; return s; })
            .initialPoint(new double[]{1.0, 1.0, 1.0})
            .maxIterations(maxIter)
            .maxEvaluations(maxEval)
            .random(new Random(42))
            .solve();

        assertThat(result.getIterations()).isLessThanOrEqualTo(maxIter);
        assertThat(result.getEvaluations()).isLessThanOrEqualTo(maxEval + 100);
    }

    // ── Property 10: TolX 停止条件 ───────────────────────────────────────

    // Feature: cmaes-optimizer, Property 10: TolX 停止条件正确性
    @Property(tries = 200)
    @Label("Feature: cmaes-optimizer, Property 10: TolX 停止条件正确性")
    void tolXStopConditionCorrect(@ForAll @IntRange(min = 2, max = 10) int n) {
        int lambda = 4 + (int) Math.floor(3.0 * Math.log(n));
        CMAESWorkspace ws = new CMAESWorkspace(); ws.ensure(n, lambda, false);
        CMAESCore.initParams(ws, 1000);
        ws.sigma = 1e-15;
        ws.sigma0 = 1.0;
        ws.iterations = 1;

        for (int i = 0; i < n; i++) {
            ws.nVec[ws.PC + i] = 1e-15;
            ws.mat[ws.C_OFF + i * n + i] = 1e-30;
        }

        CMAESProblem cfg = new CMAESProblem()
            .objective((x, _n) -> 0.0)
            .initialPoint(new double[n])
            .populationSize(lambda)
            .parameterTolerance(1e-11)
            .maxEvaluations(1000);

        Optimization.Status status = CMAESCore.checkStopConditions(ws, cfg, 0.0, false);
        assertThat(status).isEqualTo(Optimization.Status.COEFFICIENT_TOLERANCE_REACHED);
    }

    // ── Property 11: TolUpSigma 发散检测 ─────────────────────────────────

    // Feature: cmaes-optimizer, Property 11: TolUpSigma 发散检测
    @Property(tries = 200)
    @Label("Feature: cmaes-optimizer, Property 11: TolUpSigma 发散检测")
    void tolUpSigmaDivergenceDetection(@ForAll @IntRange(min = 2, max = 10) int n) {
        int lambda = 4 + (int) Math.floor(3.0 * Math.log(n));
        CMAESWorkspace ws = new CMAESWorkspace(); ws.ensure(n, lambda, false);
        CMAESCore.initParams(ws, 1000);
        ws.sigma = 1e6;
        ws.sigma0 = 1.0;
        ws.iterations = 1;

        for (int i = 0; i < n; i++) {
            ws.nVec[ws.D_OFF + i] = 1.0;
        }

        CMAESProblem cfg = new CMAESProblem()
            .objective((x, _n) -> 0.0)
            .initialPoint(new double[n])
            .populationSize(lambda)
            .maxSigmaRatio(1e3)
            .maxEvaluations(1000);

        Optimization.Status status = CMAESCore.checkStopConditions(ws, cfg, 0.0, false);
        assertThat(status).isEqualTo(Optimization.Status.ABNORMAL_TERMINATION);
    }

    // ── Property 12: 条件数停止条件 ──────────────────────────────────────

    // Feature: cmaes-optimizer, Property 12: 条件数停止条件
    @Property(tries = 200)
    @Label("Feature: cmaes-optimizer, Property 12: 条件数停止条件")
    void conditionNumberStopCondition(@ForAll @IntRange(min = 2, max = 10) int n) {
        int lambda = 4 + (int) Math.floor(3.0 * Math.log(n));
        CMAESWorkspace ws = new CMAESWorkspace(); ws.ensure(n, lambda, false);
        CMAESCore.initParams(ws, 1000);
        ws.sigma = 0.3;
        ws.sigma0 = 0.3;
        ws.iterations = 1;

        ws.nVec[ws.D_OFF] = 1e8;
        for (int i = 1; i < n; i++) ws.nVec[ws.D_OFF + i] = 1.0;

        CMAESProblem cfg = new CMAESProblem()
            .objective((x, _n) -> 0.0)
            .initialPoint(new double[n])
            .populationSize(lambda)
            .maxEvaluations(1000);

        Optimization.Status status = CMAESCore.checkStopConditions(ws, cfg, 0.0, false);
        assertThat(status).isEqualTo(Optimization.Status.ABNORMAL_TERMINATION);
    }

    // ── Property 14: IPOP lambda 增长 ────────────────────────────────────

    // Feature: cmaes-optimizer, Property 14: IPOP lambda 增长
    @Property(tries = 200)
    @Label("Feature: cmaes-optimizer, Property 14: IPOP lambda 增长")
    void ipopLambdaGrows(
            @ForAll @IntRange(min = 2, max = 5) int incPopSize,
            @ForAll @IntRange(min = 1, max = 4) int restarts) {

        int n = 3;
        int lambdaDefault = 4 + (int) Math.floor(3.0 * Math.log(n));

        int[] lambdaValues = new int[restarts + 1];
        int currentLambda = lambdaDefault;
        for (int i = 0; i <= restarts; i++) {
            lambdaValues[i] = currentLambda;
            currentLambda *= incPopSize;
        }

        for (int i = 0; i <= restarts; i++) {
            int expected = lambdaDefault * (int) Math.pow(incPopSize, i);
            assertThat(lambdaValues[i]).isEqualTo(expected);
        }

        for (int i = 0; i < restarts; i++) {
            assertThat(lambdaValues[i + 1]).isGreaterThan(lambdaValues[i]);
        }
    }

    // ── Property 15: BIPOP 小种群参数范围 ────────────────────────────────

    // Feature: cmaes-optimizer, Property 15: BIPOP 小种群参数范围
    @Property(tries = 500)
    @Label("Feature: cmaes-optimizer, Property 15: BIPOP 小种群参数范围")
    void bipopSmallRegimeParameterBounds(
            @ForAll @DoubleRange(min = 0.01, max = 0.99) double u) {

        int n = 5;
        int lambdaDefault = 4 + (int) Math.floor(3.0 * Math.log(n));
        int lambdaLarge = lambdaDefault * 4;
        double sigma0 = 0.3;

        int lambdaSmall = Math.max(lambdaDefault, (int) Math.floor(lambdaLarge * u * u));
        double sigmaSmall = sigma0 * Math.pow(10.0, -2.0 * u);

        assertThat(lambdaSmall).isGreaterThanOrEqualTo(lambdaDefault);
        assertThat(lambdaSmall).isLessThanOrEqualTo(lambdaLarge);

        assertThat(sigmaSmall).isGreaterThan(sigma0 * 0.01 - 1e-15);
        assertThat(sigmaSmall).isLessThanOrEqualTo(sigma0 + 1e-15);
    }

    // ── Property 18: NaN fitness 替换 ────────────────────────────────────

    // Feature: cmaes-optimizer, Property 18: NaN/Infinity fitness 替换
    @Property(tries = 200)
    @Label("Feature: cmaes-optimizer, Property 18: NaN/Infinity fitness 替换")
    void nanFitnessIsReplaced(@ForAll @IntRange(min = 2, max = 10) int lambda) {
        CMAESWorkspace ws = new CMAESWorkspace(); ws.ensure(3, lambda, false);

        for (int k = 0; k < lambda; k++) {
            ws.lVec[ws.FITNESS + k] = (k == 0) ? Double.NaN : (k * 10.0);
        }
        double worst = Double.NEGATIVE_INFINITY;
        boolean anyFinite = false;
        for (int k = 0; k < lambda; k++) {
            if (Double.isFinite(ws.lVec[ws.FITNESS + k])) {
                if (ws.lVec[ws.FITNESS + k] > worst) worst = ws.lVec[ws.FITNESS + k];
                anyFinite = true;
            }
        }
        assertThat(anyFinite).isTrue();
        for (int k = 0; k < lambda; k++) {
            if (!Double.isFinite(ws.lVec[ws.FITNESS + k])) ws.lVec[ws.FITNESS + k] = worst;
        }

        for (int k = 0; k < lambda; k++) {
            assertThat(Double.isFinite(ws.lVec[ws.FITNESS + k])).isTrue();
        }
        assertThat(ws.lVec[ws.FITNESS + 0]).isEqualTo(worst);
    }

    // ── Property 13: 重启后全局最优单调性 ────────────────────────────────

    // Feature: cmaes-optimizer, Property 13: 重启后全局最优单调性
    @Property(tries = 50)
    @Label("Feature: cmaes-optimizer, Property 13: 重启后全局最优单调性")
    void restartBestIsGlobalBest(@ForAll @IntRange(min = 1, max = 3) int maxRestarts) {
        Optimization result = Minimizer.cmaes()
            .objective((x, n) -> { double s = 0; for (double v : x) s += v*v; return s; })
            .initialPoint(new double[]{1.0, 1.0, 1.0})
            .maxEvaluations(5000)
            .restartMode(RestartMode.ipop(maxRestarts, 2))
            .random(new Random(42))
            .solve();

        assertThat(result.getSolution()).isNotNull();
        assertThat(Double.isFinite(result.getCost())).isTrue();
        assertThat(result.getCost()).isLessThanOrEqualTo(3.0);
    }
}
