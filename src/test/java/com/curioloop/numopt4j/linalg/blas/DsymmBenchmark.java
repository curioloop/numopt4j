/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * JMH micro-benchmark: Dsymm original (naive) vs optimized (symmetric dual-update + FMA + 4×4 unroll).
 *
 * Run with:
 *   mvn test-compile exec:java -Dexec.mainClass=org.openjdk.jmh.Main \
 *     -Dexec.args="DsymmBenchmark -f 0 -wi 5 -i 10"
 */
package com.curioloop.numopt4j.linalg.blas;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Benchmarks four variants of DSYMM (Left/Right × Lower/Upper) at three matrix sizes.
 *
 * <p>Optimizations applied in the new version:
 * <ul>
 *   <li>Symmetric dual-update: A[i,k] contributes to both C[i,:] and C[k,:] in one pass,
 *       halving the number of A reads for the off-diagonal part.</li>
 *   <li>alpha applied once at write-back (not inside the k-loop).</li>
 *   <li>FMA via {@link FMA#op} for fused multiply-add.</li>
 *   <li>4-wide unroll on the n (Left) or m (Right) dimension.</li>
 * </ul>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(0)
@State(Scope.Benchmark)
public class DsymmBenchmark {

    @Param({"32", "64", "128"})
    int n;

    double[] A, B, C;

    @Setup(Level.Invocation)
    public void setup() {
        Random rng = new Random(42);
        A = new double[n * n];
        B = new double[n * n];
        C = new double[n * n];
        for (int i = 0; i < n * n; i++) {
            A[i] = rng.nextDouble() - 0.5;
            B[i] = rng.nextDouble() - 0.5;
            C[i] = rng.nextDouble() - 0.5;
        }
        // Make A symmetric positive definite (lower triangle stored)
        for (int i = 0; i < n; i++) {
            A[i * n + i] += n; // dominant diagonal
            for (int j = i + 1; j < n; j++) {
                A[j * n + i] = A[i * n + j]; // symmetrize
            }
        }
    }

    // ── Left Lower ────────────────────────────────────────────────────────────

    @Benchmark
    public double[] leftLower_new() {
        double[] c = C.clone();
        Dsymm.dsymm(BLAS.Side.Left, BLAS.Uplo.Lower, n, n, 1.0, A, 0, n, B, 0, n, 1.0, c, 0, n);
        return c;
    }

    @Benchmark
    public double[] leftLower_orig() {
        double[] c = C.clone();
        dsymmLeftLowerOrig(n, n, 1.0, A, 0, n, B, 0, n, c, 0, n);
        return c;
    }

    // ── Left Upper ────────────────────────────────────────────────────────────

    @Benchmark
    public double[] leftUpper_new() {
        double[] c = C.clone();
        Dsymm.dsymm(BLAS.Side.Left, BLAS.Uplo.Upper, n, n, 1.0, A, 0, n, B, 0, n, 1.0, c, 0, n);
        return c;
    }

    @Benchmark
    public double[] leftUpper_orig() {
        double[] c = C.clone();
        dsymmLeftUpperOrig(n, n, 1.0, A, 0, n, B, 0, n, c, 0, n);
        return c;
    }

    // ── Right Lower ───────────────────────────────────────────────────────────

    @Benchmark
    public double[] rightLower_new() {
        double[] c = C.clone();
        Dsymm.dsymm(BLAS.Side.Right, BLAS.Uplo.Lower, n, n, 1.0, A, 0, n, B, 0, n, 1.0, c, 0, n);
        return c;
    }

    @Benchmark
    public double[] rightLower_orig() {
        double[] c = C.clone();
        dsymmRightLowerOrig(n, n, 1.0, A, 0, n, B, 0, n, c, 0, n);
        return c;
    }

    // ── Right Upper ───────────────────────────────────────────────────────────

    @Benchmark
    public double[] rightUpper_new() {
        double[] c = C.clone();
        Dsymm.dsymm(BLAS.Side.Right, BLAS.Uplo.Upper, n, n, 1.0, A, 0, n, B, 0, n, 1.0, c, 0, n);
        return c;
    }

    @Benchmark
    public double[] rightUpper_orig() {
        double[] c = C.clone();
        dsymmRightUpperOrig(n, n, 1.0, A, 0, n, B, 0, n, c, 0, n);
        return c;
    }

    // =========================================================================
    // Original (naive) implementations — reference baseline
    // =========================================================================

    static void dsymmLeftUpperOrig(int m, int n, double alpha,
                                   double[] A, int aOff, int lda,
                                   double[] B, int bOff, int ldb,
                                   double[] C, int cOff, int ldc) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * B[bOff + i * ldb + j];
                double temp2 = 0.0;
                C[cOff + i * ldc + j] += temp1 * A[aOff + i * lda + i];
                for (int k = 0; k < i; k++) {
                    double aKI = A[aOff + k * lda + i];
                    C[cOff + k * ldc + j] += temp1 * aKI;
                    temp2 += aKI * B[bOff + k * ldb + j];
                }
                for (int k = i + 1; k < m; k++) {
                    double aIK = A[aOff + i * lda + k];
                    C[cOff + k * ldc + j] += temp1 * aIK;
                    temp2 += aIK * B[bOff + k * ldb + j];
                }
                C[cOff + i * ldc + j] += alpha * temp2;
            }
        }
    }

    static void dsymmLeftLowerOrig(int m, int n, double alpha,
                                   double[] A, int aOff, int lda,
                                   double[] B, int bOff, int ldb,
                                   double[] C, int cOff, int ldc) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * B[bOff + i * ldb + j];
                double temp2 = 0.0;
                C[cOff + i * ldc + j] += temp1 * A[aOff + i * lda + i];
                for (int k = 0; k < i; k++) {
                    double aIK = A[aOff + i * lda + k];
                    C[cOff + k * ldc + j] += temp1 * aIK;
                    temp2 += aIK * B[bOff + k * ldb + j];
                }
                for (int k = i + 1; k < m; k++) {
                    double aKI = A[aOff + k * lda + i];
                    C[cOff + k * ldc + j] += temp1 * aKI;
                    temp2 += aKI * B[bOff + k * ldb + j];
                }
                C[cOff + i * ldc + j] += alpha * temp2;
            }
        }
    }

    static void dsymmRightUpperOrig(int m, int n, double alpha,
                                    double[] A, int aOff, int lda,
                                    double[] B, int bOff, int ldb,
                                    double[] C, int cOff, int ldc) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * B[bOff + i * ldb + j];
                double temp2 = 0.0;
                C[cOff + i * ldc + j] += temp1 * A[aOff + j * lda + j];
                for (int k = 0; k < j; k++) {
                    double aKJ = A[aOff + k * lda + j];
                    C[cOff + i * ldc + k] += temp1 * aKJ;
                    temp2 += aKJ * B[bOff + i * ldb + k];
                }
                for (int k = j + 1; k < n; k++) {
                    double aJK = A[aOff + j * lda + k];
                    C[cOff + i * ldc + k] += temp1 * aJK;
                    temp2 += aJK * B[bOff + i * ldb + k];
                }
                C[cOff + i * ldc + j] += alpha * temp2;
            }
        }
    }

    static void dsymmRightLowerOrig(int m, int n, double alpha,
                                    double[] A, int aOff, int lda,
                                    double[] B, int bOff, int ldb,
                                    double[] C, int cOff, int ldc) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double temp1 = alpha * B[bOff + i * ldb + j];
                double temp2 = 0.0;
                C[cOff + i * ldc + j] += temp1 * A[aOff + j * lda + j];
                for (int k = 0; k < j; k++) {
                    double aJK = A[aOff + j * lda + k];
                    C[cOff + i * ldc + k] += temp1 * aJK;
                    temp2 += aJK * B[bOff + i * ldb + k];
                }
                for (int k = j + 1; k < n; k++) {
                    double aKJ = A[aOff + k * lda + j];
                    C[cOff + i * ldc + k] += temp1 * aKJ;
                    temp2 += aKJ * B[bOff + i * ldb + k];
                }
                C[cOff + i * ldc + j] += alpha * temp2;
            }
        }
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
            .include(DsymmBenchmark.class.getSimpleName())
            .forks(0).warmupIterations(5).measurementIterations(10).build();
        new Runner(opt).run();
    }
}
