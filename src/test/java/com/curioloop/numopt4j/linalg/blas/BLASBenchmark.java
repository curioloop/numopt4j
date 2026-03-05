/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Micro-benchmark comparing original vs loop-interchange optimized implementations
 * of Dtrmm (Left Upper/Lower NN blocked).
 * Dsyrk NoTrans: loop interchange was benchmarked but showed ~40% regression;
 * original i-j-p order is already optimal — no optimization applied.
 *
 * Run with:
 *   mvn test-compile exec:java -Dexec.mainClass=org.openjdk.jmh.Main \
 *     -Dexec.args="BLASBenchmark -f 0 -wi 5 -i 10"
 */
package com.curioloop.numopt4j.linalg.blas;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(0)
@State(Scope.Benchmark)
public class BLASBenchmark {

    // Matrix sizes representative of LM optimizer workloads:
    //   poly20: n=20, poly30: n=30, poly50: n=50
    @Param({"20", "50", "100"})
    int n;

    double[] A, Borig, Corig;

    @Setup(Level.Invocation)
    public void setup() {
        Random rng = new Random(42);
        A = new double[n * n];
        Borig = new double[n * n];
        Corig = new double[n * n];
        for (int i = 0; i < n * n; i++) {
            A[i] = rng.nextDouble();
            Borig[i] = rng.nextDouble();
            Corig[i] = rng.nextDouble();
        }
        // Make A upper triangular
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++)
                A[i * n + j] = 0.0;
    }

    // ── Dtrmm Left Upper NN ───────────────────────────────────────────────────

    @Benchmark
    public double[] dtrmm_upper_new() {
        double[] b = Borig.clone();
        dtrmmLeftUpperNNNew(n, n, 1.0, A, 0, n, b, 0, n, false);
        return b;
    }

    @Benchmark
    public double[] dtrmm_upper_orig() {
        double[] b = Borig.clone();
        dtrmmLeftUpperNNOrig(n, n, 1.0, A, 0, n, b, 0, n, false);
        return b;
    }

    // ── Dtrmm Left Lower NN ───────────────────────────────────────────────────

    @Benchmark
    public double[] dtrmm_lower_new() {
        double[] b = Borig.clone();
        dtrmmLeftLowerNNNew(n, n, 1.0, A, 0, n, b, 0, n, false);
        return b;
    }

    @Benchmark
    public double[] dtrmm_lower_orig() {
        double[] b = Borig.clone();
        dtrmmLeftLowerNNOrig(n, n, 1.0, A, 0, n, b, 0, n, false);
        return b;
    }

    // ── Dtrmm Right Upper NN ─────────────────────────────────────────────────

    @Benchmark
    public double[] dtrmm_right_upper_new() {
        double[] b = Borig.clone();
        dtrmmRightUpperNNNew(n, n, 1.0, A, 0, n, b, 0, n, false);
        return b;
    }

    @Benchmark
    public double[] dtrmm_right_upper_orig() {
        double[] b = Borig.clone();
        dtrmmRightUpperNNOrig(n, n, 1.0, A, 0, n, b, 0, n, false);
        return b;
    }

    // ── Dtrmm Right Lower NN ─────────────────────────────────────────────────

    @Benchmark
    public double[] dtrmm_right_lower_new() {
        double[] b = Borig.clone();
        dtrmmRightLowerNNNew(n, n, 1.0, A, 0, n, b, 0, n, false);
        return b;
    }

    @Benchmark
    public double[] dtrmm_right_lower_orig() {
        double[] b = Borig.clone();
        dtrmmRightLowerNNOrig(n, n, 1.0, A, 0, n, b, 0, n, false);
        return b;
    }

    // ── Dsyrk NoTrans (upper) ─────────────────────────────────────────────────

    @Benchmark
    public double[] dsyrk_notrans_new() {
        double[] c = Corig.clone();
        syrkNoTransNew(n, n, 1.0, A, 0, n, c, 0, n, true, 0);
        return c;
    }

    @Benchmark
    public double[] dsyrk_notrans_orig() {
        double[] c = Corig.clone();
        syrkNoTransOrig(n, n, 1.0, A, 0, n, c, 0, n, true, 0);
        return c;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // NEW implementations (loop-interchange / scatter order)
    // ═════════════════════════════════════════════════════════════════════════

    static void dtrmmLeftUpperNNNew(int m, int n, double alpha,
                                    double[] A, int aOff, int lda,
                                    double[] B, int bOff, int ldb,
                                    boolean unitDiag) {
        int n4 = (n / 4) * 4;
        for (int k = 0; k < m; k++) {
            int bRowK = bOff + k * ldb;
            for (int i = 0; i < k; i++) {
                double aik = alpha * A[aOff + i * lda + k];
                if (aik == 0.0) continue;
                int bRowI = bOff + i * ldb;
                int j = 0;
                for (; j < n4; j += 4) {
                    B[bRowI+j]   = FMA.op(aik, B[bRowK+j],   B[bRowI+j]);
                    B[bRowI+j+1] = FMA.op(aik, B[bRowK+j+1], B[bRowI+j+1]);
                    B[bRowI+j+2] = FMA.op(aik, B[bRowK+j+2], B[bRowI+j+2]);
                    B[bRowI+j+3] = FMA.op(aik, B[bRowK+j+3], B[bRowI+j+3]);
                }
                for (; j < n; j++) B[bRowI+j] = FMA.op(aik, B[bRowK+j], B[bRowI+j]);
            }
            double diag = unitDiag ? alpha : alpha * A[aOff + k * lda + k];
            int j = 0;
            for (; j < n4; j += 4) {
                B[bRowK+j] *= diag; B[bRowK+j+1] *= diag;
                B[bRowK+j+2] *= diag; B[bRowK+j+3] *= diag;
            }
            for (; j < n; j++) B[bRowK+j] *= diag;
        }
    }

    static void dtrmmLeftLowerNNNew(int m, int n, double alpha,
                                    double[] A, int aOff, int lda,
                                    double[] B, int bOff, int ldb,
                                    boolean unitDiag) {
        int n4 = (n / 4) * 4;
        for (int k = m - 1; k >= 0; k--) {
            int bRowK = bOff + k * ldb;
            for (int i = k + 1; i < m; i++) {
                double aik = alpha * A[aOff + i * lda + k];
                if (aik == 0.0) continue;
                int bRowI = bOff + i * ldb;
                int j = 0;
                for (; j < n4; j += 4) {
                    B[bRowI+j]   = FMA.op(aik, B[bRowK+j],   B[bRowI+j]);
                    B[bRowI+j+1] = FMA.op(aik, B[bRowK+j+1], B[bRowI+j+1]);
                    B[bRowI+j+2] = FMA.op(aik, B[bRowK+j+2], B[bRowI+j+2]);
                    B[bRowI+j+3] = FMA.op(aik, B[bRowK+j+3], B[bRowI+j+3]);
                }
                for (; j < n; j++) B[bRowI+j] = FMA.op(aik, B[bRowK+j], B[bRowI+j]);
            }
            double diag = unitDiag ? alpha : alpha * A[aOff + k * lda + k];
            int j = 0;
            for (; j < n4; j += 4) {
                B[bRowK+j] *= diag; B[bRowK+j+1] *= diag;
                B[bRowK+j+2] *= diag; B[bRowK+j+3] *= diag;
            }
            for (; j < n; j++) B[bRowK+j] *= diag;
        }
    }

    // NOTE: p-i-j scatter order was benchmarked and showed ~40% regression vs i-j-p.
    // Root cause: inner j-loop accesses A[j*lda+p] with stride=lda (cache-unfriendly).
    // Original i-j-p is already optimal for NoTrans — no loop interchange applied.
    // This method is identical to syrkNoTransOrig to confirm parity.
    static void syrkNoTransNew(int n, int k, double alpha,
                               double[] A, int aOff, int lda,
                               double[] C, int cOff, int ldc,
                               boolean upper, int kStart) {
        if (upper) {
            for (int i = 0; i < n; i++) {
                int aiOff = aOff + i * lda + kStart;
                for (int j = i; j < n; j++) {
                    int ajOff = aOff + j * lda + kStart;
                    double sum = 0.0;
                    int p = 0;
                    for (; p + 3 < k; p += 4) {
                        sum = FMA.op(A[aiOff+p],   A[ajOff+p],   sum);
                        sum = FMA.op(A[aiOff+p+1], A[ajOff+p+1], sum);
                        sum = FMA.op(A[aiOff+p+2], A[ajOff+p+2], sum);
                        sum = FMA.op(A[aiOff+p+3], A[ajOff+p+3], sum);
                    }
                    for (; p < k; p++) sum = FMA.op(A[aiOff+p], A[ajOff+p], sum);
                    C[cOff+i*ldc+j] += alpha * sum;
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                int aiOff = aOff + i * lda + kStart;
                for (int j = 0; j <= i; j++) {
                    int ajOff = aOff + j * lda + kStart;
                    double sum = 0.0;
                    int p = 0;
                    for (; p + 3 < k; p += 4) {
                        sum = FMA.op(A[aiOff+p],   A[ajOff+p],   sum);
                        sum = FMA.op(A[aiOff+p+1], A[ajOff+p+1], sum);
                        sum = FMA.op(A[aiOff+p+2], A[ajOff+p+2], sum);
                        sum = FMA.op(A[aiOff+p+3], A[ajOff+p+3], sum);
                    }
                    for (; p < k; p++) sum = FMA.op(A[aiOff+p], A[ajOff+p], sum);
                    C[cOff+i*ldc+j] += alpha * sum;
                }
            }
        }
    }

    static void dtrmmRightUpperNNNew(int m, int n, double alpha,
                                     double[] A, int aOff, int lda,
                                     double[] B, int bOff, int ldb,
                                     boolean unitDiag) {
        int m4 = (m / 4) * 4;
        for (int k = n - 1; k >= 0; k--) {
            int aRowK = aOff + k * lda;
            for (int j = k + 1; j < n; j++) {
                double akj = alpha * A[aRowK + j];
                if (akj == 0.0) continue;
                int i = 0;
                for (; i < m4; i += 4) {
                    int b0 = bOff + i * ldb, b1 = bOff + (i+1) * ldb;
                    int b2 = bOff + (i+2) * ldb, b3 = bOff + (i+3) * ldb;
                    B[b0+j] = FMA.op(akj, B[b0+k], B[b0+j]);
                    B[b1+j] = FMA.op(akj, B[b1+k], B[b1+j]);
                    B[b2+j] = FMA.op(akj, B[b2+k], B[b2+j]);
                    B[b3+j] = FMA.op(akj, B[b3+k], B[b3+j]);
                }
                for (; i < m; i++) { int bi = bOff + i * ldb; B[bi+j] = FMA.op(akj, B[bi+k], B[bi+j]); }
            }
            double diag = unitDiag ? alpha : alpha * A[aRowK + k];
            int i = 0;
            for (; i < m4; i += 4) {
                B[bOff+i*ldb+k] *= diag; B[bOff+(i+1)*ldb+k] *= diag;
                B[bOff+(i+2)*ldb+k] *= diag; B[bOff+(i+3)*ldb+k] *= diag;
            }
            for (; i < m; i++) B[bOff+i*ldb+k] *= diag;
        }
    }

    static void dtrmmRightLowerNNNew(int m, int n, double alpha,
                                     double[] A, int aOff, int lda,
                                     double[] B, int bOff, int ldb,
                                     boolean unitDiag) {
        int m4 = (m / 4) * 4;
        for (int k = 0; k < n; k++) {
            int aRowK = aOff + k * lda;
            for (int j = 0; j < k; j++) {
                double akj = alpha * A[aRowK + j];
                if (akj == 0.0) continue;
                int i = 0;
                for (; i < m4; i += 4) {
                    int b0 = bOff + i * ldb, b1 = bOff + (i+1) * ldb;
                    int b2 = bOff + (i+2) * ldb, b3 = bOff + (i+3) * ldb;
                    B[b0+j] = FMA.op(akj, B[b0+k], B[b0+j]);
                    B[b1+j] = FMA.op(akj, B[b1+k], B[b1+j]);
                    B[b2+j] = FMA.op(akj, B[b2+k], B[b2+j]);
                    B[b3+j] = FMA.op(akj, B[b3+k], B[b3+j]);
                }
                for (; i < m; i++) { int bi = bOff + i * ldb; B[bi+j] = FMA.op(akj, B[bi+k], B[bi+j]); }
            }
            double diag = unitDiag ? alpha : alpha * A[aRowK + k];
            int i = 0;
            for (; i < m4; i += 4) {
                B[bOff+i*ldb+k] *= diag; B[bOff+(i+1)*ldb+k] *= diag;
                B[bOff+(i+2)*ldb+k] *= diag; B[bOff+(i+3)*ldb+k] *= diag;
            }
            for (; i < m; i++) B[bOff+i*ldb+k] *= diag;
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // ORIGINAL implementations (i-j-k gather order)
    // ═════════════════════════════════════════════════════════════════════════

    static void dtrmmLeftUpperNNOrig(int m, int n, double alpha,
                                     double[] A, int aOff, int lda,
                                     double[] B, int bOff, int ldb,
                                     boolean unitDiag) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                double sum = unitDiag ? B[bOff+i*ldb+j] : A[aOff+i*lda+i] * B[bOff+i*ldb+j];
                for (int k = i + 1; k < m; k++)
                    sum = FMA.op(A[aOff+i*lda+k], B[bOff+k*ldb+j], sum);
                B[bOff+i*ldb+j] = alpha * sum;
            }
        }
    }

    static void dtrmmLeftLowerNNOrig(int m, int n, double alpha,
                                     double[] A, int aOff, int lda,
                                     double[] B, int bOff, int ldb,
                                     boolean unitDiag) {
        for (int i = m - 1; i >= 0; i--) {
            for (int j = 0; j < n; j++) {
                int kEnd = unitDiag ? i : i + 1;
                double sum = 0.0;
                for (int k = 0; k < kEnd; k++)
                    sum = FMA.op(A[aOff+i*lda+k], B[bOff+k*ldb+j], sum);
                if (unitDiag) sum += B[bOff+i*ldb+j];
                B[bOff+i*ldb+j] = alpha * sum;
            }
        }
    }

    static void dtrmmRightUpperNNOrig(int m, int n, double alpha,
                                      double[] A, int aOff, int lda,
                                      double[] B, int bOff, int ldb,
                                      boolean unitDiag) {
        int m4 = (m / 4) * 4;
        for (int j = n - 1; j >= 0; j--) {
            for (int ii = 0; ii < m4; ii += 4) {
                for (int i = ii; i < ii + 4 && i < m; i++) {
                    double sum = 0.0;
                    int kEnd = unitDiag ? j : j + 1;
                    for (int k = 0; k < kEnd; k++) sum = FMA.op(B[bOff+i*ldb+k], A[aOff+k*lda+j], sum);
                    if (unitDiag) sum += B[bOff+i*ldb+j];
                    B[bOff+i*ldb+j] = alpha * sum;
                }
            }
            if (m > m4) {
                for (int i = m4; i < m; i++) {
                    double sum = 0.0;
                    int kEnd = unitDiag ? j : j + 1;
                    for (int k = 0; k < kEnd; k++) sum = FMA.op(B[bOff+i*ldb+k], A[aOff+k*lda+j], sum);
                    if (unitDiag) sum += B[bOff+i*ldb+j];
                    B[bOff+i*ldb+j] = alpha * sum;
                }
            }
        }
    }

    static void dtrmmRightLowerNNOrig(int m, int n, double alpha,
                                      double[] A, int aOff, int lda,
                                      double[] B, int bOff, int ldb,
                                      boolean unitDiag) {
        int m4 = (m / 4) * 4;
        for (int j = 0; j < n; j++) {
            for (int ii = 0; ii < m4; ii += 4) {
                for (int i = ii; i < ii + 4 && i < m; i++) {
                    double sum = 0.0;
                    int kStart = unitDiag ? j + 1 : j;
                    for (int k = kStart; k < n; k++) sum = FMA.op(B[bOff+i*ldb+k], A[aOff+k*lda+j], sum);
                    if (unitDiag) sum += B[bOff+i*ldb+j];
                    B[bOff+i*ldb+j] = alpha * sum;
                }
            }
            if (m > m4) {
                for (int i = m4; i < m; i++) {
                    double sum = 0.0;
                    int kStart = unitDiag ? j + 1 : j;
                    for (int k = kStart; k < n; k++) sum = FMA.op(B[bOff+i*ldb+k], A[aOff+k*lda+j], sum);
                    if (unitDiag) sum += B[bOff+i*ldb+j];
                    B[bOff+i*ldb+j] = alpha * sum;
                }
            }
        }
    }

    static void syrkNoTransOrig(int n, int k, double alpha,
                                double[] A, int aOff, int lda,
                                double[] C, int cOff, int ldc,
                                boolean upper, int kStart) {
        if (upper) {
            for (int i = 0; i < n; i++) {
                int aiOff = aOff + i * lda + kStart;
                for (int j = i; j < n; j++) {
                    int ajOff = aOff + j * lda + kStart;
                    double sum = 0.0;
                    int p = 0;
                    for (; p + 3 < k; p += 4) {
                        sum = FMA.op(A[aiOff+p],   A[ajOff+p],   sum);
                        sum = FMA.op(A[aiOff+p+1], A[ajOff+p+1], sum);
                        sum = FMA.op(A[aiOff+p+2], A[ajOff+p+2], sum);
                        sum = FMA.op(A[aiOff+p+3], A[ajOff+p+3], sum);
                    }
                    for (; p < k; p++) sum = FMA.op(A[aiOff+p], A[ajOff+p], sum);
                    C[cOff+i*ldc+j] += alpha * sum;
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                int aiOff = aOff + i * lda + kStart;
                for (int j = 0; j <= i; j++) {
                    int ajOff = aOff + j * lda + kStart;
                    double sum = 0.0;
                    int p = 0;
                    for (; p + 3 < k; p += 4) {
                        sum = FMA.op(A[aiOff+p],   A[ajOff+p],   sum);
                        sum = FMA.op(A[aiOff+p+1], A[ajOff+p+1], sum);
                        sum = FMA.op(A[aiOff+p+2], A[ajOff+p+2], sum);
                        sum = FMA.op(A[aiOff+p+3], A[ajOff+p+3], sum);
                    }
                    for (; p < k; p++) sum = FMA.op(A[aiOff+p], A[ajOff+p], sum);
                    C[cOff+i*ldc+j] += alpha * sum;
                }
            }
        }
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
            .include(BLASBenchmark.class.getSimpleName())
            .forks(0).warmupIterations(5).measurementIterations(10).build();
        new Runner(opt).run();
    }
}
