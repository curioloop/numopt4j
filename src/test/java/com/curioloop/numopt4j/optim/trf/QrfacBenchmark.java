/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * JMH micro-benchmark: TRFCore.qrfac (optimized) vs qrfacOrig (original column-stride).
 *
 * Run with:
 *   mvn test-compile exec:java -Dexec.mainClass=org.openjdk.jmh.Main \
 *     -Dexec.args="QrfacBenchmark -f 0 -wi 3 -i 5"
 */
package com.curioloop.numopt4j.optim.trf;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import com.curioloop.numopt4j.optim.trf.TRFCore;

import java.util.Random;
import java.util.concurrent.TimeUnit;

import static com.curioloop.numopt4j.optim.trf.TRFConstants.EPSMCH;

/**
 * Compares optimized {@link TRFCore#qrfac} (row-major norm scan + row-major Householder)
 * against the original column-stride implementation on Jacobian-sized matrices:
 *
 * <ul>
 *   <li>small  — 250×8   (Gauss1)</li>
 *   <li>medium — 500×20  (poly20)</li>
 *   <li>large  — 1000×30 (poly30)</li>
 *   <li>xlarge — 2000×50 (poly50)</li>
 * </ul>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(0)
@State(Scope.Benchmark)
public class QrfacBenchmark {

    private static final int SM = 250,  SN = 8;
    private static final int MM = 500,  MN = 20;
    private static final int LM = 1000, LN = 30;
    private static final int XM = 2000, XN = 50;

    private double[] A_small,  A_medium,  A_large,  A_xlarge;
    private double[] rdiag_s,  rdiag_m,   rdiag_l,  rdiag_x;
    private double[] acnorm_s, acnorm_m,  acnorm_l, acnorm_x;
    private double[] wa_s,     wa_m,      wa_l,     wa_x;
    private double[] dot_s,    dot_m,     dot_l,    dot_x;
    private int[]    ipvt_s,   ipvt_m,    ipvt_l,   ipvt_x;

    @Setup
    public void setup() {
        Random rng = new Random(42);
        A_small  = randomMatrix(rng, SM, SN);
        A_medium = randomMatrix(rng, MM, MN);
        A_large  = randomMatrix(rng, LM, LN);
        A_xlarge = randomMatrix(rng, XM, XN);

        rdiag_s  = new double[SN];  acnorm_s = new double[SN];  wa_s = new double[SN];  dot_s = new double[SN];  ipvt_s = new int[SN];
        rdiag_m  = new double[MN];  acnorm_m = new double[MN];  wa_m = new double[MN];  dot_m = new double[MN];  ipvt_m = new int[MN];
        rdiag_l  = new double[LN];  acnorm_l = new double[LN];  wa_l = new double[LN];  dot_l = new double[LN];  ipvt_l = new int[LN];
        rdiag_x  = new double[XN];  acnorm_x = new double[XN];  wa_x = new double[XN];  dot_x = new double[XN];  ipvt_x = new int[XN];
    }

    private static double[] randomMatrix(Random rng, int m, int n) {
        double[] A = new double[m * n];
        for (int i = 0; i < A.length; i++) A[i] = rng.nextGaussian();
        return A;
    }

    /**
     * Original MINPACK column-stride qrfac (baseline for comparison).
     * Identical algorithm to TRFCore.qrfac before the row-major optimization.
     */
    static void qrfacOrig(int m, int n, double[] a, int[] ipvt,
                          double[] rdiag, double[] acnorm, double[] wa) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int i = 0; i < m; i++) { double v = a[i*n+j]; s += v*v; }
            acnorm[j] = Math.sqrt(s);
            rdiag[j]  = acnorm[j];
            wa[j]     = rdiag[j];
            ipvt[j]   = j;
        }
        int minmn = Math.min(m, n);
        for (int j = 0; j < minmn; j++) {
            int kmax = j;
            for (int k = j+1; k < n; k++) if (rdiag[k] > rdiag[kmax]) kmax = k;
            if (kmax != j) {
                for (int i = 0; i < m; i++) {
                    double tmp = a[i*n+j]; a[i*n+j] = a[i*n+kmax]; a[i*n+kmax] = tmp;
                }
                rdiag[kmax] = rdiag[j]; wa[kmax] = wa[j];
                int itmp = ipvt[j]; ipvt[j] = ipvt[kmax]; ipvt[kmax] = itmp;
            }
            double ajnorm = 0.0;
            for (int i = j; i < m; i++) { double v = a[i*n+j]; ajnorm += v*v; }
            ajnorm = Math.sqrt(ajnorm);
            if (ajnorm == 0.0) { rdiag[j] = 0.0; continue; }
            if (a[j*n+j] < 0.0) ajnorm = -ajnorm;
            for (int i = j; i < m; i++) a[i*n+j] /= ajnorm;
            a[j*n+j] += 1.0;
            for (int k = j+1; k < n; k++) {
                double sum = 0.0;
                for (int i = j; i < m; i++) sum += a[i*n+j] * a[i*n+k];
                double tmp = sum / a[j*n+j];
                for (int i = j; i < m; i++) a[i*n+k] -= tmp * a[i*n+j];
                if (rdiag[k] != 0.0) {
                    double t = a[j*n+k] / rdiag[k];
                    rdiag[k] *= Math.sqrt(Math.max(0.0, 1.0 - t*t));
                    if (0.05 * (rdiag[k]/wa[k]) * (rdiag[k]/wa[k]) <= EPSMCH) {
                        double s2 = 0.0;
                        for (int i = j+1; i < m; i++) { double v = a[i*n+k]; s2 += v*v; }
                        rdiag[k] = Math.sqrt(s2); wa[k] = rdiag[k];
                    }
                }
            }
            rdiag[j] = -ajnorm;
        }
    }

    // ── small (250×8) ─────────────────────────────────────────────────────────

    @Benchmark
    public double[] small_orig() {
        double[] A = A_small.clone();
        qrfacOrig(SM, SN, A, ipvt_s, rdiag_s, acnorm_s, wa_s);
        return A;
    }

    @Benchmark
    public double[] small_opt() {
        double[] A = A_small.clone();
        TRFCore.qrfac(SM, SN, A, ipvt_s, rdiag_s, 0, acnorm_s, 0, wa_s, 0, dot_s, 0);
        return A;
    }

    // ── medium (500×20) ───────────────────────────────────────────────────────

    @Benchmark
    public double[] medium_orig() {
        double[] A = A_medium.clone();
        qrfacOrig(MM, MN, A, ipvt_m, rdiag_m, acnorm_m, wa_m);
        return A;
    }

    @Benchmark
    public double[] medium_opt() {
        double[] A = A_medium.clone();
        TRFCore.qrfac(MM, MN, A, ipvt_m, rdiag_m, 0, acnorm_m, 0, wa_m, 0, dot_m, 0);
        return A;
    }

    // ── large (1000×30) ───────────────────────────────────────────────────────

    @Benchmark
    public double[] large_orig() {
        double[] A = A_large.clone();
        qrfacOrig(LM, LN, A, ipvt_l, rdiag_l, acnorm_l, wa_l);
        return A;
    }

    @Benchmark
    public double[] large_opt() {
        double[] A = A_large.clone();
        TRFCore.qrfac(LM, LN, A, ipvt_l, rdiag_l, 0, acnorm_l, 0, wa_l, 0, dot_l, 0);
        return A;
    }

    // ── xlarge (2000×50) ──────────────────────────────────────────────────────

    @Benchmark
    public double[] xlarge_orig() {
        double[] A = A_xlarge.clone();
        qrfacOrig(XM, XN, A, ipvt_x, rdiag_x, acnorm_x, wa_x);
        return A;
    }

    @Benchmark
    public double[] xlarge_opt() {
        double[] A = A_xlarge.clone();
        TRFCore.qrfac(XM, XN, A, ipvt_x, rdiag_x, 0, acnorm_x, 0, wa_x, 0, dot_x, 0);
        return A;
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
            .include(QrfacBenchmark.class.getSimpleName())
            .forks(0).warmupIterations(3).measurementIterations(5).build();
        new Runner(opt).run();
    }
}
