package com.curioloop.numopt4j.optim;

import com.curioloop.numopt4j.optim.root.HYBRSolver;
import com.curioloop.numopt4j.optim.root.HYBRWorkspace;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;

/**
 * JMH benchmark: HYBRSolver (col-major) numerical vs analytical Jacobian.
 *
 * Problem: Extended Rosenbrock system (n equations, n variables, n must be even)
 *   f_{2i}   = 1 - x_{2i}
 *   f_{2i+1} = 10 * (x_{2i+1} - x_{2i}^2)
 * Solution: x = (1, 1, ..., 1)
 * Initial:  x = (-1.2, 1, -1.2, 1, ...)
 *
 * Two scenarios:
 *   hybrd  — numerical finite-difference Jacobian (n+1 fcn calls per outer iter)
 *   hybrj  — analytical Jacobian supplied by user
 *
 * Run:
 *   mvn package -Pbenchmarks -DskipTests &&
 *   java -jar target/benchmarks.jar "HYBRBenchmark" -wi 3 -i 5 -f 1 -tu us -bm avgt
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class HYBRBenchmark {

    @Param({"50", "100", "200", "500"})
    public int n;

    // ── problem helpers ───────────────────────────────────────────────────────

    static double[] x0(int n) {
        double[] x = new double[n];
        for (int i = 0; i < n; i += 2) { x[i] = -1.2; x[i + 1] = 1.0; }
        return x;
    }

    static void fcn(int n, double[] x, double[] f) {
        for (int i = 0; i < n; i += 2) {
            f[i]     = 1.0 - x[i];
            f[i + 1] = 10.0 * (x[i + 1] - x[i] * x[i]);
        }
    }

    /** col-major Jacobian: fjac[row + lda*col] */
    static void jacCol(int n, double[] x, double[] fjac, int lda) {
        for (int k = 0; k < n * lda; k++) fjac[k] = 0.0;
        for (int i = 0; i < n; i += 2) {
            fjac[i       + lda * i]       = -1.0;
            fjac[(i + 1) + lda * i]       = -20.0 * x[i];
            fjac[(i + 1) + lda * (i + 1)] = 10.0;
        }
    }

    // =========================================================================
    // hybrd — numerical Jacobian
    // =========================================================================

    /** HYBRSolver — col-major fjac, numerical Jacobian */
    @Benchmark
    public int hybrd_colMajor() {
        HYBRWorkspace ws = new HYBRWorkspace(n);
        BiConsumer<double[], double[]> fn = (x, f) -> fcn(n, x, f);
        Multivariate eval = NumericalJacobian.FORWARD.wrap(fn, n, n, true);
        return HYBRSolver.solve(eval, x0(n), 1e-10, 200 * (n + 1), ws)
                          .getStatus().ordinal();
    }

    // =========================================================================
    // hybrj — analytical Jacobian
    // =========================================================================

    /** HYBRSolver — col-major fjac, analytical Jacobian */
    @Benchmark
    public int hybrj_colMajor() {
        HYBRWorkspace ws = new HYBRWorkspace(n);
        Multivariate eval = (x, f, jac) -> {
            fcn(n, x, f);
            if (jac != null) jacCol(n, x, jac, n);
        };
        return HYBRSolver.solve(eval, x0(n), 1e-10, 200 * (n + 1), ws)
                          .getStatus().ordinal();
    }

    // =========================================================================
    // main
    // =========================================================================

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(HYBRBenchmark.class.getSimpleName())
                .warmupIterations(3)
                .measurementIterations(5)
                .forks(1)
                .build();
        new Runner(opt).run();
    }
}
