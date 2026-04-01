/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import com.curioloop.numopt4j.optim.Optimization;
import com.curioloop.numopt4j.optim.TestTemplates;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.concurrent.TimeUnit;

/**
 * Detailed JMH Benchmark for L-BFGS-B to identify performance bottlenecks.
 *
 * This benchmark focuses on high-dimensional Rosenbrock problems.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 5, time = 2)
@Measurement(iterations = 10, time = 2)
@Fork(0)
public class LBFGSBDetailedBenchmark {

    @Param({"20", "50", "100"})
    public int dimension;

    private LBFGSBProblem problem;
    private LBFGSBWorkspace workspace;

    private double[] initialPoint;

    private static final int CORRECTIONS = 10;
    private static final double FUNCTION_TOLERANCE = 1e10;
    private static final double GRADIENT_TOLERANCE = 1e-6;
    private static final int MAX_ITERATIONS = 2000;

    @Setup(Level.Trial)
    public void setup() {
        initialPoint = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            initialPoint[i] = (i % 2 == 0) ? -1.0 : 1.0;
        }

        problem = new LBFGSBProblem()
                .objective(TestTemplates.rosenbrock())
                .corrections(CORRECTIONS)
                .functionTolerance(FUNCTION_TOLERANCE)
                .gradientTolerance(GRADIENT_TOLERANCE)
                .maxIterations(MAX_ITERATIONS)
                .initialPoint(initialPoint.clone());

        workspace = problem.alloc();

        Optimization result = problem.solve(workspace);
        System.out.printf("Dimension %d: iterations=%d%n", dimension, result.getIterations());
    }

    @Setup(Level.Invocation)
    public void resetX() {
        problem.initialPoint(initialPoint.clone());
    }

    @Benchmark
    public void lbfgsb_rosenbrock(Blackhole bh) {
        bh.consume(problem.solve(workspace));
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(".*LBFGSBDetailedBenchmark.*")
                .param("dimension", "20", "50", "100")
                .warmupIterations(5)
                .measurementIterations(10)
                .forks(0)
                .build();
        new Runner(opt).run();
    }
}
