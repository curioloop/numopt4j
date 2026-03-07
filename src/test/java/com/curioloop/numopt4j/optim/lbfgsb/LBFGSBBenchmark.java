/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.lbfgsb;

import com.curioloop.numopt4j.optim.Univariate;
import com.curioloop.numopt4j.optim.TestTemplates;
import com.curioloop.numopt4j.optim.Bound;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH Benchmark for L-BFGS-B pure Java implementation.
 *
 * Tests various problem types:
 * - Rosenbrock function (unconstrained)
 * - Quadratic function (unconstrained)
 * - Bounded optimization
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(0)
public class LBFGSBBenchmark {

    @Param({"2", "5", "10", "20", "50"})
    public int dimension;

    @Param({"rosenbrock", "quadratic", "bounded"})
    public String problemType;

    private LBFGSBProblem problem;
    private LBFGSBWorkspace workspace;
    private double[] initialPoint;

    private static final int CORRECTIONS = 5;
    private static final double FUNCTION_TOLERANCE = 1e7;
    private static final double GRADIENT_TOLERANCE = 1e-8;
    private static final int MAX_ITERATIONS = 1000;

    @Setup(Level.Trial)
    public void setup() {
        Random rand = new Random(42);
        initialPoint = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            initialPoint[i] = rand.nextDouble() * 2 - 1;
        }

        Univariate objective = problemType.equals("rosenbrock")
                ? TestTemplates.rosenbrock()
                : (x, g) -> {
                    double f = 0.0;
                    for (int i = 0; i < x.length; i++) {
                        double coef = i + 1;
                        f += coef * x[i] * x[i];
                        if (g != null) g[i] = 2.0 * coef * x[i];
                    }
                    return f;
                };

        LBFGSBProblem p = new LBFGSBProblem()
                .objective(objective)
                .corrections(CORRECTIONS)
                .functionTolerance(FUNCTION_TOLERANCE)
                .gradientTolerance(GRADIENT_TOLERANCE)
                .maxIterations(MAX_ITERATIONS)
                .initialPoint(initialPoint.clone());

        if (problemType.equals("bounded")) {
            Bound[] bounds = new Bound[dimension];
            Arrays.fill(bounds, Bound.between(-5.0, 5.0));
            p.bounds(bounds);
        }

        problem = p;
        workspace = problem.alloc();
    }

    @Setup(Level.Invocation)
    public void resetX() {
        problem.initialPoint(initialPoint.clone());
    }

    @Benchmark
    public void lbfgsb(Blackhole bh) {
        bh.consume(problem.solve(workspace));
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(".*LBFGSBBenchmark.*")
                .param("dimension", "2", "5", "10", "20", "50")
                .param("problemType", "rosenbrock", "quadratic", "bounded")
                .warmupIterations(3)
                .measurementIterations(5)
                .forks(0)
                .build();
        new Runner(opt).run();
    }
}
