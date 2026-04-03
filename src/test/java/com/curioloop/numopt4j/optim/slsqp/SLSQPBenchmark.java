/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.slsqp;

import com.curioloop.numopt4j.optim.Univariate;
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
 * JMH Benchmark for SLSQP pure Java implementation.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(0)
public class SLSQPBenchmark {

    @Param({"2", "5", "10", "20"})
    public int dimension;

    @Param({"unconstrained", "equality", "inequality", "bounded", "mixed"})
    public String problemType;

    private SLSQPProblem problem;
    private SLSQPWorkspace workspace;
    private double[] initialPoint;

    private static final double ACCURACY = 1e-8;
    private static final int MAX_ITERATIONS = 500;

    @Setup(Level.Trial)
    public void setup() {
        Random rand = new Random(42);
        initialPoint = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            initialPoint[i] = rand.nextDouble() * 2 - 1;
        }

        Univariate objective = (x, n, g) -> {
            double f = 0.0;
            for (int i = 0; i < x.length; i++) {
                double coef = i + 1;
                f += coef * x[i] * x[i];
                if (g != null) g[i] = 2.0 * coef * x[i];
            }
            return f;
        };

        SLSQPProblem p = new SLSQPProblem()
                .objective(objective)
                .maxIterations(MAX_ITERATIONS)
                .functionTolerance(ACCURACY)
                .nnlsIterations(100)
                .initialPoint(initialPoint.clone());

        switch (problemType) {
            case "equality":
            case "mixed":
                p.equalityConstraints((Univariate) (x, n, g) -> {
                    double sum = 0.0;
                    for (int i = 0; i < x.length; i++) {
                        sum += x[i];
                        if (g != null) g[i] = 1.0;
                    }
                    return sum - 1.0;
                });
                break;
        }

        switch (problemType) {
            case "inequality": {
                p.inequalityConstraints((Univariate) (x, n, g) -> {
                    if (g != null) { Arrays.fill(g, 0.0); g[0] = 1.0; }
                    return x[0] - 0.5;
                });
                break;
            }
            case "mixed": {
                Univariate[] ineq = new Univariate[dimension];
                for (int j = 0; j < dimension; j++) {
                    final int idx = j;
                    ineq[j] = (x, n, g) -> {
                        if (g != null) { Arrays.fill(g, 0.0); g[idx] = 1.0; }
                        return x[idx] + 1.0;
                    };
                }
                p.inequalityConstraints(ineq);
                break;
            }
        }

        if (problemType.equals("bounded") || problemType.equals("mixed")) {
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
    public void slsqp(Blackhole bh) {
        bh.consume(problem.solve(workspace));
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(".*SLSQPBenchmark.*")
                .param("dimension", "2", "5", "10", "20")
                .param("problemType", "unconstrained", "equality", "inequality", "bounded", "mixed")
                .warmupIterations(3)
                .measurementIterations(5)
                .forks(0)
                .build();
        new Runner(opt).run();
    }
}
