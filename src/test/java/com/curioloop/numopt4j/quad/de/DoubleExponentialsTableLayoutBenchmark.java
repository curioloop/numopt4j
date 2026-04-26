package com.curioloop.numopt4j.quad.de;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--enable-preview", "--add-modules=jdk.incubator.vector"})
public class DoubleExponentialsTableLayoutBenchmark {

    @State(Scope.Thread)
    public static class LayoutState {
        private static final int PASSES = 64;

        @Param({"exp_sinh", "sinh_sinh", "tanh_sinh"})
        public String rule;

        double[] abscissas;
        double[] weights;
        double[] interleaved;
        int[] rowData;

        @Setup(Level.Trial)
        public void setup() {
            switch (rule) {
                case "exp_sinh" -> {
                    abscissas = DoubleExponentialsTables.EXP_SINH_ABSCISSAS;
                    weights = DoubleExponentialsTables.EXP_SINH_WEIGHTS;
                    rowData = DoubleExponentialsTables.EXP_SINH_ROWS;
                }
                case "sinh_sinh" -> {
                    abscissas = DoubleExponentialsTables.SINH_SINH_ABSCISSAS;
                    weights = DoubleExponentialsTables.SINH_SINH_WEIGHTS;
                    rowData = DoubleExponentialsTables.SINH_SINH_ROWS;
                }
                case "tanh_sinh" -> {
                    abscissas = DoubleExponentialsTables.TANH_SINH_ABSCISSAS;
                    weights = DoubleExponentialsTables.TANH_SINH_WEIGHTS;
                    rowData = DoubleExponentialsTables.TANH_SINH_ROWS;
                }
                default -> throw new IllegalStateException("Unknown rule: " + rule);
            }

            interleaved = new double[abscissas.length << 1];
            for (int index = 0; index < abscissas.length; index++) {
                int pairIndex = index << 1;
                interleaved[pairIndex] = abscissas[index];
                interleaved[pairIndex + 1] = weights[index];
            }
        }
    }

    @Benchmark
    public double separateArraysLinear(LayoutState state) {
        return sumLinearSeparate(state.abscissas, state.weights);
    }

    @Benchmark
    public double mergedPairsLinear(LayoutState state) {
        return sumLinearInterleaved(state.interleaved);
    }

    @Benchmark
    public double separateArraysRowWise(LayoutState state) {
        return sumRowsSeparate(state.abscissas, state.weights, state.rowData);
    }

    @Benchmark
    public double mergedPairsRowWise(LayoutState state) {
        return sumRowsInterleaved(state.interleaved, state.rowData);
    }

    private static double sumLinearSeparate(double[] abscissas, double[] weights) {
        double sink = 0.0;
        for (int pass = 0; pass < LayoutState.PASSES; pass++) {
            for (int index = 0; index < abscissas.length; index++) {
                sink += pairContribution(abscissas[index], weights[index]);
            }
        }
        return sink;
    }

    private static double sumLinearInterleaved(double[] interleaved) {
        double sink = 0.0;
        for (int pass = 0; pass < LayoutState.PASSES; pass++) {
            for (int pairIndex = 0; pairIndex < interleaved.length; pairIndex += 2) {
                sink += pairContribution(interleaved[pairIndex], interleaved[pairIndex + 1]);
            }
        }
        return sink;
    }

    private static double sumRowsSeparate(double[] abscissas, double[] weights, int[] rowData) {
        double sink = 0.0;
        for (int pass = 0; pass < LayoutState.PASSES; pass++) {
            for (int row = 0; row < rowData.length; row += 2) {
                int rowStart = rowData[row];
                int rowEnd = rowStart + rowData[row + 1];
                for (int index = rowStart; index < rowEnd; index++) {
                    sink += pairContribution(abscissas[index], weights[index]);
                }
            }
        }
        return sink;
    }

    private static double sumRowsInterleaved(double[] interleaved, int[] rowData) {
        double sink = 0.0;
        for (int pass = 0; pass < LayoutState.PASSES; pass++) {
            for (int row = 0; row < rowData.length; row += 2) {
                int rowStart = rowData[row];
                int rowEnd = rowStart + rowData[row + 1];
                for (int index = rowStart; index < rowEnd; index++) {
                    int pairIndex = index << 1;
                    sink += pairContribution(interleaved[pairIndex], interleaved[pairIndex + 1]);
                }
            }
        }
        return sink;
    }

    private static double pairContribution(double abscissa, double weight) {
        double scaledAbscissa = Math.abs(abscissa) + 0.25;
        return scaledAbscissa * weight + abscissa;
    }
}