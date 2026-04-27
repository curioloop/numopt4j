package com.curioloop.numopt4j.quad;

import com.curioloop.numopt4j.quad.adapt.AdaptiveIntegral;
import com.curioloop.numopt4j.quad.adapt.AdaptivePool;
import com.curioloop.numopt4j.quad.adapt.AdaptiveRule;
import com.curioloop.numopt4j.quad.de.DEOpts;
import com.curioloop.numopt4j.quad.de.DEPool;
import com.curioloop.numopt4j.quad.gauss.FixedIntegral;
import com.curioloop.numopt4j.quad.gauss.GaussPool;
import com.curioloop.numopt4j.quad.gauss.GaussRule;
import com.curioloop.numopt4j.quad.gauss.WeightedIntegral;
import com.curioloop.numopt4j.quad.ode.IVPIntegral;
import com.curioloop.numopt4j.quad.ode.IVPMethod;
import com.curioloop.numopt4j.quad.ode.IVPPool;
import com.curioloop.numopt4j.quad.ode.ODE;
import com.curioloop.numopt4j.quad.sampled.FilonOpts;
import com.curioloop.numopt4j.quad.sampled.SampledRule;
import com.curioloop.numopt4j.quad.special.EndpointOpts;
import com.curioloop.numopt4j.quad.special.EndpointSingularIntegral;
import com.curioloop.numopt4j.quad.special.ImproperIntegral;
import com.curioloop.numopt4j.quad.special.ImproperOpts;
import com.curioloop.numopt4j.quad.special.OscillatoryIntegral;
import com.curioloop.numopt4j.quad.special.OscillatoryOpts;
import com.curioloop.numopt4j.quad.special.OscillatoryPool;
import com.curioloop.numopt4j.quad.special.PrincipalValueIntegral;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.concurrent.TimeUnit;
import java.util.function.DoubleUnaryOperator;

/**
 * Allocation-focused JMH benchmark across the quadrature builders exposed by {@link Integrator}.
 *
 * <p>Run with the GC profiler to get per-op allocation metrics:</p>
 * <pre>
 * java --add-modules jdk.incubator.vector -cp target/test-classes:target/classes:$(cat tmp/test.classpath) \
 *   org.openjdk.jmh.Main QuadratureAllocationBenchmark -prof gc -wi 2 -i 3 -f 1 -tu us -bm avgt
 * </pre>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 3, time = 1)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector"})
public class QuadratureAllocationBenchmark {

    private static final DoubleUnaryOperator FIXED_FUNCTION = x -> Math.exp(-x) * Math.cos(0.5 * x);
    private static final DoubleUnaryOperator WEIGHTED_FUNCTION = x -> 1.0 + x * x;
    private static final DoubleUnaryOperator ADAPTIVE_FUNCTION = Math::sin;
    private static final DoubleUnaryOperator DE_FINITE_FUNCTION = x -> Math.exp(-x * x);
    private static final DoubleUnaryOperator DE_HALF_LINE_FUNCTION = x -> Math.exp(-x);
    private static final DoubleUnaryOperator DE_WHOLE_LINE_FUNCTION = x -> Math.exp(-x * x);
    private static final DoubleUnaryOperator OSCILLATORY_FINITE_FUNCTION = x -> Math.exp(20.0 * (x - 1.0));
    private static final DoubleUnaryOperator OSCILLATORY_UPPER_FUNCTION = x -> Math.exp(-2.5 * x);
    private static final DoubleUnaryOperator PRINCIPAL_VALUE_FUNCTION = Math::exp;
    private static final DoubleUnaryOperator ENDPOINT_ALGEBRAIC_FUNCTION = x -> 1.0 / (1.0 + x + Math.pow(2.0, -1.5));
    private static final DoubleUnaryOperator ENDPOINT_LOG_FUNCTION = x -> 1.0;
    private static final DoubleUnaryOperator IMPROPER_FUNCTION = x -> Math.exp(-x);
    private static final DoubleUnaryOperator FILON_FUNCTION = x -> Math.exp(20.0 * (x - 1.0));
    private static final DoubleUnaryOperator SAMPLED_FUNCTION = x -> Math.exp(-x) * (1.0 + 0.5 * x);
    private static final ODE.Equation ODE_EXP_DECAY = (t, y, dydt) -> dydt[0] = -y[0];

    @State(Scope.Thread)
    public static class IntegrationState {
        double[] sampledY;
        double[] sampledX;
        double[] odeEvalAt;

        FixedIntegral fixedProblem;
        GaussPool fixedWorkspace;

        WeightedIntegral weightedProblem;
        GaussPool weightedWorkspace;

        AdaptiveIntegral adaptiveGk15Problem;
        AdaptivePool adaptiveGk15Workspace;

        AdaptiveIntegral adaptiveLobattoProblem;
        AdaptivePool adaptiveLobattoWorkspace;

        com.curioloop.numopt4j.quad.de.DoubleExponentialIntegral tanhSinhProblem;
        DEPool tanhSinhWorkspace;

        com.curioloop.numopt4j.quad.de.DoubleExponentialIntegral expSinhProblem;
        DEPool expSinhWorkspace;

        com.curioloop.numopt4j.quad.de.DoubleExponentialIntegral sinhSinhProblem;
        DEPool sinhSinhWorkspace;

        OscillatoryIntegral oscillatoryFiniteProblem;
        OscillatoryPool oscillatoryFiniteWorkspace;

        OscillatoryIntegral oscillatoryUpperProblem;
        OscillatoryPool oscillatoryUpperWorkspace;

        PrincipalValueIntegral principalValueProblem;
        AdaptivePool principalValueWorkspace;

        EndpointSingularIntegral endpointAlgebraicProblem;
        GaussPool endpointAlgebraicWorkspace;

        EndpointSingularIntegral endpointLogProblem;
        DEPool endpointLogWorkspace;

        ImproperIntegral.Fixed improperFixedProblem;
        GaussPool improperFixedWorkspace;

        ImproperIntegral.Adaptive improperAdaptiveProblem;
        AdaptivePool improperAdaptiveWorkspace;

        IVPIntegral odeProblem;
        IVPPool odeWorkspace;

        @Setup(Level.Trial)
        public void setup() {
            int sampleCount = 257;
            sampledY = new double[sampleCount];
            sampledX = new double[sampleCount];
            double dx = 1.0 / (sampleCount - 1);
            for (int i = 0; i < sampleCount; i++) {
                double x = i * dx;
                sampledX[i] = x;
                sampledY[i] = SAMPLED_FUNCTION.applyAsDouble(x);
            }

            odeEvalAt = new double[33];
            double dt = 5.0 / (odeEvalAt.length - 1);
            for (int i = 0; i < odeEvalAt.length; i++) {
                odeEvalAt[i] = i * dt;
            }

            fixedProblem = Integrator.fixed()
                    .function(FIXED_FUNCTION)
                    .bounds(0.0, 1.0)
                    .points(64);
            fixedWorkspace = new GaussPool();

            weightedProblem = Integrator.weighted()
                    .function(WEIGHTED_FUNCTION)
                    .points(64)
                    .rule(GaussRule.laguerre());
            weightedWorkspace = new GaussPool();

            adaptiveGk15Problem = Integrator.adaptive()
                    .function(ADAPTIVE_FUNCTION)
                    .bounds(0.0, Math.PI)
                    .tolerances(1e-12, 1e-12);
            adaptiveGk15Workspace = new AdaptivePool();

            adaptiveLobattoProblem = Integrator.adaptive()
                    .function(ADAPTIVE_FUNCTION)
                    .bounds(0.0, Math.PI)
                    .tolerances(1e-12, 1e-12)
                    .rule(AdaptiveRule.GAUSS_LOBATTO);
            adaptiveLobattoWorkspace = new AdaptivePool();

            tanhSinhProblem = Integrator.doubleExponential(DEOpts.TANH_SINH)
                    .function(DE_FINITE_FUNCTION)
                    .bounds(0.0, 1.0)
                    .tolerance(1e-10);
            tanhSinhWorkspace = new DEPool();

            expSinhProblem = Integrator.doubleExponential(DEOpts.EXP_SINH)
                    .function(DE_HALF_LINE_FUNCTION)
                    .bounds(0.0, Double.POSITIVE_INFINITY)
                    .tolerance(1e-10);
            expSinhWorkspace = new DEPool();

            sinhSinhProblem = Integrator.doubleExponential(DEOpts.SINH_SINH)
                    .function(DE_WHOLE_LINE_FUNCTION)
                    .tolerance(1e-10);
            sinhSinhWorkspace = new DEPool();

            oscillatoryFiniteProblem = Integrator.oscillatory(OscillatoryOpts.COS)
                    .function(OSCILLATORY_FINITE_FUNCTION)
                    .lowerBound(0.0)
                    .upperBound(1.0)
                    .omega(7.5)
                    .tolerances(1e-11, 1e-11);
                oscillatoryFiniteWorkspace = new OscillatoryPool();

            oscillatoryUpperProblem = Integrator.oscillatory(OscillatoryOpts.COS_UPPER)
                    .function(OSCILLATORY_UPPER_FUNCTION)
                    .lowerBound(0.0)
                    .omega(2.3)
                    .tolerances(1e-10, 1e-10);
                oscillatoryUpperWorkspace = new OscillatoryPool();

            principalValueProblem = Integrator.principalValue()
                    .function(PRINCIPAL_VALUE_FUNCTION)
                    .bounds(-1.0, 1.0)
                    .pole(0.0)
                    .tolerances(1e-10, 1e-10);
            principalValueWorkspace = new AdaptivePool();

            endpointAlgebraicProblem = Integrator.endpointSingular(EndpointOpts.ALGEBRAIC)
                    .function(ENDPOINT_ALGEBRAIC_FUNCTION)
                    .bounds(-1.0, 1.0)
                    .exponents(-0.5, -0.5)
                    .tolerances(1e-10, 1e-10);
            endpointAlgebraicWorkspace = new GaussPool();

            endpointLogProblem = Integrator.endpointSingular(EndpointOpts.LOG_LEFT)
                    .function(ENDPOINT_LOG_FUNCTION)
                    .bounds(0.0, 1.0)
                    .exponents(0.0, 0.0)
                    .tolerances(1e-10, 1e-10);
            endpointLogWorkspace = new DEPool();

            improperFixedProblem = Integrator.improperFixed(ImproperOpts.UPPER)
                    .function(IMPROPER_FUNCTION)
                    .lowerBound(0.0)
                    .points(64);
            improperFixedWorkspace = new GaussPool();

            improperAdaptiveProblem = Integrator.improper(ImproperOpts.UPPER)
                    .function(IMPROPER_FUNCTION)
                    .lowerBound(0.0)
                    .tolerances(1e-10, 1e-10);
            improperAdaptiveWorkspace = new AdaptivePool();

            odeProblem = Integrator.ode(IVPMethod.RK45)
                    .equation(ODE_EXP_DECAY)
                    .bounds(0.0, 5.0)
                    .initialState(1.0)
                    .tolerances(1e-6, 1e-9)
                    .evalAt(odeEvalAt);
            odeWorkspace = IVPIntegral.workspace(IVPMethod.RK45);
        }
    }

    @Benchmark
    public void fixedFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.fixed()
                .function(FIXED_FUNCTION)
                .bounds(0.0, 1.0)
                .points(64)
                .integrate());
    }

    @Benchmark
    public void fixedReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.fixedProblem.integrate(state.fixedWorkspace));
    }

    @Benchmark
    public void weightedFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.weighted()
                .function(WEIGHTED_FUNCTION)
                .points(64)
                .rule(GaussRule.laguerre())
                .integrate());
    }

    @Benchmark
    public void weightedReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.weightedProblem.integrate(state.weightedWorkspace));
    }

    @Benchmark
    public void adaptiveGk15FreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.adaptive()
                .function(ADAPTIVE_FUNCTION)
                .bounds(0.0, Math.PI)
                .tolerances(1e-12, 1e-12)
                .integrate());
    }

    @Benchmark
    public void adaptiveGk15ReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.adaptiveGk15Problem.integrate(state.adaptiveGk15Workspace));
    }

    @Benchmark
    public void adaptiveLobattoFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.adaptive()
                .function(ADAPTIVE_FUNCTION)
                .bounds(0.0, Math.PI)
                .tolerances(1e-12, 1e-12)
                .rule(AdaptiveRule.GAUSS_LOBATTO)
                .integrate());
    }

    @Benchmark
    public void adaptiveLobattoReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.adaptiveLobattoProblem.integrate(state.adaptiveLobattoWorkspace));
    }

    @Benchmark
    public void tanhSinhFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.doubleExponential(DEOpts.TANH_SINH)
                .function(DE_FINITE_FUNCTION)
                .bounds(0.0, 1.0)
                .tolerance(1e-10)
                .integrate());
    }

    @Benchmark
    public void tanhSinhReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.tanhSinhProblem.integrate(state.tanhSinhWorkspace));
    }

    @Benchmark
    public void expSinhFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.doubleExponential(DEOpts.EXP_SINH)
                .function(DE_HALF_LINE_FUNCTION)
                .bounds(0.0, Double.POSITIVE_INFINITY)
                .tolerance(1e-10)
                .integrate());
    }

    @Benchmark
    public void expSinhReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.expSinhProblem.integrate(state.expSinhWorkspace));
    }

    @Benchmark
    public void sinhSinhFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.doubleExponential(DEOpts.SINH_SINH)
                .function(DE_WHOLE_LINE_FUNCTION)
                .tolerance(1e-10)
                .integrate());
    }

    @Benchmark
    public void sinhSinhReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.sinhSinhProblem.integrate(state.sinhSinhWorkspace));
    }

    @Benchmark
    public void oscillatoryFiniteFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.oscillatory(OscillatoryOpts.COS)
                .function(OSCILLATORY_FINITE_FUNCTION)
                .lowerBound(0.0)
                .upperBound(1.0)
                .omega(7.5)
                .tolerances(1e-11, 1e-11)
                .integrate());
    }

    @Benchmark
    public void oscillatoryFiniteReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.oscillatoryFiniteProblem.integrate(state.oscillatoryFiniteWorkspace));
    }

    @Benchmark
    public void oscillatoryUpperFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.oscillatory(OscillatoryOpts.COS_UPPER)
                .function(OSCILLATORY_UPPER_FUNCTION)
                .lowerBound(0.0)
                .omega(2.3)
                .tolerances(1e-10, 1e-10)
                .integrate());
    }

    @Benchmark
    public void oscillatoryUpperReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.oscillatoryUpperProblem.integrate(state.oscillatoryUpperWorkspace));
    }

    @Benchmark
    public void principalValueFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.principalValue()
                .function(PRINCIPAL_VALUE_FUNCTION)
                .bounds(-1.0, 1.0)
                .pole(0.0)
                .tolerances(1e-10, 1e-10)
                .integrate());
    }

    @Benchmark
    public void principalValueReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.principalValueProblem.integrate(state.principalValueWorkspace));
    }

    @Benchmark
    public void endpointAlgebraicFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.endpointSingular(EndpointOpts.ALGEBRAIC)
                .function(ENDPOINT_ALGEBRAIC_FUNCTION)
                .bounds(-1.0, 1.0)
                .exponents(-0.5, -0.5)
                .tolerances(1e-10, 1e-10)
                .integrate());
    }

    @Benchmark
    public void endpointAlgebraicReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.endpointAlgebraicProblem.integrate(state.endpointAlgebraicWorkspace));
    }

    @Benchmark
    public void endpointLogFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.endpointSingular(EndpointOpts.LOG_LEFT)
                .function(ENDPOINT_LOG_FUNCTION)
                .bounds(0.0, 1.0)
                .exponents(0.0, 0.0)
                .tolerances(1e-10, 1e-10)
                .integrate());
    }

    @Benchmark
    public void endpointLogReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.endpointLogProblem.integrateLogarithmic(state.endpointLogWorkspace));
    }

    @Benchmark
    public void improperFixedFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.improperFixed(ImproperOpts.UPPER)
                .function(IMPROPER_FUNCTION)
                .lowerBound(0.0)
                .points(64)
                .integrate());
    }

    @Benchmark
    public void improperFixedReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.improperFixedProblem.integrate(state.improperFixedWorkspace));
    }

    @Benchmark
    public void improperAdaptiveFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.improper(ImproperOpts.UPPER)
                .function(IMPROPER_FUNCTION)
                .lowerBound(0.0)
                .tolerances(1e-10, 1e-10)
                .integrate());
    }

    @Benchmark
    public void improperAdaptiveReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.improperAdaptiveProblem.integrate(state.improperAdaptiveWorkspace));
    }

    @Benchmark
    public void filonFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.filon(FilonOpts.COS)
                .function(FILON_FUNCTION)
                .bounds(0.0, 1.0)
                .frequency(64.0)
                .intervals(64)
                .integrate());
    }

    @Benchmark
    public void sampledSimpsonFunctionFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.sampled(SampledRule.SIMPSON)
                .function(SAMPLED_FUNCTION, 257, 0.0, 1.0)
                .integrate());
    }

    @Benchmark
    public void sampledRombergFunctionFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.sampled(SampledRule.ROMBERG)
                .function(SAMPLED_FUNCTION, 257, 0.0, 1.0)
                .integrate());
    }

    @Benchmark
    public void cumulativeSimpsonFreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.cumulative(SampledRule.SIMPSON)
                .samples(state.sampledX, state.sampledY)
                .integrate());
    }

    @Benchmark
    public void odeRk45FreshBuilder(IntegrationState state, Blackhole bh) {
        bh.consume(Integrator.ode(IVPMethod.RK45)
                .equation(ODE_EXP_DECAY)
                .bounds(0.0, 5.0)
                .initialState(1.0)
                .tolerances(1e-6, 1e-9)
                .evalAt(state.odeEvalAt)
                .integrate());
    }

    @Benchmark
    public void odeRk45ReuseWorkspace(IntegrationState state, Blackhole bh) {
        bh.consume(state.odeProblem.integrate(state.odeWorkspace));
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(QuadratureAllocationBenchmark.class.getSimpleName())
                .warmupIterations(2)
                .measurementIterations(3)
                .forks(1)
                .build();
        new Runner(opt).run();
    }
}