/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad;

import com.curioloop.numopt4j.quad.adapt.AdaptivePool;
import com.curioloop.numopt4j.quad.adapt.AdaptiveIntegral;
import com.curioloop.numopt4j.quad.gauss.FixedIntegral;
import com.curioloop.numopt4j.quad.gauss.JacobiRule;
import com.curioloop.numopt4j.quad.gauss.GaussRule;
import com.curioloop.numopt4j.quad.special.EndpointOpts;
import com.curioloop.numopt4j.quad.special.ImproperOpts;
import com.curioloop.numopt4j.quad.special.OscillatoryOpts;
import com.curioloop.numopt4j.quad.gauss.GaussPool;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.data.Offset.offset;

class QuadratureTest {

    private static final double EPS = 1e-12;
    private static final double LOOSE = 1e-8;

    @Test
    void fixedProblemMatchesStaticFixedQuadrature() {
        FixedIntegral problem = Integrator.fixed().function(x -> x * x * x * x * x).bounds(0.0, 1.0)
                .points(3);

        GaussPool workspace = problem.alloc();
        double integral = problem.integrate(workspace);

        assertThat(integral).isCloseTo(1.0 / 6.0, offset(EPS));
        assertThat(workspace.arena()).isNotNull();
    }

    @Test
    void ruleIntegralProblemMatchesHermiteRule() {
        double integral = Integrator.weighted().function(x -> 1.0)
                .points(1)
                .rule(GaussRule.Hermite)
                .integrate();

        assertThat(integral).isCloseTo(Math.sqrt(Math.PI), offset(EPS));
    }

    @Test
    void adaptiveProblemAllocAndSolveReuseWorkspace() {
        AdaptiveIntegral problem = Integrator.adaptive().function(Math::sin).bounds(0.0, Math.PI)
                .tolerances(1e-12, 1e-12);

        AdaptivePool workspace = problem.alloc();
        double[] arena = workspace.arena();
        Quadrature result = problem.integrate(workspace);

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(2.0, offset(1e-12));
        assertThat(workspace.arena()).isSameAs(arena);
    }

    @Test
    void typedWorkspacesAreNoLongerLegacyPools() {
        assertThat(new GaussPool()).isNotNull();
        assertThat(new AdaptivePool()).isNotNull();
    }

    @Test
    void principalValueProblemMatchesSymmetricConstantCase() {
        Quadrature result = Integrator.principalValue()
                .function(x -> 1.0).bounds(0.0, 1.0).pole(0.5)
                .tolerances(1e-12, 1e-12)
                .integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(0.0, offset(EPS));
    }

    @Test
    void endpointSingularProblemMatchesUnitWeightCase() {
        Quadrature result = Integrator.endpointSingular(EndpointOpts.ALGEBRAIC)
                .function(x -> 1.0).bounds(0.0, 1.0).exponents(-0.5, 0.0)
                .tolerances(1e-10, 1e-10)
                .integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(2.0, offset(1e-8));
    }

    @Test
    void oscillatoryProblemUpperMatchesDecayingExponential() {
        Quadrature result = Integrator.oscillatory(OscillatoryOpts.COS_UPPER)
                .function(x -> Math.exp(-2.5 * x)).lowerBound(0.0).omega(2.3)
                .tolerances(1e-10, 1e-10)
                .integrate();

        double expected = 2.5 / (2.5 * 2.5 + 2.3 * 2.3);
        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(expected, offset(1e-9));
    }

    @Test
    void fixedProblemRequiresPointsBeforeSolve() {
        FixedIntegral problem = Integrator.fixed().function(Math::sin).bounds(0.0, 1.0);

        assertThatThrownBy(problem::integrate)
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("Missing required parameter: points");
    }

    @Test
    void fixedLegendreIsExactForDegreeFivePolynomial() {
        double integral = Integrator.fixed().function(x -> x * x * x * x * x).bounds(0.0, 1.0).points(3).integrate();

        assertThat(integral).isCloseTo(1.0 / 6.0, offset(EPS));
    }

    @Test
    void canonicalLegendreRuleIntegratesOnMinusOneToOne() {
        double integral = Integrator.weighted().function(x -> x * x).points(2).rule(GaussRule.Legendre).integrate();

        assertThat(integral).isCloseTo(2.0 / 3.0, offset(EPS));
    }

    @Test
    void hermiteRuleIntegratesConstantWeightExactly() {
        double integral = Integrator.weighted().function(x -> 1.0).points(1).rule(GaussRule.Hermite).integrate();

        assertThat(integral).isCloseTo(Math.sqrt(Math.PI), offset(EPS));
    }

    @Test
    void laguerreRuleIntegratesLinearMomentExactly() {
        double integral = Integrator.weighted().function(x -> x).points(1).rule(GaussRule.Laguerre).integrate();

        assertThat(integral).isCloseTo(1.0, offset(EPS));
    }

    @Test
    void jacobiRuleIntegratesChebyshevWeightExactly() {
        double integral = Integrator.weighted().function(x -> 1.0).points(1).rule(new JacobiRule(-0.5, -0.5)).integrate();

        assertThat(integral).isCloseTo(Math.PI, offset(EPS));
    }

    @Test
    void jacobiRuleRejectsInvalidExponent() {
        assertThatThrownBy(() -> new JacobiRule(-1.0, 0.0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("alpha must be > -1");
    }

    @Test
    void adaptiveMatchesSineIntegral() {
        Quadrature result = Integrator.adaptive().function(Math::sin).bounds(0.0, Math.PI).tolerances(1e-12, 1e-12).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(2.0, offset(1e-12));
        assertThat(result.getEstimatedError()).isLessThan(1e-8);
        assertThat(result.getEvaluations()).isPositive();
    }

    @Test
    void adaptiveImproperUpperMatchesDecayingExponential() {
        Quadrature result = Integrator.improper(ImproperOpts.UPPER)
                .function(x -> Math.exp(-x)).lowerBound(0.0)
                .tolerances(1e-10, 1e-10).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(1.0, offset(1e-9));
    }

    @Test
    void adaptiveImproperWholeLineMatchesLorentzianIntegral() {
        Quadrature result = Integrator.improper(ImproperOpts.WHOLE_LINE)
                .function(x -> 1.0 / (1.0 + x * x))
                .tolerances(1e-10, 1e-10).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(Math.PI, offset(1e-9));
    }

    @Test
    void oscillatorySineFiniteMatchesReferenceFormula() {
        double omega = Math.pow(2.0, 3.4);
        Quadrature result = Integrator.oscillatory(OscillatoryOpts.SIN)
                .function(x -> Math.exp(20.0 * (x - 1.0))).lowerBound(0.0).upperBound(1.0).omega(omega)
                .tolerances(1e-11, 1e-11).integrate();

        double expected = (20.0 * Math.sin(omega)
            - omega * Math.cos(omega)
            + omega * Math.exp(-20.0))
            / (400.0 + omega * omega);

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(expected, offset(1e-9));
    }

    @Test
    void oscillatoryCosFiniteMatchesReferenceFormula() {
        double omega = 7.5;
        Quadrature result = Integrator.oscillatory(OscillatoryOpts.COS)
                .function(x -> Math.exp(20.0 * (x - 1.0))).lowerBound(0.0).upperBound(1.0).omega(omega)
                .tolerances(1e-11, 1e-11).integrate();

        double expected = (20.0 * Math.cos(omega)
            + omega * Math.sin(omega)
            - 20.0 * Math.exp(-20.0))
            / (400.0 + omega * omega);

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(expected, offset(1e-9));
    }

    @Test
    void oscillatorySineUpperMatchesDecayingExponential() {
        Quadrature result = Integrator.oscillatory(OscillatoryOpts.SIN_UPPER)
                .function(x -> Math.exp(-4.0 * x)).lowerBound(0.0).omega(3.0)
                .tolerances(1e-10, 1e-10).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(3.0 / 25.0, offset(1e-9));
    }

    @Test
    void oscillatoryCosUpperMatchesDecayingExponential() {
        Quadrature result = Integrator.oscillatory(OscillatoryOpts.COS_UPPER)
                .function(x -> Math.exp(-2.5 * x)).lowerBound(0.0).omega(2.3)
                .tolerances(1e-10, 1e-10).integrate();

        double expected = 2.5 / (2.5 * 2.5 + 2.3 * 2.3);
        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(expected, offset(1e-9));
    }

    @Test
    void oscillatorySinUpperWithZeroFrequencyIsExactlyZero() {
        Quadrature result = Integrator.oscillatory(OscillatoryOpts.SIN_UPPER)
                .function(x -> Math.exp(-x)).lowerBound(0.0).omega(0.0)
                .tolerances(1e-10, 1e-10).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isZero();
        assertThat(result.getEvaluations()).isZero();
    }

    @Test
    void oscillatoryUpperReportsCycleLimit() {
        Quadrature result = Integrator.oscillatory(OscillatoryOpts.COS_UPPER)
                .function(x -> Math.exp(-0.01 * x)).lowerBound(0.0).omega(1.0)
                .tolerances(1e-12, 1e-12)
                .maxCycles(1)
                .integrate();

        assertThat(result.getStatus()).isEqualTo(Quadrature.Status.MAX_CYCLES_REACHED);
        assertThat(result.isSuccessful()).isFalse();
    }

    @Test
    void oscillatoryRejectsNonFiniteFrequency() {
        assertThatThrownBy(() -> Integrator.oscillatory(OscillatoryOpts.COS)
                .function(Math::sin).lowerBound(0.0).upperBound(1.0).omega(Double.NaN)
                .tolerances(1e-8, 1e-8).integrate())
            .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("omega must be finite");
    }

    @Test
    void endpointSingularMatchesSciPyReferenceCase() {
        double a = 1.5;
        Quadrature result = Integrator.endpointSingular(EndpointOpts.ALGEBRAIC)
                .function(x -> 1.0 / (1.0 + x + Math.pow(2.0, -a)))
                .bounds(-1.0, 1.0).exponents(-0.5, -0.5)
                .tolerances(1e-10, 1e-10).integrate();

        double expected = Math.PI / Math.sqrt(Math.pow(1.0 + Math.pow(2.0, -a), 2.0) - 1.0);
        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(expected, offset(1e-8));
    }

    @Test
    void endpointSingularLogLeftMatchesUnitIntervalConstant() {
        Quadrature result = Integrator.endpointSingular(EndpointOpts.LOG_LEFT)
                .function(x -> 1.0).bounds(0.0, 1.0).exponents(0.0, 0.0)
                .tolerances(1e-10, 1e-10).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(-1.0, offset(1e-8));
    }

    @Test
    void endpointSingularLogRightMatchesUnitIntervalConstant() {
        Quadrature result = Integrator.endpointSingular(EndpointOpts.LOG_RIGHT)
                .function(x -> 1.0).bounds(0.0, 1.0).exponents(0.0, 0.0)
                .tolerances(1e-10, 1e-10).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(-1.0, offset(1e-8));
    }

    @Test
    void endpointSingularDoubleLogMatchesKnownIntegral() {
        Quadrature result = Integrator.endpointSingular(EndpointOpts.LOG_BOTH)
                .function(x -> 1.0).bounds(0.0, 1.0).exponents(0.0, 0.0)
                .tolerances(1e-10, 1e-10).integrate();

        double expected = 2.0 - Math.PI * Math.PI / 6.0;
        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(expected, offset(1e-8));
    }

    @Test
    void endpointSingularRejectsInvalidExponent() {
        assertThatThrownBy(() -> Integrator.endpointSingular(EndpointOpts.ALGEBRAIC)
                .function(x -> 1.0).bounds(0.0, 1.0).exponents(-1.0, 0.0)
                .tolerances(1e-8, 1e-8).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("alpha must be > -1");
    }

    @Test
    void principalValueMatchesSciPyReferenceCase() {
        double a = 0.4;
        Quadrature result = Integrator.principalValue()
                .function(x -> Math.pow(2.0, -a) / ((x - 1.0) * (x - 1.0) + Math.pow(4.0, -a)))
                .bounds(0.0, 5.0).pole(2.0)
                .tolerances(1e-10, 1e-10).integrate();

        double expected = ((Math.pow(2.0, -0.4) * Math.log(1.5)
            - Math.pow(2.0, -1.4) * Math.log((Math.pow(4.0, -a) + 16.0) / (Math.pow(4.0, -a) + 1.0))
            - Math.atan(Math.pow(2.0, a + 2.0))
            - Math.atan(Math.pow(2.0, a)))
            / (Math.pow(4.0, -a) + 1.0));

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(expected, offset(2e-8));
    }

    @Test
    void principalValueOfConstantOnSymmetricIntervalIsZero() {
        Quadrature result = Integrator.principalValue()
                .function(x -> 1.0).bounds(0.0, 1.0).pole(0.5)
                .tolerances(1e-12, 1e-12).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(0.0, offset(EPS));
    }

    @Test
    void principalValueOutsideIntervalMatchesOrdinaryWeightedIntegral() {
        Quadrature result = Integrator.principalValue()
                .function(Math::sin).bounds(0.0, 1.0).pole(2.0)
                .tolerances(1e-12, 1e-12).integrate();
        Quadrature reference = Integrator.adaptive().function(x -> Math.sin(x) / (x - 2.0)).bounds(0.0, 1.0).tolerances(1e-12, 1e-12).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(reference.getValue(), offset(1e-12));
    }

    @Test
    void principalValueRejectsEndpointPole() {
        assertThatThrownBy(() -> Integrator.principalValue()
                .function(Math::sin).bounds(0.0, 1.0).pole(0.0)
                .tolerances(1e-8, 1e-8).integrate())
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("pole must not coincide");
    }

    @Test
    void adaptiveRejectsZeroTolerances() {
        assertThatThrownBy(() -> Integrator.adaptive().function(Math::sin).bounds(0.0, 1.0).tolerances(0.0, 0.0).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must not both be zero");
    }

    @Test
    void adaptiveReportsSubdivisionLimit() {
        Quadrature result = Integrator.adaptive().function(x -> x < 0.1 ? 0.0 : 1.0).bounds(0.0, 1.0)
                .tolerances(1e-12, 1e-12)
                .maxSubdivisions(1)
                .integrate();

        assertThat(result.getStatus()).isEqualTo(Quadrature.Status.MAX_SUBDIVISIONS_REACHED);
        assertThat(result.isSuccessful()).isFalse();
    }

    @Test
    void adaptiveWithBreakpointsHandlesDiscontinuity() {
        Quadrature result = Integrator.adaptive().function(x -> x < 0.0 ? 0.0 : 1.0).bounds(-1.0, 1.0).breakpoints(0.0).tolerances(1e-12, 1e-12).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(1.0, offset(EPS));
    }

    @Test
    void adaptiveBreakpointsRejectOutOfRangePoint() {
        assertThatThrownBy(() -> Integrator.adaptive().function(Math::sin).bounds(0.0, 1.0).breakpoints(1.0).tolerances(1e-8, 1e-8).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("strictly inside");
    }

    @Test
    void improperUpperRejectsNonFiniteLowerBound() {
        assertThatThrownBy(() -> Integrator.improperFixed(ImproperOpts.UPPER)
                .function(x -> Math.exp(-x)).lowerBound(Double.NaN).points(16).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("min must be finite");
    }

    @Test
    void improperUpperMatchesDecayingExponential() {
        double integral = Integrator.improperFixed(ImproperOpts.UPPER)
                .function(x -> Math.exp(-x)).lowerBound(0.0).points(32)
                .integrate().getValue();

        assertThat(integral).isCloseTo(1.0, offset(LOOSE));
    }

    @Test
    void improperLowerMatchesDecayingExponential() {
        double integral = Integrator.improperFixed(ImproperOpts.LOWER)
                .function(Math::exp).upperBound(0.0).points(32)
                .integrate().getValue();

        assertThat(integral).isCloseTo(1.0, offset(LOOSE));
    }

    @Test
    void improperWholeLineMatchesLorentzianIntegral() {
        double integral = Integrator.improperFixed(ImproperOpts.WHOLE_LINE)
                .function(x -> 1.0 / (1.0 + x * x)).points(48)
                .integrate().getValue();

        assertThat(integral).isCloseTo(Math.PI, offset(LOOSE));
    }

    @Test
    void fixedRejectsInfiniteBounds() {
        assertThatThrownBy(() -> Integrator.fixed().function(Math::sin).bounds(0.0, Double.POSITIVE_INFINITY).points(8).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("finite bounds");
    }

    @Test
    void legacyWorkspaceStillBridgesRuleCalls() {
        GaussPool pool = new GaussPool();

        double fixed = Integrator.fixed().function(x -> x * x).bounds(0.0, 1.0).points(8).integrate(pool);
        double weighted = Integrator.weighted().function(x -> x).points(4).rule(GaussRule.Laguerre).integrate(pool);

        assertThat(fixed).isCloseTo(1.0 / 3.0, offset(EPS));
        assertThat(weighted).isCloseTo(1.0, offset(EPS));
    }

    @Test
    void legacyWorkspaceStillBridgesAdaptiveCalls() {
        AdaptivePool pool = new AdaptivePool();

        Quadrature result = Integrator.adaptive().function(Math::sin).bounds(0.0, Math.PI)
                .tolerances(1e-12, 1e-12)
                .integrate(pool);

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(2.0, offset(1e-12));
    }

    @Test
    void rulePoolReusesArenaOnSmallerRequests() {
        GaussPool pool = new GaussPool();

        Integrator.fixed().function(x -> x * x).bounds(0.0, 1.0).points(8).integrate(pool);
        double[] arena = pool.arena();

        Integrator.weighted().function(x -> x).points(4).rule(GaussRule.Laguerre).integrate(pool);

        assertThat(pool.arena()).isSameAs(arena);
        assertThat(pool.arena().length).isGreaterThanOrEqualTo(96);
    }

    @Test
    void adaptivePoolReusesArenaOnRepeatedAdaptiveCalls() {
        AdaptivePool pool = new AdaptivePool();

        Quadrature first = Integrator.adaptive().function(Math::sin).bounds(0.0, Math.PI)
                .tolerances(1e-12, 1e-12).integrate(pool);
        double[] arena = pool.arena();
        Quadrature second = Integrator.adaptive().function(x -> x * x).bounds(0.0, 1.0)
                .tolerances(1e-10, 1e-10).integrate(pool);

        assertThat(first.isSuccessful()).isTrue();
        assertThat(second.isSuccessful()).isTrue();
        assertThat(pool.arena()).isSameAs(arena);
        assertThat(pool.arena().length).isGreaterThanOrEqualTo(1024);
    }

    @Test
    void endpointSingularReusesGaussPool() {
        GaussPool pool = new GaussPool();

        Quadrature first = Integrator.endpointSingular(EndpointOpts.ALGEBRAIC)
                .function(x -> 1.0 / (1.0 + x)).bounds(0.0, 1.0).exponents(-0.5, 0.0)
                .tolerances(1e-10, 1e-10).integrate(pool);
        double[] arena = pool.arena();
        Quadrature second = Integrator.endpointSingular(EndpointOpts.ALGEBRAIC)
                .function(x -> 1.0).bounds(0.0, 1.0).exponents(-0.5, 0.0)
                .tolerances(1e-10, 1e-10).integrate(pool);

        assertThat(first.isSuccessful()).isTrue();
        assertThat(second.isSuccessful()).isTrue();
        assertThat(second.getValue()).isCloseTo(2.0, offset(1e-8));
        assertThat(pool.arena()).isSameAs(arena);
    }

    @Test
    void fixedRejectsNullRule() {
        assertThatThrownBy(() -> Integrator.fixed().function(Math::sin).bounds(0.0, 1.0).points(8).rule(null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("rule must not be null");
    }

    @Test
    void fixedRejectsNaturalDomainRule() {
        assertThatThrownBy(() -> Integrator.fixed().function(Math::sin).bounds(0.0, 1.0).points(8).rule(GaussRule.Hermite))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("GaussRule.Legendre");
    }

    // -----------------------------------------------------------------------
    // ImproperIntegral.Adaptive LOWER branch
    // -----------------------------------------------------------------------

    @Test
    void adaptiveImproperLowerMatchesDecayingExponential() {
        // ∫_{-∞}^{0} e^x dx = 1
        Quadrature result = Integrator.improper(ImproperOpts.LOWER)
                .function(Math::exp).upperBound(0.0)
                .tolerances(1e-10, 1e-10).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(1.0, offset(1e-9));
    }

    // -----------------------------------------------------------------------
    // adaptive MAX_EVALUATIONS_REACHED
    // -----------------------------------------------------------------------

    @Test
    void adaptiveReportsEvaluationLimit() {
        // Highly oscillatory integrand forces many evaluations
        Quadrature result = Integrator.adaptive()
                .function(x -> Math.sin(1000.0 * x)).bounds(0.0, 1.0)
                .tolerances(1e-14, 1e-14)
                .maxEvaluations(30)
                .integrate();

        assertThat(result.getStatus()).isEqualTo(Quadrature.Status.MAX_EVALUATIONS_REACHED);
        assertThat(result.isSuccessful()).isFalse();
    }

    // -----------------------------------------------------------------------
    // endpointSingular MAX_REFINEMENTS_REACHED
    // -----------------------------------------------------------------------

    @Test
    void endpointSingularReportsRefinementLimit() {
        // Very tight tolerance forces many refinement levels
        Quadrature result = Integrator.endpointSingular(EndpointOpts.ALGEBRAIC)
                .function(x -> Math.sin(100.0 * x)).bounds(0.0, 1.0).exponents(-0.5, -0.5)
                .tolerances(1e-15, 1e-15)
                .maxRefinements(1)
                .integrate();

        assertThat(result.getStatus()).isEqualTo(Quadrature.Status.MAX_REFINEMENTS_REACHED);
        assertThat(result.isSuccessful()).isFalse();
    }

    // -----------------------------------------------------------------------
    // oscillatory MAX_EVALUATIONS_REACHED
    // -----------------------------------------------------------------------

    @Test
    void oscillatoryUpperReportsEvaluationLimit() {
        Quadrature result = Integrator.oscillatory(OscillatoryOpts.COS_UPPER)
                .function(x -> Math.exp(-0.001 * x)).lowerBound(0.0).omega(1.0)
                .tolerances(1e-14, 1e-14)
                .maxEvaluations(30)
                .integrate();

        assertThat(result.getStatus()).isEqualTo(Quadrature.Status.MAX_EVALUATIONS_REACHED);
        assertThat(result.isSuccessful()).isFalse();
    }

    // -----------------------------------------------------------------------
    // principalValue ABNORMAL_TERMINATION when f(pole) is non-finite
    // -----------------------------------------------------------------------

    @Test
    void principalValueReturnsAbnormalWhenFunctionNonFiniteAtPole() {
        Quadrature result = Integrator.principalValue()
                .function(x -> Double.NaN).bounds(0.0, 1.0).pole(0.5)
                .tolerances(1e-8, 1e-8).integrate();

        assertThat(result.getStatus()).isEqualTo(Quadrature.Status.ABNORMAL_TERMINATION);
        assertThat(result.isSuccessful()).isFalse();
    }

    // -----------------------------------------------------------------------
    // scipy test_indefinite: ∫₀^∞ -e^{-x}·ln(x) dx = γ (Euler-Mascheroni)
    // -----------------------------------------------------------------------

    @Test
    void adaptiveImproperUpperMatchesEulerMascheroniConstant() {
        // ∫₀^∞ -e^{-x}·ln(x) dx = γ ≈ 0.5772156649015329
        double gamma = 0.5772156649015329;
        Quadrature result = Integrator.improper(ImproperOpts.UPPER)
                .function(x -> -Math.exp(-x) * Math.log(x)).lowerBound(0.0)
                .tolerances(1e-8, 1e-8).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(gamma, offset(1e-7));
    }

    // -----------------------------------------------------------------------
    // scipy test_singular: breakpoints at discontinuities
    // -----------------------------------------------------------------------

    @Test
    void adaptiveWithMultipleBreakpointsHandlesPiecewiseFunction() {
        // f(x) = sin(x) for x ∈ (0, 2.5], exp(-x) for x ∈ (2.5, 5], 0 otherwise
        // ∫₀^10 f(x) dx = 1 - cos(2.5) + exp(-2.5) - exp(-5)
        double expected = 1.0 - Math.cos(2.5) + Math.exp(-2.5) - Math.exp(-5.0);
        Quadrature result = Integrator.adaptive()
                .function(x -> {
                    if (x > 0 && x <= 2.5) return Math.sin(x);
                    if (x > 2.5 && x <= 5.0) return Math.exp(-x);
                    return 0.0;
                })
                .bounds(0.0, 10.0)
                .breakpoints(2.5, 5.0)
                .tolerances(1e-10, 1e-10).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(expected, offset(1e-8));
    }

    // -----------------------------------------------------------------------
    // scipy test_cosine_weighted_infinite via substitution x→-t
    // -----------------------------------------------------------------------

    @Test
    void oscillatoryCosLowerViaSubstitution() {
        // ∫_{-∞}^{0} e^{2.5x}·cos(2.3x) dx = 2.5/(2.5²+2.3²)
        // via x→-t: ∫_{0}^{+∞} e^{-2.5t}·cos(2.3t) dt  (same value, cos is even)
        double a = 2.5, ome = 2.3;
        double expected = a / (a * a + ome * ome);
        Quadrature result = Integrator.oscillatory(OscillatoryOpts.COS_UPPER)
                .function(t -> Math.exp(-a * t)).lowerBound(0.0).omega(ome)
                .tolerances(1e-10, 1e-10).integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(expected, offset(1e-9));
    }
}
