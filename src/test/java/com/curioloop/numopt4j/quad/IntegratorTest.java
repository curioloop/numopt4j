/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad;

import com.curioloop.numopt4j.quad.de.DEPool;
import com.curioloop.numopt4j.quad.de.DoubleExponentialIntegral;
import com.curioloop.numopt4j.quad.de.DEOpts;
import com.curioloop.numopt4j.quad.sampled.SampledRule;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.data.Offset.offset;

class IntegratorTest {

    private static final double EPS = 1e-12;

    @Test
    void cumulativeTrapezoidalAccumulatesEquallySpacedSamples() {
        double[] y = {0.0, 1.0, 2.0};

        double[] cumulative = Integrator.cumulative(SampledRule.TRAPEZOIDAL).samples(y, 1.0).integrate();

        assertThat(cumulative).containsExactly(0.0, 0.5, 2.0);
    }

    @Test
    void cumulativeSimpsonMatchesQuadraticPrefixes() {
        double[] y = {0.0, 1.0, 4.0, 9.0, 16.0};

        double[] cumulative = Integrator.cumulative(SampledRule.SIMPSON).samples(y, 1.0).integrate();

        assertThat(cumulative[0]).isCloseTo(0.0, offset(EPS));
        assertThat(cumulative[1]).isCloseTo(1.0 / 3.0, offset(EPS));
        assertThat(cumulative[2]).isCloseTo(8.0 / 3.0, offset(EPS));
        assertThat(cumulative[3]).isCloseTo(9.0, offset(EPS));
        assertThat(cumulative[4]).isCloseTo(64.0 / 3.0, offset(EPS));
    }

    @Test
    void cumulativeSimpsonMatchesIrregularQuadraticPrefixes() {
        double[] x = {0.0, 0.5, 2.0};
        double[] y = {0.0, 0.25, 4.0};

        double[] cumulative = Integrator.cumulative(SampledRule.SIMPSON).samples(x, y).integrate();

        assertThat(cumulative[0]).isCloseTo(0.0, offset(EPS));
        assertThat(cumulative[1]).isCloseTo(1.0 / 24.0, offset(EPS));
        assertThat(cumulative[2]).isCloseTo(8.0 / 3.0, offset(EPS));
    }

    @Test
    void trapezoidalIsExactForLinearSamples() {
        double[] x = {0.0, 0.5, 1.5};
        double[] y = {1.0, 2.0, 4.0};

        double integral = Integrator.sampled(SampledRule.TRAPEZOIDAL).samples(x, y).integrate();

        assertThat(integral).isCloseTo(3.75, offset(EPS));
    }

    @Test
    void simpsonIsExactForEquallySpacedCubic() {
        double[] y = {0.0, 1.0, 8.0, 27.0, 64.0};

        double integral = Integrator.sampled(SampledRule.SIMPSON).samples(y, 1.0).integrate();

        assertThat(integral).isCloseTo(64.0, offset(EPS));
    }

    @Test
    void simpsonIsExactForIrregularQuadratic() {
        double[] x = {0.0, 0.5, 2.0};
        double[] y = {0.0, 0.25, 4.0};

        double integral = Integrator.sampled(SampledRule.SIMPSON).samples(x, y).integrate();

        assertThat(integral).isCloseTo(8.0 / 3.0, offset(EPS));
    }

    @Test
    void rombergExtrapolatesQuadraticSamples() {
        double dx = 0.25;
        double[] y = {0.0, 0.0625, 0.25, 0.5625, 1.0};

        double integral = Integrator.sampled(SampledRule.ROMBERG).samples(y, dx).integrate();

        assertThat(integral).isCloseTo(1.0 / 3.0, offset(EPS));
    }

    @Test
    void trapezoidalRejectsUnsortedCoordinates() {
        double[] x = {0.0, 2.0, 1.0};
        double[] y = {1.0, 2.0, 3.0};

        assertThatThrownBy(() -> Integrator.sampled(SampledRule.TRAPEZOIDAL).samples(x, y).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("strictly increasing");
    }

    @Test
    void simpsonRejectsRepeatedCoordinates() {
        double[] x = {0.0, 1.0, 1.0};
        double[] y = {0.0, 1.0, 1.0};

        assertThatThrownBy(() -> Integrator.sampled(SampledRule.SIMPSON).samples(x, y).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("strictly increasing");
    }

    @Test
    void rombergRejectsInvalidSampleCount() {
        double[] y = {0.0, 0.25, 1.0, 2.25};

        assertThatThrownBy(() -> Integrator.sampled(SampledRule.ROMBERG).samples(y, 0.5).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("2^k + 1");
    }

    @Test
    void equallySpacedMethodsRejectNonPositiveSpacing() {
        double[] y = {0.0, 1.0, 4.0};

        assertThatThrownBy(() -> Integrator.sampled(SampledRule.TRAPEZOIDAL).samples(y, 0.0).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("dx must be > 0");
        assertThatThrownBy(() -> Integrator.sampled(SampledRule.SIMPSON).samples(y, -1.0).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("dx must be > 0");
        assertThatThrownBy(() -> Integrator.sampled(SampledRule.ROMBERG).samples(y, 0.0).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("dx must be > 0");
        assertThatThrownBy(() -> Integrator.cumulative(SampledRule.TRAPEZOIDAL).samples(y, 0.0).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("dx must be > 0");
        assertThatThrownBy(() -> Integrator.cumulative(SampledRule.SIMPSON).samples(y, -1.0).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("dx must be > 0");
    }

    // -----------------------------------------------------------------------
    // Simpson even-count end-panel correction
    // -----------------------------------------------------------------------

    @Test
    void simpsonEvenCountUsesEndPanelCorrection() {
        // 4 samples (even): ∫₀³ x² dx = 9, last panel uses corrected formula
        double[] y = {0.0, 1.0, 4.0, 9.0};

        double integral = Integrator.sampled(SampledRule.SIMPSON).samples(y, 1.0).integrate();

        assertThat(integral).isCloseTo(9.0, offset(1e-10));
    }

    @Test
    void simpsonIrregularEvenCountUsesEndPanelCorrection() {
        // 4 irregular samples: ∫₀³ x² dx = 9
        double[] x = {0.0, 1.0, 2.0, 3.0};
        double[] y = {0.0, 1.0, 4.0, 9.0};

        double integral = Integrator.sampled(SampledRule.SIMPSON).samples(x, y).integrate();

        assertThat(integral).isCloseTo(9.0, offset(1e-10));
    }

    // -----------------------------------------------------------------------
    // CumulativeIntegral n=2 boundary (trapezoidal and Simpson)
    // -----------------------------------------------------------------------

    @Test
    void cumulativeTrapezoidalHandlesTwoSamples() {
        double[] y = {0.0, 2.0};

        double[] cumulative = Integrator.cumulative(SampledRule.TRAPEZOIDAL).samples(y, 1.0).integrate();

        assertThat(cumulative).hasSize(2);
        assertThat(cumulative[0]).isCloseTo(0.0, offset(EPS));
        assertThat(cumulative[1]).isCloseTo(1.0, offset(EPS));
    }

    @Test
    void cumulativeSimpsonFallsBackToTrapezoidalForTwoSamples() {
        // n=2: Simpson falls back to trapezoidal for the single panel
        double[] y = {0.0, 2.0};

        double[] cumulative = Integrator.cumulative(SampledRule.SIMPSON).samples(y, 1.0).integrate();

        assertThat(cumulative).hasSize(2);
        assertThat(cumulative[0]).isCloseTo(0.0, offset(EPS));
        assertThat(cumulative[1]).isCloseTo(1.0, offset(EPS));
    }

    @Test
    void cumulativeSimpsonIrregularTwoSamples() {
        double[] x = {0.0, 2.0};
        double[] y = {1.0, 3.0};

        double[] cumulative = Integrator.cumulative(SampledRule.SIMPSON).samples(x, y).integrate();

        assertThat(cumulative[0]).isCloseTo(0.0, offset(EPS));
        assertThat(cumulative[1]).isCloseTo(4.0, offset(EPS)); // (1+3)/2 * 2
    }

    // -----------------------------------------------------------------------
    // scipy fixed_quad: ∫₀^1 x^{2n-1} dx = 1/(2n), exact for n-point rule
    // -----------------------------------------------------------------------

    @Test
    void fixedQuadIsExactForHighDegreePolynomial() {
        // n=4 points: exact for degree ≤ 7, so x^7 is exact
        // ∫₀^1 x^7 dx = 1/8
        double integral = Integrator.fixed()
                .function(x -> Math.pow(x, 7)).bounds(0.0, 1.0).points(4).integrate();

        assertThat(integral).isCloseTo(1.0 / 8.0, offset(EPS));
    }

    @Test
    void doubleExponentialTanhSinhMatchesPolynomialIntegral() {
        Quadrature result = Integrator.doubleExponential(DEOpts.TANH_SINH)
                .function(x -> x * x)
                .bounds(0.0, 2.0)
                .tolerance(1e-10)
                .integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(8.0 / 3.0, offset(1e-10));
    }

    @Test
    void doubleExponentialExpSinhMatchesExponentialTail() {
        Quadrature result = Integrator.doubleExponential(DEOpts.EXP_SINH)
                .function(x -> Math.exp(-x))
                .bounds(0.0, Double.POSITIVE_INFINITY)
                .tolerance(1e-10)
                .integrate();

        assertThat(result.isSuccessful()).isTrue();
        assertThat(result.getValue()).isCloseTo(1.0, offset(1e-10));
    }

    @Test
    void doubleExponentialBuilderAcceptsReusablePool() {
        DEPool pool = new DEPool();
        DoubleExponentialIntegral problem = Integrator.doubleExponential(DEOpts.TANH_SINH)
                .function(x -> x * x)
                .bounds(0.0, 2.0)
                .tolerance(1e-10);

        Quadrature first = problem.integrate(pool);
        Quadrature second = problem.integrate(pool);

        assertThat(first.isSuccessful()).isTrue();
        assertThat(second.isSuccessful()).isTrue();
        assertThat(first.getValue()).isCloseTo(8.0 / 3.0, offset(1e-10));
        assertThat(second.getValue()).isCloseTo(first.getValue(), offset(0.0));
    }

    // -----------------------------------------------------------------------
    // scipy romb_gh_3731: Romberg on cos(0.2x) over [0, 16]
    // -----------------------------------------------------------------------

    @Test
    void rombergMatchesCosineIntegral() {
        // ∫₀^16 cos(0.2x) dx = sin(3.2) / 0.2 ≈ 4.7696...
        int n = 16;
        double dx = 1.0;
        double[] y = new double[n + 1];
        for (int i = 0; i <= n; i++) y[i] = Math.cos(0.2 * i * dx);

        double integral = Integrator.sampled(SampledRule.ROMBERG).samples(y, dx).integrate();
        double expected = Math.sin(0.2 * n * dx) / 0.2;

        assertThat(integral).isCloseTo(expected, offset(1e-8));
    }

    // -----------------------------------------------------------------------
    // SampledIntegral function-based path (zero allocation)
    // -----------------------------------------------------------------------

    @Test
    void sampledTrapezoidalFunctionMatchesSineIntegral() {
        // ∫₀^π sin(x) dx = 2
        double result = Integrator.sampled(SampledRule.TRAPEZOIDAL)
                .function(Math::sin, 1001, 0.0, Math.PI).integrate();
        assertThat(result).isCloseTo(2.0, offset(1e-5));
    }

    @Test
    void sampledSimpsonFunctionMatchesSineIntegral() {
        // ∫₀^π sin(x) dx = 2, Simpson is O(h⁴) so 101 points is very accurate
        double result = Integrator.sampled(SampledRule.SIMPSON)
                .function(Math::sin, 101, 0.0, Math.PI).integrate();
        assertThat(result).isCloseTo(2.0, offset(1e-7));
    }

    @Test
    void sampledRombergFunctionMatchesCubicIntegral() {
        // ∫₀^1 x³ dx = 0.25, Romberg requires n = 2^k + 1
        double result = Integrator.sampled(SampledRule.ROMBERG)
                .function(x -> x * x * x, 17, 0.0, 1.0).integrate();
        assertThat(result).isCloseTo(0.25, offset(EPS));
    }

    @Test
    void sampledFunctionRejectsInvalidBounds() {
        assertThatThrownBy(() -> Integrator.sampled(SampledRule.TRAPEZOIDAL)
                .function(Math::sin, 10, 1.0, 0.0).integrate())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("a < b");
    }
}
