/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad;

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
}
