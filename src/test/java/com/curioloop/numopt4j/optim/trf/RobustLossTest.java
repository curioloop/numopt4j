/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

import com.curioloop.numopt4j.optim.Optimization;
import org.junit.jupiter.api.Test;

import com.curioloop.numopt4j.optim.Multivariate;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Tests verifying that robust loss functions improve fitting accuracy
 * in the presence of outliers compared to standard least-squares (LINEAR).
 *
 * <p>Scenario: fit a line y = a*x + b to data that contains a few large outliers.
 * The true parameters are a=2.0, b=1.0. With LINEAR loss the outliers pull the
 * solution away from the truth; robust losses should recover it more accurately.</p>
 */
class RobustLossTest {

    // True parameters
    static final double TRUE_A = 2.0;
    static final double TRUE_B = 1.0;

    // Clean data: y = 2x + 1 for x = 0..9
    static final int M_CLEAN = 10;
    static final double[] X_DATA = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    static final double[] Y_CLEAN = new double[M_CLEAN];

    // Contaminated data: same clean data + 3 large outliers appended
    static final int M_OUTLIER = M_CLEAN + 3;
    static final double[] Y_OUTLIER = new double[M_OUTLIER];
    static final double[] X_OUTLIER = new double[M_OUTLIER];

    static {
        for (int i = 0; i < M_CLEAN; i++) {
            Y_CLEAN[i] = TRUE_A * X_DATA[i] + TRUE_B;
        }
        System.arraycopy(X_DATA, 0, X_OUTLIER, 0, M_CLEAN);
        System.arraycopy(Y_CLEAN, 0, Y_OUTLIER, 0, M_CLEAN);
        // Append 3 outliers far from the true line
        X_OUTLIER[10] = 2.0;  Y_OUTLIER[10] = 50.0;   // outlier: true y=5, observed=50
        X_OUTLIER[11] = 5.0;  Y_OUTLIER[11] = -30.0;  // outlier: true y=11, observed=-30
        X_OUTLIER[12] = 7.0;  Y_OUTLIER[12] = 80.0;   // outlier: true y=15, observed=80
    }

    /** Residual function for linear model y = a*x + b on the contaminated dataset. */
    static Multivariate.Objective residualFn() {
        return (c, n, r, m) -> {
            for (int i = 0; i < M_OUTLIER; i++) {
                r[i] = Y_OUTLIER[i] - (c[0] * X_OUTLIER[i] + c[1]);
            }
        };
    }

    /** Solve with given loss and return the solution. */
    static double[] solve(RobustLoss loss, double lossScale) {
        Optimization r = new TRFProblem()
                .residuals(residualFn(), M_OUTLIER)
                .initialPoint(1.0, 0.0)
                .loss(loss)
                .lossScale(lossScale)
                .functionTolerance(1e-10)
                .parameterTolerance(1e-10)
                .maxEvaluations(2000)
                .solve();
        assertThat(r.isSuccessful()).isTrue();
        return r.getSolution();
    }

    // ── Baseline: clean data with LINEAR should recover exact parameters ──────

    @Test
    void linear_cleanData_recoversExactParameters() {
        Multivariate.Objective fn = (c, n, r, m) -> {
            for (int i = 0; i < M_CLEAN; i++) {
                r[i] = Y_CLEAN[i] - (c[0] * X_DATA[i] + c[1]);
            }
        };
        Optimization r = new TRFProblem()
                .residuals(fn, M_CLEAN)
                .initialPoint(1.0, 0.0)
                .loss(RobustLoss.LINEAR)
                .functionTolerance(1e-12)
                .parameterTolerance(1e-12)
                .solve();
        assertThat(r.isSuccessful()).isTrue();
        assertThat(r.getSolution()[0]).isCloseTo(TRUE_A, within(1e-6));
        assertThat(r.getSolution()[1]).isCloseTo(TRUE_B, within(1e-6));
    }

    // ── LINEAR is pulled away from truth by outliers ──────────────────────────

    @Test
    void linear_outlierData_degradedAccuracy() {
        double[] sol = solve(RobustLoss.LINEAR, 1.0);
        double errA = Math.abs(sol[0] - TRUE_A);
        double errB = Math.abs(sol[1] - TRUE_B);
        // LINEAR is significantly biased by the 3 large outliers
        assertThat(errA + errB).isGreaterThan(1.0);
    }

    // ── Robust losses recover true parameters despite outliers ────────────────

    @Test
    void softL1_outlierData_recoversParameters() {
        // lossScale=2.0: outlier residuals ~40-65 >> scale, strongly down-weighted
        double[] sol = solve(RobustLoss.SOFT_L1, 2.0);
        assertThat(sol[0]).isCloseTo(TRUE_A, within(0.1));
        assertThat(sol[1]).isCloseTo(TRUE_B, within(0.5));
    }

    @Test
    void huber_outlierData_recoversParameters() {
        double[] sol = solve(RobustLoss.HUBER, 2.0);
        assertThat(sol[0]).isCloseTo(TRUE_A, within(0.1));
        assertThat(sol[1]).isCloseTo(TRUE_B, within(0.5));
    }

    @Test
    void cauchy_outlierData_recoversParameters() {
        double[] sol = solve(RobustLoss.CAUCHY, 5.0);
        assertThat(sol[0]).isCloseTo(TRUE_A, within(0.1));
        assertThat(sol[1]).isCloseTo(TRUE_B, within(0.5));
    }

    @Test
    void arctan_outlierData_recoversParameters() {
        double[] sol = solve(RobustLoss.ARCTAN, 5.0);
        assertThat(sol[0]).isCloseTo(TRUE_A, within(0.1));
        assertThat(sol[1]).isCloseTo(TRUE_B, within(0.5));
    }

    // ── Robust losses are strictly better than LINEAR on this dataset ─────────

    @Test
    void robustLosses_allBetterThanLinear() {
        double[] linearSol = solve(RobustLoss.LINEAR, 1.0);
        double linearErr = Math.abs(linearSol[0] - TRUE_A) + Math.abs(linearSol[1] - TRUE_B);

        // Use lossScale=2.0 for soft_l1/huber (slower decay), 5.0 for cauchy/arctan (faster decay)
        double[] scales = {2.0, 2.0, 5.0, 5.0};
        RobustLoss[] losses = {RobustLoss.SOFT_L1, RobustLoss.HUBER, RobustLoss.CAUCHY, RobustLoss.ARCTAN};
        for (int k = 0; k < losses.length; k++) {
            double[] sol = solve(losses[k], scales[k]);
            double err = Math.abs(sol[0] - TRUE_A) + Math.abs(sol[1] - TRUE_B);
            assertThat(err)
                    .as("loss=%s should outperform LINEAR (linearErr=%.3f, err=%.3f)", losses[k], linearErr, err)
                    .isLessThan(linearErr);
        }
    }

    // ── lossScale sensitivity: too large weakens outlier suppression ──────────

    @Test
    void lossScale_tooLarge_degradesAccuracy() {
        // lossScale=100: outlier residuals ~40-65 << scale, so they are barely down-weighted
        double[] sol = solve(RobustLoss.SOFT_L1, 100.0);
        double err = Math.abs(sol[0] - TRUE_A) + Math.abs(sol[1] - TRUE_B);
        // With scale too large, all residuals look like inliers → worse than good scale
        double[] goodSol = solve(RobustLoss.SOFT_L1, 2.0);
        double goodErr = Math.abs(goodSol[0] - TRUE_A) + Math.abs(goodSol[1] - TRUE_B);
        assertThat(err).isGreaterThan(goodErr);
    }
}
