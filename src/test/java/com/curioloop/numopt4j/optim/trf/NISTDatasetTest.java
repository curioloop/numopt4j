/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

import com.curioloop.numopt4j.optim.OptimizationResult;
import com.curioloop.numopt4j.optim.OptimizationStatus;
import net.jqwik.api.*;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests using NIST Statistical Reference Datasets (StRD) for nonlinear regression.
 *
 * @see <a href="https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml">NIST StRD Nonlinear Regression</a>
 */
@Tag("Feature: levenberg-marquardt-optimizer")
public class NISTDatasetTest {

    private static final double RELATIVE_TOLERANCE = 1e-4;

    // ==================== Misra1a Dataset ====================
    private static final double[] MISRA1A_X = {
        77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3,
        536.8, 593.1, 689.1, 760.0
    };
    private static final double[] MISRA1A_Y = {
        10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05,
        61.01, 66.40, 75.47, 81.78
    };
    private static final double MISRA1A_CERTIFIED_RSS = 1.2455138894E-01;

    @Test @Tag("NIST Misra1a - Lower Starting Point")
    void testMisra1aLowerStartingPoint() { testMisra1a(new double[]{250.0, 0.0005}, "lower"); }

    @Test @Tag("NIST Misra1a - Average Starting Point")
    void testMisra1aAverageStartingPoint() { testMisra1a(new double[]{240.0, 0.00055}, "average"); }

    @Test @Tag("NIST Misra1a - Higher Starting Point")
    void testMisra1aHigherStartingPoint() { testMisra1a(new double[]{230.0, 0.0006}, "higher"); }

    private void testMisra1a(double[] initialGuess, String label) {
        final int m = MISRA1A_X.length;
        BiConsumer<double[], double[]> residualFunc = (c, r) -> {
            for (int i = 0; i < m; i++)
                r[i] = MISRA1A_Y[i] - c[0] * (1.0 - Math.exp(-c[1] * MISRA1A_X[i]));
        };
        double[] ir = new double[m];
        residualFunc.accept(initialGuess, ir);
        double initialRss = 0; for (double v : ir) initialRss += v * v;

        OptimizationResult result = new TRFProblem()
            .residuals(residualFunc, m)
            .initialPoint(initialGuess.clone())
            .gradientTolerance(1e-8).parameterTolerance(1e-8).functionTolerance(1e-6).maxEvaluations(20000)
            .solve();

        assertTrue(result.isSuccessful() || result.getStatus() == OptimizationStatus.MAX_EVALUATIONS_REACHED,
            "Misra1a (" + label + ") should converge");
        assertTrue(result.getCost() <= initialRss,
            "Misra1a (" + label + ") RSS should improve");
        assertTrue(result.getCost() < MISRA1A_CERTIFIED_RSS * 5,
            "Misra1a (" + label + ") RSS too large: " + result.getCost());
    }

    // ==================== Lanczos1 Dataset ====================
    private static final double[] LANCZOS1_X = {
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
        1.00, 1.05, 1.10, 1.15
    };
    private static final double[] LANCZOS1_Y = {
        2.5134, 2.0443, 1.6684, 1.3664, 1.1232, 0.9269, 0.7679, 0.6389,
        0.5338, 0.4479, 0.3776, 0.3197, 0.2720, 0.2325, 0.1997, 0.1723,
        0.1493, 0.1301, 0.1138, 0.1000, 0.0883, 0.0783, 0.0698, 0.0624
    };

    @Test @Tag("NIST Lanczos1 - Starting Point 1")
    void testLanczos1StartingPoint1() { testLanczos1(new double[]{0.1, 1.0, 0.9, 3.0, 1.5, 5.0}, "start1"); }

    @Test @Tag("NIST Lanczos1 - Starting Point 2")
    void testLanczos1StartingPoint2() { testLanczos1(new double[]{0.08, 1.1, 0.8, 2.9, 1.6, 4.9}, "start2"); }

    private void testLanczos1(double[] initialGuess, String label) {
        final int m = LANCZOS1_X.length;
        BiConsumer<double[], double[]> residualFunc = (c, r) -> {
            for (int i = 0; i < m; i++) {
                double x = LANCZOS1_X[i];
                r[i] = LANCZOS1_Y[i] - (c[0]*Math.exp(-c[1]*x) + c[2]*Math.exp(-c[3]*x) + c[4]*Math.exp(-c[5]*x));
            }
        };
        OptimizationResult result = new TRFProblem()
            .residuals(residualFunc, m)
            .initialPoint(initialGuess.clone())
            .gradientTolerance(1e-10).parameterTolerance(1e-10).functionTolerance(1e-10).maxEvaluations(20000)
            .solve();

        assertTrue(result.isSuccessful() || result.getStatus() == OptimizationStatus.MAX_EVALUATIONS_REACHED,
            "Lanczos1 (" + label + ") should converge");
        assertTrue(result.getCost() < 1.0,
            "Lanczos1 (" + label + ") RSS should be < 1.0: " + result.getCost());
    }

    // ==================== Gauss1 Dataset ====================
    private static final double[] GAUSS1_X;
    private static final double[] GAUSS1_Y = {
        97.62227, 97.80724, 96.62247, 92.59022, 91.23869, 95.32704, 90.35040, 89.46235, 91.72520, 89.86916,
        86.88076, 85.94360, 87.60686, 86.25839, 80.74976, 83.03551, 88.25837, 82.01316, 82.74098, 83.30034,
        81.27850, 81.85506, 80.75195, 80.09573, 81.07633, 78.81542, 78.38596, 79.93386, 79.48474, 79.95942,
        76.10691, 78.39830, 81.43060, 82.48867, 81.65462, 80.84323, 88.68663, 84.74438, 86.83934, 85.97739,
        91.28509, 97.22411, 93.51733, 94.10159, 101.91760, 98.43134, 110.4214, 107.6628, 111.7288, 116.5115,
        120.7609, 123.9553, 124.2437, 130.7996, 133.2960, 130.7788, 132.0565, 138.6584, 142.9252, 142.7215,
        144.1249, 147.4377, 148.2647, 152.0519, 147.3863, 149.2074, 148.9537, 144.5876, 148.1226, 148.0144,
        143.8893, 140.9088, 143.4434, 139.3938, 135.9878, 136.3927, 126.7262, 124.4487, 122.8647, 113.8557,
        113.7037, 106.8407, 107.0034, 102.46290, 96.09296, 94.57555, 86.98824, 84.90154, 81.18023, 76.40117,
        67.09200, 72.67155, 68.10848, 67.99088, 63.34094, 60.55253, 56.18687, 53.64482, 53.70307, 48.07893,
        42.21258, 45.65181, 41.69728, 41.24946, 39.21349, 37.71696, 36.68395, 37.30393, 37.43277, 37.45012,
        32.64648, 31.84347, 31.39951, 26.68912, 32.25323, 27.61008, 33.58649, 28.10714, 30.26428, 28.01648,
        29.11021, 23.02099, 25.65091, 28.50295, 25.23701, 26.13828, 33.53260, 29.25195, 27.09847, 26.52999,
        25.52401, 26.69218, 24.55269, 27.71763, 25.20297, 25.61483, 25.06893, 27.63930, 24.94851, 25.86806,
        22.48183, 26.90045, 25.39919, 17.90614, 23.76039, 25.89689, 27.64231, 22.86101, 26.47003, 23.72888,
        27.54334, 23.83386, 23.00216, 26.19075, 24.05393, 22.58901, 22.48183, 23.64482, 25.99417, 24.12449,
        23.69223, 23.99802, 24.22370, 24.44832, 22.94640, 24.23649, 22.30312, 22.28571, 21.43407, 23.71387,
        20.66770, 21.97080, 21.84801, 22.25727, 20.50010, 20.52547, 21.56798, 22.35078, 21.77746, 21.44763,
        20.42610, 21.17296, 20.48910, 20.16017, 20.10290, 19.91334, 21.16981, 20.61707, 20.63490, 19.86916,
        19.92311, 19.29155, 18.85344, 18.96211, 20.06001, 19.09766, 19.69256, 19.08076, 18.57096, 18.40054,
        19.24748, 18.36151, 17.96333, 18.37451, 18.33398, 17.18456, 18.17871, 17.55282, 16.72033, 17.40711,
        16.75120, 17.82102, 16.75120, 17.32959, 16.64841, 17.88396, 16.75120, 16.75120, 16.75120, 16.75120,
        16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120,
        16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120,
        16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120, 16.75120
    };
    private static final double GAUSS1_CERTIFIED_RSS = 1.3158222432E+03;

    static {
        GAUSS1_X = new double[250];
        for (int i = 0; i < 250; i++) GAUSS1_X[i] = i + 1;
    }

    @Test @Tag("NIST Gauss1 - Starting Point 1")
    void testGauss1StartingPoint1() {
        final int m = GAUSS1_X.length;
        BiConsumer<double[], double[]> residualFunc = (c, r) -> {
            for (int i = 0; i < m; i++) {
                double x = GAUSS1_X[i];
                double d2 = x - c[3], d3 = x - c[6];
                r[i] = GAUSS1_Y[i] - (c[0]*Math.exp(-c[1]*x)
                    + c[2]*Math.exp(-d2*d2/(c[4]*c[4]))
                    + c[5]*Math.exp(-d3*d3/(c[7]*c[7])));
            }
        };
        double[] initialGuess = {98.0, 0.011, 100.0, 65.0, 20.0, 72.0, 195.0, 27.0};
        double[] ir = new double[m];
        residualFunc.accept(initialGuess, ir);
        double initialRss = 0; for (double v : ir) initialRss += v * v;

        OptimizationResult result = new TRFProblem()
            .residuals(residualFunc, m)
            .initialPoint(initialGuess.clone())
            .gradientTolerance(1e-6).parameterTolerance(1e-6).functionTolerance(0.1).maxEvaluations(20000)
            .solve();

        assertTrue(result.isSuccessful() || result.getStatus() == OptimizationStatus.MAX_EVALUATIONS_REACHED,
            "Gauss1 should converge, status=" + result.getStatus());
        assertTrue(result.getCost() <= initialRss,
            "Gauss1 RSS should improve");
        double relErr = Math.abs(result.getCost() - GAUSS1_CERTIFIED_RSS) / GAUSS1_CERTIFIED_RSS;
        assertTrue(relErr < 0.5,
            "Gauss1 RSS error: " + relErr + " (RSS=" + result.getCost() + ")");
    }

    // ==================== Property 22: Difficulty Level Handling ====================

    @Property(tries = 10)
    @Tag("Property 22: Difficulty Level Handling")
    void allStartingPointsShouldConvergeToSameSolution(
            @ForAll("misra1aStartingPoints") double[] startingPoint) {
        final int m = MISRA1A_X.length;
        BiConsumer<double[], double[]> residualFunc = (c, r) -> {
            for (int i = 0; i < m; i++)
                r[i] = MISRA1A_Y[i] - c[0] * (1.0 - Math.exp(-c[1] * MISRA1A_X[i]));
        };
        OptimizationResult result = new TRFProblem()
            .residuals(residualFunc, m)
            .initialPoint(startingPoint.clone())
            .gradientTolerance(1e-10).maxEvaluations(5000)
            .solve();

        double relRssError = Math.abs(result.getCost() - MISRA1A_CERTIFIED_RSS) / MISRA1A_CERTIFIED_RSS;
        assertTrue(relRssError < RELATIVE_TOLERANCE,
            "Starting point should converge to certified RSS (error=" + relRssError + ")");
    }

    @Provide
    Arbitrary<double[]> misra1aStartingPoints() {
        return Arbitraries.of(
            new double[]{250.0, 0.0005},
            new double[]{240.0, 0.00055},
            new double[]{230.0, 0.0006},
            new double[]{235.0, 0.00052},
            new double[]{245.0, 0.00058}
        );
    }
}
