/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.trf;

import com.curioloop.numopt4j.optim.Optimization;

import net.jqwik.api.*;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for multi-dimensional fitting support.
 */
@Tag("Feature: levenberg-marquardt-optimizer")
public class MultiDimensionalFittingTest {

    // ==================== 2D Surface Fitting ====================

    @Test @Tag("2D-Surface-Bilinear")
    void testBilinearSurfaceFitting() {
        final int nx = 5, ny = 5, m = nx*ny;
        double[] trueCoeffs = {1.0, 2.0, -1.5, 0.5};
        double[] xData = new double[m], yData = new double[m], zData = new double[m];
        int idx = 0;
        for (int i = 0; i < nx; i++) for (int j = 0; j < ny; j++) {
            double x = -2.0 + i, y = -2.0 + j;
            xData[idx] = x; yData[idx] = y;
            zData[idx] = trueCoeffs[0] + trueCoeffs[1]*x + trueCoeffs[2]*y + trueCoeffs[3]*x*y;
            zData[idx++] += 0.01*(Math.random()-0.5);
        }
        BiConsumer<double[], double[]> residualFunc = (c, r) -> {
            for (int i = 0; i < m; i++)
                r[i] = zData[i] - (c[0] + c[1]*xData[i] + c[2]*yData[i] + c[3]*xData[i]*yData[i]);
        };
        Optimization result = new TRFProblem()
            .residuals(residualFunc, m)
            .initialPoint(0.5, 1.5, -1.0, 0.3)
            .gradientTolerance(1e-10).maxEvaluations(2000)
            .solve();

        assertTrue(result.isSuccessful(), "Bilinear surface fitting should converge");
        assertTrue(result.getCost() < 0.1,
            "Bilinear surface chi² should be small: " + result.getCost());
    }

    @Test @Tag("2D-Surface-Gaussian")
    void testGaussianSurfaceFitting() {
        final int nx = 7, ny = 7, m = nx*ny;
        double amplitude = 5.0, xCenter = 0.5, yCenter = -0.3, width = 1.2;
        double[] xData = new double[m], yData = new double[m], zData = new double[m];
        int idx = 0;
        for (int i = 0; i < nx; i++) for (int j = 0; j < ny; j++) {
            double x = -3.0 + i, y = -3.0 + j;
            xData[idx] = x; yData[idx] = y;
            double dx = x-xCenter, dy = y-yCenter;
            zData[idx] = amplitude*Math.exp(-(dx*dx+dy*dy)/(2*width*width));
            zData[idx++] += 0.02*(Math.random()-0.5);
        }
        BiConsumer<double[], double[]> residualFunc = (c, r) -> {
            for (int i = 0; i < m; i++) {
                double dx = xData[i]-c[1], dy = yData[i]-c[2];
                r[i] = zData[i] - c[0]*Math.exp(-(dx*dx+dy*dy)/(2*c[3]*c[3]));
            }
        };
        Optimization result = new TRFProblem()
            .residuals(residualFunc, m)
            .initialPoint(4.5, 0.3, -0.5, 1.0)
            .gradientTolerance(1e-8).maxEvaluations(2000)
            .solve();

        assertTrue(result.isSuccessful(), "Gaussian surface fitting should converge");
        assertTrue(result.getCost() < 0.5,
            "Gaussian surface chi² should be small: " + result.getCost());
    }

    @Test @Tag("2D-Surface-Polynomial")
    void testPolynomialSurfaceFitting() {
        final int nx = 6, ny = 6, m = nx*ny;
        double[] trueCoeffs = {1.0, 0.5, -0.3, 0.2, 0.1, -0.15};
        double[] xData = new double[m], yData = new double[m], zData = new double[m];
        int idx = 0;
        for (int i = 0; i < nx; i++) for (int j = 0; j < ny; j++) {
            double x = -2.5 + i, y = -2.5 + j;
            xData[idx] = x; yData[idx] = y;
            zData[idx] = trueCoeffs[0] + trueCoeffs[1]*x + trueCoeffs[2]*y
                       + trueCoeffs[3]*x*x + trueCoeffs[4]*y*y + trueCoeffs[5]*x*y;
            zData[idx++] += 0.01*(Math.random()-0.5);
        }
        BiConsumer<double[], double[]> residualFunc = (c, r) -> {
            for (int i = 0; i < m; i++)
                r[i] = zData[i] - (c[0] + c[1]*xData[i] + c[2]*yData[i]
                     + c[3]*xData[i]*xData[i] + c[4]*yData[i]*yData[i] + c[5]*xData[i]*yData[i]);
        };
        Optimization result = new TRFProblem()
            .residuals(residualFunc, m)
            .initialPoint(0.8, 0.4, -0.2, 0.15, 0.08, -0.1)
            .gradientTolerance(1e-10).maxEvaluations(2000)
            .solve();

        assertTrue(result.isSuccessful(), "Polynomial surface fitting should converge");
        assertTrue(result.getCost() < 0.1,
            "Polynomial surface chi² should be small: " + result.getCost());
    }

    // ==================== Property 21: Multi-Dimensional Fitting ====================

    @Property(tries = 100)
    @Tag("Property 21: Multi-Dimensional Fitting")
    void multiDimensionalFittingShouldConverge(
            @ForAll("surfaceFittingProblems") SurfaceFittingProblem problem) {
        double[] ir = new double[problem.m];
        problem.residualFunction.accept(problem.initialGuess, ir);
        double initialChiSq = 0; for (double v : ir) initialChiSq += v*v;

        Optimization result = new TRFProblem()
            .residuals(problem.residualFunction, problem.m)
            .initialPoint(problem.initialGuess.clone())
            .gradientTolerance(1e-8).maxEvaluations(2000)
            .solve();

        assertTrue(result.isSuccessful() || result.getStatus() == Optimization.Status.MAX_EVALUATIONS_REACHED,
            "Multi-dimensional fitting should converge or reach limit");
        assertTrue(result.getCost() <= initialChiSq,
            "Chi² should improve: initial=" + initialChiSq + ", final=" + result.getCost());
    }

    @Provide
    Arbitrary<SurfaceFittingProblem> surfaceFittingProblems() {
        return Combinators.combine(
            Arbitraries.integers().between(4, 6),
            Arbitraries.doubles().between(0.5, 2.0),
            Arbitraries.doubles().between(-1.0, 1.0),
            Arbitraries.doubles().between(-1.0, 1.0)
        ).as(SurfaceFittingProblem::new);
    }

    static class SurfaceFittingProblem {
        final int m, n = 4;
        final double[] initialGuess;
        final BiConsumer<double[], double[]> residualFunction;

        SurfaceFittingProblem(int gridSize, double amplitude, double xOffset, double yOffset) {
            this.m = gridSize * gridSize;
            double[] xData = new double[m], yData = new double[m], zData = new double[m];
            double[] trueCoeffs = {amplitude, 0.5+xOffset, -0.3+yOffset, 0.2};
            int idx = 0;
            for (int i = 0; i < gridSize; i++) for (int j = 0; j < gridSize; j++) {
                double x = -2.0 + i*4.0/(gridSize-1), y = -2.0 + j*4.0/(gridSize-1);
                xData[idx] = x; yData[idx] = y;
                zData[idx] = trueCoeffs[0] + trueCoeffs[1]*x + trueCoeffs[2]*y + trueCoeffs[3]*x*y;
                zData[idx++] += 0.01*(Math.random()-0.5);
            }
            this.initialGuess = new double[]{trueCoeffs[0]*0.8, trueCoeffs[1]*0.9, trueCoeffs[2]*1.1, trueCoeffs[3]*0.85};
            this.residualFunction = (c, r) -> {
                for (int i = 0; i < m; i++)
                    r[i] = zData[i] - (c[0] + c[1]*xData[i] + c[2]*yData[i] + c[3]*xData[i]*yData[i]);
            };
        }
    }

    @Test @Tag("MultiDim-WithClosure")
    void testMultiDimensionalWithClosure() {
        final int m = 25;
        final double[][] gridData = new double[m][3];
        double[] trueCoeffs = {1.0, 2.0, -1.5, 0.5};
        int idx = 0;
        for (int i = 0; i < 5; i++) for (int j = 0; j < 5; j++) {
            double x = -2.0 + i, y = -2.0 + j;
            gridData[idx][0] = x; gridData[idx][1] = y;
            gridData[idx++][2] = trueCoeffs[0] + trueCoeffs[1]*x + trueCoeffs[2]*y + trueCoeffs[3]*x*y;
        }
        BiConsumer<double[], double[]> residualFunc = (c, r) -> {
            for (int i = 0; i < m; i++) {
                double x = gridData[i][0], y = gridData[i][1], z = gridData[i][2];
                r[i] = z - (c[0] + c[1]*x + c[2]*y + c[3]*x*y);
            }
        };
        Optimization result = new TRFProblem()
            .residuals(residualFunc, m)
            .initialPoint(0.5, 1.5, -1.0, 0.3)
            .gradientTolerance(1e-10).maxEvaluations(2000)
            .solve();

        assertTrue(result.isSuccessful(), "Multi-dimensional with closure should converge");
        assertTrue(result.getCost() < 0.1,
            "Chi² should be small: " + result.getCost());
    }
}
