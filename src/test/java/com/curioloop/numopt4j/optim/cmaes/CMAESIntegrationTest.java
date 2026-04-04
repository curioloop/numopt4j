/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim.cmaes;

import com.curioloop.numopt4j.optim.Bound;
import com.curioloop.numopt4j.optim.Minimizer;
import com.curioloop.numopt4j.optim.Optimization;
import com.curioloop.numopt4j.optim.Univariate;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration / convergence regression tests for CMA-ES.
 *
 * <p>Test functions adapted from:
 * <ul>
 *   <li>Hipparchus CMAESOptimizerTest (Apache License 2.0)</li>
 *   <li>pycma fitness_functions.py (BSD License)</li>
 * </ul>
 *
 * <p>DIM=13, LAMBDA=4+floor(3*ln(13))=11 (same as hipparchus reference).</p>
 */
class CMAESIntegrationTest {

    static final int DIM = 13;
    static final int LAMBDA = 4 + (int)(3.0 * Math.log(DIM));  // = 11

    // ── Standard test functions ───────────────────────────────────────────

    /** f(x) = Σ xᵢ²,  optimum 0 at origin */
    static double sphere(double[] x, int n) {
        double f = 0; for (int i = 0; i < n; i++) f += x[i] * x[i]; return f;
    }

    /** f(x) = x₀² + 1e6·Σᵢ₌₁ xᵢ²,  optimum 0 (ill-conditioned) */
    static double cigar(double[] x, int n) {
        double f = x[0] * x[0];
        for (int i = 1; i < n; i++) f += 1e6 * x[i] * x[i];
        return f;
    }

    /** f(x) = 1e6·x₀² + Σᵢ₌₁ xᵢ²,  optimum 0 */
    static double tablet(double[] x, int n) {
        double f = 1e6 * x[0] * x[0];
        for (int i = 1; i < n; i++) f += x[i] * x[i];
        return f;
    }

    /** f(x) = x₀²/1e4 + 1e4·x_{n-1}² + Σ middle xᵢ²,  optimum 0 */
    static double cigTab(double[] x, int n) {
        int end = n - 1;
        double f = x[0] * x[0] / 1e4 + 1e4 * x[end] * x[end];
        for (int i = 1; i < end; i++) f += x[i] * x[i];
        return f;
    }

    /** f(x) = 1e6·Σᵢ<n/2 xᵢ² + Σᵢ≥n/2 xᵢ²,  optimum 0 */
    static double twoAxes(double[] x, int n) {
        double f = 0;
        for (int i = 0; i < n; i++)
            f += (i < n / 2 ? 1e6 : 1.0) * x[i] * x[i];
        return f;
    }

    /** f(x) = Σ (1e3)^(i/(n-1)) · xᵢ²,  optimum 0 (ellipsoid) */
    static double elli(double[] x, int n) {
        double f = 0;
        for (int i = 0; i < n; i++)
            f += Math.pow(1e3, (double) i / (n - 1)) * x[i] * x[i];
        return f;
    }

    /** Rotated ellipsoid using a fixed orthogonal basis */
    static double elliRotated(double[] x, int n) {
        return elli(ROTATION.rotate(x), n);
    }

    /** Rosenbrock: f(x) = Σ [100(xᵢ²-xᵢ₊₁)² + (xᵢ-1)²],  optimum 0 at ones */
    static double rosen(double[] x, int n) {
        double f = 0;
        for (int i = 0; i < n - 1; i++) {
            double a = x[i] * x[i] - x[i + 1];
            double b = x[i] - 1.0;
            f += 100.0 * a * a + b * b;
        }
        return f;
    }

    /** Rastrigin: f(x) = 10n + Σ [xᵢ² - 10·cos(2π·xᵢ)],  optimum 0 at origin */
    static double rastrigin(double[] x, int n) {
        double f = 10.0 * n;
        for (int i = 0; i < n; i++) f += x[i] * x[i] - 10.0 * Math.cos(2.0 * Math.PI * x[i]);
        return f;
    }

    /** Ackley: optimum 0 at origin */
    static double ackley(double[] x, int n) {
        double sumSq = 0, sumCos = 0;
        for (int i = 0; i < n; i++) { sumSq += x[i] * x[i]; sumCos += Math.cos(2.0 * Math.PI * x[i]); }
        return 20.0 - 20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / n))
             + Math.E - Math.exp(sumCos / n);
    }

    /** DiffPow: f(x) = Σ |xᵢ|^(2+10i/(n-1)),  optimum 0 */
    static double diffPow(double[] x, int n) {
        double f = 0;
        for (int i = 0; i < n; i++)
            f += Math.pow(Math.abs(x[i]), 2.0 + 10.0 * i / (n - 1));
        return f;
    }

    /** SsDiffPow: f(x) = diffPow(x)^0.25 */
    static double ssDiffPow(double[] x, int n) {
        return Math.pow(diffPow(x, n), 0.25);
    }

    // ── Fixed orthogonal rotation matrix (seed=2, same as hipparchus) ─────

    static final OrthoBasis ROTATION = new OrthoBasis(DIM, 2);

    static class OrthoBasis {
        final double[][] basis;
        OrthoBasis(int n, long seed) {
            Random rng = new Random(seed);
            basis = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) basis[i][j] = rng.nextGaussian();
                for (int j = i - 1; j >= 0; --j) {
                    double sp = 0;
                    for (int k = 0; k < n; k++) sp += basis[i][k] * basis[j][k];
                    for (int k = 0; k < n; k++) basis[i][k] -= sp * basis[j][k];
                }
                double norm = 0;
                for (int k = 0; k < n; k++) norm += basis[i][k] * basis[i][k];
                norm = Math.sqrt(norm);
                for (int k = 0; k < n; k++) basis[i][k] /= norm;
            }
        }
        double[] rotate(double[] x) {
            int n = x.length;
            double[] y = new double[n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) y[i] += basis[i][j] * x[j];
            return y;
        }
    }

    // ── Helper ────────────────────────────────────────────────────────────

    static double[] point(int n, double v) {
        double[] x = new double[n]; Arrays.fill(x, v); return x;
    }

    static Bound[] bounds(int n, double lo, double hi) {
        Bound[] b = new Bound[n];
        Arrays.fill(b, Bound.between(lo, hi));
        return b;
    }

    /**
     * Runs CMA-ES and asserts convergence.
     * Mirrors hipparchus doTest() signature for easy cross-reference.
     *
     * <p>Uses {@code stopFitness = expectedF + fTol} so the algorithm runs until
     * it actually reaches the target precision, rather than stopping early on
     * TolX/TolFun. This matches hipparchus's {@code stopValue} parameter.</p>
     *
     * <p>TolX and TolFun are set to very small values to prevent early stopping
     * before stopFitness is reached.</p>
     */
    void doTest(Univariate.Objective fn,
                double[] x0, double sigma, int lambda, boolean diag,
                Bound[] bds, int maxEval,
                double fTol, double xTol,
                double[] expectedX, double expectedF) {

        CMAESProblem p = Minimizer.cmaes()
            .objective(fn)
            .initialPoint(x0)
            .initialSigma(sigma)
            .updateMode(diag ? UpdateMode.SEP_CMA : UpdateMode.ACTIVE_CMA)
            .maxIterations(maxEval)   // prevent maxIterations from stopping early
            .maxEvaluations(maxEval)
            .stopFitness(expectedF + fTol)   // drive convergence to target precision
            .parameterTolerance(1e-30)             // disable TolX early stopping
            .functionTolerance(1e-30)           // disable TolFun early stopping
            .random(new Random(42));
        if (lambda > 0) p.populationSize(lambda);
        if (bds != null) p.bounds(bds);

        Optimization r = p.solve();

        assertEquals(expectedF, r.getCost(), fTol,
            "f-value: got " + r.getCost() + " after " + r.getEvaluations() + " evals");

        if (expectedX != null) {
            double[] sol = r.getSolution();
            for (int i = 0; i < expectedX.length; i++) {
                assertEquals(expectedX[i], sol[i], xTol,
                    "x[" + i + "]: got " + sol[i]);
            }
        }
    }

    // ── Sphere ────────────────────────────────────────────────────────────

    @Test void testSphere() {
        doTest(CMAESIntegrationTest::sphere, point(DIM, 1.0), 0.1, LAMBDA, false,
               null, 100_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    @Test void testSphere_diag() {
        doTest(CMAESIntegrationTest::sphere, point(DIM, 1.0), 0.1, LAMBDA, true,
               null, 100_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    // ── Cigar ─────────────────────────────────────────────────────────────

    @Test void testCigar() {
        doTest(CMAESIntegrationTest::cigar, point(DIM, 1.0), 0.1, LAMBDA, false,
               null, 200_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    @Test void testCigar_diag() {
        doTest(CMAESIntegrationTest::cigar, point(DIM, 1.0), 0.1, LAMBDA, true,
               null, 500_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    @Test void testCigarWithBounds() {
        // lower bound = -1e100 (effectively unbounded below)
        Bound[] bds = new Bound[DIM];
        Arrays.fill(bds, Bound.atLeast(-1e100));
        doTest(CMAESIntegrationTest::cigar, point(DIM, 1.0), 0.1, LAMBDA, false,
               bds, 200_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    // ── Tablet ────────────────────────────────────────────────────────────

    @Test void testTablet() {
        doTest(CMAESIntegrationTest::tablet, point(DIM, 1.0), 0.1, LAMBDA, false,
               null, 100_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    @Test void testTablet_diag() {
        doTest(CMAESIntegrationTest::tablet, point(DIM, 1.0), 0.1, LAMBDA, true,
               null, 1_000_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    // ── CigTab ────────────────────────────────────────────────────────────

    @Test void testCigTab() {
        doTest(CMAESIntegrationTest::cigTab, point(DIM, 1.0), 0.3, LAMBDA, false,
               null, 100_000, 1e-13, 5e-5, point(DIM, 0.0), 0.0);
    }

    @Test void testCigTab_diag() {
        doTest(CMAESIntegrationTest::cigTab, point(DIM, 1.0), 0.3, LAMBDA, true,
               null, 200_000, 1e-13, 5e-5, point(DIM, 0.0), 0.0);
    }

    // ── TwoAxes ───────────────────────────────────────────────────────────

    @Test void testTwoAxes() {
        doTest(CMAESIntegrationTest::twoAxes, point(DIM, 1.0), 0.1, 2 * LAMBDA, false,
               null, 200_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    // ── Ellipsoid ─────────────────────────────────────────────────────────

    @Test void testElli() {
        doTest(CMAESIntegrationTest::elli, point(DIM, 1.0), 0.1, LAMBDA, false,
               null, 100_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    @Test void testElli_diag() {
        doTest(CMAESIntegrationTest::elli, point(DIM, 1.0), 0.1, LAMBDA, true,
               null, 100_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    @Test void testElliRotated() {
        doTest(CMAESIntegrationTest::elliRotated, point(DIM, 1.0), 0.1, LAMBDA, false,
               null, 100_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    @Test void testElliRotated_diag() {
        doTest(CMAESIntegrationTest::elliRotated, point(DIM, 1.0), 0.1, LAMBDA, true,
               null, 100_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    // ── Rosenbrock ────────────────────────────────────────────────────────

    @Test void testRosen() {
        doTest(CMAESIntegrationTest::rosen, point(DIM, 0.1), 0.1, LAMBDA, false,
               null, 100_000, 1e-13, 1e-6, point(DIM, 1.0), 0.0);
    }

    @Test void testRosen_diag() {
        // diagonal mode needs more evals on Rosenbrock
        doTest(CMAESIntegrationTest::rosen, point(DIM, 0.1), 0.1, LAMBDA, true,
               null, 1_000_000, 1e-10, 1e-4, point(DIM, 1.0), 0.0);
    }

    @Test void testConstrainedRosen() {
        doTest(CMAESIntegrationTest::rosen, point(DIM, 0.1), 0.1, 2 * LAMBDA, false,
               bounds(DIM, -1.0, 2.0), 100_000, 1e-13, 1e-6, point(DIM, 1.0), 0.0);
    }

    // ── Ackley ────────────────────────────────────────────────────────────

    @Test void testAckley() {
        doTest(CMAESIntegrationTest::ackley, point(DIM, 1.0), 1.0, 2 * LAMBDA, false,
               null, 100_000, 1e-9, 1e-5, point(DIM, 0.0), 0.0);
    }

    @Test void testAckley_diag() {
        doTest(CMAESIntegrationTest::ackley, point(DIM, 1.0), 1.0, 2 * LAMBDA, true,
               null, 100_000, 1e-9, 1e-5, point(DIM, 0.0), 0.0);
    }

    // ── Rastrigin (large lambda, near-origin start) ───────────────────────

    @Test void testRastrigin() {
        // hipparchus uses lambda = 200*sqrt(DIM) ≈ 721 for reliable convergence
        int bigLambda = (int)(200 * Math.sqrt(DIM));
        doTest(CMAESIntegrationTest::rastrigin, point(DIM, 0.1), 0.1, bigLambda, false,
               null, 200_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    @Test void testRastrigin_diag() {
        int bigLambda = (int)(200 * Math.sqrt(DIM));
        doTest(CMAESIntegrationTest::rastrigin, point(DIM, 0.1), 0.1, bigLambda, true,
               null, 1_000_000, 1e-13, 1e-6, point(DIM, 0.0), 0.0);
    }

    // ── DiffPow ───────────────────────────────────────────────────────────

    @Test void testDiffPow() {
        doTest(CMAESIntegrationTest::diffPow, point(DIM, 1.0), 0.1, 10, false,
               null, 100_000, 1e-8, 2e-1, point(DIM, 0.0), 0.0);
    }

    @Test void testDiffPow_diag() {
        doTest(CMAESIntegrationTest::diffPow, point(DIM, 1.0), 0.1, 10, true,
               null, 100_000, 1e-8, 3e-1, point(DIM, 0.0), 0.0);
    }

    @Test void testSsDiffPow() {
        doTest(CMAESIntegrationTest::ssDiffPow, point(DIM, 1.0), 0.1, 10, false,
               null, 200_000, 1e-4, 1e-1, point(DIM, 0.0), 0.0);
    }

    // ── 1-D boundary regression (MATH-864 equivalent) ────────────────────

    @Test void testOneDimWithUpperBound() {
        // minimize (1 - x)², x ∈ [-1e6, 1.5], optimum at x=1
        Optimization r = Minimizer.cmaes()
            .objective((x, n) -> { double e = 1.0 - x[0]; return e * e; })
            .initialPoint(0.0)
            .initialSigma(0.1)
            .populationSize(5)
            .bounds(Bound.between(-1e6, 1.5))
            .maxEvaluations(10_000)
            .random(new Random(42))
            .solve();
        assertTrue(r.getSolution()[0] <= 1.5,
            "Solution must respect upper bound, got " + r.getSolution()[0]);
        assertEquals(1.0, r.getSolution()[0], 1e-3,
            "Should converge to x=1, got " + r.getSolution()[0]);
    }

    // ── Boundary accuracy regression (MATH-867 equivalent) ───────────────

    @Test void testFitAccuracyNearBoundary() {
        // minimize (11.1 - x)², verify that proximity to bounds doesn't degrade accuracy
        // Cf. hipparchus testFitAccuracyDependsOnBoundary (MATH-867)
        Univariate.Objective fn =
            (x, n) -> { double e = 11.1 - x[0]; return e * e; };

        // No bounds — baseline
        Optimization noBound = Minimizer.cmaes()
            .objective(fn)
            .initialPoint(1.0)
            .initialSigma(0.1)
            .populationSize(5)
            .maxEvaluations(100_000)
            .random(new Random(42))
            .solve();

        // Optimum near lower bound: x ∈ [-20, 5e16]
        Optimization nearLo = Minimizer.cmaes()
            .objective(fn)
            .initialPoint(1.0)
            .initialSigma(10.0)
            .populationSize(5)
            .bounds(Bound.between(-20.0, 5e16))
            .maxEvaluations(100_000)
            .random(new Random(42))
            .solve();

        // Optimum near upper bound: x ∈ [-5e16, 20]
        Optimization nearHi = Minimizer.cmaes()
            .objective(fn)
            .initialPoint(1.0)
            .initialSigma(10.0)
            .populationSize(5)
            .bounds(Bound.between(-5e16, 20.0))
            .maxEvaluations(100_000)
            .random(new Random(42))
            .solve();

        double resNoBound = noBound.getSolution()[0];
        double resNearLo  = nearLo.getSolution()[0];
        double resNearHi  = nearHi.getSolution()[0];

        assertEquals(resNoBound, resNearLo, 1e-3,
            "Near-lower-bound result should match no-bound result");
        assertEquals(resNoBound, resNearHi, 1e-3,
            "Near-upper-bound result should match no-bound result");
    }

    // ── IPOP: Rastrigin global optimum ────────────────────────────────────

    @Test void testRastriginIPOP() {
        Optimization r = Minimizer.cmaes()
            .objective(CMAESIntegrationTest::rastrigin)
            .initialPoint(point(5, 2.0))
            .initialSigma(2.0)
            .maxEvaluations(200_000)
            .restartMode(RestartMode.ipop(9, 2))
            .stopFitness(1e-6)
            .random(new Random(42))
            .solve();
        assertTrue(r.getCost() < 1e-4,
            "IPOP Rastrigin n=5 should find global optimum, got " + r.getCost());
    }

    // ── BIPOP: convergence ────────────────────────────────────────────────

    @Test void testBIPOP() {
        Optimization r = Minimizer.cmaes()
            .objective(CMAESIntegrationTest::sphere)
            .initialPoint(point(5, 1.0))
            .initialSigma(0.5)
            .maxEvaluations(50_000)
            .restartMode(RestartMode.bipop(5))
            .stopFitness(1e-8)
            .random(new Random(42))
            .solve();
        assertTrue(r.getCost() < 1e-6,
            "BIPOP sphere should converge, got " + r.getCost());
    }

    // ── sep-CMA-ES high-dimensional sphere ────────────────────────────────

    @Test void testSepCMAES_highDim() {
        int n = 50;
        doTest(CMAESIntegrationTest::sphere, point(n, 1.0), 0.5, 0, true,
               null, 200_000, 1e-8, 1e-4, null, 0.0);
    }

    // ── NaN objective → CALLBACK_ERROR ───────────────────────────────────

    @Test void testAllNaNObjectiveReturnsCallbackError() {
        Optimization r = Minimizer.cmaes()
            .objective((x, n) -> Double.NaN)
            .initialPoint(1.0, 1.0)
            .maxEvaluations(1000)
            .random(new Random(42))
            .solve();
        assertEquals(Optimization.Status.CALLBACK_ERROR, r.getStatus());
    }

    @Test void testPartialNaNObjectiveContinues() {
        // Some NaN, some finite — should continue and converge
        int[] callCount = {0};
        Optimization r = Minimizer.cmaes()
            .objective((x, n) -> {
                callCount[0]++;
                return (callCount[0] % 5 == 0) ? Double.NaN : sphere(x, n);
            })
            .initialPoint(1.0, 1.0, 1.0)
            .maxEvaluations(10_000)
            .stopFitness(1e-6)
            .random(new Random(42))
            .solve();
        assertNotEquals(Optimization.Status.CALLBACK_ERROR, r.getStatus());
        assertTrue(r.getCost() < 1e-4,
            "Should converge despite partial NaN, got " + r.getCost());
    }

    // ── Workspace reuse ───────────────────────────────────────────────────

    @Test void testWorkspaceReuse() {
        CMAESProblem p = Minimizer.cmaes()
            .objective(CMAESIntegrationTest::sphere)
            .initialPoint(point(5, 1.0))
            .populationSize(10)
            .maxEvaluations(5_000)
            .stopFitness(1e-8)
            .random(new Random(42));

        CMAESWorkspace ws = new CMAESWorkspace();
        ws.ensure(5, 10, p.updateMode().separable);
        Optimization r1 = p.solve(ws);
        Optimization r2 = p.solve(ws);

        assertTrue(r1.getCost() < 1e-6, "First run: " + r1.getCost());
        assertTrue(r2.getCost() < 1e-6, "Second run: " + r2.getCost());
    }

    // ── Workspace dimension mismatch ──────────────────────────────────────

    @Test void testWorkspaceDimensionMismatch() {
        CMAESProblem p = Minimizer.cmaes()
            .objective((x, n) -> { double s = 0; for (double v : x) s += v*v; return s; })
            .initialPoint(new double[5])
            .populationSize(10)
            .maxEvaluations(100);
        // Workspace with wrong dimensions — should auto-resize instead of throwing
        CMAESWorkspace ws1 = new CMAESWorkspace(); ws1.ensure(3, 10, false);
        assertDoesNotThrow(() -> p.solve(ws1));
        assertEquals(5, ws1.n);
        CMAESWorkspace ws2 = new CMAESWorkspace(); ws2.ensure(5, 20, false);
        assertDoesNotThrow(() -> p.solve(ws2));
        assertEquals(10, ws2.lambda);
    }

    // ── Stop condition tests ──────────────────────────────────────────────

    /** MaxEval stop: set maxEvaluations = lambda, verify MAX_EVALUATIONS_REACHED */
    @Test void testMaxEvalStop() {
        int lambda = LAMBDA;
        Optimization r = Minimizer.cmaes()
            .objective(CMAESIntegrationTest::sphere)
            .initialPoint(point(DIM, 1.0))
            .populationSize(lambda)
            .maxEvaluations(lambda)   // only one iteration
            .maxIterations(lambda)
            .random(new Random(42))
            .solve();
        assertEquals(Optimization.Status.MAX_EVALUATIONS_REACHED, r.getStatus(),
            "Should stop with MAX_EVALUATIONS_REACHED, got " + r.getStatus());
    }

    /** StopFitness stop: set stopFitness = 1e-5, verify FUNCTION_TOLERANCE_REACHED on Sphere */
    @Test void testStopFitnessStop() {
        Optimization r = Minimizer.cmaes()
            .objective(CMAESIntegrationTest::sphere)
            .initialPoint(point(DIM, 1.0))
            .initialSigma(0.1)
            .populationSize(LAMBDA)
            .maxEvaluations(100_000)
            .stopFitness(1e-5)
            .parameterTolerance(1e-30)
            .functionTolerance(1e-30)
            .random(new Random(42))
            .solve();
        assertEquals(Optimization.Status.FUNCTION_TOLERANCE_REACHED, r.getStatus(),
            "Should stop with FUNCTION_TOLERANCE_REACHED, got " + r.getStatus());
        assertTrue(r.getCost() < 1e-5,
            "Cost should be < 1e-5, got " + r.getCost());
    }

    /** All NaN objective: verify CALLBACK_ERROR */
    @Test void testAllNaNObjective() {
        Optimization r = Minimizer.cmaes()
            .objective((x, n) -> Double.NaN)
            .initialPoint(point(DIM, 1.0))
            .maxEvaluations(1000)
            .random(new Random(42))
            .solve();
        assertEquals(Optimization.Status.CALLBACK_ERROR, r.getStatus(),
            "Should return CALLBACK_ERROR for all-NaN objective");
    }

    /** Active CMA condition number: run on Ellipsoid, verify condition number <= 1e14 */
    @Test void testActiveCMAConditionNumber() {
        // Run on Ellipsoid (condition number 1e6) with Active CMA
        Optimization r = Minimizer.cmaes()
            .objective(CMAESIntegrationTest::elli)
            .initialPoint(point(DIM, 1.0))
            .initialSigma(0.1)
            .populationSize(LAMBDA)
            .maxEvaluations(50_000)
            .stopFitness(1e-10)
            .parameterTolerance(1e-30)
            .functionTolerance(1e-30)
            .updateMode(UpdateMode.ACTIVE_CMA)
            .random(new Random(42))
            .solve();

        // Verify convergence
        assertTrue(r.getCost() < 1e-8,
            "Active CMA should converge on Ellipsoid, got " + r.getCost());

        // Verify condition number via workspace
        CMAESProblem p = Minimizer.cmaes()
            .objective(CMAESIntegrationTest::elli)
            .initialPoint(point(DIM, 1.0))
            .initialSigma(0.1)
            .populationSize(LAMBDA)
            .maxEvaluations(50_000)
            .stopFitness(1e-10)
            .parameterTolerance(1e-30)
            .functionTolerance(1e-30)
            .updateMode(UpdateMode.ACTIVE_CMA)
            .random(new Random(42));
        CMAESWorkspace ws = new CMAESWorkspace();
        ws.ensure(DIM, LAMBDA, p.updateMode().separable);
        p.solve(ws);

        // Check condition number of final covariance matrix
        double maxD = ws.nVec[ws.D_OFF], minD = ws.nVec[ws.D_OFF];
        for (int i = 1; i < DIM; i++) {
            if (ws.nVec[ws.D_OFF + i] > maxD) maxD = ws.nVec[ws.D_OFF + i];
            if (ws.nVec[ws.D_OFF + i] < minD) minD = ws.nVec[ws.D_OFF + i];
        }
        if (minD > 0) {
            double condNum = (maxD * maxD) / (minD * minD);
            assertTrue(condNum <= 1e14,
                "Condition number should be <= 1e14, got " + condNum);
        }
    }
}
