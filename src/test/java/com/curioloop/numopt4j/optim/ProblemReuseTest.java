/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.optim;

import com.curioloop.numopt4j.optim.cmaes.CMAESProblem;
import com.curioloop.numopt4j.optim.cmaes.CMAESWorkspace;
import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBProblem;
import com.curioloop.numopt4j.optim.lbfgsb.LBFGSBWorkspace;
import com.curioloop.numopt4j.optim.root.BrentqProblem;
import com.curioloop.numopt4j.optim.root.BroydenProblem;
import com.curioloop.numopt4j.optim.root.BroydenWorkspace;
import com.curioloop.numopt4j.optim.root.HYBRProblem;
import com.curioloop.numopt4j.optim.root.HYBRWorkspace;
import com.curioloop.numopt4j.optim.slsqp.SLSQPProblem;
import com.curioloop.numopt4j.optim.slsqp.SLSQPWorkspace;
import com.curioloop.numopt4j.optim.subplex.SubplexProblem;
import com.curioloop.numopt4j.optim.subplex.SubplexWorkspace;
import com.curioloop.numopt4j.optim.trf.TRFProblem;
import com.curioloop.numopt4j.optim.trf.TRFWorkspace;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Verifies that all Problem implementations are reusable objects:
 * repeated solve() calls on the same instance produce identical results,
 * and explicit workspace reuse does not cause state leakage between runs.
 */
class ProblemReuseTest {

    private static final double TOL = 1e-6;
    private static final double[] X0_2D = {1.0, 1.0};

    // ── helpers ──────────────────────────────────────────────────────────────

    /** Sphere: f(x) = Σ xᵢ², minimum 0 at origin. */
    private static final Univariate SPHERE = TestTemplates.quadratic();

    /** Residuals: rᵢ = xᵢ − targetᵢ, zero at target. */
    private static Multivariate.Objective residuals(double... target) {
        return (x, n, r, m) -> { for (int i = 0; i < m; i++) r[i] = x[i] - target[i]; };
    }

    /** Equations: fᵢ(x) = xᵢ − targetᵢ = 0. */
    private static Multivariate.Objective equations(double... target) {
        return (x, n, f, m) -> { for (int i = 0; i < m; i++) f[i] = x[i] - target[i]; };
    }

    // ── L-BFGS-B ─────────────────────────────────────────────────────────────

    @Test
    void lbfgsbProblemIsReusable() {
        LBFGSBProblem p = Minimizer.lbfgsb().objective(SPHERE).initialPoint(X0_2D.clone());

        Optimization r1 = p.solve();
        Optimization r2 = p.initialPoint(X0_2D.clone()).solve();
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    @Test
    void lbfgsbWorkspaceIsReusable() {
        LBFGSBProblem p = Minimizer.lbfgsb().objective(SPHERE);
        LBFGSBWorkspace ws = LBFGSBProblem.workspace();

        Optimization r1 = p.initialPoint(X0_2D.clone()).solve(ws);
        Optimization r2 = p.initialPoint(X0_2D.clone()).solve(ws);
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    // ── SLSQP ────────────────────────────────────────────────────────────────

    @Test
    void slsqpProblemIsReusable() {
        SLSQPProblem p = Minimizer.slsqp().objective(SPHERE).initialPoint(X0_2D.clone());

        Optimization r1 = p.solve();
        Optimization r2 = p.initialPoint(X0_2D.clone()).solve();
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    @Test
    void slsqpWorkspaceIsReusable() {
        SLSQPProblem p = Minimizer.slsqp().objective(SPHERE);
        SLSQPWorkspace ws = SLSQPProblem.workspace();

        Optimization r1 = p.initialPoint(X0_2D.clone()).solve(ws);
        Optimization r2 = p.initialPoint(X0_2D.clone()).solve(ws);
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    // ── TRF ──────────────────────────────────────────────────────────────────

    @Test
    void trfProblemIsReusable() {
        TRFProblem p = Minimizer.trf()
                .residuals(residuals(2.0, 3.0), 2)
                .initialPoint(0.0, 0.0);

        Optimization r1 = p.solve();
        Optimization r2 = p.initialPoint(0.0, 0.0).solve();
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    @Test
    void trfWorkspaceIsReusable() {
        TRFProblem p = Minimizer.trf().residuals(residuals(2.0, 3.0), 2);
        TRFWorkspace ws = TRFProblem.workspace();

        Optimization r1 = p.initialPoint(0.0, 0.0).solve(ws);
        Optimization r2 = p.initialPoint(0.0, 0.0).solve(ws);
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    // ── Subplex ───────────────────────────────────────────────────────────────

    @Test
    void subplexProblemIsReusable() {
        SubplexProblem p = Minimizer.subplex()
                .objective((x, n) -> x[0]*x[0] + x[1]*x[1])
                .initialPoint(X0_2D.clone());

        Optimization r1 = p.solve();
        Optimization r2 = p.initialPoint(X0_2D.clone()).solve();
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    @Test
    void subplexWorkspaceIsReusable() {
        SubplexProblem p = Minimizer.subplex()
                .objective((x, n) -> x[0]*x[0] + x[1]*x[1]);
        SubplexWorkspace ws = SubplexProblem.workspace();

        Optimization r1 = p.initialPoint(X0_2D.clone()).solve(ws);
        Optimization r2 = p.initialPoint(X0_2D.clone()).solve(ws);
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    // ── CMA-ES ────────────────────────────────────────────────────────────────

    @Test
    void cmaesProblemIsReusable() {
        CMAESProblem p = Minimizer.cmaes()
                .objective((x, n) -> x[0]*x[0] + x[1]*x[1])
                .maxEvaluations(2000)
                .initialPoint(X0_2D.clone());

        Optimization r1 = p.solve();
        Optimization r2 = p.initialPoint(X0_2D.clone()).solve();
        assertThat(r1.isSuccessful()).isTrue();
        assertThat(r2.isSuccessful()).isTrue();
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(1e-4));
    }

    @Test
    void cmaesWorkspaceIsReusable() {
        CMAESProblem p = Minimizer.cmaes()
                .objective((x, n) -> x[0]*x[0] + x[1]*x[1])
                .maxEvaluations(2000);
        CMAESWorkspace ws = CMAESProblem.workspace();

        Optimization r1 = p.initialPoint(X0_2D.clone()).solve(ws);
        Optimization r2 = p.initialPoint(X0_2D.clone()).solve(ws);
        assertThat(r1.isSuccessful()).isTrue();
        assertThat(r2.isSuccessful()).isTrue();
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(1e-4));
    }

    // ── Brentq ────────────────────────────────────────────────────────────────

    @Test
    void brentqProblemIsReusable() {
        BrentqProblem p = RootFinder.brentq(Math::sin)
                .bracket(Bound.between(3.0, 4.0));

        Optimization r1 = p.solve();
        Optimization r2 = p.solve();
        assertThat(r1.getRoot()).isCloseTo(r2.getRoot(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    // ── HYBR ─────────────────────────────────────────────────────────────────

    @Test
    void hybrProblemIsReusable() {
        HYBRProblem p = RootFinder.hybr(equations(2.0, 3.0), 2)
                .initialPoint(0.0, 0.0);

        Optimization r1 = p.solve();
        Optimization r2 = p.initialPoint(0.0, 0.0).solve();
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    @Test
    void hybrWorkspaceIsReusable() {
        HYBRProblem p = RootFinder.hybr(equations(2.0, 3.0), 2);
        HYBRWorkspace ws = HYBRProblem.workspace();

        Optimization r1 = p.initialPoint(0.0, 0.0).solve(ws);
        Optimization r2 = p.initialPoint(0.0, 0.0).solve(ws);
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    // ── Broyden ───────────────────────────────────────────────────────────────

    @Test
    void broydenProblemIsReusable() {
        BroydenProblem p = RootFinder.broyden(equations(2.0, 3.0), 2)
                .initialPoint(0.0, 0.0);

        Optimization r1 = p.solve();
        Optimization r2 = p.initialPoint(0.0, 0.0).solve();
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    @Test
    void broydenWorkspaceIsReusable() {
        BroydenProblem p = RootFinder.broyden(equations(2.0, 3.0), 2);
        BroydenWorkspace ws = BroydenProblem.workspace();

        Optimization r1 = p.initialPoint(0.0, 0.0).solve(ws);
        Optimization r2 = p.initialPoint(0.0, 0.0).solve(ws);
        assertThat(r1.getCost()).isCloseTo(r2.getCost(), within(TOL));
        assertThat(r1.getStatus()).isEqualTo(r2.getStatus());
    }

    // ── Dimension change: workspace auto-resizes ──────────────────────────────

    @Test
    void lbfgsbWorkspaceResizesAcrossDimensions() {
        LBFGSBWorkspace ws = LBFGSBProblem.workspace();

        // First run: 2D
        Optimization r1 = Minimizer.lbfgsb().objective(SPHERE)
                .initialPoint(1.0, 1.0).solve(ws);
        assertThat(r1.isSuccessful()).isTrue();

        // Second run: 5D — workspace must resize
        Optimization r2 = Minimizer.lbfgsb().objective(SPHERE)
                .initialPoint(1.0, 1.0, 1.0, 1.0, 1.0).solve(ws);
        assertThat(r2.isSuccessful()).isTrue();
        assertThat(r2.getCost()).isCloseTo(0.0, within(TOL));

        // Third run: back to 2D — workspace reuses existing (larger) allocation
        Optimization r3 = Minimizer.lbfgsb().objective(SPHERE)
                .initialPoint(1.0, 1.0).solve(ws);
        assertThat(r3.isSuccessful()).isTrue();
        assertThat(r3.getCost()).isCloseTo(r1.getCost(), within(TOL));
    }

    @Test
    void slsqpWorkspaceResizesAcrossDimensions() {
        SLSQPWorkspace ws = SLSQPProblem.workspace();

        Optimization r1 = Minimizer.slsqp().objective(SPHERE)
                .initialPoint(1.0, 1.0).solve(ws);
        assertThat(r1.isSuccessful()).isTrue();

        Optimization r2 = Minimizer.slsqp().objective(SPHERE)
                .initialPoint(1.0, 1.0, 1.0, 1.0, 1.0).solve(ws);
        assertThat(r2.isSuccessful()).isTrue();
        assertThat(r2.getCost()).isCloseTo(0.0, within(TOL));
    }

    @Test
    void trfWorkspaceResizesAcrossDimensions() {
        TRFWorkspace ws = TRFProblem.workspace();

        // 2 residuals, 2 params
        Optimization r1 = Minimizer.trf().residuals(residuals(1.0, 2.0), 2)
                .initialPoint(0.0, 0.0).solve(ws);
        assertThat(r1.isSuccessful()).isTrue();

        // 4 residuals, 3 params — workspace must resize
        Optimization r2 = Minimizer.trf()
                .residuals((x, n, r, m) -> { r[0]=x[0]-1; r[1]=x[1]-2; r[2]=x[2]-3; r[3]=x[0]+x[1]-3; }, 4)
                .initialPoint(0.0, 0.0, 0.0).solve(ws);
        assertThat(r2.isSuccessful()).isTrue();
    }

    @Test
    void subplexWorkspaceResizesAcrossDimensions() {
        SubplexWorkspace ws = SubplexProblem.workspace();

        Optimization r1 = Minimizer.subplex()
                .objective((x, n) -> x[0]*x[0] + x[1]*x[1])
                .initialPoint(1.0, 1.0).solve(ws);
        assertThat(r1.isSuccessful()).isTrue();

        Optimization r2 = Minimizer.subplex()
                .objective((x, n) -> { double s=0; for(int i=0;i<n;i++) s+=x[i]*x[i]; return s; })
                .functionTolerance(1e-8).parameterTolerance(1e-8).maxEvaluations(10000)
                .initialPoint(1.0, 1.0, 1.0, 1.0, 1.0).solve(ws);
        assertThat(r2.isSuccessful()).isTrue();
        assertThat(r2.getCost()).isCloseTo(0.0, within(1e-4));
    }

    @Test
    void hybrWorkspaceResizesAcrossDimensions() {
        HYBRWorkspace ws = HYBRProblem.workspace();

        Optimization r1 = RootFinder.hybr(equations(1.0, 2.0), 2)
                .initialPoint(0.0, 0.0).solve(ws);
        assertThat(r1.isSuccessful()).isTrue();

        Optimization r2 = RootFinder.hybr(equations(1.0, 2.0, 3.0), 3)
                .initialPoint(0.0, 0.0, 0.0).solve(ws);
        assertThat(r2.isSuccessful()).isTrue();
    }

    @Test
    void broydenWorkspaceResizesAcrossDimensions() {
        BroydenWorkspace ws = BroydenProblem.workspace();

        Optimization r1 = RootFinder.broyden(equations(1.0, 2.0), 2)
                .initialPoint(0.0, 0.0).solve(ws);
        assertThat(r1.isSuccessful()).isTrue();

        Optimization r2 = RootFinder.broyden(equations(1.0, 2.0, 3.0), 3)
                .initialPoint(0.0, 0.0, 0.0).solve(ws);
        assertThat(r2.isSuccessful()).isTrue();
    }

    // ── Parameter mutation: objective change ──────────────────────────────────

    @Test
    void lbfgsbObjectiveChangeProducesNewResult() {
        LBFGSBProblem p = Minimizer.lbfgsb().initialPoint(1.0, 1.0);

        // First: minimize x² + y², solution at (0, 0)
        Optimization r1 = p.objective(SPHERE).solve();
        assertThat(r1.isSuccessful()).isTrue();
        assertThat(r1.getCost()).isCloseTo(0.0, within(TOL));

        // Second: minimize (x-2)² + (y-3)², solution at (2, 3)
        Optimization r2 = p.objective(TestTemplates.quadraticWithTarget(new double[]{2.0, 3.0}))
                .initialPoint(0.0, 0.0).solve();
        assertThat(r2.isSuccessful()).isTrue();
        assertThat(r2.getSolution()[0]).isCloseTo(2.0, within(TOL));
        assertThat(r2.getSolution()[1]).isCloseTo(3.0, within(TOL));
    }

    @Test
    void trfResidualsChangeProducesNewResult() {
        TRFProblem p = Minimizer.trf().initialPoint(0.0, 0.0);

        // First: target (1, 2)
        Optimization r1 = p.residuals(residuals(1.0, 2.0), 2).solve();
        assertThat(r1.isSuccessful()).isTrue();
        assertThat(r1.getSolution()[0]).isCloseTo(1.0, within(TOL));

        // Second: target (5, 7)
        Optimization r2 = p.residuals(residuals(5.0, 7.0), 2).initialPoint(0.0, 0.0).solve();
        assertThat(r2.isSuccessful()).isTrue();
        assertThat(r2.getSolution()[0]).isCloseTo(5.0, within(TOL));
        assertThat(r2.getSolution()[1]).isCloseTo(7.0, within(TOL));
    }

    // ── Parameter mutation: bounds change ─────────────────────────────────────

    @Test
    void lbfgsbBoundsChangeAffectsSolution() {
        LBFGSBProblem p = Minimizer.lbfgsb()
                .objective(SPHERE)
                .initialPoint(2.0, 2.0);

        // No bounds: solution at (0, 0)
        Optimization r1 = p.solve();
        assertThat(r1.getSolution()[0]).isCloseTo(0.0, within(TOL));

        // Add bounds x >= 1, y >= 1: solution at (1, 1)
        Optimization r2 = p.bounds(Bound.atLeast(1.0), Bound.atLeast(1.0))
                .initialPoint(2.0, 2.0).solve();
        assertThat(r2.getSolution()[0]).isCloseTo(1.0, within(TOL));
        assertThat(r2.getSolution()[1]).isCloseTo(1.0, within(TOL));

        // Remove bounds: solution back at (0, 0)
        Optimization r3 = p.bounds((Bound[]) null)
                .initialPoint(2.0, 2.0).solve();
        assertThat(r3.getSolution()[0]).isCloseTo(0.0, within(TOL));
    }

    @Test
    void trfBoundsChangeAffectsSolution() {
        TRFProblem p = Minimizer.trf()
                .residuals(residuals(-2.0, -2.0), 2)
                .initialPoint(0.0, 0.0);

        // No bounds: solution at (-2, -2)
        Optimization r1 = p.solve();
        assertThat(r1.getSolution()[0]).isCloseTo(-2.0, within(TOL));

        // Add bounds x >= 0, y >= 0: solution clamped to (0, 0)
        Optimization r2 = p.bounds(Bound.atLeast(0.0), Bound.atLeast(0.0))
                .initialPoint(0.0, 0.0).solve();
        assertThat(r2.getSolution()[0]).isCloseTo(0.0, within(TOL));
        assertThat(r2.getSolution()[1]).isCloseTo(0.0, within(TOL));
    }

    // ── Parameter mutation: constraints change (SLSQP) ────────────────────────

    @Test
    void slsqpConstraintsChangeAffectsSolution() {
        SLSQPProblem p = Minimizer.slsqp().objective(SPHERE).initialPoint(1.0, 1.0);

        // No constraints: solution at (0, 0)
        Optimization r1 = p.solve();
        assertThat(r1.getCost()).isCloseTo(0.0, within(TOL));

        // Equality constraint x + y = 2: solution at (1, 1)
        Optimization r2 = p.equalityConstraints(TestTemplates.sumConstraint(2.0))
                .initialPoint(1.0, 1.0).solve();
        assertThat(r2.isSuccessful()).isTrue();
        assertThat(r2.getSolution()[0] + r2.getSolution()[1]).isCloseTo(2.0, within(TOL));

        // Remove constraints: solution back at (0, 0)
        Optimization r3 = p.equalityConstraints((Univariate[]) null)
                .initialPoint(1.0, 1.0).solve();
        assertThat(r3.getCost()).isCloseTo(0.0, within(TOL));
    }

    // ── Parameter mutation: tolerance tightening ──────────────────────────────

    @Test
    void lbfgsbTighterToleranceImprovesPrecision() {
        LBFGSBProblem p = Minimizer.lbfgsb()
                .objective(TestTemplates.rosenbrock())
                .initialPoint(0.0, 0.0);

        Optimization loose = p.gradientTolerance(1e-3).solve();
        Optimization tight = p.gradientTolerance(1e-8).initialPoint(0.0, 0.0).solve();

        assertThat(loose.isSuccessful()).isTrue();
        assertThat(tight.isSuccessful()).isTrue();
        // Tighter tolerance should yield lower cost
        assertThat(tight.getCost()).isLessThanOrEqualTo(loose.getCost() + 1e-6);
    }

    @Test
    void trfTighterToleranceImprovesPrecision() {
        double[] xd = {0, 1, 2, 3, 4}, yd = {1, 3, 5, 7, 9};
        Multivariate.Objective fn = (x, n, r, m) -> {
            for (int i = 0; i < m; i++) r[i] = yd[i] - (x[0] + x[1] * xd[i]);
        };
        TRFProblem p = Minimizer.trf().residuals(fn, 5).initialPoint(0.0, 0.0);

        Optimization loose = p.functionTolerance(1e-4).solve();
        Optimization tight = p.functionTolerance(1e-12).parameterTolerance(1e-12)
                .initialPoint(0.0, 0.0).solve();

        assertThat(loose.isSuccessful()).isTrue();
        assertThat(tight.isSuccessful()).isTrue();
        assertThat(tight.getCost()).isLessThanOrEqualTo(loose.getCost() + 1e-8);
    }

    // ── Parameter mutation: corrections change (LBFGSB workspace resize) ──────

    @Test
    void lbfgsbCorrectionsChangeResizesWorkspace() {
        LBFGSBWorkspace ws = LBFGSBProblem.workspace();
        LBFGSBProblem p = Minimizer.lbfgsb().objective(SPHERE);

        // First run: corrections=3
        Optimization r1 = p.corrections(3).initialPoint(1.0, 1.0).solve(ws);
        assertThat(r1.isSuccessful()).isTrue();

        // Second run: corrections=15 — workspace must resize
        Optimization r2 = p.corrections(15).initialPoint(1.0, 1.0).solve(ws);
        assertThat(r2.isSuccessful()).isTrue();
        assertThat(r2.getCost()).isCloseTo(r1.getCost(), within(TOL));
    }

    // ── Parameter mutation: equations change (HYBR / Broyden) ────────────────

    @Test
    void hybrEquationsChangeProducesNewResult() {
        HYBRProblem p = RootFinder.hybr(equations(1.0, 2.0), 2).initialPoint(0.0, 0.0);

        Optimization r1 = p.solve();
        assertThat(r1.getSolution()[0]).isCloseTo(1.0, within(TOL));

        // Change equations: new root at (3, 4)
        p.equations(equations(3.0, 4.0), 2).initialPoint(0.0, 0.0);
        Optimization r2 = p.solve();
        assertThat(r2.getSolution()[0]).isCloseTo(3.0, within(TOL));
        assertThat(r2.getSolution()[1]).isCloseTo(4.0, within(TOL));
    }

    @Test
    void broydenEquationsChangeProducesNewResult() {
        BroydenProblem p = RootFinder.broyden(equations(1.0, 2.0), 2).initialPoint(0.0, 0.0);

        Optimization r1 = p.solve();
        assertThat(r1.getSolution()[0]).isCloseTo(1.0, within(TOL));

        p.equations(equations(3.0, 4.0), 2).initialPoint(0.0, 0.0);
        Optimization r2 = p.solve();
        assertThat(r2.getSolution()[0]).isCloseTo(3.0, within(TOL));
        assertThat(r2.getSolution()[1]).isCloseTo(4.0, within(TOL));
    }
}
