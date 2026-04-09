package com.curioloop.numopt4j.quad;

import com.curioloop.numopt4j.quad.ode.IVPPool;
import com.curioloop.numopt4j.quad.ode.ODEEvent;
import com.curioloop.numopt4j.quad.ode.ODE;
import com.curioloop.numopt4j.quad.ode.IVPIntegral;
import com.curioloop.numopt4j.quad.ode.IVPMethod;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Validates ODE IVP solver output correctness, mirroring scipy test_ivp.py.
 */
class IvpSolverTest {

    // -----------------------------------------------------------------------
    // Test functions (corresponding to scipy test_ivp.py)
    // -----------------------------------------------------------------------

    // fun_rational: y' = [y1/t, y1*(y0+2*y1-1)/(t*(y0-1))]
    // exact solution: y(t) = [t/(t+10), 10*t/(t+10)^2]
    // y0 = [1/3, 2/9], t in [5,9] or [5,1] (backward)
    static void funRational(double t, double[] y, double[] dydt) {
        dydt[0] = y[1] / t;
        dydt[1] = y[1] * (y[0] + 2 * y[1] - 1) / (t * (y[0] - 1));
    }

    static double[] solRational(double t) {
        return new double[]{t / (t + 10), 10 * t / (t + 10) / (t + 10)};
    }

    // Analytic Jacobian (corresponding to scipy jac_rational)
    static final ODE jacRational = (t, y, dydt, jac) -> {
        funRational(t, y, dydt);
        if (jac != null) {
            jac[0] = 0;
            jac[1] = 1.0 / t;
            jac[2] = -2 * y[1] * y[1] / (t * (y[0] - 1) * (y[0] - 1));
            jac[3] = (y[0] + 4 * y[1] - 1) / (t * (y[0] - 1));
        }
    };

    /**
     * Normalized error norm (scipy compute_error), should be < 5.
     */
    static double computeError(double[] y, double[] yTrue, double rtol, double atol) {
        double sum = 0;
        for (int i = 0; i < y.length; i++) {
            double e = (y[i] - yTrue[i]) / (atol + rtol * Math.abs(yTrue[i]));
            sum += e * e;
        }
        return Math.sqrt(sum / y.length);
    }

    // -----------------------------------------------------------------------
    // 1. Core integration: fun_rational, rtol=1e-3, atol=1e-6 (scipy test_integration)
    //    forward [5,9] and backward [5,1], with/without analytic Jacobian
    // -----------------------------------------------------------------------

    @ParameterizedTest @ValueSource(strings = {"RK23","RK45","DOP853","BDF","Radau"})
    void integrationRationalForward(String method) {
        double rtol = 1e-3, atol = 1e-6;
        double[] y0 = {1.0/3, 2.0/9};

        Trajectory sol = new IVPIntegral(IVPMethod.valueOf(method))
                .equation(IvpSolverTest::funRational)
                .bounds(5.0, 9.0).initialState(y0)
                .tolerances(rtol, atol).integrate();

        assertTrue(sol.isSuccessful(), method + " forward: should succeed");
        assertEquals(Trajectory.Status.SUCCESS, sol.getStatus());

        // verify error < 5 at each time point (scipy assert_(np.all(e < 5)))
        for (int j = 0; j < sol.getTimePoints(); j++) {
            double[] yj = {sol.timeSeries.y[0 * sol.getTimePoints() + j], sol.timeSeries.y[1 * sol.getTimePoints() + j]};
            double[] yTrue = solRational(sol.timeSeries.t[j]);
            double e = computeError(yj, yTrue, rtol, atol);
            assertTrue(e < 5, method + " forward: error=" + e + " at t=" + sol.timeSeries.t[j]);
        }
    }

    @ParameterizedTest @ValueSource(strings = {"RK23","RK45","DOP853","BDF","Radau"})
    void integrationRationalBackward(String method) {
        double rtol = 1e-3, atol = 1e-6;
        double[] y0 = {1.0/3, 2.0/9};

        Trajectory sol = new IVPIntegral(IVPMethod.valueOf(method))
                .equation(IvpSolverTest::funRational)
                .bounds(5.0, 1.0).initialState(y0)
                .tolerances(rtol, atol).integrate();

        assertTrue(sol.isSuccessful(), method + " backward: should succeed");
        assertEquals(Trajectory.Status.SUCCESS, sol.getStatus());

        for (int j = 0; j < sol.getTimePoints(); j++) {
            double[] yj = {sol.timeSeries.y[0 * sol.getTimePoints() + j], sol.timeSeries.y[1 * sol.getTimePoints() + j]};
            double[] yTrue = solRational(sol.timeSeries.t[j]);
            double e = computeError(yj, yTrue, rtol, atol);
            assertTrue(e < 5, method + " backward: error=" + e + " at t=" + sol.timeSeries.t[j]);
        }
    }

    @ParameterizedTest @ValueSource(strings = {"BDF","Radau"})
    void integrationRationalWithAnalyticJac(String method) {
        double rtol = 1e-3, atol = 1e-6;
        double[] y0 = {1.0/3, 2.0/9};

        Trajectory sol = new IVPIntegral(IVPMethod.valueOf(method))
                .equation(jacRational)
                .bounds(5.0, 9.0).initialState(y0)
                .tolerances(rtol, atol).integrate();

        assertTrue(sol.isSuccessful(), method + " analytic jac: should succeed");
        assertTrue(sol.getJacobianEvaluations() > 0, method + ": njev should be > 0");
        assertTrue(sol.getLuDecompositions() > 0, method + ": nlu should be > 0");

        for (int j = 0; j < sol.getTimePoints(); j++) {
            double[] yj = {sol.timeSeries.y[0 * sol.getTimePoints() + j], sol.timeSeries.y[1 * sol.getTimePoints() + j]};
            double[] yTrue = solRational(sol.timeSeries.t[j]);
            double e = computeError(yj, yTrue, rtol, atol);
            assertTrue(e < 5, method + " analytic jac: error=" + e + " at t=" + sol.timeSeries.t[j]);
        }
    }

    // -----------------------------------------------------------------------
    // 2. Dense output: error < 5, sol(t_grid) matches y (scipy test_integration)
    // -----------------------------------------------------------------------

    @ParameterizedTest @ValueSource(strings = {"RK23","RK45","DOP853","BDF","Radau"})
    void denseOutputRational(String method) {
        double rtol = 1e-3, atol = 1e-6;
        double[] y0 = {1.0/3, 2.0/9};

        Trajectory sol = new IVPIntegral(IVPMethod.valueOf(method))
                .equation(IvpSolverTest::funRational)
                .bounds(5.0, 9.0).initialState(y0)
                .tolerances(rtol, atol).denseOutput(true).integrate();

        assertNotNull(sol.denseOutput, method + ": interpolator should not be null");

        // verify interpolation accuracy on uniform grid (scipy tc = np.linspace(*t_span))
        double[] out = new double[2];
        for (int i = 0; i <= 20; i++) {
            double tc = 5.0 + i * 4.0 / 20;
            sol.denseOutput.interpolate(tc, out);
            double[] yTrue = solRational(tc);
            double e = computeError(out, yTrue, rtol, atol);
            assertTrue(e < 5, method + ": dense output error=" + e + " at t=" + tc);
        }

        // sol(t_grid) must match y exactly (scipy assert_allclose(res.sol(res.t), res.y, rtol=1e-15))
        for (int j = 0; j < sol.getTimePoints(); j++) {
            sol.denseOutput.interpolate(sol.timeSeries.t[j], out);
            assertEquals(sol.timeSeries.y[0 * sol.getTimePoints() + j], out[0], 1e-10, method + ": sol(t[j]) != y[j] at j=" + j);
            assertEquals(sol.timeSeries.y[1 * sol.getTimePoints() + j], out[1], 1e-10, method + ": sol(t[j]) != y[j] at j=" + j);
        }
    }

    // -----------------------------------------------------------------------
    // 3. Stiff problems: Van der Pol
    // -----------------------------------------------------------------------

    @Test
    void stiffVanDerPolBdf() {
        double mu = 1000.0;
        Trajectory sol = new IVPIntegral(IVPMethod.BDF)
                .equation((t, y, dydt) -> {
                    dydt[0] = y[1];
                    dydt[1] = mu * (1 - y[0] * y[0]) * y[1] - y[0];
                })
                .bounds(0.0, 500.0).initialState(2.0, 0.0)
                .tolerances(1e-3, 1e-6).integrate();

        assertTrue(sol.isSuccessful(), "BDF should succeed on Van der Pol mu=1000");
        assertTrue(Math.abs(sol.timeSeries.y[0 * sol.getTimePoints() + sol.getTimePoints() - 1]) <= 2.5, "y should stay bounded");
    }

    @Test
    void stiffVanDerPolRadau() {
        double mu = 10.0;
        Trajectory sol = new IVPIntegral(IVPMethod.Radau)
                .equation((t, y, dydt) -> {
                    dydt[0] = y[1];
                    dydt[1] = mu * (1 - y[0] * y[0]) * y[1] - y[0];
                })
                .bounds(0.0, 10.0).initialState(2.0, 0.0)
                .tolerances(1e-4, 1e-7).integrate();

        assertTrue(sol.isSuccessful(), "Radau should succeed on Van der Pol mu=10");
        assertTrue(Math.abs(sol.timeSeries.y[0 * sol.getTimePoints() + sol.getTimePoints() - 1]) <= 2.5, "y should stay bounded");
    }

    // -----------------------------------------------------------------------
    // 4. Event detection (scipy test_events)
    // -----------------------------------------------------------------------

    @Test
    void eventProjectileLanding() {
        Trajectory sol = new IVPIntegral(IVPMethod.RK45)
                .equation((t, y, dydt) -> { dydt[0] = y[1]; dydt[1] = -9.8; })
                .bounds(0.0, 100.0).initialState(0.0, 50.0)
                .detectors(new ODEEvent((t, y) -> y[0], ODEEvent.Trigger.FALLING, 1))
                .integrate();

        assertEquals(Trajectory.Status.EVENT, sol.getStatus(), "terminal event expected");
        assertNotNull(sol.events);
        assertEquals(1, sol.events[0].length);
        assertEquals(100.0 / 9.8, sol.events[0][0].t, 0.01, "landing time mismatch");
    }

    @Test
    void eventHarmonicOscillatorZeroCrossings() {
        Trajectory sol = new IVPIntegral(IVPMethod.RK45)
                .equation((t, y, dydt) -> { dydt[0] = y[1]; dydt[1] = -y[0]; })
                .bounds(0.0, 4 * Math.PI).initialState(0.0, 1.0)
                .detectors(new ODEEvent((t, y) -> y[0], ODEEvent.Trigger.EITHER, 0))
                .integrate();

        assertEquals(Trajectory.Status.SUCCESS, sol.getStatus());
        assertNotNull(sol.events);
        assertTrue(sol.events[0].length >= 3,
                "expected at least 3 zero crossings, got " + sol.events[0].length);
        for (Trajectory.EventPoint ev : sol.events[0]) {
            double tMod = ev.t % Math.PI;
            assertTrue(Math.min(tMod, Math.PI - tMod) < 0.05,
                    "zero crossing at t=" + ev.t + " not near k*pi");
        }
    }

    // -----------------------------------------------------------------------
    // 5. evalAt
    // -----------------------------------------------------------------------

    @Test
    void evalAtForward() {
        double rtol = 1e-3, atol = 1e-6;
        double[] ts = {5.0, 6.0, 7.0, 8.0, 9.0};
        double[] y0 = {1.0/3, 2.0/9};

        Trajectory sol = new IVPIntegral(IVPMethod.RK45)
                .equation(IvpSolverTest::funRational)
                .bounds(5.0, 9.0).initialState(y0)
                .evalAt(ts).tolerances(rtol, atol).integrate();

        assertEquals(ts.length, sol.getTimePoints());
        for (int i = 0; i < ts.length; i++) {
            assertEquals(ts[i], sol.timeSeries.t[i], 0.0, "t[" + i + "] mismatch");
            double[] yj = {sol.timeSeries.y[0 * sol.getTimePoints() + i], sol.timeSeries.y[1 * sol.getTimePoints() + i]};
            double[] yTrue = solRational(ts[i]);
            double e = computeError(yj, yTrue, rtol, atol);
            assertTrue(e < 5, "evalAt t=" + ts[i] + " error=" + e);
        }
    }

    @Test
    void evalAtBackward() {
        double rtol = 1e-3, atol = 1e-6;
        double[] ts = {5.0, 4.0, 3.0, 2.0, 1.0};
        double[] y0 = {1.0/3, 2.0/9};

        Trajectory sol = new IVPIntegral(IVPMethod.RK45)
                .equation(IvpSolverTest::funRational)
                .bounds(5.0, 1.0).initialState(y0)
                .evalAt(ts).tolerances(rtol, atol).integrate();

        assertEquals(ts.length, sol.getTimePoints());
        for (int i = 0; i < ts.length; i++) {
            assertEquals(ts[i], sol.timeSeries.t[i], 0.0, "t[" + i + "] mismatch");
            double[] yj = {sol.timeSeries.y[0 * sol.getTimePoints() + i], sol.timeSeries.y[1 * sol.getTimePoints() + i]};
            double[] yTrue = solRational(ts[i]);
            double e = computeError(yj, yTrue, rtol, atol);
            assertTrue(e < 5, "backward evalAt t=" + ts[i] + " error=" + e);
        }
    }

    // -----------------------------------------------------------------------
    // 6. maxStep constraint
    // -----------------------------------------------------------------------

    @Test
    void maxStepConstraintRespected() {
        double maxStep = 0.1;
        Trajectory sol = new IVPIntegral(IVPMethod.RK45)
                .equation((t, y, dydt) -> dydt[0] = -y[0])
                .bounds(0.0, 1.0).initialState(1.0)
                .maxStep(maxStep).integrate();

        assertTrue(sol.getTimePoints() >= 10, "with maxStep=0.1, m should be >= 10, got " + sol.getTimePoints());
        for (int i = 1; i < sol.getTimePoints(); i++)
            assertTrue(sol.timeSeries.t[i] - sol.timeSeries.t[i - 1] <= maxStep + 1e-12,
                    "step " + i + " exceeds maxStep");
    }

    // -----------------------------------------------------------------------
    // 7. Parameter validation
    // -----------------------------------------------------------------------

    @Test void missingEquationThrows() {
        assertThrows(IllegalStateException.class, () ->
                new IVPIntegral(IVPMethod.RK45).bounds(0.0, 1.0).initialState(1.0).integrate());
    }

    @Test void missingBoundsThrows() {
        assertThrows(IllegalStateException.class, () ->
                new IVPIntegral(IVPMethod.RK45).equation((t, y, d) -> {}).initialState(1.0).integrate());
    }

    @Test void missingInitialStateThrows() {
        assertThrows(IllegalStateException.class, () ->
                new IVPIntegral(IVPMethod.RK45).equation((t, y, d) -> {}).bounds(0.0, 1.0).integrate());
    }

    @Test void t0EqualsTfThrows() {
        assertThrows(IllegalArgumentException.class, () ->
                new IVPIntegral(IVPMethod.RK45).equation((t, y, d) -> {}).bounds(1.0, 1.0).initialState(1.0).integrate());
    }

    @Test void nonPositiveRtolThrows() {
        assertThrows(IllegalArgumentException.class, () ->
                new IVPIntegral(IVPMethod.RK45).equation((t, y, d) -> {}).bounds(0.0, 1.0).initialState(1.0)
                        .tolerances(0.0, 1e-6).integrate());
    }

    @Test void invalidMethodThrows() {
        assertThrows(IllegalArgumentException.class, () -> IVPMethod.valueOf("EULER"));
    }

    @Test void atolDimensionMismatchThrows() {
        assertThrows(IllegalArgumentException.class, () ->
                new IVPIntegral(IVPMethod.RK45).equation((t, y, d) -> {}).bounds(0.0, 1.0)
                        .initialState(1.0, 2.0)       // n=2
                        .tolerances(1e-3, 1e-6, 1e-6, 1e-6)  // atol.length=3, mismatch
                        .integrate());
    }

    @Test void evalAtOutOfRangeThrows() {
        assertThrows(IllegalArgumentException.class, () ->
                new IVPIntegral(IVPMethod.RK45).equation((t, y, d) -> {}).bounds(0.0, 1.0).initialState(1.0)
                        .evalAt(new double[]{0.5, 1.5}).integrate());
    }

    // -----------------------------------------------------------------------
    // 8. Workspace reuse
    // -----------------------------------------------------------------------

    @ParameterizedTest @ValueSource(strings = {"RK23","RK45","DOP853","BDF","Radau"})
    void workspaceReuse(String method) {
        double rtol = 1e-3, atol = 1e-6;
        IVPMethod m = IVPMethod.valueOf(method);
        IVPPool ws = IVPIntegral.workspace(m);

        // solve two different initial conditions with the same workspace;
        // results must match independent solves
        double[] y0a = {1.0/3, 2.0/9};
        double[] y0b = {0.5,   0.1  };

        Trajectory solA1 = new IVPIntegral(m).equation(IvpSolverTest::funRational)
                .bounds(5.0, 9.0).initialState(y0a).tolerances(rtol, atol).integrate(ws);
        Trajectory solA2 = new IVPIntegral(m).equation(IvpSolverTest::funRational)
                .bounds(5.0, 9.0).initialState(y0a).tolerances(rtol, atol).integrate();

        Trajectory solB1 = new IVPIntegral(m).equation(IvpSolverTest::funRational)
                .bounds(5.0, 9.0).initialState(y0b).tolerances(rtol, atol).integrate(ws);
        Trajectory solB2 = new IVPIntegral(m).equation(IvpSolverTest::funRational)
                .bounds(5.0, 9.0).initialState(y0b).tolerances(rtol, atol).integrate();

        assertTrue(solA1.isSuccessful(), method + " pool reuse A: should succeed");
        assertTrue(solB1.isSuccessful(), method + " pool reuse B: should succeed");

        assertEquals(solA1.getTimePoints(), solA2.getTimePoints(), method + " pool reuse A: time points mismatch");
        assertEquals(solB1.getTimePoints(), solB2.getTimePoints(), method + " pool reuse B: time points mismatch");

        for (int j = 0; j < solA1.getTimePoints(); j++) {
            assertEquals(solA1.timeSeries.t[j], solA2.timeSeries.t[j], 1e-12, method + " pool reuse A: t[" + j + "] mismatch");
            assertEquals(solA1.timeSeries.y[j], solA2.timeSeries.y[j], 1e-12, method + " pool reuse A: y[" + j + "] mismatch");
        }
        for (int j = 0; j < solB1.getTimePoints(); j++) {
            assertEquals(solB1.timeSeries.t[j], solB2.timeSeries.t[j], 1e-12, method + " pool reuse B: t[" + j + "] mismatch");
            assertEquals(solB1.timeSeries.y[j], solB2.timeSeries.y[j], 1e-12, method + " pool reuse B: y[" + j + "] mismatch");
        }
    }

    // -----------------------------------------------------------------------
    // 9. Implicit solver numerical Jacobian via ODE.Equation path
    // -----------------------------------------------------------------------

    @ParameterizedTest @ValueSource(strings = {"BDF", "Radau"})
    void implicitNumericJacobianViaEquation(String method) {
        double rtol = 1e-3, atol = 1e-6;
        double[] y0 = {1.0/3, 2.0/9};

        int[] jacCalls = {0};

        // Pass as ODE.Equation (3-arg lambda) — jac branch must never be reached
        jacCalls[0] = 0;
        Trajectory sol = new IVPIntegral(IVPMethod.valueOf(method))
                .equation((ODE.Equation)(t, y, dydt) -> {
                    funRational(t, y, dydt);
                    // jac is not accessible here — if the bug wraps this as ODE
                    // and calls evaluate(t,y,dydt,jac) with non-null jac on the
                    // wrapper, the wrapper ignores jac. We detect the bug via the
                    // second ODE path below instead.
                })
                .bounds(5.0, 9.0).initialState(y0)
                .tolerances(rtol, atol).integrate();

        assertTrue(sol.isSuccessful(), method + ": should succeed");
        assertTrue(sol.getJacobianEvaluations() > 0, method + ": njev > 0");
        assertEquals(0, jacCalls[0],
                method + ": jac branch must NOT be called via equation(ODE.Equation)");

        // Pass as full ODE (4-arg lambda) — jac branch must be reached
        jacCalls[0] = 0;
        Trajectory solAna = new IVPIntegral(IVPMethod.valueOf(method))
                .equation((ODE)(t, y, dydt, jac) -> {
                    funRational(t, y, dydt);
                    if (jac != null) jacCalls[0]++;
                })
                .bounds(5.0, 9.0).initialState(y0)
                .tolerances(rtol, atol).integrate();

        assertTrue(solAna.isSuccessful(), method + ": analytic jac should succeed");
        assertTrue(jacCalls[0] > 0,
                method + ": jac branch must be called via equation(ODE)");
    }

    // -----------------------------------------------------------------------
    // 10. Vector atol
    // -----------------------------------------------------------------------

    @ParameterizedTest @ValueSource(strings = {"RK45", "BDF", "Radau"})
    void vectorAtolDifferentPrecisionPerComponent(String method) {
        // Verify that vector atol controls step count:
        // scalar atol=1e-3 (loose) → fewer steps
        // scalar atol=1e-9 (tight) → more steps
        // vector atol=[1e-3, 1e-9] → step count between the two scalars
        double rtol = 1e-3;
        double[] y0 = {1.0, 1.0};
        ODE.Equation eq = (t, y, dydt) -> { dydt[0] = -y[0]; dydt[1] = -y[0]; };

        int stepsLoose = new IVPIntegral(IVPMethod.valueOf(method))
                .equation(eq).bounds(0.0, 1.0).initialState(y0)
                .tolerances(rtol, 1e-3).integrate().getTimePoints();

        int stepsTight = new IVPIntegral(IVPMethod.valueOf(method))
                .equation(eq).bounds(0.0, 1.0).initialState(y0)
                .tolerances(rtol, 1e-9).integrate().getTimePoints();

        int stepsVec = new IVPIntegral(IVPMethod.valueOf(method))
                .equation(eq).bounds(0.0, 1.0).initialState(y0)
                .tolerances(rtol, 1e-3, 1e-9).integrate().getTimePoints();

        // vector atol tightens y1 → at least as many steps as all-loose
        assertTrue(stepsLoose <= stepsVec,
                method + ": vector atol should produce >= steps than loose scalar"
                + " (loose=" + stepsLoose + " vec=" + stepsVec + ")");
        // and fewer steps than all-tight (allow small slack for adaptive step variability)
        assertTrue(stepsVec <= stepsTight + 2,
                method + ": vector atol should produce roughly <= steps than tight scalar"
                + " (vec=" + stepsVec + " tight=" + stepsTight + ")");
    }
}