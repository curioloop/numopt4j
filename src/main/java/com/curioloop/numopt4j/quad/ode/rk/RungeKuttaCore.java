package com.curioloop.numopt4j.quad.ode.rk;

import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.quad.ode.ODE;
import com.curioloop.numopt4j.quad.ode.IVPCore;

/**
 * Generic explicit Runge-Kutta stepping core, strictly following scipy rk.py RungeKutta._step_impl and rk_step.
 *
 * <p>Supports three methods: RK23 (Bogacki-Shampine), RK45 (Dormand-Prince), DOP853,
 * created via static factory methods {@link #rk23}, {@link #rk45}, {@link #dop853}.</p>
 *
 * <p>All vector operations use hand-written loops, reusing buffers in {@link RungeKuttaPool}, zero allocation in hot path.</p>
 */
public class RungeKuttaCore extends IVPCore<RungeKuttaPool> {

    // Step size control constants (corresponding to scipy rk.py)
    private static final double SAFETY     = 0.9;
    private static final double MIN_FACTOR = 0.2;
    private static final double MAX_FACTOR = 10.0;

    // right-hand side function
    private final ODE.Equation fun;

    // Butcher tableau
    private final ButcherTableau tab;

    private final double maxStep;
    private final double rtol;
    private final double[] atol;


    // -----------------------------------------------------------------------
    // Constructor (private, created via factory methods)
    // -----------------------------------------------------------------------

    private RungeKuttaCore(
            ODE.Equation fun,
            double t0, double[] y0, double tBound,
            double rtol, double[] atol, double maxStep, double firstStep,
            RungeKuttaPool ws,
            ButcherTableau tab) {

        super(y0.length, t0, y0, tBound, ws);
        ws.ensure(tab.nStages, n, tab.nStagesExtended, tab.interpOrder);

        this.fun           = fun;
        this.tab           = tab;
        this.rtol          = rtol;
        this.atol          = atol;
        this.maxStep       = maxStep;

        // Compute initial f(t0, y0)
        fun.evaluate(t0, this.y, this.f);
        nfev++;

        // Initial step size
        if (Double.isNaN(firstStep)) {
            this.hAbs = selectInitialStep(fun, t0, y0, this.f, tab.order, rtol, atol, maxStep,
                    this.ws.yTmp, this.ws.fTmp);
        } else {
            this.hAbs = Math.abs(firstStep);
        }
    }

    // -----------------------------------------------------------------------
    // Static factory methods
    // -----------------------------------------------------------------------

    /** Creates a RK23 (Bogacki-Shampine 3(2) order) solver. */
    public static RungeKuttaCore rk23(
            ODE.Equation fun, double t0, double[] y0, double tBound,
            double rtol, double[] atol, double maxStep, double firstStep,
            RungeKuttaPool ws) {
        return new RungeKuttaCore(fun, t0, y0, tBound, rtol, atol, maxStep, firstStep, ws,
                ButcherTableau.RK23);
    }

    /** Creates a RK45 (Dormand-Prince 5(4) order) solver. */
    public static RungeKuttaCore rk45(
            ODE.Equation fun, double t0, double[] y0, double tBound,
            double rtol, double[] atol, double maxStep, double firstStep,
            RungeKuttaPool ws) {
        return new RungeKuttaCore(fun, t0, y0, tBound, rtol, atol, maxStep, firstStep, ws,
                ButcherTableau.RK45);
    }

    /** Creates a DOP853 (8(5,3) order) solver. */
    public static RungeKuttaCore dop853(
            ODE.Equation fun, double t0, double[] y0, double tBound,
            double rtol, double[] atol, double maxStep, double firstStep,
            RungeKuttaPool ws) {
        return new RungeKuttaCore(fun, t0, y0, tBound, rtol, atol, maxStep, firstStep, ws,
                ButcherTableau.DOP853);
    }

    /** Creates a solver from a custom {@link ButcherTableau}. */
    public static RungeKuttaCore of(
            ODE.Equation fun, double t0, double[] y0, double tBound,
            double rtol, double[] atol, double maxStep, double firstStep,
            RungeKuttaPool ws, ButcherTableau tableau) {
        return new RungeKuttaCore(fun, t0, y0, tBound, rtol, atol, maxStep, firstStep, ws, tableau);
    }

    // -----------------------------------------------------------------------
    // step() — corresponding to scipy RungeKutta._step_impl
    // -----------------------------------------------------------------------

    @Override
    protected boolean step() {
        double t   = this.t;
        double[] y = this.y;

        double minStep = 10.0 * Math.ulp(t);

        double hAbs;
        if (this.hAbs > maxStep) {
            hAbs = maxStep;
        } else if (this.hAbs < minStep) {
            hAbs = minStep;
        } else {
            hAbs = this.hAbs;
        }

        boolean stepAccepted = false;
        boolean stepRejected = false;

        double tNew = t;
        double h    = 0;

        while (!stepAccepted) {
            if (hAbs < minStep) {
                return false;  // TOO_SMALL_STEP
            }

            h    = hAbs * direction;
            tNew = t + h;

            // Clamp to tBound
            if (direction * (tNew - tBound) > 0) {
                tNew = tBound;
            }
            h    = tNew - t;
            hAbs = Math.abs(h);

            // Execute rk_step, results written to ws.yTmp (y_new) and ws.fTmp (f_new)
            rkStep(t, y, this.f, h);

            // scale = atol + max(|y|, |y_new|) * rtol
            double[] yNew  = ws.yTmp;
            double[] scale = ws.errBuf;
            for (int i = 0; i < n; i++) {
                scale[i] = atol(atol, i) + Math.max(Math.abs(y[i]), Math.abs(yNew[i])) * rtol;
            }

            // Error norm
            double errorNorm = estimateErrorNorm(h, scale);

            if (errorNorm < 1.0) {
                double factor;
                if (errorNorm == 0.0) {
                    factor = MAX_FACTOR;
                } else {
                    factor = Math.min(MAX_FACTOR, SAFETY * Math.pow(errorNorm, tab.errorExponent));
                }
                if (stepRejected) {
                    factor = Math.min(1.0, factor);
                }
                hAbs *= factor;
                stepAccepted = true;
            } else {
                hAbs *= Math.max(MIN_FACTOR, SAFETY * Math.pow(errorNorm, tab.errorExponent));
                stepRejected = true;
            }
        }

        // Step accepted, update state
        // Save y_new and f_new first (computeDenseOutput will overwrite ws.yTmp/ws.fTmp)
        double[] yNew = ws.yTmp;
        double[] fNew = ws.fTmp;

        // Compute dense output coefficients and update workspace
        computeDenseOutput(h);

        // Update ws interpolation state
        ws.tOld = t;
        ws.tCur = tNew;
        System.arraycopy(y, 0, ws.yOld, 0, n);

        // Update solver state (using saved y_new/f_new references, but DOP853 has overwritten ws.yTmp)
        // For DOP853, need to recover y_new from ws.interpCoeffs F[0]: y_new = yOld + F[0]
        this.tOld = t;
        this.t    = tNew;
        if (tab.D != null) {
            // DOP853: F[0] = delta_y = y_new - y_old, so y_new = y_old + F[0]
            double[] F = ws.interpCoeffs;
            for (int i = 0; i < n; i++) {
                this.y[i] = y[i] + F[i];  // F[0*n+i] = delta_y[i]
            }
            // f_new has also been overwritten, need to recompute
            fun.evaluate(tNew, this.y, this.f);
            nfev++;
        } else {
            System.arraycopy(yNew, 0, this.y, 0, n);
            System.arraycopy(fNew, 0, this.f, 0, n);
        }
        this.hAbs = hAbs;

        return true;
    }

    // -----------------------------------------------------------------------
    // rk_step — corresponding to scipy rk_step function
    // Results written to ws.yTmp (y_new) and ws.fTmp (f_new), K written to ws.K
    // -----------------------------------------------------------------------

    private void rkStep(double t, double[] y, double[] f, double h) {
        double[] K = ws.K;

        // K[0] = f
        System.arraycopy(f, 0, K, 0, n);

        // Compute stage slopes K[1..nStages-1]
        // Lower triangular structure: row s has s elements, offset = s*(s-1)/2
        for (int s = 1; s < tab.nStages; s++) {
            int aOff = s * (s - 1) / 2;
            // yTmp = y + h * K(s×n)^T · A[aOff..aOff+s-1]
            System.arraycopy(y, 0, ws.yTmp, 0, n);
            BLAS.dgemv(BLAS.Trans.Trans, s, n, h, K, 0, n, tab.A, aOff, 1, 1.0, ws.yTmp, 0, 1);
            fun.evaluate(t + tab.C[s] * h, ws.yTmp, ws.fTmp);
            System.arraycopy(ws.fTmp, 0, K, s * n, n);
            nfev++;
        }

        // y_new = y + h * K(nStages×n)^T · B
        double[] yNew = ws.yTmp;
        System.arraycopy(y, 0, yNew, 0, n);
        BLAS.dgemv(BLAS.Trans.Trans, tab.nStages, n, h, K, 0, n, tab.B, 0, 1, 1.0, yNew, 0, 1);

        // f_new = fun(t + h, y_new)
        fun.evaluate(t + h, ws.yTmp, ws.fTmp);
        nfev++;

        // K[-1] = f_new (K[nStages])
        System.arraycopy(ws.fTmp, 0, K, tab.nStages * n, n);
    }

    // -----------------------------------------------------------------------
    // Error norm estimation
    // -----------------------------------------------------------------------

    /** Standard RK error norm: norm(E·K·h / scale), RMS norm. */
    private double estimateErrorNorm(double h, double[] scale) {
        if (tab.E3 != null) {
            return estimateErrorNormDop853(h, scale);
        }
        double[] K = ws.K;
        double[] E = tab.E;
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double err = 0.0;
            for (int s = 0; s <= tab.nStages; s++) {
                err += E[s] * K[s * n + i];
            }
            err = (err * h) / scale[i];
            sum += err * err;
        }
        return Math.sqrt(sum / n);
    }

    /**
     * DOP853 error norm:
     *   err5 = dot(K.T, E5) / scale
     *   err3 = dot(K.T, E3) / scale
     *   denom = err5_norm^2 + 0.01 * err3_norm^2
     *   return |h| * err5_norm^2 / sqrt(denom * n)
     */
    private double estimateErrorNormDop853(double h, double[] scale) {
        double[] K  = ws.K;
        double[] E5 = tab.E5, E3 = tab.E3;
        double err5NormSq = 0.0, err3NormSq = 0.0;
        for (int i = 0; i < n; i++) {
            double e5 = 0.0, e3 = 0.0;
            for (int s = 0; s <= tab.nStages; s++) {
                double ki = K[s * n + i];
                e5 += E5[s] * ki;
                e3 += E3[s] * ki;
            }
            e5 /= scale[i];
            e3 /= scale[i];
            err5NormSq += e5 * e5;
            err3NormSq += e3 * e3;
        }
        if (err5NormSq == 0.0 && err3NormSq == 0.0) return 0.0;
        return Math.abs(h) * err5NormSq / Math.sqrt((err5NormSq + 0.01 * err3NormSq) * n);
    }

    // -----------------------------------------------------------------------
    // Dense output coefficient computation
    // -----------------------------------------------------------------------

    /**
     * Computes dense output coefficients, writing to ws.interpCoeffs.
     *
     * <p>Standard methods (RK23/RK45): Q = K^T · P, shape (n, interpOrder+1).</p>
     * <p>DOP853: first compute extended stages, then build F matrix (7×n), stored in ws.interpCoeffs.</p>
     */
    private void computeDenseOutput(double h) {
        if (tab.D != null) {
            computeDenseOutputDop853(h);
        } else {
            computeDenseOutputStandard();
        }
    }

    /**
     * Standard RK dense output: Q = K^T · P.
     * Q shape (n, interpOrder), row-major stored in ws.interpCoeffs.
     * Corresponds to scipy: Q = self.K.T.dot(self.P)
     */
    private void computeDenseOutputStandard() {
        int cols = tab.interpOrder;
        // Q(n×cols) = K^T((tab.nStages+1)×n transposed) · P((tab.nStages+1)×cols)
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, n, cols, tab.nStages + 1,
                1.0, ws.K, 0, n, tab.P, 0, cols, 0.0, ws.interpCoeffs, 0, cols);
    }

    /**
     * DOP853 dense output:
     * 1. Compute extended stages K[nStages+1..N_STAGES_EXTENDED-1]
     * 2. Build F matrix (7×n)
     * 3. Store in ws.interpCoeffs (row-major, shape (7, n))
     *
     * Corresponds to scipy DOP853._dense_output_impl
     */
    private void computeDenseOutputDop853(double h) {
        double   tOld = this.t;   // step start (this.t not yet updated in step())
        double[] yOld = this.y;   // step start state
        double[] yNew = ws.yTmp;  // y_new (written by rkStep)
        double[] fNew = ws.fTmp;  // f_new (written by rkStep)
        double[] K    = ws.KExtended;

        // K shares the same array as ws.K (merged in ensure), no copy needed

        // Save F[0..2] needed values (y_new/f_new) to interpQ first, to avoid being overwritten by subsequent computation
        // F[0] = delta_y, F[1] = h*f_old - delta_y, F[2] = 2*delta_y - h*(f_new+f_old)
        double[] F = ws.interpCoeffs;
        for (int i = 0; i < n; i++) {
            double deltaY = yNew[i] - yOld[i];
            double fOldI  = K[i];       // K[0*n + i] = f_old
            double fNewI  = fNew[i];
            F[0 * n + i] = deltaY;
            F[1 * n + i] = h * fOldI - deltaY;
            F[2 * n + i] = 2.0 * deltaY - h * (fNewI + fOldI);
        }

        // Compute extended stages K[nStages+1..N_STAGES_EXTENDED-1]
        for (int idx = 0; idx < tab.nExtraStages; idx++) {
            int r     = tab.nStages + 1 + idx;
            int aOff  = r * (r - 1) / 2;
            double cs = tab.C_EXTRA[idx];

            // stageIn = yOld + h * K(r×n)^T · A[aOff..aOff+r-1]
            System.arraycopy(yOld, 0, ws.yTmp, 0, n);
            BLAS.dgemv(BLAS.Trans.Trans, r, n, h, K, 0, n, tab.A, aOff, 1, 1.0, ws.yTmp, 0, 1);

            fun.evaluate(tOld + cs * h, ws.yTmp, ws.fTmp);
            System.arraycopy(ws.fTmp, 0, K, r * n, n);
            nfev++;
        }

        // F[3..6](4×n) = h * D(4×16) · K(16×n), row-major dgemm
        int nExt = tab.nStagesExtended;
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, tab.dRows, n, nExt,
                h, tab.D, 0, nExt, K, 0, n, 0.0, F, 3 * n, n);
    }

    // -----------------------------------------------------------------------
    // interpolate — standard RK delegates to ws.interpolate; DOP853 uses dedicated scheme
    // -----------------------------------------------------------------------

    @Override
    protected void interpolate(double t, double[] out) {
        if (tab.D != null) {
            interpolateDop853(t, out);
        } else {
            ws.interpolate(t, out);
        }
    }

    /**
     * DOP853 dense output interpolation, corresponding to scipy Dop853DenseOutput._call_impl:
     *   x = (t - tOld) / h
     *   y = F[6] * x
     *   y = (y + F[5]) * (1-x)
     *   y = (y + F[4]) * x
     *   y = (y + F[3]) * (1-x)
     *   y = (y + F[2]) * x
     *   y = (y + F[1]) * (1-x)
     *   y = (y + F[0]) * x
     *   y += yOld
     *
     * ws.interpCoeffs stores F, shape (7, n), row-major.
     */
    private void interpolateDop853(double t, double[] out) {
        double h = ws.tCur - ws.tOld;
        double x = (t - ws.tOld) / h;
        double[] F    = ws.interpCoeffs;
        double[] yOld = ws.yOld;
        int power = tab.interpOrder;  // 7

        // Horner-like scheme: iterate reversed(F), alternating multiply by x and (1-x)
        // i=0 (F[6]): y = F[6]; y *= x
        // i=1 (F[5]): y += F[5]; y *= (1-x)
        // i=2 (F[4]): y += F[4]; y *= x
        // ...
        for (int i = 0; i < n; i++) {
            double val = 0.0;
            for (int fi = 0; fi < power; fi++) {
                // reversed index: power-1-fi
                int ri = power - 1 - fi;
                val += F[ri * n + i];
                if (fi % 2 == 0) {
                    val *= x;
                } else {
                    val *= (1.0 - x);
                }
            }
            out[i] = yOld[i] + val;
        }
    }
}
