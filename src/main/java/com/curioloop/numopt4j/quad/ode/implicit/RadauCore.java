package com.curioloop.numopt4j.quad.ode.implicit;

import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.quad.ode.ODE;

/**
 * Radau IIA implicit Runge-Kutta solver core (5th order, 3 stages), strictly following scipy radau.py Radau._step_impl.
 *
 * <p>Linear algebra operations directly call {@link BLAS#dgetrf} / {@link BLAS#dgetrs} without wrapper classes.
 * Complex eigenvalues are handled by expanding into a 2n×2n real system.</p>
 */
public class RadauCore extends ImplicitCore<RadauPool> {

    // -----------------------------------------------------------------------
    // Radau static constants (corresponding to scipy radau.py)
    // -----------------------------------------------------------------------

    private static final double S6 = Math.sqrt(6.0);

    /** Collocation nodes C[0..2], corresponding to scipy C */
    private static final double[] C_NODES = {
        (4.0 - S6) / 10.0,
        (4.0 + S6) / 10.0,
        1.0
    };

    /** Error coefficients E[0..2], corresponding to scipy E */
    private static final double[] E_COEFF = {
        (-13.0 - 7.0 * S6) / 3.0,
        (-13.0 + 7.0 * S6) / 3.0,
        -1.0 / 3.0
    };

    /** Real eigenvalue, corresponding to scipy MU_REAL */
    private static final double MU_REAL = 3.0 + Math.pow(3.0, 2.0/3.0) - Math.pow(3.0, 1.0/3.0);

    /** Real part of complex eigenvalue, corresponding to scipy MU_COMPLEX.real */
    private static final double MU_COMPLEX_REAL =
        3.0 + 0.5 * (Math.pow(3.0, 1.0/3.0) - Math.pow(3.0, 2.0/3.0));

    /** Imaginary part of complex eigenvalue (negative), corresponding to scipy MU_COMPLEX.imag */
    private static final double MU_COMPLEX_IMAG =
        -0.5 * (Math.pow(3.0, 5.0/6.0) + Math.pow(3.0, 7.0/6.0));

    /**
     * Transformation matrix T (3×3, row-major), corresponding to scipy T.
     * T[i][j] = T_FLAT[i*3+j]
     */
    private static final double[] T_FLAT = {
         0.09443876248897524, -0.14125529502095421,  0.03002919410514742,
         0.25021312296533332,  0.20412935229379994, -0.38294211275726192,
         1.0,                  1.0,                  0.0
    };

    /**
     * Inverse transformation matrix TI (3×3, row-major), corresponding to scipy TI.
     * TI[i][j] = TI_FLAT[i*3+j]
     */
    private static final double[] TI_FLAT = {
         4.17871859155190428,  0.32768282076106237,  0.52337644549944951,
        -4.17871859155190428, -0.32768282076106237,  0.47662355450055044,
         0.50287263494578682, -2.57192694985560522,  0.59603920482822492
    };

    /** Row 0 of TI (real system), corresponding to scipy TI_REAL */
    private static final double[] TI_REAL = {
         4.17871859155190428,  0.32768282076106237,  0.52337644549944951
    };

    /** Row 1 of TI (real part of complex system), corresponding to scipy TI_COMPLEX.real */
    private static final double[] TI_CPLX_RE = {
        -4.17871859155190428, -0.32768282076106237,  0.47662355450055044
    };

    /** Row 2 of TI (imaginary part of complex system), corresponding to scipy TI_COMPLEX.imag */
    private static final double[] TI_CPLX_IM = {
         0.50287263494578682, -2.57192694985560522,  0.59603920482822492
    };

    /**
     * Interpolation coefficient matrix P (3×3, row-major), corresponding to scipy P.
     * P[i][j] = P_FLAT[i*3+j]
     */
    private static final double[] P_FLAT = {
        13.0/3.0 + 7.0*S6/3.0, -23.0/3.0 - 22.0*S6/3.0, 10.0/3.0 + 5.0*S6,
        13.0/3.0 - 7.0*S6/3.0, -23.0/3.0 + 22.0*S6/3.0, 10.0/3.0 - 5.0*S6,
        1.0/3.0,                -8.0/3.0,                  10.0/3.0
    };

    // -----------------------------------------------------------------------
    // Algorithm constants
    // -----------------------------------------------------------------------

    private static final int    NEWTON_MAXITER = 6;
    private static final double MIN_FACTOR     = 0.2;
    private static final double MAX_FACTOR     = 10.0;

    // -----------------------------------------------------------------------
    // Fields
    // -----------------------------------------------------------------------

    private double  hAbsOld;      // previous step size (used for predict_factor)
    private double  errorNormOld; // previous step error norm (used for predict_factor)

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /**
     * Creates a Radau IIA solver.
     *
     * @param fun       right-hand side function (ODE.Equation)
     * @param ode       full interface with optional analytic Jacobian (null = numerical diff)
     * @param t0        initial time
     * @param y0        initial state (length n)
     * @param tBound    end time
     * @param rtol      relative tolerance
     * @param atol      absolute tolerance
     * @param maxStep   maximum step size
     * @param firstStep initial step size (NaN = auto-estimate)
     * @param ws        workspace (null = auto-allocate)
     */
    public RadauCore(ODE.Equation fun, ODE ode,
                     double t0, double[] y0, double tBound,
                     double rtol, double atol, double maxStep, double firstStep,
                     RadauPool ws) {
        super(y0.length, t0, y0, tBound, ws, fun, ode, rtol, atol, maxStep);
        ws.ensure(n);

        // Compute initial f(t0, y0)
        fun.evaluate(t0, this.y, this.f);
        nfev++;

        // Initial step size
        if (Double.isNaN(firstStep)) {
            this.hAbs = selectInitialStep(fun, t0, y0, this.f, 3, rtol, atol, maxStep,
                    this.ws.yTmp, this.ws.fTmp);
        } else {
            this.hAbs = Math.abs(firstStep);
        }

        this.luValid     = false;
        this.jacCurrent  = false;
        this.hAbsOld     = Double.NaN;
        this.errorNormOld = Double.NaN;

        // Initial Jacobian
        computeJacobian(t0, this.y);
    }

    // -----------------------------------------------------------------------
    // step() — corresponding to scipy Radau._step_impl
    // -----------------------------------------------------------------------

    @Override
    protected boolean step() {
        double t = this.t;

        double minStep = 10.0 * Math.ulp(t);

        double hAbs = this.hAbs;
        double hAbsOld = this.hAbsOld;
        double errorNormOld = this.errorNormOld;

        // Clamp step size to maxStep / minStep (same as scipy)
        if (hAbs > maxStep) {
            hAbs = maxStep;
            hAbsOld = Double.NaN;
            errorNormOld = Double.NaN;
        } else if (hAbs < minStep) {
            hAbs = minStep;
            hAbsOld = Double.NaN;
            errorNormOld = Double.NaN;
        }

        boolean rejected = false;
        boolean stepAccepted = false;

        while (!stepAccepted) {
            if (hAbs < minStep) {
                return false;  // TOO_SMALL_STEP
            }

            double h    = hAbs * direction;
            double tNew = t + h;

            // Clamp to tBound
            if (direction * (tNew - tBound) > 0) {
                tNew = tBound;
            }
            h    = tNew - t;
            hAbs = Math.abs(h);

            // Initial guess Z0: use dense-output interpolation from previous step (scipy approach),
            // which gives a better starting point than reusing the previous Z directly.
            // Falls back to zero on the first step or when no interpolation data is available.
            double[] Z = ws.Z;
            if (Double.isNaN(tOld) || ws.interpCoeffs == null) {
                // First step or no dense output yet: Z0 = 0
                for (int i = 0; i < 3 * n; i++) Z[i] = 0.0;
            } else {
                // Z0[s] = sol(t + h * C[s]) - y,  s = 0, 1, 2
                double[] tmp = ws.yTmp;
                for (int s = 0; s < 3; s++) {
                    ws.interpolate(t + h * C_NODES[s], tmp);
                    for (int i = 0; i < n; i++) Z[s * n + i] = tmp[i] - y[i];
                }
            }

            // scale = atol + |y| * rtol
            double[] scale = ws.scaleBuf;
            for (int i = 0; i < n; i++) {
                scale[i] = atol + Math.abs(y[i]) * rtol;
            }

            boolean converged = false;
            int nIter = 0;
            double rate = Double.NaN;

            while (!converged) {
                // Build LU factorization (if needed)
                if (!luValid) {
                    buildLU(h);
                    luValid = true;
                }

                // Newton iteration, results written to ws.newtonConverged/newtonIter/newtonRate
                solveCollocationSystem(t, y, h, Z, scale);
                converged = ws.newtonConverged;
                nIter     = ws.newtonIter;
                rate      = ws.newtonRate;

                if (!converged) {
                    if (jacCurrent) {
                        break;  // still not converged with latest Jacobian
                    }
                    // Recompute Jacobian and retry
                    computeJacobian(t, y);
                    jacCurrent = true;
                    luValid = false;
                }
            }

            if (!converged) {
                hAbs *= 0.5;
                luValid = false;
                continue;
            }

            // y_new = y + Z[2]
            double[] yNew = ws.yTmp;
            System.arraycopy(y, 0, yNew, 0, n);
            BLAS.daxpy(n, 1.0, Z, 2 * n, 1, yNew, 0, 1);

            // rhsReal = f + Z^T · E_COEFF / h, use dgemv instead of manual loop
            double[] rhsReal = ws.newtonBuf;
            System.arraycopy(f, 0, rhsReal, 0, n);
            BLAS.dgemv(BLAS.Trans.Trans, 3, n, 1.0 / h, Z, 0, n, E_COEFF, 0, 1, 1.0, rhsReal, 0, 1);

            // error = solve_lu(LU_real, f + ZE)
            BLAS.dgetrs(BLAS.Trans.NoTrans, n, 1, ws.luReal, 0, n, ws.ipivReal, 0, rhsReal, 0, 1);
            double[] error = rhsReal;  // solved in-place, rhsReal is now error

            // scale = atol + max(|y|, |y_new|) * rtol
            for (int i = 0; i < n; i++) {
                scale[i] = atol + Math.max(Math.abs(y[i]), Math.abs(yNew[i])) * rtol;
            }

            double safety = 0.9 * (2.0 * NEWTON_MAXITER + 1.0) / (2.0 * NEWTON_MAXITER + nIter);

            // If rejected and error norm > 1, re-estimate error (corresponds to scipy rejected branch)
            double errorNorm = normScaled(error, scale, n);

            if (rejected && errorNorm > 1.0) {
                // error = solve_lu(LU_real, fun(t, y + error) + ZE)
                double[] yPlusErr = ws.fTmp;
                System.arraycopy(y, 0, yPlusErr, 0, n);
                BLAS.daxpy(n, 1.0, error, 0, 1, yPlusErr, 0, 1);
                fun.evaluate(t, yPlusErr, rhsReal);
                nfev++;
                BLAS.dgemv(BLAS.Trans.Trans, 3, n, 1.0 / h, Z, 0, n, E_COEFF, 0, 1, 1.0, rhsReal, 0, 1);
                BLAS.dgetrs(BLAS.Trans.NoTrans, n, 1, ws.luReal, 0, n, ws.ipivReal, 0, rhsReal, 0, 1);
                errorNorm = normScaled(rhsReal, scale, n);
            }

            if (errorNorm > 1.0) {
                double factor = predictFactor(hAbs, hAbsOld, errorNorm, errorNormOld);
                hAbs *= Math.max(MIN_FACTOR, safety * factor);
                luValid = false;
                rejected = true;
                continue;
            }

            // Step accepted
            stepAccepted = true;

            // Jacobian reuse strategy: reuse when n_iter <= 2 && rate <= 1e-3
            boolean recomputeJac = (ode != null) && (nIter > 2) && (!Double.isNaN(rate) && rate > 1e-3);

            double factor = predictFactor(hAbs, hAbsOld, errorNorm, errorNormOld);
            factor = Math.min(MAX_FACTOR, safety * factor);

            if (!recomputeJac && factor < 1.2) {
                factor = 1.0;
            } else {
                luValid = false;
            }

            // Compute f_new
            double[] fNew = ws.fTmp;
            fun.evaluate(tNew, yNew, fNew);
            nfev++;

            if (recomputeJac) {
                computeJacobian(tNew, yNew);
                jacCurrent = true;
            } else if (ode != null) {
                jacCurrent = false;
            }

            // Update state
            this.hAbsOld      = hAbs;
            this.errorNormOld = errorNorm;
            this.hAbs         = hAbs * factor;
            this.tOld         = t;
            this.t            = tNew;

            // Dense output: Q = Z^T · P, shape (n, 3)
            // Must save yOld before updating this.y
            saveDenseOutput(tNew, t, this.y, Z);

            System.arraycopy(yNew, 0, this.y, 0, n);
            System.arraycopy(fNew, 0, this.f, 0, n);
            // ws.Z has been updated in solveCollocationSystem
        }

        return true;
    }

    // -----------------------------------------------------------------------
    // interpolate — delegates to RadauPool
    // -----------------------------------------------------------------------

    @Override
    protected void interpolate(double t, double[] out) {
        ws.interpolate(t, out);
    }

    // -----------------------------------------------------------------------
    // Private helper methods
    // -----------------------------------------------------------------------

    /**
     * Builds real system matrix (n×n) and real-expanded complex system matrix (2n×2n), then performs in-place LU factorization.
     * Corresponds to scipy:
     *   LU_real    = lu(MU_REAL/h * I - J)
     *   LU_complex = lu(MU_COMPLEX/h * I - J)  → real-expanded to 2n×2n
     */
    private void buildLU(double h) {
        double[] J       = ws.jacBuf;
        double[] luReal  = ws.luReal;
        double[] luCplx  = ws.luCplx;
        int[]    ipivR   = ws.ipivReal;
        int[]    ipivC   = ws.ipivCplx;
        int      cn      = 2 * n;

        double mReal = MU_REAL / h;
        double a     = MU_COMPLEX_REAL / h;  // real part of complex eigenvalue / h
        double b     = MU_COMPLEX_IMAG / h;  // imaginary part of complex eigenvalue / h

        // Real system: luReal = mReal*I - J
        System.arraycopy(J, 0, luReal, 0, n * n);
        BLAS.dscal(n * n, -1.0, luReal, 0, 1);
        for (int i = 0; i < n; i++) luReal[i * n + i] += mReal;
        BLAS.dgetrf(n, n, luReal, 0, n, ipivR, 0);
        nlu++;

        // Real-expanded complex system (2n×2n):
        // [a*I - J,  -b*I]
        // [b*I,       a*I - J]
        // Zero out the entire matrix first
        java.util.Arrays.fill(luCplx, 0, cn * cn, 0.0);
        // Fill upper-left and lower-right blocks (a*I - J): copy -J, then add diagonal a
        for (int i = 0; i < n; i++) {
            // Upper-left block [i, 0..n-1]
            BLAS.daxpy(n, -1.0, J, i * n, 1, luCplx, i * cn, 1);
            luCplx[i * cn + i] += a;
            // Lower-right block [n+i, n..2n-1]
            BLAS.daxpy(n, -1.0, J, i * n, 1, luCplx, (n + i) * cn + n, 1);
            luCplx[(n + i) * cn + (n + i)] += a;
            // Upper-right block [i, n+i]: -b*I (diagonal only)
            luCplx[i * cn + (n + i)] = -b;
            // Lower-left block [n+i, i]: b*I (diagonal only)
            luCplx[(n + i) * cn + i] = b;
        }
        BLAS.dgetrf(cn, cn, luCplx, 0, cn, ipivC, 0);
        nlu++;
    }

    /**
     * Newton iteration to solve the collocation system.
     * Results written to ws.newtonConverged / ws.newtonIter / ws.newtonRate, zero allocation.
     */
    private void solveCollocationSystem(double t, double[] y, double h,
                                        double[] Z, double[] scale) {
        double mReal    = MU_REAL / h;
        double mCplxRe  = MU_COMPLEX_REAL / h;
        double mCplxIm  = MU_COMPLEX_IMAG / h;

        double[] W    = ws.W;
        double[] F    = ws.F;
        double[] dW   = ws.F;   // dW reuses F: F is read before dW is written, same array
        double[] rhs  = ws.newtonBuf;
        // Complex system RHS written directly to dW[n..3n-1], saving an independent buffer
        int      cn   = 2 * n;

        // W = TI · Z0 (3×n), expand 3×3 matrix multiplication
        for (int i = 0; i < n; i++) {
            double z0 = Z[i], z1 = Z[n + i], z2 = Z[2 * n + i];
            W[i]           = TI_FLAT[0] * z0 + TI_FLAT[1] * z1 + TI_FLAT[2] * z2;
            W[n + i]       = TI_FLAT[3] * z0 + TI_FLAT[4] * z1 + TI_FLAT[5] * z2;
            W[2 * n + i]   = TI_FLAT[6] * z0 + TI_FLAT[7] * z1 + TI_FLAT[8] * z2;
        }

        double dWNormOld = Double.NaN;
        boolean converged = false;
        double rate = Double.NaN;
        int k;

        for (k = 0; k < NEWTON_MAXITER; k++) {
            // Compute stage values F[s] = fun(t + h*C[s], y + Z[s])
            double[] yTmp = ws.yTmp;
            for (int s = 0; s < 3; s++) {
                for (int i = 0; i < n; i++) {
                    yTmp[i] = y[i] + Z[s * n + i];
                }
                fun.evaluate(t + h * C_NODES[s], yTmp, ws.fTmp);
                System.arraycopy(ws.fTmp, 0, F, s * n, n);
                nfev++;
            }

            // Check if F is finite
            boolean finite = true;
            for (int i = 0; i < 3 * n; i++) {
                if (!Double.isFinite(F[i])) { finite = false; break; }
            }
            if (!finite) break;

            // Build RHS for three systems, merged into one loop for better cache locality
            for (int i = 0; i < n; i++) {
                double f0 = F[i], f1 = F[n + i], f2 = F[2 * n + i];
                double w0 = W[i], w1 = W[n + i], w2 = W[2 * n + i];
                // Real system RHS
                rhs[i] = f0 * TI_REAL[0] + f1 * TI_REAL[1] + f2 * TI_REAL[2] - mReal * w0;
                // Complex system RHS (real/imaginary parts written directly to dW[n..3n-1])
                dW[n + i]     = f0 * TI_CPLX_RE[0] + f1 * TI_CPLX_RE[1] + f2 * TI_CPLX_RE[2]
                                - (mCplxRe * w1 - mCplxIm * w2);
                dW[2 * n + i] = f0 * TI_CPLX_IM[0] + f1 * TI_CPLX_IM[1] + f2 * TI_CPLX_IM[2]
                                - (mCplxIm * w1 + mCplxRe * w2);
            }

            // Solve real system: dW[0..n-1] = solve_lu(LU_real, rhs)
            System.arraycopy(rhs, 0, dW, 0, n);
            BLAS.dgetrs(BLAS.Trans.NoTrans, n, 1, ws.luReal, 0, n, ws.ipivReal, 0, dW, 0, 1);

            // Solve real-expanded complex system (2n×2n): dW[n..3n-1] already contains RHS, solve in-place
            BLAS.dgetrs(BLAS.Trans.NoTrans, cn, 1, ws.luCplx, 0, cn, ws.ipivCplx, 0, dW, n, 1);

            // dW_norm = norm(dW / scale), expand s=0,1,2
            double dWNormSq = 0.0;
            for (int i = 0; i < n; i++) {
                double si = scale[i];
                double v0 = dW[i] / si, v1 = dW[n + i] / si, v2 = dW[2 * n + i] / si;
                dWNormSq += v0 * v0 + v1 * v1 + v2 * v2;
            }
            double dWNorm = Math.sqrt(dWNormSq / (3 * n));

            if (!Double.isNaN(dWNormOld)) {
                rate = dWNorm / dWNormOld;
                if (rate >= 1.0 || Math.pow(rate, NEWTON_MAXITER - k) / (1.0 - rate) * dWNorm > newtonTol) {
                    break;
                }
            }

            // W += dW
            BLAS.daxpy(3 * n, 1.0, dW, 0, 1, W, 0, 1);

            // Z = T · W (3×n), expand 3×3 matrix multiplication
            for (int i = 0; i < n; i++) {
                double w0 = W[i], w1 = W[n + i], w2 = W[2 * n + i];
                Z[i]           = T_FLAT[0] * w0 + T_FLAT[1] * w1 + T_FLAT[2] * w2;
                Z[n + i]       = T_FLAT[3] * w0 + T_FLAT[4] * w1 + T_FLAT[5] * w2;
                Z[2 * n + i]   = T_FLAT[6] * w0 + T_FLAT[7] * w1 + T_FLAT[8] * w2;
            }

            if (dWNorm == 0.0 || (!Double.isNaN(rate) && rate / (1.0 - rate) * dWNorm < newtonTol)) {
                converged = true;
                break;
            }

            dWNormOld = dWNorm;
        }

        ws.newtonConverged = converged;
        ws.newtonIter      = k + 1;
        ws.newtonRate      = rate;
    }

    /**
     * Dense output: Q = Z^T · P, shape (n, 3).
     * Writes to ws.interpCoeffs, format: [yOld[0..n-1], Q[0..n-1][0..2] flat (n×3, row-major)]
     * Corresponds to scipy _compute_dense_output: Q = np.dot(Z.T, P)
     */
    private void saveDenseOutput(double tCur, double tOld, double[] yOld, double[] Z) {
        int coeffSize = n + n * 3;
        if (ws.interpCoeffs == null || ws.interpCoeffs.length < coeffSize) {
            ws.interpCoeffs = new double[coeffSize];
        }

        // yOld[0..n-1]
        System.arraycopy(yOld, 0, ws.interpCoeffs, 0, n);

        // Q(n×3) = Z^T((3×n) transposed) · P_FLAT(3×3), written to interpCoeffs[n..]
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, n, 3, 3,
                1.0, Z, 0, n, P_FLAT, 0, 3, 0.0, ws.interpCoeffs, n, 3);

        ws.tOld = tOld;
        ws.tCur = tCur;
        ws.n    = n;
    }

    /**
     * Step size prediction factor, corresponding to scipy predict_factor.
     */
    private static double predictFactor(double hAbs, double hAbsOld,
                                         double errorNorm, double errorNormOld) {
        double multiplier;
        if (Double.isNaN(errorNormOld) || Double.isNaN(hAbsOld) || errorNorm == 0.0) {
            multiplier = 1.0;
        } else {
            multiplier = hAbs / hAbsOld * Math.pow(errorNormOld / errorNorm, 0.25);
        }
        if (errorNorm == 0.0) return MAX_FACTOR;
        return Math.min(1.0, multiplier) * Math.pow(errorNorm, -0.25);
    }
}
