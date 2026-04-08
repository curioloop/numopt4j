package com.curioloop.numopt4j.quad.ode.implicit;

import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.quad.ode.ODE;

/**
 * BDF (Backward Differentiation Formula) implicit ODE solver core, strictly following scipy bdf.py BDF._step_impl.
 *
 * <p>Variable-order method, order automatically varies from 1 to 5 (MAX_ORDER).
 * Linear algebra operations directly call {@link BLAS#dgetrf} / {@link BLAS#dgetrs} without wrapper classes.</p>
 *
 * <p>Difference table D layout: {@code D[k*n + i]} = i-th component of k-th order difference (k=0..MAX_ORDER+2).</p>
 */
public class BDFCore extends ImplicitCore<BDFPool> {

    // -----------------------------------------------------------------------
    // Constants (corresponding to scipy bdf.py)
    // -----------------------------------------------------------------------

    private static final int    MAX_ORDER      = BDFPool.MAX_ORDER;  // 5
    private static final int    NEWTON_MAXITER = 4;
    private static final double MIN_FACTOR     = 0.2;
    private static final double MAX_FACTOR     = 10.0;

    // NDF correction coefficients kappa[0..5]
    private static final double[] KAPPA = {0.0, -0.1850, -1.0/9, -0.0823, -0.0415, 0.0};

    // gamma[k] = sum(1/j for j in 1..k), gamma[0]=0
    private static final double[] GAMMA;
    // alpha[k] = (1 - kappa[k]) * gamma[k]
    private static final double[] ALPHA;
    // error_const[k] = kappa[k]*gamma[k] + 1/(k+1)
    private static final double[] ERROR_CONST;

    // U_ALL[order] = row-major expansion of computeR(order, 1.0), offset at U_OFF[order]
    // Pre-computed U matrices for all orders (0..MAX_ORDER) to avoid repeated computation in changeD hot path
    private static final double[] U_ALL;
    private static final int[]    U_OFF;

    static {
        GAMMA       = new double[MAX_ORDER + 1];
        ALPHA       = new double[MAX_ORDER + 1];
        ERROR_CONST = new double[MAX_ORDER + 2];

        GAMMA[0] = 0.0;
        for (int k = 1; k <= MAX_ORDER; k++) {
            GAMMA[k] = GAMMA[k - 1] + 1.0 / k;
        }
        for (int k = 0; k <= MAX_ORDER; k++) {
            ALPHA[k]       = (1.0 - KAPPA[k]) * GAMMA[k];
            ERROR_CONST[k] = KAPPA[k] * GAMMA[k] + 1.0 / (k + 1);
        }
        double gammaNext = GAMMA[MAX_ORDER] + 1.0 / (MAX_ORDER + 1);
        ERROR_CONST[MAX_ORDER + 1] = 0.0 * gammaNext + 1.0 / (MAX_ORDER + 2);

        // Pre-compute U = computeR(order, 1.0) for order = 0..MAX_ORDER
        U_OFF = new int[MAX_ORDER + 1];
        int total = 0;
        for (int order = 0; order <= MAX_ORDER; order++) {
            U_OFF[order] = total;
            total += (order + 1) * (order + 1);
        }
        U_ALL = new double[total];
        for (int order = 0; order <= MAX_ORDER; order++) {
            int sz  = order + 1;
            int off = U_OFF[order];
            for (int j = 0; j < sz; j++) U_ALL[off + j] = 1.0;
            for (int i = 1; i < sz; i++) {
                for (int j = 1; j < sz; j++) {
                    double mij = ((i - 1) - (double) j) / (double) i;
                    U_ALL[off + i * sz + j] = U_ALL[off + (i - 1) * sz + j] * mij;
                }
                U_ALL[off + i * sz] = 0.0;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Fields
    // -----------------------------------------------------------------------

    private int    order;
    private int    nEqualSteps;
    private boolean bdfConverged;
    private int     bdfIter;

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /**
     * Creates a BDF solver.
     *
     * @param fun      right-hand side function (ODE.Equation)
     * @param ode      full interface with optional analytic Jacobian (null = numerical diff)
     * @param t0       initial time
     * @param y0       initial state (length n)
     * @param tBound   end time
     * @param rtol     relative tolerance
     * @param atol     absolute tolerance
     * @param maxStep  maximum step size
     * @param firstStep initial step size (NaN = auto-estimate)
     * @param ws       workspace (null = auto-allocate)
     */
    public BDFCore(ODE.Equation fun, ODE ode,
                   double t0, double[] y0, double tBound,
                   double rtol, double atol, double maxStep, double firstStep,
                   BDFPool ws) {
        super(y0.length, t0, y0, tBound, ws, fun, ode, rtol, atol, maxStep);
        ws.ensure(n);

        // Compute initial f(t0, y0)
        fun.evaluate(t0, this.y, this.f);
        nfev++;

        // Initial step size
        if (Double.isNaN(firstStep)) {
            this.hAbs = selectInitialStep(fun, t0, y0, this.f, 1, rtol, atol, maxStep,
                    this.ws.yTmp, this.ws.fTmp);
        } else {
            this.hAbs = Math.abs(firstStep);
        }

        // Initialize difference table D
        // D[0] = y0, D[1] = f0 * h * direction
        double[] D = this.ws.D;
        System.arraycopy(this.y, 0, D, 0, n);
        for (int i = 0; i < n; i++) {
            D[n + i] = this.f[i] * this.hAbs * this.direction;
        }
        // D[2..MAX_ORDER+2] = 0 (already zero-initialized by ensure)

        this.order       = 1;
        this.nEqualSteps = 0;
        this.luValid     = false;
        this.jacCurrent  = false;

        // Initial Jacobian (numerical diff)
        computeJacobian(t0, this.y);
    }

    // -----------------------------------------------------------------------
    // step() — corresponding to scipy BDF._step_impl
    // -----------------------------------------------------------------------

    @Override
    protected boolean step() {
        double t   = this.t;
        double[] D = ws.D;

        double minStep = 10.0 * Math.ulp(t);

        // Clamp step size to maxStep / minStep
        double hAbs = this.hAbs;
        if (hAbs > maxStep) {
            changeD(D, order, maxStep / hAbs);
            hAbs = maxStep;
            nEqualSteps = 0;
            luValid = false;
        } else if (hAbs < minStep) {
            changeD(D, order, minStep / hAbs);
            hAbs = minStep;
            nEqualSteps = 0;
            luValid = false;
        }

        int    order      = this.order;
        double safety     = 0.9 * (2.0 * NEWTON_MAXITER + 1.0) / (2.0 * NEWTON_MAXITER + 1.0);
        // safety is recomputed after Newton iteration based on n_iter

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
                double factor = Math.abs(tNew - t) / hAbs;
                changeD(D, order, factor);
                nEqualSteps = 0;
                luValid = false;
            }

            h    = tNew - t;
            hAbs = Math.abs(h);

            // Predict y_predict and psi, merged into a single loop
            double[] yPredict = ws.yTmp;
            double[] psi = ws.psiVec;
            double invAlpha = 1.0 / ALPHA[order];
            for (int i = 0; i < n; i++) {
                double yp = D[i], ps = 0.0;
                for (int k = 1; k <= order; k++) {
                    double dk = D[k * n + i];
                    yp += dk;
                    ps += dk * GAMMA[k];
                }
                yPredict[i] = yp;
                psi[i] = ps * invAlpha;
            }

            // scale = atol + rtol * |y_predict|
            double[] scale = ws.scaleBuf;
            for (int i = 0; i < n; i++) {
                scale[i] = atol + rtol * Math.abs(yPredict[i]);
            }

            double c = h / ALPHA[order];

            // Newton iteration (up to two rounds: first with old Jacobian, then with new Jacobian)
            boolean converged = false;
            int nIter = 0;
            // Align with scipy: current_jac = self.jac is None
            // When using numerical diff (ode==null), the Jacobian is always considered current
            // because we just computed it; no need to recompute on Newton failure.
            boolean currentJac = (ode == null) || jacCurrent;

            while (!converged) {
                // Build LU factorization (if needed)
                if (!luValid) {
                    buildLU(c);
                    luValid = true;
                }

                // Newton iteration
                solveBdfSystem(fun, tNew, yPredict, c, psi, scale);
                converged = bdfConverged;
                nIter     = bdfIter;

                if (!converged) {
                    if (currentJac) {
                        break;  // still not converged with latest Jacobian, give up
                    }
                    // Recompute Jacobian and retry
                    computeJacobian(tNew, yPredict);
                    currentJac = true;
                    luValid = false;
                }
            }

            if (!converged) {
                // Halve the step size
                double factor = 0.5;
                hAbs *= factor;
                changeD(D, order, factor);
                nEqualSteps = 0;
                luValid = false;
                continue;
            }

            // safety = 0.9 * (2*NEWTON_MAXITER+1) / (2*NEWTON_MAXITER+n_iter)
            safety = 0.9 * (2.0 * NEWTON_MAXITER + 1.0) / (2.0 * NEWTON_MAXITER + nIter);

            // Error estimate: error = error_const[order] * d
            double[] d     = ws.dBuf;
            double[] yNew  = ws.fTmp;  // reuse fBuf to store y_new (written by solveBdfSystem)

            // Recompute scale using y_new
            for (int i = 0; i < n; i++) {
                scale[i] = atol + rtol * Math.abs(yNew[i]);
            }

            double errorConst = ERROR_CONST[order];
            double errorNormSq = 0.0;
            for (int i = 0; i < n; i++) {
                double e = errorConst * d[i] / scale[i];
                errorNormSq += e * e;
            }
            double errorNorm = Math.sqrt(errorNormSq / n);

            if (errorNorm > 1.0) {
                double factor = Math.max(MIN_FACTOR, safety * Math.pow(errorNorm, -1.0 / (order + 1)));
                hAbs *= factor;
                changeD(D, order, factor);
                nEqualSteps = 0;
                // do not reset LU (convergence issue is not in Jacobian)
                continue;
            }

            // Step accepted
            stepAccepted = true;

            // Jacobian reuse strategy: reuse when nIter <= 2
            jacCurrent = (nIter <= 2);
            if (!jacCurrent) {
                luValid = false;
            }

            nEqualSteps++;

            // Update difference table D
            // D[order+2] = d - D[order+1]
            // D[order+1] = d
            System.arraycopy(d, 0, D, (order + 2) * n, n);
            BLAS.daxpy(n, -1.0, D, (order + 1) * n, 1, D, (order + 2) * n, 1);
            System.arraycopy(d, 0, D, (order + 1) * n, n);
            for (int k = order; k >= 0; k--) {
                BLAS.daxpy(n, 1.0, D, (k + 1) * n, 1, D, k * n, 1);
            }

            // Update solver state
            this.tOld = t;
            this.t    = tNew;
            System.arraycopy(yNew, 0, this.y, 0, n);
            fun.evaluate(tNew, this.y, this.f);
            nfev++;
            this.hAbs = hAbs;

            // Update interpolation coefficients (dense output)
            saveDenseOutput(h, order);

            // Order selection (after order+1 consecutive equal-size steps)
            if (nEqualSteps < order + 1) {
                this.order = order;
                return true;
            }

            double errorMNorm = Double.POSITIVE_INFINITY;
            double errorPNorm = Double.POSITIVE_INFINITY;

            if (order > 1) {
                double ecM = ERROR_CONST[order - 1];
                double sumM = 0.0;
                for (int i = 0; i < n; i++) {
                    double e = ecM * D[order * n + i] / scale[i];
                    sumM += e * e;
                }
                errorMNorm = Math.sqrt(sumM / n);
            }

            if (order < MAX_ORDER) {
                double ecP = ERROR_CONST[order + 1];
                double sumP = 0.0;
                for (int i = 0; i < n; i++) {
                    double e = ecP * D[(order + 2) * n + i] / scale[i];
                    sumP += e * e;
                }
                errorPNorm = Math.sqrt(sumP / n);
            }

            // factors = [errorMNorm, errorNorm, errorPNorm] ^ (-1/[order, order+1, order+2])
            double fM = (errorMNorm == 0.0) ? Double.MAX_VALUE : Math.pow(errorMNorm, -1.0 / order);
            double fC = (errorNorm  == 0.0) ? Double.MAX_VALUE : Math.pow(errorNorm,  -1.0 / (order + 1));
            double fP = (errorPNorm == 0.0) ? Double.MAX_VALUE : Math.pow(errorPNorm, -1.0 / (order + 2));

            int deltaOrder;
            double maxFactor;
            if (fM >= fC && fM >= fP) {
                deltaOrder = -1;
                maxFactor  = fM;
            } else if (fP >= fC && fP >= fM) {
                deltaOrder = +1;
                maxFactor  = fP;
            } else {
                deltaOrder = 0;
                maxFactor  = fC;
            }

            order += deltaOrder;
            order  = Math.max(1, Math.min(order, MAX_ORDER));

            double factor = Math.min(MAX_FACTOR, safety * maxFactor);
            hAbs *= factor;
            changeD(D, order, factor);
            nEqualSteps = 0;
            luValid = false;

            this.order = order;
            this.hAbs  = hAbs;
        }

        return true;
    }

    // -----------------------------------------------------------------------
    // interpolate — delegates to BDFPool
    // -----------------------------------------------------------------------

    @Override
    protected void interpolate(double t, double[] out) {
        ws.interpolate(t, out);
    }

    // -----------------------------------------------------------------------
    // Private helper methods
    // -----------------------------------------------------------------------

    /**
     * Builds system matrix I - c*J and performs in-place LU factorization, writing to ws.luBuf / ws.ipiv.
     * Corresponds to scipy: LU = lu(I - c * J)
     */
    private void buildLU(double c) {
        double[] J     = ws.jacBuf;
        double[] luBuf = ws.luBuf;
        int[]    ipiv  = ws.ipiv;

        // luBuf = I - c*J: copy J, scale by -c, then add diagonal
        System.arraycopy(J, 0, luBuf, 0, n * n);
        BLAS.dscal(n * n, -c, luBuf, 0, 1);
        for (int i = 0; i < n; i++) luBuf[i * n + i] += 1.0;

        int info = BLAS.dgetrf(n, n, luBuf, 0, n, ipiv, 0);
        if (info != 0) {
            // Singular matrix: mark LU invalid, force Jacobian recomputation
            luValid = false;
        }
        nlu++;
    }

    /**
     * Newton iteration to solve the BDF algebraic system.
     * Corresponds to scipy solve_bdf_system.
     *
     * <p>Result: y_new written to ws.fTmp, d written to ws.dBuf.</p>
     */
    private void solveBdfSystem(ODE.Equation fun, double tNew,
                                double[] yPredict, double c,
                                double[] psi, double[] scale) {
        double[] d    = ws.dBuf;
        double[] y    = ws.fTmp;
        double[] dyf  = ws.newtonBuf;

        System.arraycopy(yPredict, 0, y, 0, n);
        java.util.Arrays.fill(d, 0, n, 0.0);

        double dyNormOld = Double.NaN;
        boolean converged = false;
        int k;

        for (k = 0; k < NEWTON_MAXITER; k++) {
            // f = fun(t_new, y)
            fun.evaluate(tNew, y, dyf);
            nfev++;

            // Check if f is finite
            boolean finite = true;
            for (int i = 0; i < n; i++) {
                if (!Double.isFinite(dyf[i])) { finite = false; break; }
            }
            if (!finite) break;

            // dyf = c*dyf - psi - d
            BLAS.dscal(n, c, dyf, 0, 1);
            BLAS.daxpy(n, -1.0, psi, 0, 1, dyf, 0, 1);
            BLAS.daxpy(n, -1.0, d, 0, 1, dyf, 0, 1);
            // Solve in-place: (I - c*J) * dy = rhs
            BLAS.dgetrs(BLAS.Trans.NoTrans, n, 1, ws.luBuf, 0, n, ws.ipiv, 0, dyf, 0, 1);

            // dy_norm = norm(dy / scale)
            double dyNormSq = 0.0;
            for (int i = 0; i < n; i++) {
                double v = dyf[i] / scale[i];
                dyNormSq += v * v;
            }
            double dyNorm = Math.sqrt(dyNormSq / n);

            if (!Double.isNaN(dyNormOld)) {
                double rate = dyNorm / dyNormOld;
                if (rate >= 1.0 || Math.pow(rate, NEWTON_MAXITER - k) / (1.0 - rate) * dyNorm > newtonTol) {
                    break;
                }
                // convergence check reuses the same rate
                for (int i = 0; i < n; i++) { y[i] += dyf[i]; d[i] += dyf[i]; }
                if (dyNorm == 0.0 || rate / (1.0 - rate) * dyNorm < newtonTol) {
                    converged = true;
                    break;
                }
            } else {
                for (int i = 0; i < n; i++) { y[i] += dyf[i]; d[i] += dyf[i]; }
                if (dyNorm == 0.0) {
                    converged = true;
                    break;
                }
            }

            dyNormOld = dyNorm;
        }

        bdfConverged = converged;
        bdfIter      = k + 1;
    }

    /**
     * Writes dense output coefficients to ws.interpCoeffs.
     * Format: [h, order, t_shift[0..order-1], denom[0..order-1], D[0..order] flat]
     * Corresponds to scipy BdfDenseOutput.__init__:
     *   t_shift[k] = t - h * k  (k=0..order-1)
     *   denom[k]   = h * (1 + k)
     */
    private void saveDenseOutput(double h, int order) {
        // coefficient array size: 2 + 2*order + (order+1)*n
        int coeffSize = 2 + 2 * order + (order + 1) * n;
        if (ws.interpCoeffs == null || ws.interpCoeffs.length < coeffSize) {
            ws.interpCoeffs = new double[coeffSize];
        }

        double tCur = this.t;
        ws.interpCoeffs[0] = h;
        ws.interpCoeffs[1] = order;

        for (int k = 0; k < order; k++) {
            ws.interpCoeffs[2 + k]         = tCur - h * k;   // t_shift[k]
            ws.interpCoeffs[2 + order + k] = h * (1.0 + k);  // denom[k]
        }

        // D[0..order] flat (n elements per row)
        int dOffset = 2 + 2 * order;
        double[] D = ws.D;
        for (int k = 0; k <= order; k++) {
            System.arraycopy(D, k * n, ws.interpCoeffs, dOffset + k * n, n);
        }

        ws.tOld = this.tOld;
        ws.tCur = tCur;
        ws.n    = n;
    }

    // -----------------------------------------------------------------------
    // change_D — update difference table when step size / order changes
    // -----------------------------------------------------------------------

    /**
     * Corresponds to scipy change_D(D, order, factor).
     * Updates difference table D[:order+1] in-place, using ws.RU and ws.changeDTmp to avoid allocation.
     */
    private void changeD(double[] D, int order, double factor) {
        int sz = order + 1;
        double[] RU  = ws.RU;
        double[] tmp = ws.changeDTmp;  // first sz^2 elements used as R temp matrix

        computeRU(order, factor, RU, tmp);

        // tmp = snapshot of D[:order+1] (reuse changeDTmp)
        double[] snap = ws.changeDTmp;
        for (int k = 0; k < sz; k++) {
            System.arraycopy(D, k * n, snap, k * n, n);
        }

        // D(sz×n) = RU^T(sz×sz) · snap(sz×n), row-major dgemm
        BLAS.dgemm(BLAS.Trans.Trans, BLAS.Trans.NoTrans, sz, n, sz,
                1.0, RU, 0, sz, snap, 0, n, 0.0, D, 0, n);
    }

    /** Computes RU = R(order,factor) · U_ALL[order], result written to buf; R temp stored in tmp (length >= sz^2). */
    private static void computeRU(int order, double factor, double[] buf, double[] tmp) {
        int sz  = order + 1;
        int uOff = U_OFF[order];

        // R = computeR(order, factor), stored in tmp
        for (int j = 0; j < sz; j++) tmp[j] = 1.0;
        for (int i = 1; i < sz; i++) {
            for (int j = 1; j < sz; j++) {
                double mij = ((i - 1) - factor * j) / (double) i;
                tmp[i * sz + j] = tmp[(i - 1) * sz + j] * mij;
            }
            tmp[i * sz] = 0.0;
        }

        // buf = R · U_ALL[order], row-major dgemm (sz max 6)
        BLAS.dgemm(BLAS.Trans.NoTrans, BLAS.Trans.NoTrans, sz, sz, sz,
                1.0, tmp, 0, sz, U_ALL, uOff, sz, 0.0, buf, 0, sz);
    }
}
