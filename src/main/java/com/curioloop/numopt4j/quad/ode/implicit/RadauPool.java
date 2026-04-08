package com.curioloop.numopt4j.quad.ode.implicit;




/**
 * Radau IIA solver dedicated workspace.
 * Real-expanded complex system: 2n×2n real LU replaces complex LU.
 */
public class RadauPool extends ImplicitPool {

    public double[] luReal;    // real eigenvalue system LU, n×n
    public int[]    ipivReal;  // pivots, length n
    public double[] luCplx;    // real-expanded complex system LU, 2n×2n
    public int[]    ipivCplx;  // pivots, length 2n

    public double[] Z;         // collocation increments, 3×n, Z[s*n+i]
    public double[] W;         // transformed W, 3×n
    public double[] F;         // stage F values / Newton increments, 3×n
    // rhsReal reuses base class newtonBuf (length n), non-overlapping usage

    // solveCollocationSystem output fields (avoid new double[3] each call)
    public boolean newtonConverged;
    public int     newtonIter;
    public double  newtonRate;

    @Override
    public RadauPool ensure(int n) {
        super.ensure(n);
        if (luReal == null || luReal.length < n * n) luReal = new double[n * n];
        if (ipivReal == null || ipivReal.length < n) ipivReal = new int[n];
        int cn = 2 * n;
        if (luCplx == null || luCplx.length < cn * cn) luCplx = new double[cn * cn];
        if (ipivCplx == null || ipivCplx.length < cn) ipivCplx = new int[cn];
        if (Z == null || Z.length < 3 * n) Z = new double[3 * n];
        if (W == null || W.length < 3 * n) W = new double[3 * n];
        if (F == null || F.length < 3 * n) F = new double[3 * n];
        return this;
    }

    /**
     * Radau dense output interpolation from snapshot (cubic polynomial).
     * snapshot format: [yOld[0..n-1], Q[0..n-1][0..2] flat (n×3, row-major)]
     */
    @Override
    public void interpolate(double t, double[] snapshot, double tOld, double tCur, double[] out) {
        double h = tCur - tOld;
        double x = (t - tOld) / h;
        double x2 = x * x, x3 = x2 * x;
        for (int i = 0; i < n; i++) {
            out[i] = snapshot[i]
                   + snapshot[n + i * 3]     * x
                   + snapshot[n + i * 3 + 1] * x2
                   + snapshot[n + i * 3 + 2] * x3;
        }
    }

    /**
     * Radau dense output interpolation (cubic polynomial satisfying collocation conditions).
     * Corresponds to scipy RadauDenseOutput._call_impl:
     *   x = (t - tOld) / h
     *   p = [x, x^2, x^3] (cumprod)
     *   y = Q · p + yOld
     * interpCoeffs stores [yOld[0..n-1], Q[0..n-1][0..2] flat (n×3, row-major)]
     */
    @Override
    public void interpolate(double t, double[] out) {
        if (interpCoeffs == null) throw new IllegalStateException("No interpolation data");
        double h = tCur - tOld;
        double x = (t - tOld) / h;
        double x2 = x * x;
        double x3 = x2 * x;
        // out[i] = yOld[i] + Q[i][0]*x + Q[i][1]*x^2 + Q[i][2]*x^3
        for (int i = 0; i < n; i++) {
            double yOldI = interpCoeffs[i];
            double q0 = interpCoeffs[n + i * 3];
            double q1 = interpCoeffs[n + i * 3 + 1];
            double q2 = interpCoeffs[n + i * 3 + 2];
            out[i] = yOldI + q0 * x + q1 * x2 + q2 * x3;
        }
    }
}
