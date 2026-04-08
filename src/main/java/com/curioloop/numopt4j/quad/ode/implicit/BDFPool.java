package com.curioloop.numopt4j.quad.ode.implicit;


/**
 * BDF solver dedicated workspace.
 * Difference table D layout: D[k*n + i] = i-th component of k-th order difference (k=0..MAX_ORDER+2)
 */
public class BDFPool extends ImplicitPool {

    public static final int MAX_ORDER = 5;

    public double[] D;       // difference table, (MAX_ORDER+3) × n
    public double[] luBuf;   // LU factors, n×n, overwritten in-place by dgetrf
    public int[]    ipiv;    // pivot indices, length n
    public double[] dBuf;    // Newton iteration accumulator d, length n
    public double[] psiVec;  // psi vector, length n
    public double[] changeDTmp; // changeD temp buffer, max(6n, 36); first 36 elements also used as R temp matrix in computeRU
    public double[] RU;         // changeD matrix buffer, (MAX_ORDER+1)^2

    /** Per-component adaptive step-size factor for numerical Jacobian. */
    public double[] jacFactor;

    @Override
    public BDFPool ensure(int n) {
        super.ensure(n);
        int dSize = (MAX_ORDER + 3) * n;
        if (D == null || D.length < dSize) D = new double[dSize];
        if (luBuf == null || luBuf.length < n * n) luBuf = new double[n * n];
        if (ipiv == null || ipiv.length < n) ipiv = new int[n];
        if (dBuf == null || dBuf.length < n) dBuf = new double[n];
        if (psiVec == null || psiVec.length < n) psiVec = new double[n];
        int szMax = MAX_ORDER + 1;
        int szSq  = szMax * szMax;
        // changeDTmp also used as R temp matrix in computeRU (first szSq elements)
        // needs max(szMax*n, szSq) elements
        int tmpSize = Math.max(szMax * n, szSq);
        if (changeDTmp == null || changeDTmp.length < tmpSize) changeDTmp = new double[tmpSize];
        if (RU  == null || RU.length  < szSq) RU  = new double[szSq];
        return this;
    }

    /**
     * BDF dense output interpolation from snapshot (Newton backward difference polynomial).
     * snapshot format: [h, order, t_shift[0..order-1], denom[0..order-1], D[0..order] flat]
     */
    @Override
    public void interpolate(double t, double[] snapshot, double tOld, double tCur, double[] out) {
        int order = (int) snapshot[1];
        double[] p = new double[order];
        for (int k = 0; k < order; k++) {
            p[k] = (t - snapshot[2 + k]) / snapshot[2 + order + k];
        }
        for (int k = 1; k < order; k++) p[k] *= p[k - 1];
        int dOffset = 2 + 2 * order;
        for (int i = 0; i < n; i++) out[i] = snapshot[dOffset + i];
        for (int k = 0; k < order; k++) {
            int dk = dOffset + (k + 1) * n;
            for (int i = 0; i < n; i++) out[i] += snapshot[dk + i] * p[k];
        }
    }

    /**
     * BDF dense output interpolation (Newton backward difference polynomial).
     * Corresponds to scipy BdfDenseOutput._call_impl:
     *   x[k] = (t - t_shift[k]) / denom[k], k=0..order-1
     *   p = cumprod(x)
     *   y = D[1:order+1]^T · p + D[0]
     * interpCoeffs stores [h, order, t_shift[0..order-1], denom[0..order-1], D[0..order] flat]
     */
    @Override
    public void interpolate(double t, double[] out) {
        if (interpCoeffs == null) throw new IllegalStateException("No interpolation data");
        int order = (int) interpCoeffs[1];
        // Reuse first order elements of changeDTmp for p, avoiding allocation (interpolate and changeD are not called simultaneously)
        double[] p = changeDTmp;
        for (int k = 0; k < order; k++) {
            double tShift = interpCoeffs[2 + k];
            double denom  = interpCoeffs[2 + order + k];
            p[k] = (t - tShift) / denom;
        }
        for (int k = 1; k < order; k++) p[k] *= p[k - 1];

        int dOffset = 2 + 2 * order;
        for (int i = 0; i < n; i++) {
            out[i] = interpCoeffs[dOffset + i];
        }
        for (int k = 0; k < order; k++) {
            int dk = dOffset + (k + 1) * n;
            for (int i = 0; i < n; i++) {
                out[i] += interpCoeffs[dk + i] * p[k];
            }
        }
    }
}
