/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.mat;

import com.curioloop.numopt4j.linalg.Decomposition;
import com.curioloop.numopt4j.linalg.blas.BLAS;
import com.curioloop.numopt4j.linalg.blas.Select;

import static java.lang.Math.max;

/**
 * Real Schur decomposition: A = Z * T * Z^T.
 *
 * <p>T is quasi-upper-triangular (real Schur form) and Z is orthogonal.
 * Eigenvalues appear on the diagonal of T as 1x1 (real) or 2x2 (complex pair) blocks.</p>
 *
 * <p>Also provides Lyapunov equation solvers based on the Bartels-Stewart algorithm:</p>
 * <ul>
 *   <li>{@link #solveContinuousLyapunov} - solves AX + XA^T = sign*Q</li>
 *   <li>{@link #solveDiscreteLyapunov}   - solves AXA^T - X = Q</li>
 * </ul>
 */
public final class Schur implements Decomposition {

    /**
     * Pool for reusing Schur decomposition result arrays across multiple calls.
     * Directly extends {@link Decomposition.Workspace} so it can be passed as a Workspace.
     * Uses lazy-init and grow-only strategy.
     */
    public static final class Pool extends Decomposition.Workspace {
        public double[] wr;
        public double[] wi;
        public double[] Z;
        public double[] lyap;

        /**
         * Ensures all result arrays are allocated and large enough for the given parameters.
         * Uses lazy-init and grow-only strategy (never shrinks existing arrays).
         *
         * @param n        matrix dimension
         * @param computeZ whether the orthogonal matrix Z is needed
         * @return this Pool instance (for chaining)
         */
        public Pool ensure(int n, boolean computeZ) {
            // wr, wi: always ensure length >= n
            if (wr == null || wr.length < n) {
                wr = new double[n];
            }
            if (wi == null || wi.length < n) {
                wi = new double[n];
            }

            // Z: computeZ=true → length >= n*n; computeZ=false → do not allocate
            if (computeZ) {
                int zSize = n * n;
                if (Z == null || Z.length < zSize) {
                    Z = new double[zSize];
                }
            } else {
                Z = null;
            }

            // iwork: always ensure length >= n (reused as ipiv in solveDiscreteLyapunov)
            ensureIwork(n);

            return this;
        }

        /**
         * Ensures the {@code lyap} buffer is large enough for an n×n temporary matrix.
         * Prefers reusing {@code work} (zero extra allocation) when it is large enough.
         *
         * @param n matrix dimension
         * @return this Pool instance (for chaining)
         */
        public Pool ensureLyap(int n) {
            int needed = n * n;
            if (work != null && work.length >= needed) {
                // Reuse work array — zero extra allocation
                lyap = work;
            } else {
                // Grow-only: only reallocate if current lyap is too small
                if (lyap == null || lyap.length < needed) {
                    lyap = new double[needed];
                }
            }
            return this;
        }

        /**
         * Returns true when all allocated arrays satisfy the minimum size requirements
         * for the given parameters.
         *
         * @param n        matrix dimension
         * @param computeZ whether the orthogonal matrix Z is needed
         * @return true if compatible
         */
        public boolean isCompatible(int n, boolean computeZ) {
            if (wr == null || wr.length < n) return false;
            if (wi == null || wi.length < n) return false;
            if (computeZ && (Z == null || Z.length < n * n)) return false;
            return true;
        }
    }

    /** Configuration options for Schur decomposition. */
    public enum Opts {
        /** Compute the orthogonal matrix Z. Default: not computed. */
        WANT_Z,
        /** Sort eigenvalues: left half plane (real part &lt; 0). */
        SORT_LHP,
        /** Sort eigenvalues: right half plane (real part &ge; 0). */
        SORT_RHP,
        /** Sort eigenvalues: inside unit circle (|λ| &le; 1). */
        SORT_IUC,
        /** Sort eigenvalues: outside unit circle (|λ| &gt; 1). */
        SORT_OUC
    }

    /**
     * Left Half Plane selector: selects eigenvalues with negative real part (wr &lt; 0).
     * Used for continuous-time system stability analysis.
     */
    public static final Select LHP = (wr, wi) -> wr < 0;

    /**
     * Right Half Plane selector: selects eigenvalues with non-negative real part (wr &gt;= 0).
     */
    public static final Select RHP = (wr, wi) -> wr >= 0;

    /**
     * Inside Unit Circle selector: selects eigenvalues inside or on the unit circle
     * (|wr + wi*i| &lt;= 1). Used for discrete-time system stability analysis.
     */
    public static final Select IUC = (wr, wi) -> Math.hypot(wr, wi) <= 1.0;

    /**
     * Outside Unit Circle selector: selects eigenvalues outside the unit circle
     * (|wr + wi*i| &gt; 1).
     */
    public static final Select OUC = (wr, wi) -> Math.hypot(wr, wi) > 1.0;

    private Schur.Pool pool;
    private double[] T;
    private int n;
    private int sdim;
    private boolean ok;

    private Schur() {}

    public static Pool workspace(int n) {
        Pool pool = new Pool();
        // Query dgees for optimal work size
        double[] tmp = new double[1];
        BLAS.dgees('V', 'N', null, n, null, n, null, null, null, n, tmp, 0, -1, null);
        pool.ensureWork((int) tmp[0]);
        pool.ensureBwork(n);
        return pool;
    }

    /** Opts-based entry point. */
    public static Schur decompose(double[] A, int n, Workspace ws, Opts... opts) {
        boolean computeZ = contains(opts, Opts.WANT_Z);
        Select sort = null;
        if (contains(opts, Opts.SORT_LHP))      sort = LHP;
        else if (contains(opts, Opts.SORT_RHP)) sort = RHP;
        else if (contains(opts, Opts.SORT_IUC)) sort = IUC;
        else if (contains(opts, Opts.SORT_OUC)) sort = OUC;
        return decompose(A, n, computeZ, sort, ws);
    }

    private static boolean contains(Opts[] opts, Opts target) {
        if (opts == null) return false;
        for (Opts o : opts) if (o == target) return true;
        return false;
    }

    public static Schur decompose(double[] A, int n) {
        return decompose(A, n, false, null, null);
    }

    public static Schur decompose(double[] A, int n, boolean computeZ) {
        return decompose(A, n, computeZ, null, null);
    }

    public static Schur decompose(double[] A, int n, boolean computeZ, Select sort) {
        return decompose(A, n, computeZ, sort, null);
    }

    public static Schur decompose(double[] A, int n, boolean computeZ, Select sort, Workspace ws) {
        Schur schur = new Schur();
        schur.doDecompose(A, n, computeZ, sort, ws);
        return schur;
    }

    private void doDecompose(double[] A, int n, boolean computeZ, Select sort, Workspace ws) {
        if (A == null || A.length < n * n) {
            throw new IllegalArgumentException("Matrix A must have length >= n*n");
        }
        if (n <= 0) {
            throw new IllegalArgumentException("Matrix dimension must be positive");
        }
        // NOTE: Do NOT pass a matrix containing NaN or Inf values.
        // NaN inputs propagate into Dgebal.gotoScaling() where they cause an infinite loop
        // (the convergence check "c + r >= factor * s" always evaluates to false under NaN,
        // so the balancing loop never terminates). Validate input before calling decompose().

        this.T = A;
        this.n = n;
        this.ok = false;

        if (ws != null && !(ws instanceof Pool)) {
            throw new IllegalArgumentException("Workspace must be an instance of Schur.Pool");
        }

        if (ws == null) {
            ws = workspace(n);
        }
        this.pool = (Pool) ws;

        Pool pool = (Pool) ws;
        char sortChar = (sort != null) ? 'S' : 'N';
        // Query dgees for optimal work size
        double[] tmp = new double[1];
        BLAS.dgees(computeZ ? 'V' : 'N', sortChar, sort, n, null, n, null, null, null, n, tmp, 0, -1, null);
        pool.ensureWork((int) tmp[0]);
        if (sort != null) pool.ensureBwork(n);
        pool.ensure(n, computeZ);

        int info = BLAS.dgees(computeZ ? 'V' : 'N', sortChar, sort, n, A, n, pool.wr, pool.wi, pool.Z, n, pool.work(), 0, pool.work().length, pool.bwork());
        this.sdim = (int) pool.work()[0];
        this.ok = (info == 0);
    }

    public double[] T() { return T; }

    public double[] Z() { return pool.Z; }

    public double[] eigenvalues() { return pool.wr; }

    public double[] eigenvaluesImag() { return pool.wi; }

    public int sdim() { return sdim; }

    public boolean ok() { return ok; }

    public int n() { return n; }

    public double cond() {
        if (!ok || pool.wr == null) return Double.NaN;
        double max = 0, min = Double.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            double absVal = Math.hypot(pool.wr[i], pool.wi[i]);
            if (absVal > max) max = absVal;
            if (absVal > 0 && absVal < min) min = absVal;
        }
        if (min == 0) return Double.POSITIVE_INFINITY;
        return max / min;
    }

    public boolean isStable() {
        if (!ok || pool.wr == null) return false;
        for (int i = 0; i < n; i++) {
            if (pool.wr[i] >= 0) return false;
        }
        return true;
    }

    public boolean isDiscreteStable() {
        if (!ok || pool.wr == null) return false;
        for (int i = 0; i < n; i++) {
            if (Math.hypot(pool.wr[i], pool.wi[i]) > 1.0) return false;
        }
        return true;
    }

    public double[] invariantSubspace() {
        return invariantSubspaceTo(null);
    }

    public double[] invariantSubspaceTo(double[] dst) {
        if (!ok || pool.Z == null || sdim == 0) return null;
        int size = n * sdim;
        if (dst == null || dst.length < size) dst = new double[size];
        for (int j = 0; j < sdim; j++) {
            for (int i = 0; i < n; i++) {
                dst[i * sdim + j] = pool.Z[i * n + j];
            }
        }
        return dst;
    }

    public double[] unstableSubspace() {
        return unstableSubspaceTo(null);
    }

    public double[] unstableSubspaceTo(double[] dst) {
        if (!ok || pool.Z == null || sdim == n) return null;
        int udim = n - sdim;
        if (udim == 0) return null;
        int size = n * udim;
        if (dst == null || dst.length < size) dst = new double[size];
        for (int j = 0; j < udim; j++) {
            for (int i = 0; i < n; i++) {
                dst[i * udim + j] = pool.Z[i * n + sdim + j];
            }
        }
        return dst;
    }

    /**
     * Reorders the Schur factorization so that eigenvalues matching {@code sort} appear first.
     * Updates T, Z, wr, wi, and sdim in place.
     *
     * @param sort eigenvalue selector (eigenvalues returning true are moved to the front)
     * @return true if reordering succeeded
     */
    public boolean reorder(Select sort) {
        if (!ok || sort == null) return false;
        boolean[] selects = new boolean[n];
        for (int i = 0; i < n; i++) {
            selects[i] = sort.select(pool.wr[i], pool.wi[i]);
        }
        return reorder(selects);
    }

    /**
     * Reorders the Schur factorization so that eigenvalues with {@code selects[i]=true} appear first.
     * Updates T, Z, wr, wi, and sdim in place.
     *
     * @param selects boolean array of length n; true selects the eigenvalue to move to the front
     * @return true if reordering succeeded
     */
    public boolean reorder(boolean[] selects) {
        if (!ok || selects == null || selects.length < n) return false;
        pool.ensureWork(max(1, n * n));
        pool.ensureIwork(max(1, n * n));
        boolean result = BLAS.dtrsen('N', pool.Z != null, selects, n,
                T, 0, n, pool.Z, 0, n, pool.wr, pool.wi,
                pool.work(), pool.work().length,
                pool.iwork(), pool.iwork().length);
        this.sdim = pool.iwork()[1];
        this.ok = result;
        return result;
    }

    // ==================== Lyapunov equation solvers ====================

    /**
     * Solves the continuous Lyapunov equation: AX + XA^T = sign*Q (Bartels-Stewart algorithm).
     *
     * <p>Steps:</p>
     * <ol>
     *   <li>Schur decompose A = UTU^T</li>
     *   <li>Transform right-hand side: F = sign * U^T * Q * U</li>
     *   <li>Solve Sylvester equation: TY + YT^T = F</li>
     *   <li>Back-transform: X = U * (scale*Y) * U^T</li>
     * </ol>
     * <p>Uses the already-computed Schur decomposition (T and Z) stored in this instance.
     * The solution X is written back into Q.</p>
     *
     * @param Q    n×n right-hand side (row-major, overwritten with solution X)
     * @param sign +1 to solve AX + XA^T = Q, -1 to solve AX + XA^T = -Q
     * @return true on success
     * @throws IllegalStateException if decomposition was not completed or Z is not available
     */
    public boolean lyapunov(double[] Q, double sign) {
        if (!ok) {
            throw new IllegalStateException("Schur decomposition not completed or failed");
        }
        if (pool.Z == null) {
            throw new IllegalStateException("Orthogonal matrix Z not available (computeZ was false)");
        }

        // Special case n=1: 2*T[0]*X[0] = sign*Q[0]  =>  X[0] = sign*Q[0] / (2*T[0])
        if (n == 1) {
            Q[0] = sign * Q[0] / (2.0 * T[0]);
            return true;
        }

        // F = sign * Z^T * Q * Z  (stored back into Q)
        final double[] tmp = pool.ensureLyap(n).lyap;
        BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, n, n, n,
                sign, Q, 0, n, pool.Z, 0, n, 0.0, tmp, 0, n);
        BLAS.dgemm(BLAS.Transpose.Trans, BLAS.Transpose.NoTrans, n, n, n,
                1.0, pool.Z, 0, n, tmp, 0, n, 0.0, Q, 0, n);

        // Solve TY + YT^T = F  (isgn=+1, tranb=true means B^T)
        double scale = BLAS.dtrsyl(false, true, 1, n, n,
                T, 0, n, T, 0, n, Q, 0, n, null);

        // X = Z * (scale*Y) * Z^T  (stored back into Q)
        BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.Trans, n, n, n,
                scale, Q, 0, n, pool.Z, 0, n, 0.0, tmp, 0, n);
        BLAS.dgemm(BLAS.Transpose.NoTrans, BLAS.Transpose.NoTrans, n, n, n,
                1.0, pool.Z, 0, n, tmp, 0, n, 0.0, Q, 0, n);

        return true;
    }

    /**
     * Solves the continuous Lyapunov equation: AX + XA^T = Q (Bartels-Stewart algorithm).
     *
     * <p>Convenience overload equivalent to {@code lyapunov(Q, 1.0)}.</p>
     *
     * @param Q n×n right-hand side (row-major, overwritten with solution X)
     * @return true on success
     * @throws IllegalStateException if decomposition was not completed or Z is not available
     */
    public boolean lyapunov(double[] Q) {
        return lyapunov(Q, 1.0);
    }

    /**
     * Solves the discrete Lyapunov equation: AXA^T - X = Q (Bilinear Transformation Method).
     *
     * <p>Uses the already-computed Schur decomposition to validate state, then performs
     * the bilinear transformation on the provided matrix A.</p>
     *
     * <p>Steps:</p>
     * <ol>
     *   <li>Bilinear transform: B = (A-I)(A+I)^{-1}</li>
     *   <li>Transform right-hand side: C = (A+I)^{-1} * Q * (A^T+I)^{-1}</li>
     *   <li>Schur decompose B, then solve continuous equation: BX + XB^T = 2C</li>
     * </ol>
     *
     * @param A n×n matrix (row-major, will be overwritten)
     * @param Q n×n right-hand side symmetric matrix (row-major, overwritten with solution X)
     * @return true on success; false if LU factorization is singular
     * @throws IllegalStateException if this Schur decomposition was not completed or Z is not available
     */
    public boolean discreteLyapunov(double[] A, double[] Q) {
        if (!ok) {
            throw new IllegalStateException("Schur decomposition not completed or failed");
        }
        if (pool.Z == null) {
            throw new IllegalStateException("Orthogonal matrix Z not available (computeZ was false)");
        }
        if (n == 0) return true;

        // Ensure lyap is large enough for n*n
        pool.ensureLyap(n);
        final int[] ipiv = pool.iwork();

        // Phase 2 first: build aa = A+I into pool.lyap, transform Q
        // (independent of phase 1, so order can be swapped)
        final double[] aa = pool.lyap;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aa[i * n + j] = A[i * n + j];
            }
        }
        for (int i = 0; i < n; i++) {
            aa[i * n + i] += 1.0; // aa = A + I
        }
        if (BLAS.dgetrf(n, n, aa, 0, n, ipiv, 0) != 0) return false;
        // Solve (A+I) * Y^T = Q^T  (Q is symmetric so Q^T = Q)
        BLAS.dgetrs(BLAS.Transpose.NoTrans, n, n, aa, 0, n, ipiv, 0, Q, 0, n);
        transposeSquare(Q, n);
        // Solve (A+I) * C = Y  →  C = (A+I)^{-1} * Y
        BLAS.dgetrs(BLAS.Transpose.NoTrans, n, n, aa, 0, n, ipiv, 0, Q, 0, n);

        // Phase 1: overwrite pool.lyap with at = A^T+I, compute B
        final double[] at = pool.lyap;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                at[i * n + j] = A[j * n + i]; // A^T
            }
        }
        for (int i = 0; i < n; i++) {
            at[i * n + i] += 1.0; // at = A^T + I
            A[i * n + i]  -= 1.0; // A = A - I
        }
        if (BLAS.dgetrf(n, n, at, 0, n, ipiv, 0) != 0) return false;
        transposeSquare(A, n);
        BLAS.dgetrs(BLAS.Transpose.NoTrans, n, n, at, 0, n, ipiv, 0, A, 0, n);
        transposeSquare(A, n);

        // Phase 3: Schur decompose B (stored in A), pass pool so lyapunov()
        // reuses pool.lyap as tmp buffer (zero extra allocation)
        Schur bSchur = Schur.decompose(A, n, true, null, pool);
        if (!bSchur.ok()) return false;
        return bSchur.lyapunov(Q, 2.0);
    }

    /** In-place transpose of an n×n square matrix (row-major). */
    private static void transposeSquare(double[] a, int n) {
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                double tmp = a[i * n + j];
                a[i * n + j] = a[j * n + i];
                a[j * n + i] = tmp;
            }
        }
    }

    @Override
    public Pool work() { return pool; }

    /** Returns the quasi-upper-triangular Schur form T as an n×n matrix. */
    public Matrix toT() {
        if (!ok) return null;
        return new Matrix(n, n, false, T);
    }

    /** Returns the orthogonal matrix Z as an n×n matrix, or null if not computed. */
    public Matrix toZ() {
        if (!ok || pool.Z == null) return null;
        return new Matrix(n, n, false, pool.Z);
    }

    /** Returns eigenvalues as an n×1 complex matrix (interleaved [wr0,wi0,...]), or null if failed. */
    public Matrix toS() {
        if (!ok || pool.wr == null) return null;
        double[] dst = new double[n * 2];
        for (int i = 0; i < n; i++) {
            dst[i * 2]     = pool.wr[i];
            dst[i * 2 + 1] = pool.wi != null ? pool.wi[i] : 0.0;
        }
        return new Matrix(n, 1, true, dst);
    }
}
