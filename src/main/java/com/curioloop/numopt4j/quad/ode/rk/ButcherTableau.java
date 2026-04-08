/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.ode.rk;

/**
 * Enum of built-in Butcher tableaux for explicit Runge-Kutta methods.
 *
 * <p>A Butcher tableau defines an explicit RK method of the form:
 * <pre>
 *   kₛ = f(t + Cₛ·h,  y + h · Σⱼ₌₀ˢ⁻¹ Aₛⱼ · kⱼ),   s = 0 .. nStages-1
 *   y_new = y + h · Σₛ Bₛ · kₛ
 * </pre>
 * The error estimate uses an embedded lower-order solution:
 * <pre>
 *   err = h · Σₛ Eₛ · kₛ
 * </pre>
 * </p>
 *
 * <p>Standard methods (RK23, RK45) use {@code E} for error estimation and {@code P} for dense output.
 * DOP853 uses {@code E3}/{@code E5} for dual error estimation and {@code D} for dense output,
 * with additional extended stages defined by {@code nExtraStages} and {@code C_EXTRA}.</p>
 *
 * <p>The {@code A} matrix is stored in compact lower-triangular form (row-major):
 * row {@code s} has {@code s} elements at offset {@code s*(s-1)/2}.</p>
 */
public enum ButcherTableau {

    /**
     * Bogacki-Shampine RK23: 3rd-order method with 2nd-order error estimate.
     * Reference: Bogacki &amp; Shampine (1989).
     */
    RK23(RK23Coeffs.ORDER, RK23Coeffs.ERROR_ORDER, RK23Coeffs.N_STAGES, RK23Coeffs.INTERP_ORDER,
         RK23Coeffs.C, RK23Coeffs.A, RK23Coeffs.B,
         RK23Coeffs.E, null, null,
         RK23Coeffs.P, null, 0, 0, null),

    /**
     * Dormand-Prince RK45: 5th-order method with 4th-order error estimate.
     * Reference: Dormand &amp; Prince (1980).
     */
    RK45(RK45Coeffs.ORDER, RK45Coeffs.ERROR_ORDER, RK45Coeffs.N_STAGES, RK45Coeffs.INTERP_ORDER,
         RK45Coeffs.C, RK45Coeffs.A, RK45Coeffs.B,
         RK45Coeffs.E, null, null,
         RK45Coeffs.P, null, 0, 0, null),

    /**
     * DOP853: 8th-order method with dual 5th/3rd-order error estimate and extended stages for dense output.
     * Reference: Hairer, Nørsett &amp; Wanner (1993), "Solving ODEs I", 2nd ed.
     */
    DOP853(DOP853Coeffs.ORDER, DOP853Coeffs.ERROR_ORDER, DOP853Coeffs.N_STAGES, DOP853Coeffs.INTERPOLATOR_POWER,
           DOP853Coeffs.C, DOP853Coeffs.A, DOP853Coeffs.B,
           null, DOP853Coeffs.E3, DOP853Coeffs.E5,
           null, DOP853Coeffs.D, DOP853Coeffs.D_ROWS,
           DOP853Coeffs.N_STAGES_EXTENDED - DOP853Coeffs.N_STAGES - 1,
           DOP853Coeffs.C_EXTRA);

    // -----------------------------------------------------------------------
    // Tableau fields
    // -----------------------------------------------------------------------

    /** Method order (solution). */
    final int order;
    /** Error estimate order (embedded lower-order solution). */
    final int errorOrder;
    /** Number of main stages (excluding the final error-estimation stage). */
    final int nStages;
    /** Dense output polynomial order. */
    final int interpOrder;

    /**
     * Precomputed step-size control exponent: {@code -1 / (errorOrder + 1)}.
     * Used in the error-based step-size update:
     * <pre>
     *   h_new = h * min(MAX_FACTOR, SAFETY * errorNorm^errorExponent)
     * </pre>
     */
    final double errorExponent;

    /**
     * Time nodes {@code C[s]}, length {@code nStages} (or {@code nStagesExtended} for DOP853).
     * Stage s is evaluated at {@code t + C[s] * h}.
     */
    final double[] C;

    /**
     * Butcher coefficient matrix {@code A}, compact lower-triangular row-major.
     * Row {@code s} has {@code s} elements at offset {@code s*(s-1)/2}.
     */
    final double[] A;

    /** High-order solution weights {@code B}, length {@code nStages}. */
    final double[] B;

    /** Standard error coefficients {@code E}, length {@code nStages+1} (null for DOP853). */
    final double[] E;
    /** DOP853 3rd-order error coefficients (null for standard methods). */
    final double[] E3;
    /** DOP853 5th-order error coefficients (null for standard methods). */
    final double[] E5;

    /**
     * Dense output coefficient matrix {@code P}, shape {@code (nStages+1) × interpOrder}, row-major.
     * Null for DOP853 (uses {@code D} instead).
     */
    final double[] P;

    /**
     * DOP853 dense output matrix {@code D}, shape {@code dRows × nStagesExtended}, row-major.
     * Null for standard methods.
     */
    final double[] D;
    /** Number of rows in the {@code D} matrix (meaningful only when {@code D != null}). */
    final int dRows;

    /** Number of extra stages beyond {@code nStages} for DOP853 dense output (0 for standard methods). */
    final int nExtraStages;
    /** Time nodes for extra stages (null for standard methods). */
    final double[] C_EXTRA;

    /**
     * Total extended stage count including main stages.
     * {@code nStages + 1 + nExtraStages} for DOP853; 0 for standard methods.
     */
    final int nStagesExtended;

    ButcherTableau(int order, int errorOrder, int nStages, int interpOrder,
                   double[] C, double[] A, double[] B,
                   double[] E, double[] E3, double[] E5,
                   double[] P, double[] D, int dRows,
                   int nExtraStages, double[] C_EXTRA) {
        this.order           = order;
        this.errorOrder      = errorOrder;
        this.nStages         = nStages;
        this.interpOrder     = interpOrder;
        this.errorExponent   = -1.0 / (errorOrder + 1);
        this.C               = C;
        this.A               = A;
        this.B               = B;
        this.E               = E;
        this.E3              = E3;
        this.E5              = E5;
        this.P               = P;
        this.D               = D;
        this.dRows           = dRows;
        this.nExtraStages    = nExtraStages;
        this.C_EXTRA         = C_EXTRA;
        this.nStagesExtended = (D != null) ? nStages + 1 + nExtraStages : 0;
    }
}
