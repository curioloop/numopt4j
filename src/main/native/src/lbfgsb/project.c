/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * L-BFGS-B projection operations module.
 * Based on the Go implementation in lbfgsb/project.go.
 *
 * This module implements projection operations for bound-constrained optimization.
 * The projection operator P(x, l, u) maps a point x to the feasible region [l, u]:
 *
 *   P(xᵢ, lᵢ, uᵢ) = lᵢ    if xᵢ < lᵢ
 *   P(xᵢ, lᵢ, uᵢ) = uᵢ    if xᵢ > uᵢ
 *   P(xᵢ, lᵢ, uᵢ) = xᵢ    otherwise
 *
 * The projected gradient measures optimality for bound-constrained problems.
 */

#include "optimizer.h"
#include <math.h>

/* Variable Status Constants (matches Go varWhere type)
 *
 * These constants track the status of each variable during optimization:
 *   varNotMove  (-3): Variable will not move in this iteration
 *   varUnbound  (-1): Variable has no bounds (free variable)
 *   varFree      (0): Variable is free to move within bounds
 *   varAtLower   (1): Variable is at its lower bound
 *   varAtUpper   (2): Variable is at its upper bound
 *   varFixed     (3): Variable is fixed (lower bound = upper bound)
 */
#define VAR_NOT_MOVE  -3
#define VAR_UNBOUND   -1
#define VAR_FREE       0
#define VAR_AT_LOWER   1
#define VAR_AT_UPPER   2
#define VAR_FIXED      3

/**
 * Subroutine projGradNorm (projgr)
 *
 * Computes the infinity norm of the projected gradient.
 *
 * For the next location xₖ₊₁ = xₖ - αₖBₖgₖ (where αₖBₖ > 0), the gradient
 * projection P(gᵢ, lᵢ, uᵢ) limits the gradient to the feasible region:
 *
 *   𝚙𝚛𝚘𝚓 gᵢ = 𝚖𝚊𝚡(xᵢ - uᵢ, gᵢ)  if gᵢ < 0
 *   𝚙𝚛𝚘𝚓 gᵢ = 𝚖𝚒𝚗(xᵢ - lᵢ, gᵢ)  if gᵢ > 0
 *   𝚙𝚛𝚘𝚓 gᵢ = gᵢ                otherwise
 *
 * This function computes ‖𝚙𝚛𝚘𝚓 g‖∞ = maxᵢ |𝚙𝚛𝚘𝚓 gᵢ|
 *
 * The projected gradient norm is used as a convergence criterion:
 * when ‖𝚙𝚛𝚘𝚓 g‖∞ < ε, the current point is approximately optimal.
 *
 * @param n Problem dimension
 * @param x Current point xₖ
 * @param g Gradient gₖ at current point
 * @param lower Lower bounds l
 * @param upper Upper bounds u
 * @param bound_type Bound types for each variable (bndNo, bndLow, bndBoth, bndUp)
 * @return Infinity norm of the projected gradient ‖𝚙𝚛𝚘𝚓 g‖∞
 *
 * Validates: Requirements 1.5
 */
double proj_grad_norm(int n, const double* x, const double* g,
                      const double* lower, const double* upper,
                      const int* bound_type) {
    int i;
    double norm = ZERO;
    double gi, xi;
    int bt;
    
    /*
     * Go implementation reference (project.go projGradNorm):
     *
     * Bound hint values:
     *   bndNo    = 0 (BOUND_NONE)   - No bounds
     *   bndLower = 1 (BOUND_LOWER)  - Lower bound only
     *   bndBoth  = 2 (BOUND_BOTH)   - Both lower and upper bounds
     *   bndUpper = 3 (BOUND_UPPER)  - Upper bound only
     *
     * The condition b.hint >= bndBoth captures BOUND_BOTH (2) and BOUND_UPPER (3),
     * i.e., variables that have an upper bound.
     *
     * The condition b.hint <= bndBoth captures BOUND_LOWER (1) and BOUND_BOTH (2),
     * i.e., variables that have a lower bound.
     */
    
    for (i = 0; i < n; i++) {
        gi = g[i];
        xi = x[i];
        bt = bound_type ? bound_type[i] : BOUND_NONE;

        if (bt != BOUND_NONE) {
            if (gi < ZERO) {
                /* gᵢ < 0: check upper bound (bt >= BOUND_BOTH means has upper bound) */
                if (bt >= BOUND_BOTH) {
                    /* 𝚙𝚛𝚘𝚓 gᵢ = 𝚖𝚊𝚡(xᵢ - uᵢ, gᵢ) */
                    double diff = xi - upper[i];
                    if (diff > gi) {
                        gi = diff;
                    }
                }
            } else {
                /* gᵢ >= 0: check lower bound (bt <= BOUND_BOTH means has lower bound) */
                if (bt <= BOUND_BOTH) {
                    /* 𝚙𝚛𝚘𝚓 gᵢ = 𝚖𝚒𝚗(xᵢ - lᵢ, gᵢ) */
                    double diff = xi - lower[i];
                    if (diff < gi) {
                        gi = diff;
                    }
                }
            }
        }

        /* Update infinity norm: ‖𝚙𝚛𝚘𝚓 g‖∞ = maxᵢ |𝚙𝚛𝚘𝚓 gᵢ| */
        double abs_gi = fabs(gi);
        if (abs_gi > norm) {
            norm = abs_gi;
        }
    }
    
    return norm;
}

/**
 * Subroutine projInitActive (active)
 *
 * Initializes the variable status array (iwhere) and projects the initial point
 * to the feasible set if necessary.
 *
 * Initial projection P(xᵢ, lᵢ, uᵢ) limits x to the feasible region:
 *
 *   𝚙𝚛𝚘𝚓 xᵢ = uᵢ    if xᵢ > uᵢ
 *   𝚙𝚛𝚘𝚓 xᵢ = lᵢ    if xᵢ < lᵢ
 *   𝚙𝚛𝚘𝚓 xᵢ = xᵢ    otherwise
 *
 * The function performs two passes:
 * 1. Project x to feasible region and count variables at bounds
 * 2. Initialize iwhere array and determine problem characteristics
 *
 * Variable status values (iwhere):
 *   varUnbound (-1): Variable has no bounds
 *   varFree     (0): Variable is free to move within bounds
 *   varFixed    (3): Variable is fixed (uᵢ - lᵢ ≤ 0)
 *
 * @param n Problem dimension
 * @param x Current point (modified in place if projection needed)
 * @param lower Lower bounds l
 * @param upper Upper bounds u
 * @param bound_type Bound types for each variable
 * @param iwhere Output: variable status array
 * @param out_projected Output: 1 if x was projected, 0 otherwise
 * @param out_constrained Output: 1 if problem has constraints, 0 otherwise
 * @param out_boxed Output: 1 if all variables have both bounds, 0 otherwise
 *
 * Validates: Requirements 1.5
 */
void proj_init_active(int n, double* x,
                      const double* lower, const double* upper,
                      const int* bound_type, int* iwhere,
                      int* out_projected, int* out_constrained, int* out_boxed) {
    int i;
    int bt;
    double xi, li, ui;
    int num_bnd = 0;
    int projected = 0;
    int constrained = 0;
    int boxed = 1;
    
    /*
     * First pass: Project x to feasible region
     *
     * For each variable with bounds:
     *   - If xᵢ ≤ lᵢ (and has lower bound): project to lᵢ
     *   - If xᵢ ≥ uᵢ (and has upper bound): project to uᵢ
     *
     * The condition bt <= BOUND_BOTH captures variables with lower bounds.
     * The condition bt >= BOUND_BOTH captures variables with upper bounds.
     */
    
    /* First pass: project x to feasible region */
    for (i = 0; i < n; i++) {
        bt = bound_type ? bound_type[i] : BOUND_NONE;

        if (bt != BOUND_NONE) {
            xi = x[i];

            /* Check lower bound (bt <= BOUND_BOTH means has lower bound) */
            if (bt <= BOUND_BOTH) {
                li = lower[i];
                if (xi <= li) {
                    if (xi < li) {
                        projected = 1;
                        x[i] = li;  /* 𝚙𝚛𝚘𝚓 xᵢ = lᵢ */
                    }
                    num_bnd++;
                    continue;  /* Skip upper bound check if at lower bound */
                }
            }

            /* Check upper bound (bt >= BOUND_BOTH means has upper bound) */
            if (bt >= BOUND_BOTH) {
                ui = upper[i];
                if (xi >= ui) {
                    if (xi > ui) {
                        projected = 1;
                        x[i] = ui;  /* 𝚙𝚛𝚘𝚓 xᵢ = uᵢ */
                    }
                    num_bnd++;
                }
            }
        }
    }
    
    /*
     * Second pass: Initialize iwhere and determine problem characteristics
     *
     * Variable status assignment:
     *   - varUnbound: No bounds on variable
     *   - varFixed:   Both bounds and uᵢ - lᵢ ≤ 0 (variable is fixed)
     *   - varFree:    Has bounds but free to move
     *
     * Problem characteristics:
     *   - boxed:       All variables have both bounds (BOUND_BOTH)
     *   - constrained: At least one variable has bounds
     */
    
    /* Second pass: initialize iwhere and determine constrained/boxed */
    for (i = 0; i < n; i++) {
        bt = bound_type ? bound_type[i] : BOUND_NONE;

        /* Update boxed flag: true only if all variables have both bounds */
        boxed = boxed && (bt == BOUND_BOTH);

        if (bt == BOUND_NONE) {
            iwhere[i] = VAR_UNBOUND;  /* No bounds on this variable */
        } else {
            constrained = 1;
            if (bt == BOUND_BOTH && upper[i] - lower[i] <= ZERO) {
                iwhere[i] = VAR_FIXED;  /* Variable is fixed: uᵢ - lᵢ ≤ 0 */
            } else {
                iwhere[i] = VAR_FREE;   /* Variable is free to move within bounds */
            }
        }
    }
    
    /* Set output flags */
    if (out_projected) *out_projected = projected;
    if (out_constrained) *out_constrained = constrained;
    if (out_boxed) *out_boxed = boxed;
    
    (void)num_bnd;  /* num_bnd is used for logging in Go, not needed here */
}

/**
 * Project x onto the feasible region defined by bounds.
 *
 * This is a simple projection operation:
 *
 *   P(xᵢ, lᵢ, uᵢ) = lᵢ    if xᵢ < lᵢ
 *   P(xᵢ, lᵢ, uᵢ) = uᵢ    if xᵢ > uᵢ
 *   P(xᵢ, lᵢ, uᵢ) = xᵢ    otherwise
 *
 * @param n Problem dimension
 * @param x Point to project (modified in place)
 * @param lower Lower bounds l
 * @param upper Upper bounds u
 * @param bound_type Bound types for each variable
 *
 * Validates: Requirements 1.5
 */
void project_x(int n, double* x, const double* lower, const double* upper,
               const int* bound_type) {
    int i;
    int bt;

    for (i = 0; i < n; i++) {
        bt = bound_type ? bound_type[i] : BOUND_NONE;

        switch (bt) {
            case BOUND_LOWER:
                /* Only lower bound: xᵢ = max(lᵢ, xᵢ) */
                if (x[i] < lower[i]) x[i] = lower[i];
                break;
            case BOUND_UPPER:
                /* Only upper bound: xᵢ = min(uᵢ, xᵢ) */
                if (x[i] > upper[i]) x[i] = upper[i];
                break;
            case BOUND_BOTH:
                /* Both bounds: xᵢ = min(uᵢ, max(lᵢ, xᵢ)) */
                if (x[i] < lower[i]) x[i] = lower[i];
                else if (x[i] > upper[i]) x[i] = upper[i];
                break;
        }
    }
}
