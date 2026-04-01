/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.quad.gauss;

import com.curioloop.numopt4j.linalg.blas.BLAS;

import java.util.Arrays;

/**
 * A Gaussian quadrature rule defined on its own canonical domain and weight.
 *
 * <p>The generated nodes and weights satisfy the rule family's native integral form. For example,
 * Gauss-Jacobi approximates {@code integral_{-1}^{1} (1 - x)^alpha (1 + x)^beta f(x) dx},
 * Gauss-Hermite approximates {@code integral exp(-x^2) f(x) dx} on the whole real line, and
 * Gauss-Laguerre approximates {@code integral exp(-x) f(x) dx} on {@code [0, +inf)}.</p>
 */
public interface GaussRule {

    /**
     * Generates nodes and weights for the given number of quadrature points.
     *
     * @param points    number of quadrature points, must be positive
     * @param workspace reusable rule-generation workspace, must not be null
     */
    default void generate(int points, GaussPool workspace) {
        fromJacobiMatrix(points, workspace);
    }

    double zeroMoment();

    /** Callback that fills the symmetric tridiagonal Jacobi matrix for a specific rule family. */
    void fillJacobi(int points, double[] arena, int diagonalOffset, int offDiagonalOffset);

    /**
     * Generates nodes and weights from a symmetric tridiagonal Jacobi matrix.
     *
     * <p>The Golub-Welsch algorithm: given the three-term recurrence
     *   p_{n+1}(x) = (x − αₙ)·pₙ(x) − βₙ·p_{n−1}(x)
     * the quadrature nodes are the eigenvalues of the symmetric tridiagonal matrix
     *   J = diag(α₀,…,αₙ₋₁) + off-diag(√β₁,…,√βₙ₋₁)
     * and the weights are wᵢ = μ₀ · v₀ᵢ², where μ₀ = ∫ w(x) dx is the zero-th
     * moment and v₀ᵢ is the first component of the i-th normalised eigenvector.</p>
     *
     * <p>The eigendecomposition is computed via LAPACK dsteqr (QR iteration on
     * symmetric tridiagonal matrix with eigenvector accumulation).</p>
     */
    default void fromJacobiMatrix(int points, GaussPool workspace) {
        if (points <= 0) throw new IllegalArgumentException("points must be > 0");
        if (workspace == null) throw new IllegalArgumentException("workspace must not be null");
        workspace.ensure(points);
        double[] arena = workspace.arena();
        int mat = workspace.matrixOffset(), spec = workspace.spectrumOffset();
        int offDiag = workspace.offDiagonalOffset(), nodes = workspace.nodesOffset();
        int weights = workspace.weightsOffset(), work = workspace.workOffset();

        Arrays.fill(arena, mat, mat + points * points, 0.0);
        Arrays.fill(arena, spec, spec + points, 0.0);
        if (points > 1) Arrays.fill(arena, offDiag, offDiag + points - 1, 0.0);
        fillJacobi(points, arena, spec, offDiag);

        int info = BLAS.dsteqr('I', points, arena, spec, arena, offDiag, arena, mat, points, arena, work);
        if (info != 0) throw new ArithmeticException("Quadrature rule generation failed, dsteqr info=" + info);

        double zeroMoment = zeroMoment();
        System.arraycopy(arena, spec, arena, nodes, points);
        for (int j = 0; j < points; j++) {
            double v0 = arena[mat + j];
            arena[weights + j] = zeroMoment * v0 * v0;
        }
    }

    GaussRule Laguerre = new Laguerre();
    GaussRule Legendre = new Legendre();
    GaussRule Hermite  = new Hermite();

    /** Gauss-Laguerre rule: ∫_{0}^{+∞} e^{−x} f(x) dx.  Zero-th moment μ₀ = 1. */
    static final class Laguerre implements GaussRule {
        public double zeroMoment() { return 1.0; }
        public void fillJacobi(int points, double[] arena, int diag, int offDiag) {
            for (int i = 0; i < points; i++) arena[diag + i] = 2.0 * i + 1.0;
            for (int i = 1; i < points; i++) arena[offDiag + i - 1] = i;
        }
    }

    /** Gauss-Legendre rule: ∫_{−1}^{1} f(x) dx.  Zero-th moment μ₀ = 2. */
    static final class Legendre implements GaussRule {
        public double zeroMoment() { return 2.0; }
        public void fillJacobi(int points, double[] arena, int diag, int offDiag) {
            for (int i = 0; i < points; i++) arena[diag + i] = 0.0;
            for (int i = 1; i < points; i++)
                arena[offDiag + i - 1] = i / Math.sqrt(4.0 * i * i - 1.0);
        }
    }

    /** Gauss-Hermite rule: ∫_{−∞}^{+∞} e^{−x²} f(x) dx.  Zero-th moment μ₀ = √π. */
    static final class Hermite implements GaussRule {
        private static final double SQRT_PI = 1.7724538509055160272981674833411451;
        public double zeroMoment() { return SQRT_PI; }
        public void fillJacobi(int points, double[] arena, int diag, int offDiag) {
            for (int i = 0; i < points; i++) arena[diag + i] = 0.0;
            for (int i = 1; i < points; i++) arena[offDiag + i - 1] = Math.sqrt(0.5 * i);
        }
    }
}
