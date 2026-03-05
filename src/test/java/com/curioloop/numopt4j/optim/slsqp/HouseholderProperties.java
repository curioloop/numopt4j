/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 *
 * Property-based tests for Householder transformations and Givens rotations.
 *
 * **Property 2: Householder Transformation Correctness**
 */
package com.curioloop.numopt4j.optim.slsqp;

import net.jqwik.api.*;
import net.jqwik.api.constraints.*;
import org.junit.jupiter.api.Tag;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Property-based tests for Householder transformations and Givens rotations.
 *
 * <p>Tests the following properties:</p>
 * <ul>
 *   <li>Property 2.1: h1 computes Householder transformation vector correctly</li>
 *   <li>Property 2.2: h2 applies Householder transformation correctly (preserves orthogonality)</li>
 *   <li>Property 2.3: Householder transformation preserves vector norms</li>
 *   <li>Property 2.4: g1 computes Givens rotation matrix correctly</li>
 *   <li>Property 2.5: g2 applies Givens rotation correctly</li>
 *   <li>Property 2.6: Givens rotation zeros out the second element</li>
 * </ul>
 */
@Tag("Feature: slsqp-java-rewrite, Property 2: Householder Transformation Correctness")
class HouseholderProperties {

    private static final double EPSILON = 1e-12;

    // ========================================================================
    // Property 2.1: h1 computes Householder transformation vector correctly
    // ========================================================================

    @Property(tries = 100)
    void h1ComputesHouseholderVectorCorrectly(
            @ForAll @IntRange(min = 3, max = 20) int m
    ) {
        // Create a random vector
        double[] u = new double[m];
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < m; i++) {
            u[i] = rand.nextDouble() * 10 - 5;  // Random values in [-5, 5]
        }

        // Save original values for verification
        double[] original = u.clone();

        // Apply h1 with pivot=0, start=1
        int pivot = 0;
        int start = 1;
        double up = Householder.h1(pivot, start, m, u, 0, 1);

        // If up is 0, it's an identity transformation (zero vector case)
        if (up == 0.0) {
            // Check if original vector was effectively zero
            double norm = 0;
            for (int i = pivot; i < m; i++) {
                norm += original[i] * original[i];
            }
            assertTrue(Math.sqrt(norm) < EPSILON, 
                    "up=0 should only occur for zero vectors");
            return;
        }

        // Verify: u[pivot] now contains s (the result of transformation)
        // The Householder transformation Q applied to original should give [s, 0, 0, ...]
        // where s = u[pivot] after h1

        // Compute the norm of the original vector (from pivot to end)
        double originalNorm = 0;
        for (int i = pivot; i < m; i++) {
            originalNorm += original[i] * original[i];
        }
        originalNorm = Math.sqrt(originalNorm);

        // |s| should equal the norm of the original vector
        assertEquals(originalNorm, Math.abs(u[pivot]), EPSILON,
                "|s| should equal the norm of the original vector");
    }

    // ========================================================================
    // Property 2.2: h2 applies Householder transformation correctly
    // ========================================================================

    @Property(tries = 100)
    void h2AppliesHouseholderTransformationCorrectly(
            @ForAll @IntRange(min = 3, max = 15) int m,
            @ForAll @IntRange(min = 1, max = 5) int nc
    ) {
        // Create a random vector for Householder transformation
        double[] u = new double[m];
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < m; i++) {
            u[i] = rand.nextDouble() * 10 - 5;
        }

        // Compute Householder transformation
        int pivot = 0;
        int start = 1;
        double up = Householder.h1(pivot, start, m, u, 0, 1);

        if (up == 0.0) {
            return;  // Identity transformation, skip
        }

        // Create a matrix C (column-major, m rows, nc columns)
        double[] c = new double[m * nc];
        for (int j = 0; j < nc; j++) {
            for (int i = 0; i < m; i++) {
                c[j * m + i] = rand.nextDouble() * 10 - 5;
            }
        }

        // Save original C for verification
        double[] cOriginal = c.clone();

        // Apply Householder transformation to C
        Householder.h2(pivot, start, m, u, 0, 1, up, c, 0, 1, m, nc);

        // Verify: The transformation should preserve column norms (orthogonality)
        for (int j = 0; j < nc; j++) {
            double originalNorm = 0;
            double transformedNorm = 0;
            for (int i = 0; i < m; i++) {
                originalNorm += cOriginal[j * m + i] * cOriginal[j * m + i];
                transformedNorm += c[j * m + i] * c[j * m + i];
            }
            assertEquals(Math.sqrt(originalNorm), Math.sqrt(transformedNorm), EPSILON,
                    "Householder transformation should preserve column norms");
        }
    }

    // ========================================================================
    // Property 2.3: Householder transformation preserves vector norms
    // ========================================================================

    @Property(tries = 100)
    void householderPreservesVectorNorms(
            @ForAll @IntRange(min = 3, max = 20) int m
    ) {
        // Create a random vector
        double[] v = new double[m];
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < m; i++) {
            v[i] = rand.nextDouble() * 10 - 5;
        }

        // Compute original norm
        double originalNorm = 0;
        for (int i = 0; i < m; i++) {
            originalNorm += v[i] * v[i];
        }
        originalNorm = Math.sqrt(originalNorm);

        // Create Householder transformation from another vector
        double[] u = new double[m];
        for (int i = 0; i < m; i++) {
            u[i] = rand.nextDouble() * 10 - 5;
        }

        int pivot = 0;
        int start = 1;
        double up = Householder.h1(pivot, start, m, u, 0, 1);

        if (up == 0.0) {
            return;  // Identity transformation
        }

        // Apply transformation to v (as a single column matrix)
        double[] vCopy = v.clone();
        Householder.h2(pivot, start, m, u, 0, 1, up, vCopy, 0, 1, m, 1);

        // Compute transformed norm
        double transformedNorm = 0;
        for (int i = 0; i < m; i++) {
            transformedNorm += vCopy[i] * vCopy[i];
        }
        transformedNorm = Math.sqrt(transformedNorm);

        // Norms should be equal (orthogonal transformation preserves norms)
        assertEquals(originalNorm, transformedNorm, EPSILON,
                "Householder transformation should preserve vector norms");
    }

    // ========================================================================
    // Property 2.4: g1 computes Givens rotation matrix correctly
    // ========================================================================

    @Property(tries = 100)
    void g1ComputesGivensRotationCorrectly(
            @ForAll @DoubleRange(min = -100, max = 100) double a,
            @ForAll @DoubleRange(min = -100, max = 100) double b
    ) {
        double[] result = new double[3];
        Householder.g1(a, b, result);

        double c = result[0];
        double s = result[1];
        double sig = result[2];

        // Property: G[a,b]^T = [r,0]^T where G = [[c,s],[-s,c]]
        // So: c*a + s*b = r and -s*a + c*b = 0

        double r = c * a + s * b;
        double zero = -s * a + c * b;

        // The second element should be zero
        assertEquals(0.0, zero, EPSILON,
                "Givens rotation should zero out the second element");

        // r should equal sig (the computed result)
        assertEquals(sig, r, EPSILON,
                "r should equal the computed sig value");

        // sig should equal sqrt(a^2 + b^2)
        double expectedSig = Math.sqrt(a * a + b * b);
        assertEquals(expectedSig, Math.abs(sig), EPSILON,
                "sig should equal sqrt(a^2 + b^2)");

        // Rotation matrix should be orthogonal: c^2 + s^2 = 1
        if (a != 0 || b != 0) {
            assertEquals(1.0, c * c + s * s, EPSILON,
                    "Rotation matrix should be orthogonal (c^2 + s^2 = 1)");
        }
    }

    // ========================================================================
    // Property 2.5: g2 applies Givens rotation correctly
    // ========================================================================

    @Property(tries = 100)
    void g2AppliesGivensRotationCorrectly(
            @ForAll @DoubleRange(min = -100, max = 100) double a,
            @ForAll @DoubleRange(min = -100, max = 100) double b,
            @ForAll @DoubleRange(min = -100, max = 100) double z1,
            @ForAll @DoubleRange(min = -100, max = 100) double z2
    ) {
        // First compute Givens rotation from (a, b)
        double[] result = new double[3];
        Householder.g1(a, b, result);

        double c = result[0];
        double s = result[1];

        // Apply rotation to (z1, z2)
        double[] ab = {z1, z2};
        Householder.g2(c, s, ab, 0, 1);

        // Verify: G[z1,z2]^T = [c*z1 + s*z2, -s*z1 + c*z2]^T
        double expectedFirst = c * z1 + s * z2;
        double expectedSecond = -s * z1 + c * z2;

        assertEquals(expectedFirst, ab[0], EPSILON,
                "First element after rotation should be c*z1 + s*z2");
        assertEquals(expectedSecond, ab[1], EPSILON,
                "Second element after rotation should be -s*z1 + c*z2");

        // Verify norm preservation: ||(z1,z2)|| = ||(ab[0],ab[1])||
        double originalNorm = Math.sqrt(z1 * z1 + z2 * z2);
        double transformedNorm = Math.sqrt(ab[0] * ab[0] + ab[1] * ab[1]);
        assertEquals(originalNorm, transformedNorm, EPSILON,
                "Givens rotation should preserve vector norm");
    }

    // ========================================================================
    // Property 2.6: Givens rotation zeros out the second element
    // ========================================================================

    @Property(tries = 100)
    void givensRotationZerosSecondElement(
            @ForAll @DoubleRange(min = -100, max = 100) double a,
            @ForAll @DoubleRange(min = -100, max = 100) double b
    ) {
        // Compute Givens rotation
        double[] result = new double[3];
        Householder.g1(a, b, result);

        double c = result[0];
        double s = result[1];
        double sig = result[2];

        // Apply rotation to (a, b) itself
        double[] ab = {a, b};
        Householder.g2(c, s, ab, 0, 1);

        // After rotation, second element should be zero
        assertEquals(0.0, ab[1], EPSILON,
                "Givens rotation should zero out the second element");

        // First element should equal sig
        assertEquals(sig, ab[0], EPSILON,
                "First element after rotation should equal sig");
    }

    // ========================================================================
    // Property 2.7: h1 handles edge cases correctly
    // ========================================================================

    @Property(tries = 100)
    void h1HandlesEdgeCasesCorrectly(
            @ForAll @IntRange(min = 3, max = 10) int m
    ) {
        // Test with zero vector
        double[] zeroVec = new double[m];
        double up = Householder.h1(0, 1, m, zeroVec, 0, 1);
        assertEquals(0.0, up, "h1 should return 0 for zero vector");

        // Test with invalid bounds (pivot >= start)
        double[] vec = {1.0, 2.0, 3.0};
        up = Householder.h1(1, 1, 3, vec, 0, 1);  // pivot == start
        assertEquals(0.0, up, "h1 should return 0 for invalid bounds (pivot == start)");

        up = Householder.h1(2, 1, 3, vec, 0, 1);  // pivot > start
        assertEquals(0.0, up, "h1 should return 0 for invalid bounds (pivot > start)");

        // Test with start >= m
        up = Householder.h1(0, 3, 3, vec, 0, 1);  // start == m
        assertEquals(0.0, up, "h1 should return 0 for invalid bounds (start >= m)");
    }

    // ========================================================================
    // Property 2.8: h2 handles edge cases correctly
    // ========================================================================

    @Property(tries = 100)
    void h2HandlesEdgeCasesCorrectly(
            @ForAll @IntRange(min = 3, max = 10) int m
    ) {
        double[] u = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] c = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] cOriginal = c.clone();

        // Test with nc = 0 (no columns to transform)
        Householder.h2(0, 1, 5, u, 0, 1, 1.0, c, 0, 1, 5, 0);
        assertArrayEquals(cOriginal, c, "h2 should not modify c when nc = 0");

        // Test with invalid bounds
        c = cOriginal.clone();
        Householder.h2(1, 1, 5, u, 0, 1, 1.0, c, 0, 1, 5, 1);  // pivot == start
        assertArrayEquals(cOriginal, c, "h2 should not modify c for invalid bounds");

        // Test with up = 0 (identity transformation)
        c = cOriginal.clone();
        Householder.h2(0, 1, 5, u, 0, 1, 0.0, c, 0, 1, 5, 1);
        // When up = 0, b = up * u[pivot] = 0, and since b >= 0, it's identity
        // So c should remain unchanged
    }

    // ========================================================================
    // Property 2.9: Householder transformation is involutory (Q = Q^(-1))
    // ========================================================================

    @Property(tries = 100)
    void householderTransformationIsInvolutory(
            @ForAll @IntRange(min = 3, max = 15) int m
    ) {
        // Create a random vector for Householder transformation
        double[] u = new double[m];
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < m; i++) {
            u[i] = rand.nextDouble() * 10 - 5;
        }

        int pivot = 0;
        int start = 1;
        double up = Householder.h1(pivot, start, m, u, 0, 1);

        if (up == 0.0) {
            return;  // Identity transformation
        }

        // Create a random vector to transform
        double[] v = new double[m];
        for (int i = 0; i < m; i++) {
            v[i] = rand.nextDouble() * 10 - 5;
        }
        double[] vOriginal = v.clone();

        // Apply transformation twice (Q * Q = I for Householder)
        Householder.h2(pivot, start, m, u, 0, 1, up, v, 0, 1, m, 1);
        Householder.h2(pivot, start, m, u, 0, 1, up, v, 0, 1, m, 1);

        // v should be back to original
        for (int i = 0; i < m; i++) {
            assertEquals(vOriginal[i], v[i], EPSILON,
                    "Applying Householder transformation twice should return original vector");
        }
    }

    // ========================================================================
    // Property 2.10: g1 handles special cases correctly
    // ========================================================================

    @Property(tries = 100)
    void g1HandlesSpecialCasesCorrectly() {
        double[] result = new double[3];

        // Test with both zero
        Householder.g1(0.0, 0.0, result);
        assertEquals(0.0, result[2], "sig should be 0 for (0,0)");

        // Test with a = 0
        Householder.g1(0.0, 5.0, result);
        assertEquals(5.0, result[2], EPSILON, "sig should be |b| when a = 0");
        assertEquals(0.0, result[0], EPSILON, "c should be 0 when a = 0");
        assertEquals(1.0, result[1], EPSILON, "s should be 1 when a = 0 and b > 0");

        // Test with b = 0
        Householder.g1(5.0, 0.0, result);
        assertEquals(5.0, result[2], EPSILON, "sig should be |a| when b = 0");
        assertEquals(1.0, result[0], EPSILON, "c should be 1 when b = 0 and a > 0");
        assertEquals(0.0, result[1], EPSILON, "s should be 0 when b = 0");

        // Test with negative a
        Householder.g1(-5.0, 0.0, result);
        assertEquals(5.0, result[2], EPSILON, "sig should be |a| when b = 0");
        assertEquals(-1.0, result[0], EPSILON, "c should be -1 when b = 0 and a < 0");
    }

    // ========================================================================
    // Property 2.11: Strided access works correctly for h1
    // ========================================================================

    @Property(tries = 100)
    void h1StridedAccessWorksCorrectly(
            @ForAll @IntRange(min = 3, max = 10) int m
    ) {
        // Create array with stride 2
        double[] u = new double[m * 2];
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < m; i++) {
            u[i * 2] = rand.nextDouble() * 10 - 5;
            u[i * 2 + 1] = 999.0;  // Filler that should not be touched
        }

        // Save original values
        double[] originalValues = new double[m];
        for (int i = 0; i < m; i++) {
            originalValues[i] = u[i * 2];
        }

        // Apply h1 with stride 2
        double up = Householder.h1(0, 1, m, u, 0, 2);

        // Verify filler values are unchanged
        for (int i = 0; i < m; i++) {
            assertEquals(999.0, u[i * 2 + 1], "Filler values should not be modified");
        }

        // Verify norm is preserved
        if (up != 0.0) {
            double originalNorm = 0;
            for (int i = 0; i < m; i++) {
                originalNorm += originalValues[i] * originalValues[i];
            }
            originalNorm = Math.sqrt(originalNorm);

            assertEquals(originalNorm, Math.abs(u[0]), EPSILON,
                    "Strided h1 should preserve norm");
        }
    }

    // ========================================================================
    // Property 2.12: Strided access works correctly for h2
    // ========================================================================

    @Property(tries = 100)
    void h2StridedAccessWorksCorrectly(
            @ForAll @IntRange(min = 3, max = 10) int m
    ) {
        // Create Householder vector with stride 1
        double[] u = new double[m];
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < m; i++) {
            u[i] = rand.nextDouble() * 10 - 5;
        }

        double up = Householder.h1(0, 1, m, u, 0, 1);
        if (up == 0.0) {
            return;
        }

        // Create matrix with stride 2 for elements
        double[] c = new double[m * 2];
        for (int i = 0; i < m; i++) {
            c[i * 2] = rand.nextDouble() * 10 - 5;
            c[i * 2 + 1] = 888.0;  // Filler
        }

        // Save original values
        double originalNorm = 0;
        for (int i = 0; i < m; i++) {
            originalNorm += c[i * 2] * c[i * 2];
        }
        originalNorm = Math.sqrt(originalNorm);

        // Apply h2 with stride 2 for c
        Householder.h2(0, 1, m, u, 0, 1, up, c, 0, 2, m * 2, 1);

        // Verify filler values are unchanged
        for (int i = 0; i < m; i++) {
            assertEquals(888.0, c[i * 2 + 1], "Filler values should not be modified");
        }

        // Verify norm is preserved
        double transformedNorm = 0;
        for (int i = 0; i < m; i++) {
            transformedNorm += c[i * 2] * c[i * 2];
        }
        transformedNorm = Math.sqrt(transformedNorm);

        assertEquals(originalNorm, transformedNorm, EPSILON,
                "Strided h2 should preserve norm");
    }
}
