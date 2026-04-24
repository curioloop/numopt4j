package com.curioloop.numopt4j.linalg.blas;

import java.util.Random;

public final class DlaqrTestSupport {

    private DlaqrTestSupport() {
    }

    static double[] randomHessenberg(int n, long seed) {
        double[] h = new double[n * n];
        fillRandomHessenberg(new Random(seed), h, n, n);
        return h;
    }

    static double[] structuredHessenberg(int n, long seed) {
        double[] h = new double[n * n];
        fillStructuredHessenberg(new Random(seed), h, n, n);
        return h;
    }

    static void fillRandomHessenberg(Random random, double[] h, int n, int ldh) {
        for (int row = 0; row < n; row++) {
            int rowOff = row * ldh;
            for (int col = 0; col < n; col++) {
                if (col < row - 1) {
                    h[rowOff + col] = 0.0;
                } else {
                    h[rowOff + col] = random.nextDouble() - 0.5;
                }
            }
        }

        for (int i = 0; i < n - 1; i++) {
            h[(i + 1) * ldh + i] += 0.25 + 0.25 * random.nextDouble();
        }
    }

    static void fillStructuredHessenberg(Random random, double[] h, int n, int ldh) {
        for (int row = 0; row < n; row++) {
            int rowOff = row * ldh;
            for (int col = 0; col < n; col++) {
                if (col < row - 1) {
                    h[rowOff + col] = 0.0;
                    continue;
                }

                if (col == row) {
                    h[rowOff + col] = 0.04 * (n - row) + 0.15 * (random.nextDouble() - 0.5);
                    continue;
                }

                if (col == row + 1) {
                    h[rowOff + col] = 0.18 * (random.nextDouble() - 0.5);
                    continue;
                }

                h[rowOff + col] = 0.03 * (random.nextDouble() - 0.5);
            }
        }

        for (int i = 0; i < n - 1; i++) {
            h[(i + 1) * ldh + i] = 0.12 + 0.08 * random.nextDouble();
        }

        for (int i = 0; i + 1 < n; i += 6) {
            h[i * ldh + i + 1] += 0.16;
            h[(i + 1) * ldh + i] += 0.10;
            h[(i + 1) * ldh + i + 1] = h[i * ldh + i] + 0.01 * (random.nextDouble() - 0.5);
        }
    }

    static void fillIdentity(double[] matrix, int n, int ldm) {
        for (int row = 0; row < n; row++) {
            int rowOff = row * ldm;
            for (int col = 0; col < n; col++) {
                matrix[rowOff + col] = row == col ? 1.0 : 0.0;
            }
        }
    }

    static void fillRealShiftPairs(double[] h, int n, int ldh, int nshfts, double[] sr, double[] si) {
        int diag = n - 1;
        for (int i = 0; i < nshfts; i += 2) {
            double shift = h[diag * ldh + diag];
            sr[i] = shift;
            si[i] = 0.0;
            if (i + 1 < nshfts) {
                sr[i + 1] = shift;
                si[i + 1] = 0.0;
            }
            diag = Math.max(0, diag - 1);
        }
    }

    static double maxOrthogonalityError(double[] q, int n, int ldq) {
        double maxError = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dot = 0.0;
                for (int k = 0; k < n; k++) {
                    dot += q[k * ldq + i] * q[k * ldq + j];
                }
                double expected = i == j ? 1.0 : 0.0;
                maxError = Math.max(maxError, Math.abs(dot - expected));
            }
        }
        return maxError;
    }
}