/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop.numopt4j.linalg.blas;

/**
 * LAPACK ILAENV: Returns environment-dependent parameters for LAPACK routines.
 *
 * <p>Routine names are encoded as integer keys to avoid string allocation and
 * enable switch-on-int dispatch:
 * <ul>
 *   <li>c2 key = name[1..2] packed as {@code (char1 << 8) | char2} (16-bit)</li>
 *   <li>c3 key = name[3..5] packed as {@code (char3 << 16) | (char4 << 8) | char5} (24-bit)</li>
 * </ul>
 */
interface Ilaenv {

    // -------------------------------------------------------------------------
    // c2 constants: (name[1] << 8) | name[2]
    // -------------------------------------------------------------------------
    int GE = ('G' << 8) | 'E';
    int PO = ('P' << 8) | 'O';
    int SY = ('S' << 8) | 'Y';
    int HE = ('H' << 8) | 'E';
    int OR = ('O' << 8) | 'R';
    int UN = ('U' << 8) | 'N';
    int GB = ('G' << 8) | 'B';
    int PB = ('P' << 8) | 'B';
    int PT = ('P' << 8) | 'T';
    int TR = ('T' << 8) | 'R';
    int LA = ('L' << 8) | 'A';
    int ST = ('S' << 8) | 'T';

    // -------------------------------------------------------------------------
    // c3 constants: (name[3] << 16) | (name[4] << 8) | name[5]
    // -------------------------------------------------------------------------
    int TRF = ('T' << 16) | ('R' << 8) | 'F';
    int QRF = ('Q' << 16) | ('R' << 8) | 'F';
    int RQF = ('R' << 16) | ('Q' << 8) | 'F';
    int LQF = ('L' << 16) | ('Q' << 8) | 'F';
    int QLF = ('Q' << 16) | ('L' << 8) | 'F';
    int HRD = ('H' << 16) | ('R' << 8) | 'D';
    int BRD = ('B' << 16) | ('R' << 8) | 'D';
    int TRI = ('T' << 16) | ('R' << 8) | 'I';
    int TRD = ('T' << 16) | ('R' << 8) | 'D';
    int GST = ('G' << 16) | ('S' << 8) | 'T';
    int TRS = ('T' << 16) | ('R' << 8) | 'S';
    int EVC = ('E' << 16) | ('V' << 8) | 'C';
    int UUM = ('U' << 16) | ('U' << 8) | 'M';
    int EBZ = ('E' << 16) | ('B' << 8) | 'Z';
    int EXC = ('E' << 16) | ('X' << 8) | 'C';
    // OR/UN group: G-prefix (generate Q)
    int GQR = ('G' << 16) | ('Q' << 8) | 'R';
    int GRQ = ('G' << 16) | ('R' << 8) | 'Q';
    int GLQ = ('G' << 16) | ('L' << 8) | 'Q';
    int GQL = ('G' << 16) | ('Q' << 8) | 'L';
    int GHR = ('G' << 16) | ('H' << 8) | 'R';
    int GTR = ('G' << 16) | ('T' << 8) | 'R';
    int GBR = ('G' << 16) | ('B' << 8) | 'R';
    // OR/UN group: M-prefix (multiply by Q)
    int MQR = ('M' << 16) | ('Q' << 8) | 'R';
    int MRQ = ('M' << 16) | ('R' << 8) | 'Q';
    int MLQ = ('M' << 16) | ('L' << 8) | 'Q';
    int MQL = ('M' << 16) | ('Q' << 8) | 'L';
    int MHR = ('M' << 16) | ('H' << 8) | 'R';
    int MTR = ('M' << 16) | ('T' << 8) | 'R';
    int MBR = ('M' << 16) | ('B' << 8) | 'R';

    // -------------------------------------------------------------------------
    // iparmq sub5 constants: name[1..5] packed into long
    // (name[1]<<32)|(name[2]<<24)|(name[3]<<16)|(name[4]<<8)|name[5]
    // -------------------------------------------------------------------------
    long GGHRD = ((long)'G' << 32) | ((long)'G' << 24) | ((long)'H' << 16) | ((long)'R' << 8) | 'D';
    long GGHD3 = ((long)'G' << 32) | ((long)'G' << 24) | ((long)'H' << 16) | ((long)'D' << 8) | '3';
    long HSEQR = ((long)'H' << 32) | ((long)'S' << 24) | ((long)'E' << 16) | ((long)'Q' << 8) | 'R';

    // -------------------------------------------------------------------------
    // iparmq sub4 constant: name[1..4] packed into int
    // (name[1]<<24)|(name[2]<<16)|(name[3]<<8)|name[4]
    // -------------------------------------------------------------------------
    int LAQR = ('L' << 24) | ('A' << 16) | ('Q' << 8) | 'R';

    // -------------------------------------------------------------------------
    // Encoding helpers (runtime, used at call sites)
    // -------------------------------------------------------------------------

    /** Extract c2 key from a LAPACK routine name (chars at index 1-2, uppercased). */
    static int c2(String name) {
        return (Character.toUpperCase(name.charAt(1)) << 8)
             |  Character.toUpperCase(name.charAt(2));
    }

    /** Extract c3 key from a LAPACK routine name (chars at index 3-5, uppercased). */
    static int c3(String name) {
        return (Character.toUpperCase(name.charAt(3)) << 16)
             | (Character.toUpperCase(name.charAt(4)) << 8)
             |  Character.toUpperCase(name.charAt(5));
    }

    /** Extract sub4 key from a LAPACK routine name (chars at index 1-4, uppercased). */
    static int sub4(String name) {
        return (Character.toUpperCase(name.charAt(1)) << 24)
             | (Character.toUpperCase(name.charAt(2)) << 16)
             | (Character.toUpperCase(name.charAt(3)) << 8)
             |  Character.toUpperCase(name.charAt(4));
    }

    /** Extract sub5 key from a LAPACK routine name (chars at index 1-5, uppercased). */
    static long sub5(String name) {
        return ((long) Character.toUpperCase(name.charAt(1)) << 32)
             | ((long) Character.toUpperCase(name.charAt(2)) << 24)
             | ((long) Character.toUpperCase(name.charAt(3)) << 16)
             | ((long) Character.toUpperCase(name.charAt(4)) << 8)
             |  (long) Character.toUpperCase(name.charAt(5));
    }

    /**
     * Returns algorithm tuning parameters for the given LAPACK routine.
     *
     * @param ispec  Parameter selector:
     *               1  = optimal block size
     *               2  = minimum block size
     *               3  = crossover point (unblocked vs blocked)
     *               4  = number of shifts
     *               5  = minimum column dimension for blocking
     *               6  = SVD crossover point (min(m,n)*1.6)
     *               7  = number of processors
     *               8  = multi-shift QR crossover
     *               9  = divide-and-conquer subproblem size
     *               10 = IEEE infinity+NaN arithmetic trusted
     *               11 = IEEE infinity arithmetic trusted
     *               12-16 = Dhseqr / Dlaqr parameters (delegated to iparmq)
     * @param name   Routine name, e.g. "DGEHRD" (first char must be S/D/C/Z)
     * @param opts   Options string (e.g. "U", "L")
     * @param n1     First dimension parameter
     * @param n2     Second dimension parameter
     * @param n3     Third dimension parameter
     * @param n4     Fourth dimension parameter
     * @return The requested parameter value
     */
    static int ilaenv(int ispec, String name, String opts, int n1, int n2, int n3, int n4) {
        switch (ispec) {
            case 1:  return blockSize1(name, n2, n4);
            case 2:  return blockSize2(name);
            case 3:  return blockSize3(name);
            case 4:  return 6;           // used by xHSEQR
            case 5:  return 2;           // not used
            case 6:  return (int) (Math.min(n1, n2) * 1.6); // used by xGELSS and xGESVD
            case 7:  return 1;           // not used
            case 8:  return 50;          // used by xHSEQR
            case 9:  return 25;          // used by xGELSD and xGESDD
            case 10: return 1;           // JVM guarantees IEEE 754
            case 11: return 1;           // JVM guarantees IEEE 754
            case 12:
            case 13:
            case 14:
            case 15:
            case 16: return iparmq(ispec, name, opts, n1, n2, n3, n4); // Dhseqr and related
            default: throw new IllegalArgumentException("ilaenv: invalid ispec=" + ispec);
        }
    }

    // -------------------------------------------------------------------------
    // ispec=1: optimal block size
    // -------------------------------------------------------------------------
    static int blockSize1(String name, int n2, int n4) {
        if (name == null || name.length() < 6) return 1;
        switch (c2(name)) {
            case GE:
                switch (c3(name)) {
                    case TRF: return 64;
                    case TRI: return 64;
                    case QRF:
                    case RQF:
                    case LQF:
                    case QLF:
                    case HRD:
                    case BRD: return 32;
                    default:  return 1;
                }
            case PO:
                switch (c3(name)) {
                    case TRF: return 64;
                    default:  return 1;
                }
            case SY:
                switch (c3(name)) {
                    case TRF:
                    case GST: return 64;
                    case TRD: return 32;
                    default:  return 1;
                }
            case HE:
                switch (c3(name)) {
                    case TRF:
                    case GST: return 64;
                    case TRD: return 32;
                    default:  return 1;
                }
            case OR:
            case UN:
                switch (c3(name)) {
                    case GQR:
                    case GRQ:
                    case GLQ:
                    case GQL:
                    case GHR:
                    case GTR:
                    case GBR:
                    case MQR:
                    case MRQ:
                    case MLQ:
                    case MQL:
                    case MHR:
                    case MTR:
                    case MBR: return 32;
                    default:  return 1;
                }
            case GB:
                switch (c3(name)) {
                    case TRF: return n4 <= 64 ? 1 : 32;
                    default:  return 1;
                }
            case PB:
                switch (c3(name)) {
                    case TRF: return n2 <= 64 ? 1 : 32;
                    default:  return 1;
                }
            case PT:
                switch (c3(name)) {
                    case TRS: return 1;
                    default:  return 1;
                }
            case TR:
                switch (c3(name)) {
                    case TRI:
                    case EVC: return 64;
                    default:  return 1;
                }
            case LA:
                switch (c3(name)) {
                    case UUM: return 64;
                    default:  return 1;
                }
            case ST:
                switch (c3(name)) {
                    case EBZ: return 1;
                    default:  return 1;
                }
            default:
                return 1;
        }
    }

    // -------------------------------------------------------------------------
    // ispec=2: minimum block size
    // -------------------------------------------------------------------------
    static int blockSize2(String name) {
        if (name == null || name.length() < 6) return 2;
        switch (c2(name)) {
            case GE:
                switch (c3(name)) {
                    case QRF:
                    case RQF:
                    case LQF:
                    case QLF:
                    case HRD:
                    case BRD:
                    case TRI: return 2;
                    default:  return 2;
                }
            case SY:
                switch (c3(name)) {
                    case TRF: return 8;
                    case TRD: return 2;
                    default:  return 2;
                }
            case HE:
                switch (c3(name)) {
                    case TRD: return 2;
                    default:  return 2;
                }
            case OR:
            case UN:
                switch (c3(name)) {
                    case GQR:
                    case GRQ:
                    case GLQ:
                    case GQL:
                    case GHR:
                    case GTR:
                    case GBR:
                    case MQR:
                    case MRQ:
                    case MLQ:
                    case MQL:
                    case MHR:
                    case MTR:
                    case MBR: return 2;
                    default:  return 2;
                }
            default:
                return 2;
        }
    }

    // -------------------------------------------------------------------------
    // ispec=3: crossover point (unblocked → blocked)
    // -------------------------------------------------------------------------
    static int blockSize3(String name) {
        if (name == null || name.length() < 6) return 128;
        switch (c2(name)) {
            case GE:
                switch (c3(name)) {
                    case QRF:
                    case RQF:
                    case LQF:
                    case QLF:
                    case HRD:
                    case BRD: return 128;
                    default:  return 128;
                }
            case SY:
                switch (c3(name)) {
                    case TRD: return 32;
                    default:  return 128;
                }
            case HE:
                switch (c3(name)) {
                    case TRD: return 32;
                    default:  return 128;
                }
            case OR:
            case UN:
                switch (c3(name)) {
                    case GQR:
                    case GRQ:
                    case GLQ:
                    case GQL:
                    case GHR:
                    case GTR:
                    case GBR: return 128;
                    default:  return 128;
                }
            default:
                return 128;
        }
    }

    // -------------------------------------------------------------------------
    // ispec=12..16: Dhseqr / Dlaqr parameters (Iparmq)
    // -------------------------------------------------------------------------
    static int iparmq(int ispec, String name, String opts, int n, int ilo, int ihi, int lwork) {
        int nh = ihi - ilo + 1;
        // ns is determined by the interval nh falls into (largest threshold first).
        int ns = 2;
        if      (nh >= 6000) ns = 256;
        else if (nh >= 3000) ns = 128;
        else if (nh >= 590)  ns = 64;
        else if (nh >= 150)  ns = Math.max(10, nh / (int)(Math.log(nh) / Math.log(2)));
        else if (nh >= 60)   ns = 10;
        else if (nh >= 30)   ns = 4;
        ns = Math.max(2, ns - (ns % 2));

        switch (ispec) {
            case 12: return 75;
            case 13: return nh <= 500 ? ns : 3 * ns / 2;
            case 14: return 14;
            case 15: return ns;
            case 16: {
                if (name == null || name.length() != 6) return 0;
                final int k22min = 14, kacmin = 14;
                int acc22 = 0;
                long k = sub5(name);
                if (k == GGHRD || k == GGHD3) {
                    acc22 = 1;
                    if (nh >= k22min) acc22 = 2;
                } else if (c3(name) == EXC) {
                    if (nh >= kacmin) acc22 = 1;
                    if (nh >= k22min) acc22 = 2;
                } else if (k == HSEQR || sub4(name) == LAQR) {
                    if (ns >= kacmin) acc22 = 1;
                    if (ns >= k22min) acc22 = 2;
                }
                return acc22;
            }
            default: throw new IllegalArgumentException("iparmq: invalid ispec=" + ispec);
        }
    }
}
