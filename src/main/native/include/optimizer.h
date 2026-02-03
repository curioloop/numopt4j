/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * High-performance optimization algorithms via JNI.
 */

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

/* ============================================================================
 * Numerical Stability Macros and Error Codes
 * ============================================================================ */

/* Numerical validity check macro - checks for NaN and Inf values */
#define IS_VALID_NUMBER(x) (!isnan(x) && !isinf(x))

/* Error codes for numerical stability issues */
#define ERR_NONE                 0  /* No error */
#define ERR_NOT_POS_DEF         -1  /* Matrix is not positive definite (Cholesky failed) */
#define ERR_NOT_POS_DEF_2ND_K   -2  /* Second K matrix not positive definite */
#define ERR_NOT_POS_DEF_T       -3  /* T matrix not positive definite */
#define ERR_DERIVATIVE          -4  /* Invalid derivative (not a descent direction) */
#define ERR_SINGULAR_TRIANGULAR -5  /* Triangular matrix is singular */
#define ERR_LINE_SEARCH_FAILED  -6  /* Line search failed to find valid step */
#define ERR_LINE_SEARCH_TOL     -7  /* Line search tolerance error */
#define ERR_TOO_MANY_RESETS     -8  /* Too many BFGS matrix resets */

/* ============================================================================
 * Common Numerical Constants
 * ============================================================================ */

#define ZERO  0.0
#define ONE   1.0
#define TWO   2.0
#define THREE 3.0
#define TEN   10.0
#define HUN   100.0

/* Machine epsilon from C standard library <float.h>
 * DBL_EPSILON is the smallest x such that 1.0 + x != 1.0
 * This matches Go's math.Nextafter(1, 2) - 1 */
#define EPS DBL_EPSILON

/* sqrt(DBL_EPSILON) - use compiler builtin if available for compile-time evaluation */
#if defined(__GNUC__) || defined(__clang__)
#define SQRT_EPS __builtin_sqrt(DBL_EPSILON)
#else
static inline double get_sqrt_eps(void) {
    static double val = 0.0;
    if (val == 0.0) val = sqrt(DBL_EPSILON);
    return val;
}
#define SQRT_EPS (get_sqrt_eps())
#endif

/* Cross-platform export macro */
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

/* ============================================================================
 * Common Types
 * ============================================================================ */

/* Bound type enumeration */
typedef enum {
    BOUND_NONE  = 0,  /* No bounds */
    BOUND_LOWER = 1,  /* Lower bound only */
    BOUND_BOTH  = 2,  /* Both lower and upper bounds */
    BOUND_UPPER = 3   /* Upper bound only */
} BoundType;

/* Optimization status codes */
typedef enum {
    STATUS_CONVERGED              =  0,  /* Optimization converged successfully */
    STATUS_MAX_ITER               =  1,  /* Maximum iterations reached */
    STATUS_MAX_EVAL               =  2,  /* Maximum evaluations reached */
    STATUS_GRAD_TOL               =  3,  /* Gradient tolerance satisfied */
    STATUS_FUNC_TOL               =  4,  /* Function tolerance satisfied */
    STATUS_MAX_TIME               =  5,  /* Maximum CPU time reached */
    STATUS_ABNORMAL               = -1,  /* Abnormal termination */
    STATUS_INVALID_ARG            = -2,  /* Invalid argument */
    STATUS_CONSTRAINT_INCOMPATIBLE = -3, /* Constraints incompatible */
    STATUS_LINE_SEARCH_FAILED     = -4,  /* Line search failed */
    STATUS_CALLBACK_ERROR         = -5   /* Callback function error */
} OptStatus;

/* Callback function types */

/**
 * Objective function evaluation callback.
 * Evaluates the objective function and optionally its gradient.
 * 
 * @param ctx User context pointer
 * @param x Current point (length n)
 * @param g Output: gradient vector (length n), can be NULL if gradient not needed
 * @param n Problem dimension
 * @return Function value f(x)
 */
typedef double (*Objective)(void* ctx, const double* x, double* g, int n);

/**
 * Batch constraint evaluation callback.
 * Evaluates all constraints at once for better performance.
 * 
 * @param ctx User context pointer
 * @param x Current point (length n)
 * @param c Output: constraint values (length m)
 * @param jac Output: constraint Jacobian in column-major order (m × n)
 *            jac[i + m*j] = ∂c_i/∂x_j
 *            Can be NULL if gradients are not needed
 * @param m Number of constraints
 * @param n Problem dimension
 */
typedef void (*Constraint)(void* ctx, const double* x, double* c, double* jac, int m, int n);

/* ============================================================================
 * L-BFGS-B Types
 * ============================================================================ */

/* Line search task status */
typedef enum {
    SEARCH_START = 0,
    SEARCH_FG = 1,      /* Need to compute f and g */
    SEARCH_CONV = 2,    /* Converged */
    SEARCH_WARN = 3,    /* Warning */
    SEARCH_ERROR = 4    /* Error */
} SearchTask;

/* Line search tolerance parameters */
typedef struct {
    double alpha;   /* Sufficient decrease parameter (default 1e-3) */
    double beta;    /* Curvature parameter (default 0.9) */
    double eps;     /* Relative tolerance */
    double lower;   /* Step lower bound */
    double upper;   /* Step upper bound */
} SearchTol;

/* Line search context */
typedef struct {
    int bracket;        /* Whether minimum is bracketed */
    int stage;          /* Search stage */
    double f0, g0;      /* Initial function value and derivative */
    double fx, gx;      /* Function value and derivative at stx */
    double fy, gy;      /* Function value and derivative at sty */
    double stx, sty;    /* Interval endpoints */
    double width[2];    /* Interval width history */
    double bound[2];    /* Step bounds */
} SearchCtx;

/**
 * L-BFGS-B workspace structure.
 * All state is stored here; algorithm functions are stateless.
 * This ensures thread safety - each thread uses its own workspace.
 */
typedef struct {
    int n;              /* Problem dimension */
    int m;              /* Number of L-BFGS corrections */
    
    /* Work arrays (allocated by caller) */
    double* ws;         /* n × m, S matrix */
    double* wy;         /* n × m, Y matrix */
    double* sy;         /* m × m, S^T Y */
    double* ss;         /* m × m, S^T S */
    double* wt;         /* m × m, Cholesky factor */
    double* wn;         /* 4 × m × m */
    double* snd;        /* 4 × m × m */
    double* z;          /* n */
    double* r;          /* n */
    double* d;          /* n */
    double* t;          /* n */
    double* xp;         /* n */
    double* g;          /* n, gradient */
    double* wa;         /* 8 × m */
    int* index;         /* 2 × n */
    int* iwhere;        /* n */
    
    /* Iteration state */
    int iter;
    int col;            /* Number of corrections stored */
    int head;           /* Head pointer for circular buffer */
    int tail;           /* Tail pointer for circular buffer */
    int total_eval;
    double f;           /* Current function value */
    double f_old;
    double theta;       /* Scaling factor */
    double sbg_norm;    /* Projected gradient norm */
    
    /* Free variable tracking (for subspace minimization) */
    int free;           /* Number of free variables */
    int active;         /* Number of active constraints */
    int enter;          /* Variables entering free set */
    int leave;          /* Variables leaving free set */
    int updated;        /* BFGS matrix updated flag */
    int updates;        /* Total number of BFGS updates */
    int word;           /* Solution status: 0=within box, 1=beyond box */
    int constrained;    /* Problem has constraints */
    int boxed;          /* All variables have both bounds */
    int seg;            /* Number of segments in piecewise linear path (GCP search) */
    
    /* Line search state */
    SearchTol search_tol;
    SearchCtx search_ctx;
    double gd;          /* Directional derivative g'd */
    double gd_old;      /* Previous directional derivative */
    double d_norm;      /* ||d||_2 */
    double d_sqrt;      /* d'd */
    int num_eval;       /* Line search function evaluations */
    int num_back;       /* Backtracking count */
    
    /* BFGS reset recovery state */
    int reset_count;    /* Number of BFGS matrix resets */
} LbfgsbWorkspace;

/**
 * L-BFGS-B configuration (read-only parameters).
 */
typedef struct {
    int n;              /* Problem dimension */
    int m;              /* Number of L-BFGS corrections */
    double* x;          /* Initial point / solution vector */
    double* lower;      /* Lower bounds */
    double* upper;      /* Upper bounds */
    int* bound_type;    /* Bound types */
    double factr;       /* Function value accuracy factor */
    double pgtol;       /* Projected gradient tolerance */
    int max_iter;       /* Maximum iterations */
    int max_eval;       /* Maximum function evaluations */
    long max_time;      /* Maximum wall-clock time in microseconds (0 = disabled) */
    void* eval_ctx;     /* Evaluation context */
    Objective eval;     /* Objective function callback */
} LbfgsbConfig;

/**
 * L-BFGS-B result structure.
 */
typedef struct {
    double f;           /* Optimal function value */
    int iterations;     /* Number of iterations */
    int evaluations;    /* Number of function evaluations */
    OptStatus status;   /* Optimization status */
} LbfgsbResult;

/* ============================================================================
 * SLSQP Types
 * ============================================================================ */

/* Exact line search mode */
typedef enum {
    FIND_NOOP = 0,
    FIND_INIT = 1,
    FIND_NEXT = 2,
    FIND_CONV = 3
} FindMode;

/* Exact line search workspace */
typedef struct {
    double a, b;        /* Search interval */
    double d, e;        /* Step information */
    double p, q, r;     /* Interpolation parameters */
    double u, v, w, x;  /* Key points */
    double m;           /* Midpoint */
    double fu, fv, fw, fx;  /* Function values */
    double tol1, tol2;  /* Tolerances */
} FindWork;

/**
 * SLSQP workspace structure.
 * All state is stored here; algorithm functions are stateless.
 * 
 * Array dimensions match Go implementation (base.go, optimize.go):
 * - l: (n+1)*(n+2)/2 = n*(n+1)/2 + n + 1 (LDL^T factor, extra space for augmented QP)
 * - r: 2n + m + 2 (multipliers for constraints and bounds)
 * - c: max(1,m) (constraint values)
 * - a: max(1,m) × (n+1) (constraint Jacobian, column-major)
 * - mu: max(1,m) (penalty multipliers)
 * 
 * Requirements: 8.1, 8.2, 8.3
 */
typedef struct {
    int n;              /* Problem dimension */
    int m;              /* Total number of constraints */
    int meq;            /* Number of equality constraints */
    
    /* Work arrays (allocated by caller)
     * Dimensions match Go implementation in optimize.go */
    double* l;          /* (n+1)*(n+2)/2, LDL^T factor (Go: ll := (n+1)*(n+2)/2) */
    double* x0;         /* n, initial position */
    double* g;          /* n+1, gradient */
    double* c;          /* max(1,m), constraint values */
    double* a;          /* max(1,m) × (n+1), constraint Jacobian (column-major) */
    double* mu;         /* max(1,m), penalty multipliers */
    double* s;          /* n+1, search direction */
    double* u;          /* n+1, lower bound difference */
    double* v;          /* n+1, upper bound difference */
    double* r;          /* 2n+m+2, multipliers (Go: lr := n+n+m+2) */
    double* w;          /* General workspace */
    int* jw;            /* Integer workspace */
    
    /* Iteration state */
    int iter;
    int mode;           /* Current mode */
    double acc;         /* Accuracy */
    double f0;          /* Initial function value */
    double alpha;       /* Line search step */
    
    /* Exact line search state */
    FindWork fw;        /* Exact line search workspace */
    int line_mode;      /* Line search mode */
    double t0;          /* Merit function initial value */
} SlsqpWorkspace;

/**
 * SLSQP configuration (read-only parameters).
 */
typedef struct {
    int n;              /* Problem dimension */
    int meq;            /* Number of equality constraints */
    int mineq;          /* Number of inequality constraints */
    double* x;          /* Initial point / solution vector */
    double* lower;      /* Lower bounds */
    double* upper;      /* Upper bounds */
    double accuracy;    /* Solution accuracy */
    int max_iter;       /* Maximum iterations */
    int exact_search;   /* Enable exact line search (0=disabled, 1=enabled) */
    int nnls_iter;      /* Maximum NNLS iterations (0 = use default 3*n) */
    long max_time;      /* Maximum wall-clock time in microseconds (0 = disabled) */

    /* Extended termination criteria (negative value = disabled) */
    double f_eval_tol;  /* Terminate when |f(x)| < tol, -1.0 = disabled */
    double f_diff_tol;  /* Terminate when |f_new - f_old| < tol, -1.0 = disabled */
    double x_diff_tol;  /* Terminate when ||x_new - x_old||_2 < tol, -1.0 = disabled */
    
    void* eval_ctx;               /* Evaluation context */
    Objective obj_eval;           /* Objective function callback */
    Constraint eq_eval;           /* Equality constraint callback */
    Constraint ineq_eval;         /* Inequality constraint callback */
} SlsqpConfig;

/**
 * SLSQP result structure.
 */
typedef struct {
    double f;           /* Optimal function value */
    int iterations;     /* Number of iterations */
    OptStatus status;   /* Optimization status */
} SlsqpResult;

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Get current wall-clock time in microseconds.
 * Uses monotonic clock to avoid issues with system time adjustments.
 * @return Current time in microseconds
 */
long get_time_us(void);

/* ============================================================================
 * L-BFGS-B Functions
 * ============================================================================ */

/**
 * Calculate the required workspace size for L-BFGS-B.
 * @param n Problem dimension
 * @param m Number of L-BFGS corrections
 * @return Required size in bytes
 */
EXPORT size_t lbfgsb_workspace_size(int n, int m);

/**
 * Initialize L-BFGS-B workspace.
 * @param ws Workspace structure to initialize
 * @param memory Pre-allocated memory block
 * @param n Problem dimension
 * @param m Number of L-BFGS corrections
 */
EXPORT void lbfgsb_workspace_init(LbfgsbWorkspace* ws, void* memory, int n, int m);

/**
 * Reset L-BFGS-B workspace for a new optimization.
 * @param ws Workspace to reset
 */
EXPORT void lbfgsb_workspace_reset(LbfgsbWorkspace* ws);

/**
 * Execute L-BFGS-B optimization (stateless function).
 * @param config Configuration parameters (read-only)
 * @param workspace Workspace (read-write)
 * @param result Result output
 * @return Optimization status
 */
EXPORT OptStatus lbfgsb_optimize(const LbfgsbConfig* config, 
                                  LbfgsbWorkspace* workspace, 
                                  LbfgsbResult* result);

/* ============================================================================
 * SLSQP Functions
 * ============================================================================ */

/**
 * Calculate the required workspace size for SLSQP.
 * @param n Problem dimension
 * @param meq Number of equality constraints
 * @param mineq Number of inequality constraints
 * @return Required size in bytes
 */
EXPORT size_t slsqp_workspace_size(int n, int meq, int mineq);

/**
 * Initialize SLSQP workspace.
 * @param ws Workspace structure to initialize
 * @param memory Pre-allocated memory block
 * @param n Problem dimension
 * @param meq Number of equality constraints
 * @param mineq Number of inequality constraints
 */
EXPORT void slsqp_workspace_init(SlsqpWorkspace* ws, void* memory, 
                                  int n, int meq, int mineq);

/**
 * Reset SLSQP workspace for a new optimization.
 * @param ws Workspace to reset
 */
EXPORT void slsqp_workspace_reset(SlsqpWorkspace* ws);

/**
 * Execute SLSQP optimization (stateless function).
 * @param config Configuration parameters (read-only)
 * @param workspace Workspace (read-write)
 * @param result Result output
 * @return Optimization status
 */
EXPORT OptStatus slsqp_optimize(const SlsqpConfig* config,
                                 SlsqpWorkspace* workspace,
                                 SlsqpResult* result);

/* ============================================================================
 * Householder Transformation Functions
 * ============================================================================ */

/**
 * Compute Householder transformation vector.
 * @param pivot Pivot index
 * @param start Start index for transformation
 * @param m Vector length
 * @param u Vector to transform (modified in place)
 * @param inc Increment for u
 * @return Pivot element up for use in h2
 */
double h1(int pivot, int start, int m, double* u, int inc);

/**
 * Apply Householder transformation.
 * @param pivot Pivot index
 * @param start Start index
 * @param m Number of rows
 * @param u Householder vector
 * @param incu Increment for u
 * @param up Pivot element from h1
 * @param c Matrix to transform (column-major)
 * @param incc Row increment for c
 * @param mdc Column stride for c
 * @param nc Number of columns to transform
 */
void h2(int pivot, int start, int m, double* u, int incu,
        double up, double* c, int incc, int mdc, int nc);

/**
 * Compute Givens rotation.
 * @param a First element
 * @param b Second element
 * @param c Output: cosine
 * @param s Output: sine
 * @param sig Output: resulting value
 */
void g1(double a, double b, double* c, double* s, double* sig);

/**
 * Apply Givens rotation.
 * @param c Cosine from g1
 * @param s Sine from g1
 * @param a First element (modified)
 * @param b Second element (modified)
 */
void g2(double c, double s, double* a, double* b);

/* ============================================================================
 * LINPACK Functions
 * ============================================================================ */

/**
 * Cholesky factorization of a symmetric positive definite matrix.
 * @param a Matrix (column-major), upper triangle contains result
 * @param lda Leading dimension of a
 * @param n Order of matrix
 * @return 0 if successful, k if leading minor k is not positive definite
 */
int dpofa(double* a, int lda, int n);

/**
 * Solve triangular system T*x = b or T'*x = b.
 * @param t Triangular matrix (column-major)
 * @param ldt Leading dimension of t
 * @param n Order of matrix
 * @param b Right-hand side, solution on exit
 * @param job Solve type: 0=L*x=b, 1=L'*x=b, 10=U*x=b, 11=U'*x=b
 * @return 0 if successful, k if t[k][k] = 0
 */
int dtrsl(double* t, int ldt, int n, double* b, int job);

/**
 * Update LDL' factorization with rank-1 modification.
 * @param n Order of matrix
 * @param l Lower triangular matrix in packed form (modified)
 * @param z Vector for rank-1 update
 * @param sigma Scalar multiplier
 * @param w Working vector (can be NULL if sigma > 0)
 */
void compositeT(int n, double* l, double* z, double sigma, double* w);

/* ============================================================================
 * Least Squares Solver Functions
 * ============================================================================ */

/**
 * HFTI - Householder Forward Triangulation with column Interchanges.
 * Solves least-squares problem AX ≅ B.
 * @param m Number of rows in A
 * @param n Number of columns in A
 * @param a Matrix A (column-major), modified on return
 * @param mda Leading dimension of A
 * @param b Matrix B (column-major), contains solution X on return
 * @param mdb Leading dimension of B
 * @param nb Number of columns in B
 * @param tau Tolerance for pseudo-rank determination
 * @param rnorm Output: residual norms
 * @param h Working array of length n
 * @param g Working array of length min(m,n)
 * @param ip Working array of length min(m,n)
 * @return Pseudo-rank k
 */
int hfti(int m, int n, double* a, int mda,
         double* b, int mdb, int nb,
         double tau, double* rnorm,
         double* h, double* g, int* ip);

/**
 * NNLS - Non-Negative Least Squares.
 * Solves min ||Ax - b||_2 subject to x >= 0.
 * @param m Number of rows in A
 * @param n Number of columns in A
 * @param a Matrix A (column-major), modified on return
 * @param mda Leading dimension of A
 * @param b Vector b, modified on return
 * @param x Output: solution vector
 * @param w Output: dual vector
 * @param z Working array of length m
 * @param index Working array of length n
 * @param maxIter Maximum iterations
 * @param rnorm Output: residual norm
 * @return Status code (0 = success)
 */
int nnls(int m, int n, double* a, int mda,
         double* b, double* x, double* w,
         double* z, int* index, int maxIter,
         double* rnorm);

/**
 * LDP - Least Distance Programming.
 * Solves min ||x||_2 subject to Gx >= h.
 * @param m Number of constraints
 * @param n Number of variables
 * @param g Constraint matrix G (column-major)
 * @param mdg Leading dimension of G
 * @param h Constraint vector h
 * @param x Output: solution vector
 * @param w Working array
 * @param jw Working array
 * @param maxIter Maximum iterations
 * @param xnorm Output: norm of solution
 * @return Status code (0 = success)
 */
int ldp(int m, int n, double* g, int mdg,
        double* h, double* x, double* w,
        int* jw, int maxIter, double* xnorm);

/**
 * LSI - Least Squares with Inequality constraints.
 * Solves min ||Ex - f||_2 subject to Gx >= h.
 * @param e Matrix E (column-major), modified on return
 * @param f Vector f, modified on return
 * @param g Matrix G (column-major), modified on return
 * @param h Vector h, modified on return
 * @param le Leading dimension of E
 * @param me Number of rows in E
 * @param lg Leading dimension of G
 * @param mg Number of constraints
 * @param n Number of variables
 * @param x Output: solution vector
 * @param w Working array
 * @param jw Working array
 * @param maxIter Maximum iterations
 * @param xnorm Output: residual norm
 * @return Status code (0 = success)
 */
int lsi(double* e, double* f, double* g, double* h,
        int le, int me, int lg, int mg, int n,
        double* x, double* w, int* jw, int maxIter,
        double* xnorm);

/**
 * LSEI - Least Squares with Equality and Inequality constraints.
 * Solves min ||Ex - f||_2 subject to Cx = d and Gx >= h.
 * @param c Matrix C (column-major), modified on return
 * @param d Vector d, modified on return
 * @param e Matrix E (column-major), modified on return
 * @param f Vector f, modified on return
 * @param g Matrix G (column-major), modified on return
 * @param h Vector h, modified on return
 * @param lc Leading dimension of C
 * @param mc Number of equality constraints
 * @param le Leading dimension of E
 * @param me Number of rows in E
 * @param lg Leading dimension of G
 * @param mg Number of inequality constraints
 * @param n Number of variables
 * @param x Output: solution vector
 * @param w Working array
 * @param jw Working array
 * @param maxIter Maximum iterations
 * @param norm Output: residual norm
 * @return Status code (0 = success)
 */
int lsei(double* c, double* d, double* e, double* f,
         double* g, double* h,
         int lc, int mc, int le, int me, int lg, int mg, int n,
         double* x, double* w, int* jw, int maxIter,
         double* norm);

/**
 * LSQ - Least Squares Quadratic Programming.
 * Solves QP subproblem for SLSQP.
 * @param m Total number of constraints
 * @param meq Number of equality constraints
 * @param n Number of variables
 * @param nl Length of l array
 * @param l L + D in packed form
 * @param g Gradient vector
 * @param a Constraint matrix A (column-major)
 * @param b Constraint vector b
 * @param xl Lower bounds
 * @param xu Upper bounds
 * @param x Output: solution vector
 * @param y Output: Lagrange multipliers
 * @param w Working array
 * @param jw Working array
 * @param maxIter Maximum iterations
 * @param infBnd Infinity bound value
 * @param norm Output: residual norm
 * @return Status code (0 = success)
 */
int lsq(int m, int meq, int n, int nl,
        double* l, double* g, double* a, double* b,
        double* xl, double* xu,
        double* x, double* y, double* w, int* jw,
        int maxIter, double infBnd, double* norm);

#endif /* OPTIMIZER_H */
