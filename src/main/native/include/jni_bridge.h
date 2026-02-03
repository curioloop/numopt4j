/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * JNI bridge header for Java-C interoperability.
 * 
 * THREAD SAFETY NOTE:
 * The JNIEnv* pointer stored in callback contexts is thread-bound. Each JNIEnv
 * is only valid for the thread that obtained it. The optimization functions
 * must be called from the same thread that initiated the JNI call.
 * 
 * For concurrent optimization, each thread must:
 * 1. Have its own JNIEnv* (obtained via AttachCurrentThread if needed)
 * 2. Use separate optimizer and workspace instances
 * 3. Not share callback contexts between threads
 */

#ifndef JNI_BRIDGE_H
#define JNI_BRIDGE_H

#include <jni.h>
#include "optimizer.h"

/* ============================================================================
 * Constants
 * ============================================================================ */

/**
 * Threshold for using PushLocalFrame/PopLocalFrame in constraint evaluation.
 * When the number of constraints exceeds this value, local frame management
 * is used to prevent local reference table overflow.
 */
#define LOCAL_REF_THRESHOLD 16

/* ============================================================================
 * JNI Callback Context Structures
 * 
 * NOTE: The JNIEnv* pointer in these structures is thread-bound. These contexts
 * must only be used from the thread that created them.
 * ============================================================================ */

/**
 * Context for objective function callbacks from C to Java.
 * Uses the user's x array directly and pre-allocated gradient array from workspace.
 * 
 * @note The env pointer is thread-bound and must not be used from other threads.
 */
typedef struct {
    JNIEnv* env;              /* JNI environment (thread-bound, do not share) */
    jobject callback;         /* Java callback object */
    jmethodID method_id;      /* Method ID for evaluate() */
    jdoubleArray x_array;     /* User's x array (passed from optimize call) */
    jdoubleArray g_array;     /* Pre-allocated array for gradient (from workspace) */
    int has_error;            /* Error flag */
} JniObjectiveContext;

/**
 * Context for batch constraint function callbacks from C to Java.
 * Uses the user's x array directly and pre-allocated arrays from workspace.
 * 
 * @note The env pointer is thread-bound and must not be used from other threads.
 */
typedef struct {
    JNIEnv* env;              /* JNI environment (thread-bound, do not share) */
    jobject callback;         /* Java batch constraint callback object */
    jmethodID method_id;      /* Method ID for evaluate(double[], double[], double[]) */
    jdoubleArray x_array;     /* User's x array (passed from optimize call) */
    jdoubleArray c_array;     /* Pre-allocated array for constraint values (from workspace, length m) */
    jdoubleArray jac_array;   /* Pre-allocated array for Jacobian (from workspace, length m*n) */
    int m;                    /* Number of constraints */
    int n;                    /* Problem dimension */
    int has_error;            /* Error flag */
} JniConstraintContext;

/**
 * Combined context for SLSQP callbacks.
 */
typedef struct {
    JniObjectiveContext obj_ctx;
    JniConstraintContext eq_ctx;
    JniConstraintContext ineq_ctx;
} JniSlsqpContext;

/* ============================================================================
 * JNI Helper Functions
 * ============================================================================ */

/**
 * Initialize objective callback context.
 * @param ctx Context to initialize
 * @param env JNI environment
 * @param callback Java ObjectiveFunction object
 * @param x_array User's x array (will be used directly in callbacks)
 * @param g_array Pre-allocated gradient array from workspace
 * @return 0 on success, -1 on error
 */
int jni_init_objective_context(JniObjectiveContext* ctx, JNIEnv* env,
                               jobject callback, jdoubleArray x_array, jdoubleArray g_array);

/**
 * Initialize batch constraint callback context.
 * @param ctx Context to initialize
 * @param env JNI environment
 * @param callback Java ConstraintFunction object (batch version)
 * @param x_array User's x array (will be used directly in callbacks)
 * @param c_array Pre-allocated constraint values array from workspace
 * @param jac_array Pre-allocated Jacobian array from workspace
 * @param m Number of constraints
 * @param n Problem dimension
 * @return 0 on success, -1 on error
 */
int jni_init_constraint_context(JniConstraintContext* ctx, JNIEnv* env,
                                jobject callback, jdoubleArray x_array,
                                jdoubleArray c_array, jdoubleArray jac_array,
                                int m, int n);

/**
 * Clean up objective callback context.
 * @param ctx Context to clean up
 */
void jni_cleanup_objective_context(JniObjectiveContext* ctx);

/**
 * Clean up constraint callback context.
 * @param ctx Context to clean up
 */
void jni_cleanup_constraint_context(JniConstraintContext* ctx);

/**
 * Objective function callback wrapper.
 * @param ctx JniObjectiveContext pointer
 * @param x Current point
 * @param g Gradient output (may be NULL)
 * @param n Dimension
 * @return Function value
 */
double jni_objective_callback(void* ctx, const double* x, double* g, int n);

/**
 * Batch constraint function callback wrapper for SLSQP.
 * Evaluates all constraints at once.
 * @param ctx JniConstraintContext pointer
 * @param x Current point (length n)
 * @param c Output: constraint values (length m)
 * @param jac Output: Jacobian in column-major order (m × n), can be NULL
 * @param m Number of constraints
 * @param n Problem dimension
 */
void jni_constraint_callback(void* ctx, const double* x, double* c, double* jac, int m, int n);

/**
 * Check for and handle JNI exceptions.
 * @param env JNI environment
 * @return 1 if exception occurred, 0 otherwise
 */
int jni_check_exception(JNIEnv* env);

/**
 * Throw a Java exception from native code.
 * @param env JNI environment
 * @param class_name Exception class name
 * @param message Exception message
 */
void jni_throw_exception(JNIEnv* env, const char* class_name, const char* message);

/* ============================================================================
 * JNI Native Method Declarations
 * ============================================================================ */

#ifdef __cplusplus
extern "C" {
#endif

/* L-BFGS-B workspace methods */
JNIEXPORT jlong JNICALL Java_com_curioloop_LbfgsbWorkspace_nativeWorkspaceSize
    (JNIEnv* env, jclass cls, jint n, jint m);

JNIEXPORT void JNICALL Java_com_curioloop_LbfgsbWorkspace_nativeWorkspaceInit
    (JNIEnv* env, jclass cls, jlong wsPtr, jlong memPtr, jint n, jint m);

/* L-BFGS-B optimizer method (always uses external workspace) */
JNIEXPORT jint JNICALL Java_com_curioloop_LbfgsbOptimizer_nativeOptimize
    (JNIEnv* env, jobject obj,
     /* Problem definition */
     jint n, jint m, jdoubleArray x,
     /* Callbacks */
     jobject objective, jdoubleArray gradient,
     /* Termination criteria */
     jdouble factr, jdouble pgtol, jint maxIter, jint maxEval, jlong maxTimeMs,
     /* Workspace */
     jobject workspace,
     jint lowerOffset, jint upperOffset, jint boundTypeOffset, jint resultOffset);

/* SLSQP workspace methods */
JNIEXPORT jlong JNICALL Java_com_curioloop_SlsqpWorkspace_nativeWorkspaceSize
    (JNIEnv* env, jclass cls, jint n, jint meq, jint mineq);

JNIEXPORT void JNICALL Java_com_curioloop_SlsqpWorkspace_nativeWorkspaceInit
    (JNIEnv* env, jclass cls, jlong wsPtr, jlong memPtr, jint n, jint meq, jint mineq);

/* SLSQP optimizer method (always uses external workspace) */
JNIEXPORT jint JNICALL Java_com_curioloop_SlsqpOptimizer_nativeOptimize
    (JNIEnv* env, jobject obj,
     /* Problem definition */
     jint n, jint meq, jint mineq, jdoubleArray x,
     /* Objective callback */
     jobject objective, jdoubleArray gradient,
     /* Constraint callbacks (share arrays since evaluated sequentially) */
     jobject eqConstraint, jobject ineqConstraint,
     jdoubleArray constraintValues, jdoubleArray constraintJacobian,
     /* Termination criteria */
     jdouble accuracy, jint maxIter, jint nnlsIter, jlong maxTime,
     jboolean exactLineSearch, jdouble fEvalTol, jdouble fDiffTol, jdouble xDiffTol,
     /* Workspace */
     jobject workspace,
     jint lowerOffset, jint upperOffset, jint resultOffset);

#ifdef __cplusplus
}
#endif

#endif /* JNI_BRIDGE_H */
