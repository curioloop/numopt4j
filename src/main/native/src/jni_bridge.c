/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 * JNI bridge implementation for Java-C interoperability.
 */

#include "jni_bridge.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * JNI Helper Functions
 * ============================================================================ */

int jni_check_exception(JNIEnv* env) {
    if ((*env)->ExceptionCheck(env)) {
        return 1;
    }
    return 0;
}

void jni_throw_exception(JNIEnv* env, const char* class_name, const char* message) {
    jclass exc_class = (*env)->FindClass(env, class_name);
    if (exc_class != NULL) {
        (*env)->ThrowNew(env, exc_class, message);
        (*env)->DeleteLocalRef(env, exc_class);
    }
}

int jni_init_objective_context(JniObjectiveContext* ctx, JNIEnv* env,
                               jobject callback, jdoubleArray x_array, jdoubleArray g_array) {
    if (!ctx || !env || !callback || !x_array || !g_array) return -1;
    
    ctx->env = env;
    ctx->callback = callback;
    ctx->has_error = 0;
    ctx->x_array = x_array;
    ctx->g_array = g_array;
    
    /* Get method ID for evaluate(double[], double[]) */
    jclass cls = (*env)->GetObjectClass(env, callback);
    if (!cls) return -1;
    
    ctx->method_id = (*env)->GetMethodID(env, cls, "evaluate", "([D[D)D");
    (*env)->DeleteLocalRef(env, cls);
    
    if (!ctx->method_id) return -1;
    
    return 0;
}

int jni_init_constraint_context(JniConstraintContext* ctx, JNIEnv* env,
                                jobject callback, jdoubleArray x_array,
                                jdoubleArray c_array, jdoubleArray jac_array,
                                int m, int n) {
    if (!ctx || !env) return -1;
    
    ctx->env = env;
    ctx->callback = callback;
    ctx->has_error = 0;
    ctx->m = m;
    ctx->n = n;
    ctx->x_array = x_array;
    ctx->c_array = c_array;
    ctx->jac_array = jac_array;
    
    if (!callback || m == 0) {
        ctx->method_id = NULL;
        return 0;
    }
    
    /* Get method ID for evaluate(double[] x, double[] c, double[] jac) */
    jclass cls = (*env)->GetObjectClass(env, callback);
    if (!cls) return -1;
    
    ctx->method_id = (*env)->GetMethodID(env, cls, "evaluate", "([D[D[D)V");
    (*env)->DeleteLocalRef(env, cls);
    
    if (!ctx->method_id) {
        /* Clear any NoSuchMethodError exception */
        if ((*env)->ExceptionCheck(env)) {
            (*env)->ExceptionClear(env);
        }
        return -1;
    }
    
    return 0;
}

void jni_cleanup_objective_context(JniObjectiveContext* ctx) {
    /* Arrays are managed by Java workspace, no cleanup needed */
    (void)ctx;
}

void jni_cleanup_constraint_context(JniConstraintContext* ctx) {
    /* Arrays are managed by Java workspace, no cleanup needed */
    (void)ctx;
}

double jni_objective_callback(void* ctx_ptr, const double* x, double* g, int n) {
    JniObjectiveContext* ctx = (JniObjectiveContext*)ctx_ptr;
    if (!ctx || ctx->has_error) return NAN;
    
    JNIEnv* env = ctx->env;
    
    /* Copy x to Java array */
    (*env)->SetDoubleArrayRegion(env, ctx->x_array, 0, n, x);
    
    /* Prepare gradient array (null if not needed) */
    jdoubleArray g_arr = (g != NULL) ? ctx->g_array : NULL;
    
    /* Call Java method */
    jdouble result = (*env)->CallDoubleMethod(env, ctx->callback, ctx->method_id,
                                               ctx->x_array, g_arr);
    
    /* Check for exception */
    if (jni_check_exception(env)) {
        ctx->has_error = 1;
        return NAN;
    }
    
    /* Copy gradient back if needed */
    if (g != NULL) {
        (*env)->GetDoubleArrayRegion(env, ctx->g_array, 0, n, g);
    }
    
    return result;
}

/**
 * Batch constraint function callback wrapper.
 * Evaluates all constraints at once for better performance.
 * 
 * @param ctx_ptr JniConstraintContext pointer
 * @param x Current point (length n)
 * @param c Output: constraint values (length m)
 * @param jac Output: Jacobian in column-major order (m × n), can be NULL
 * @param m Number of constraints
 * @param n Problem dimension
 */
void jni_constraint_callback(void* ctx_ptr, const double* x, double* c, double* jac, int m, int n) {
    JniConstraintContext* ctx = (JniConstraintContext*)ctx_ptr;
    if (!ctx || ctx->has_error || !ctx->callback) return;
    
    JNIEnv* env = ctx->env;
    
    /* Copy x to Java array */
    (*env)->SetDoubleArrayRegion(env, ctx->x_array, 0, n, x);
    
    /* Prepare Jacobian array (null if not needed) */
    jdoubleArray jac_arr = (jac != NULL) ? ctx->jac_array : NULL;
    
    /* Call Java method: void evaluate(double[] x, double[] c, double[] jac) */
    (*env)->CallVoidMethod(env, ctx->callback, ctx->method_id,
                           ctx->x_array, ctx->c_array, jac_arr);
    
    /* Check for exception */
    if (jni_check_exception(env)) {
        ctx->has_error = 1;
        return;
    }
    
    /* Copy constraint values back */
    (*env)->GetDoubleArrayRegion(env, ctx->c_array, 0, m, c);
    
    /* Copy Jacobian back if needed */
    if (jac != NULL) {
        (*env)->GetDoubleArrayRegion(env, ctx->jac_array, 0, m * n, jac);
    }
}

/* SLSQP-specific objective callback wrapper */
static double jni_slsqp_obj_callback(void* ctx_ptr, const double* x, double* g, int n) {
    JniSlsqpContext* slsqp_ctx = (JniSlsqpContext*)ctx_ptr;
    if (!slsqp_ctx) return NAN;
    return jni_objective_callback(&slsqp_ctx->obj_ctx, x, g, n);
}

/* SLSQP-specific equality constraint callback wrapper */
static void jni_slsqp_eq_callback(void* ctx_ptr, const double* x, double* c, double* jac, int m, int n) {
    JniSlsqpContext* slsqp_ctx = (JniSlsqpContext*)ctx_ptr;
    if (!slsqp_ctx) return;
    jni_constraint_callback(&slsqp_ctx->eq_ctx, x, c, jac, m, n);
}

/* SLSQP-specific inequality constraint callback wrapper */
static void jni_slsqp_ineq_callback(void* ctx_ptr, const double* x, double* c, double* jac, int m, int n) {
    JniSlsqpContext* slsqp_ctx = (JniSlsqpContext*)ctx_ptr;
    if (!slsqp_ctx) return;
    jni_constraint_callback(&slsqp_ctx->ineq_ctx, x, c, jac, m, n);
}

/* ============================================================================
 * L-BFGS-B JNI Methods
 * ============================================================================ */

JNIEXPORT jlong JNICALL Java_com_curioloop_LbfgsbWorkspace_nativeWorkspaceSize
    (JNIEnv* env, jclass cls, jint n, jint m) {
    (void)env;
    (void)cls;
    return (jlong)(lbfgsb_workspace_size(n, m) + sizeof(LbfgsbWorkspace));
}

JNIEXPORT void JNICALL Java_com_curioloop_LbfgsbWorkspace_nativeWorkspaceInit
    (JNIEnv* env, jclass cls, jlong wsPtr, jlong memPtr, jint n, jint m) {
    (void)env;
    (void)cls;
    LbfgsbWorkspace* ws = (LbfgsbWorkspace*)wsPtr;
    void* mem = (void*)memPtr;
    lbfgsb_workspace_init(ws, mem, n, m);
}

/* ============================================================================
 * SLSQP JNI Methods
 * ============================================================================ */

JNIEXPORT jlong JNICALL Java_com_curioloop_SlsqpWorkspace_nativeWorkspaceSize
    (JNIEnv* env, jclass cls, jint n, jint meq, jint mineq) {
    (void)env;
    (void)cls;
    return (jlong)(slsqp_workspace_size(n, meq, mineq) + sizeof(SlsqpWorkspace));
}

JNIEXPORT void JNICALL Java_com_curioloop_SlsqpWorkspace_nativeWorkspaceInit
    (JNIEnv* env, jclass cls, jlong wsPtr, jlong memPtr, jint n, jint meq, jint mineq) {
    (void)env;
    (void)cls;
    SlsqpWorkspace* ws = (SlsqpWorkspace*)wsPtr;
    void* mem = (void*)memPtr;
    slsqp_workspace_init(ws, mem, n, meq, mineq);
}

/* ============================================================================
 * L-BFGS-B Optimizer (always uses external workspace)
 * ============================================================================ */

JNIEXPORT jint JNICALL Java_com_curioloop_LbfgsbOptimizer_nativeOptimize
    (JNIEnv* env, jobject obj,
     /* Problem definition */
     jint n, jint m, jdoubleArray x,
     /* Callbacks */
     jobject objective, jdoubleArray gradient,
     /* Termination criteria */
     jdouble factr, jdouble pgtol, jint maxIter, jint maxEval,
     /* Workspace */
     jobject workspace,
     jint lowerOffset, jint upperOffset, jint boundTypeOffset, jint resultOffset) {
    
    (void)obj;
    
    /* Get workspace buffer address */
    char* ws_mem = (char*)(*env)->GetDirectBufferAddress(env, workspace);
    if (!ws_mem) {
        jni_throw_exception(env, "java/lang/IllegalArgumentException", 
                           "Workspace must be a direct ByteBuffer");
        return STATUS_INVALID_ARG;
    }
    
    /* Get pointers to auxiliary arrays within workspace buffer */
    double* lower_ptr = (double*)(ws_mem + lowerOffset);
    double* upper_ptr = (double*)(ws_mem + upperOffset);
    int* bound_ptr = (int*)(ws_mem + boundTypeOffset);
    double* result_ptr = (double*)(ws_mem + resultOffset);
    
    /* Get x array pointer */
    jdouble* x_ptr = (*env)->GetDoubleArrayElements(env, x, NULL);
    
    if (!x_ptr) {
        jni_throw_exception(env, "java/lang/OutOfMemoryError", "Failed to get array elements");
        return STATUS_INVALID_ARG;
    }
    
    /* Initialize callback context - use x directly for callbacks, gradient from workspace */
    JniObjectiveContext ctx;
    if (jni_init_objective_context(&ctx, env, objective, x, gradient) != 0) {
        jni_throw_exception(env, "java/lang/RuntimeException", "Failed to initialize callback");
        (*env)->ReleaseDoubleArrayElements(env, x, x_ptr, 0);
        return STATUS_INVALID_ARG;
    }
    
    /* Use external workspace - workspace structure at start, memory follows */
    LbfgsbWorkspace* ws = (LbfgsbWorkspace*)ws_mem;
    void* mem = ws_mem + sizeof(LbfgsbWorkspace);
    lbfgsb_workspace_init(ws, mem, n, m);
    
    /* Set up configuration */
    LbfgsbConfig config;
    config.n = n;
    config.m = m;
    config.x = x_ptr;
    config.lower = lower_ptr;
    config.upper = upper_ptr;
    config.bound_type = bound_ptr;
    config.factr = factr;
    config.pgtol = pgtol;
    config.max_iter = maxIter;
    config.max_eval = maxEval;
    config.callback_ctx = &ctx;
    config.eval = jni_objective_callback;
    
    /* Run optimization */
    LbfgsbResult opt_result;
    OptStatus status = lbfgsb_optimize(&config, ws, &opt_result);
    
    /* Store results in workspace buffer */
    result_ptr[0] = opt_result.f;
    result_ptr[1] = (double)opt_result.iterations;
    result_ptr[2] = (double)opt_result.evaluations;
    result_ptr[3] = (double)status;
    
    /* Get error flag before cleanup */
    int has_error = ctx.has_error;
    jni_cleanup_objective_context(&ctx);
    
    /* Release x array */
    (*env)->ReleaseDoubleArrayElements(env, x, x_ptr, 0);
    
    /* Check for callback error */
    if (has_error) {
        return STATUS_CALLBACK_ERROR;
    }
    
    return status;
}

/* ============================================================================
 * SLSQP Optimizer (always uses external workspace)
 * ============================================================================ */

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
     jdouble accuracy, jint maxIter,
     jboolean exactLineSearch, jdouble fEvalTol, jdouble fDiffTol, jdouble xDiffTol,
     /* Workspace */
     jobject workspace,
     jint lowerOffset, jint upperOffset, jint resultOffset) {
    
    (void)obj;
    
    /* Get workspace buffer address */
    char* ws_mem = (char*)(*env)->GetDirectBufferAddress(env, workspace);
    if (!ws_mem) {
        jni_throw_exception(env, "java/lang/IllegalArgumentException", 
                           "Workspace must be a direct ByteBuffer");
        return STATUS_INVALID_ARG;
    }
    
    /* Get pointers to auxiliary arrays within workspace buffer */
    double* lower_ptr = (double*)(ws_mem + lowerOffset);
    double* upper_ptr = (double*)(ws_mem + upperOffset);
    double* result_ptr = (double*)(ws_mem + resultOffset);
    
    /* Get x array pointer */
    jdouble* x_ptr = (*env)->GetDoubleArrayElements(env, x, NULL);
    
    if (!x_ptr) {
        jni_throw_exception(env, "java/lang/OutOfMemoryError", "Failed to get array elements");
        return STATUS_INVALID_ARG;
    }
    
    /* Initialize callback contexts - use x directly for callbacks, shared arrays for constraints */
    JniSlsqpContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    
    if (jni_init_objective_context(&ctx.obj_ctx, env, objective, x, gradient) != 0) {
        jni_throw_exception(env, "java/lang/RuntimeException", "Failed to initialize objective callback");
        (*env)->ReleaseDoubleArrayElements(env, x, x_ptr, 0);
        return STATUS_INVALID_ARG;
    }
    
    /* Equality and inequality constraints share the same arrays since they are evaluated sequentially */
    if (eqConstraint && meq > 0 && jni_init_constraint_context(&ctx.eq_ctx, env, eqConstraint, 
            x, constraintValues, constraintJacobian, meq, n) != 0) {
        jni_cleanup_objective_context(&ctx.obj_ctx);
        jni_throw_exception(env, "java/lang/RuntimeException", "Failed to initialize equality constraint callback");
        (*env)->ReleaseDoubleArrayElements(env, x, x_ptr, 0);
        return STATUS_INVALID_ARG;
    }
    
    if (ineqConstraint && mineq > 0 && jni_init_constraint_context(&ctx.ineq_ctx, env, ineqConstraint,
            x, constraintValues, constraintJacobian, mineq, n) != 0) {
        jni_cleanup_objective_context(&ctx.obj_ctx);
        jni_cleanup_constraint_context(&ctx.eq_ctx);
        jni_throw_exception(env, "java/lang/RuntimeException", "Failed to initialize inequality constraint callback");
        (*env)->ReleaseDoubleArrayElements(env, x, x_ptr, 0);
        return STATUS_INVALID_ARG;
    }
    
    /* Use external workspace - workspace structure at start, memory follows */
    SlsqpWorkspace* ws = (SlsqpWorkspace*)ws_mem;
    void* mem = ws_mem + sizeof(SlsqpWorkspace);
    slsqp_workspace_init(ws, mem, n, meq, mineq);
    
    /* Set up configuration */
    SlsqpConfig config;
    config.n = n;
    config.meq = meq;
    config.mineq = mineq;
    config.x = x_ptr;
    config.lower = lower_ptr;
    config.upper = upper_ptr;
    config.accuracy = accuracy;
    config.max_iter = maxIter;
    config.exact_search = exactLineSearch ? 1 : 0;
    
    /* Extended termination criteria from Java parameters */
    config.f_eval_tol = fEvalTol;
    config.f_diff_tol = fDiffTol;
    config.x_diff_tol = xDiffTol;
    
    config.callback_ctx = &ctx;
    config.obj_eval = jni_slsqp_obj_callback;
    config.eq_eval = (eqConstraint && meq > 0) ? jni_slsqp_eq_callback : NULL;
    config.ineq_eval = (ineqConstraint && mineq > 0) ? jni_slsqp_ineq_callback : NULL;
    
    /* Run optimization */
    SlsqpResult opt_result;
    OptStatus status = slsqp_optimize(&config, ws, &opt_result);
    
    /* Store results in workspace buffer */
    result_ptr[0] = opt_result.f;
    result_ptr[1] = (double)opt_result.iterations;
    result_ptr[2] = (double)status;
    
    /* Get error flag before cleanup */
    int has_error = ctx.obj_ctx.has_error || ctx.eq_ctx.has_error || ctx.ineq_ctx.has_error;
    jni_cleanup_objective_context(&ctx.obj_ctx);
    jni_cleanup_constraint_context(&ctx.eq_ctx);
    jni_cleanup_constraint_context(&ctx.ineq_ctx);
    
    /* Release x array */
    (*env)->ReleaseDoubleArrayElements(env, x, x_ptr, 0);
    
    /* Check for callback error */
    if (has_error) {
        return STATUS_CALLBACK_ERROR;
    }
    
    return status;
}
