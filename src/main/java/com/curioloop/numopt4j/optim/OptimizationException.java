package com.curioloop.numopt4j.optim;

/**
 * Unified exception for the numopt4j optimization library.
 *
 * <p>Each instance carries a structured {@code errorCode} and a message that includes
 * a documentation URL pointing to a detailed fix guide.</p>
 *
 * <p>The documentation base URL is resolved in the following priority order:</p>
 * <ol>
 *   <li>System property {@code numopt4j.doc.baseUrl}</li>
 *   <li>Environment variable {@code NUMOPT4J_DOC_BASE_URL}</li>
 *   <li>Default: {@code https://github.com/curioloop/numopt4j/wiki/}</li>
 * </ol>
 *
 * @see OptimizationResult
 * @see OptimizationStatus
 */
public class OptimizationException extends RuntimeException {

    private static final String DEFAULT_BASE_URL =
            "https://github.com/curioloop/numopt4j/wiki/";

    private final String errorCode;

    public OptimizationException(String errorCode, String message) {
        super(formatMessage(errorCode, message));
        this.errorCode = errorCode;
    }

    public OptimizationException(String errorCode, String message, Throwable cause) {
        super(formatMessage(errorCode, message), cause);
        this.errorCode = errorCode;
    }

    public String getErrorCode() {
        return errorCode;
    }

    private static String formatMessage(String errorCode, String message) {
        String baseUrl = System.getProperty("numopt4j.doc.baseUrl");
        if (baseUrl == null) {
            baseUrl = System.getenv("NUMOPT4J_DOC_BASE_URL");
        }
        if (baseUrl == null) {
            baseUrl = DEFAULT_BASE_URL;
        }
        return "[" + errorCode + "] " + message + " See: " + baseUrl + errorCode;
    }
}
