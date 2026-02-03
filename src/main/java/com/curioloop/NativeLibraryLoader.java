/*
 * Copyright (c) 2025 curioloop. All rights reserved.
 */
package com.curioloop;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Utility class for loading native libraries.
 * <p>
 * This class handles platform detection and automatic extraction of native
 * libraries from JAR resources.
 * </p>
 */
public final class NativeLibraryLoader {
    
    private static final String LIBRARY_NAME = "optimizer";
    private static volatile boolean loaded = false;
    private static final Object lock = new Object();
    
    private NativeLibraryLoader() {}
    
    /**
     * Loads the native library.
     * <p>
     * This method is thread-safe and will only load the library once.
     * </p>
     *
     * @throws OptimizationException if the library cannot be loaded
     */
    public static void load() {
        if (loaded) return;
        
        synchronized (lock) {
            if (loaded) return;
            
            // Check for custom library path
            String customPath = System.getProperty("optimizer.native.path");
            if (customPath != null && !customPath.isEmpty()) {
                try {
                    System.load(customPath);
                    loaded = true;
                    return;
                } catch (UnsatisfiedLinkError e) {
                    throw new OptimizationException(
                        "Failed to load native library from custom path: " + customPath, e);
                }
            }
            
            // Try to load from system path first
            try {
                System.loadLibrary(LIBRARY_NAME);
                loaded = true;
                return;
            } catch (UnsatisfiedLinkError e) {
                // Fall through to extract from JAR
            }
            
            // Extract and load from JAR
            try {
                Path tempLib = extractLibrary();
                System.load(tempLib.toString());
                loaded = true;
            } catch (IOException e) {
                throw new OptimizationException("Failed to extract native library", e);
            } catch (UnsatisfiedLinkError e) {
                throw new OptimizationException("Failed to load native library", e);
            }
        }
    }
    
    /**
     * Checks if the native library is loaded.
     * @return true if loaded
     */
    public static boolean isLoaded() {
        return loaded;
    }
    
    /**
     * Detects the current operating system.
     * @return OS name (windows, linux, darwin)
     */
    static String detectOS() {
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("win")) {
            return "windows";
        } else if (os.contains("mac") || os.contains("darwin")) {
            return "darwin";
        } else if (os.contains("linux")) {
            return "linux";
        } else {
            throw new OptimizationException("Unsupported operating system: " + os);
        }
    }
    
    /**
     * Detects the current CPU architecture.
     * @return Architecture name (x86_64, aarch64)
     */
    static String detectArch() {
        String arch = System.getProperty("os.arch").toLowerCase();
        if (arch.contains("amd64") || arch.contains("x86_64")) {
            return "x86_64";
        } else if (arch.contains("aarch64") || arch.contains("arm64")) {
            return "aarch64";
        } else {
            throw new OptimizationException("Unsupported architecture: " + arch);
        }
    }
    
    /**
     * Gets the library file name for the current platform.
     * @param os Operating system
     * @return Library file name
     */
    static String getLibraryName(String os) {
        switch (os) {
            case "windows":
                return LIBRARY_NAME + ".dll";
            case "darwin":
                return "lib" + LIBRARY_NAME + ".dylib";
            case "linux":
            default:
                return "lib" + LIBRARY_NAME + ".so";
        }
    }
    
    /**
     * Extracts the native library from JAR to a temporary file.
     * @return Path to extracted library
     * @throws IOException if extraction fails
     */
    private static Path extractLibrary() throws IOException {
        String os = detectOS();
        String arch = detectArch();
        String libName = getLibraryName(os);
        String resourcePath = "/native/" + os + "-" + arch + "/" + libName;
        
        try (InputStream in = NativeLibraryLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IOException("Native library not found in JAR: " + resourcePath);
            }
            
            // Create temp file with appropriate extension
            String suffix = libName.substring(libName.lastIndexOf('.'));
            Path tempFile = Files.createTempFile("optimizer_", suffix);
            tempFile.toFile().deleteOnExit();
            
            Files.copy(in, tempFile, StandardCopyOption.REPLACE_EXISTING);
            return tempFile;
        }
    }
}
