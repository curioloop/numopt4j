cmake_minimum_required(VERSION 3.10)
project(optimizer C)

# C99 standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find JNI
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src/main/native/include)

# Source files
set(SOURCES
    src/main/native/src/blas.c
    src/main/native/src/utils.c
    src/main/native/src/jni_bridge.c
    # L-BFGS-B algorithm
    src/main/native/src/lbfgsb/linpack.c
    src/main/native/src/lbfgsb/cauchy.c
    src/main/native/src/lbfgsb/update.c
    src/main/native/src/lbfgsb/subspace.c
    src/main/native/src/lbfgsb/project.c
    src/main/native/src/lbfgsb/linesearch.c
    src/main/native/src/lbfgsb/minpack.c
    src/main/native/src/lbfgsb/lbfgsb.c
    # SLSQP algorithm
    src/main/native/src/slsqp/householder.c
    src/main/native/src/slsqp/hfti.c
    src/main/native/src/slsqp/nnls.c
    src/main/native/src/slsqp/ldp.c
    src/main/native/src/slsqp/lsei.c
    src/main/native/src/slsqp/lsq.c
    src/main/native/src/slsqp/slsqp.c
)

# Create shared library
add_library(optimizer SHARED ${SOURCES})

# Platform-specific settings
if(WIN32)
    set_target_properties(optimizer PROPERTIES
        PREFIX ""
        SUFFIX ".dll"
        OUTPUT_NAME "optimizer"
    )
    target_compile_definitions(optimizer PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif(APPLE)
    set_target_properties(optimizer PROPERTIES
        PREFIX "lib"
        SUFFIX ".dylib"
        OUTPUT_NAME "optimizer"
    )
    # Enable symbol visibility
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
else()
    # Linux
    set_target_properties(optimizer PROPERTIES
        PREFIX "lib"
        SUFFIX ".so"
        OUTPUT_NAME "optimizer"
    )
    # Enable symbol visibility
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
    # Link math library
    target_link_libraries(optimizer m)
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(optimizer PRIVATE /W4)
else()
    target_compile_options(optimizer PRIVATE -Wall -Wextra -pedantic)
endif()

# Output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Install target
install(TARGETS optimizer
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
