cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0135 NEW)
project(mini_sklearn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------- Libreria ----------
add_library(msk STATIC
        include/msk/core/matrix.h
        src/core/matrix.cpp
        include/msk/metrics/regression.h
        src/metrics/regression.cpp
        include/msk/preprocessing/standard_scaler.h
        src/preprocessing/standard_scaler.cpp
        include/msk/preprocessing/min_max_scaler.h
        src/preprocessing/min_max_scaler.cpp
        include/msk/core/base.h
        include/msk/linear_model/linear_regression.h
        src/linear_model/linear_regression.cpp
)
target_include_directories(msk PUBLIC include)
if(NOT MSVC)
    target_compile_options(msk PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------- Ejemplos ----------
# add_executable(example_linreg examples/linreg.cpp)
# target_link_libraries(example_linreg PRIVATE msk)


# ---------- Tests (GoogleTest) ----------
option(MSK_BUILD_TESTS "Compilar tests" ON)
if(MSK_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(msk_tests
            tests/test_matrix.cpp
            tests/test_matrix.cpp
            tests/test_metrics.cpp
            tests/test_utils.h
            tests/test_standard_scaler.cpp
            tests/test_min_max_scaler.cpp
            tests/test_transformer.cpp
            tests/test_linear_regression.cpp
    )
    target_link_libraries(msk_tests PRIVATE msk GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(msk_tests)
endif()
