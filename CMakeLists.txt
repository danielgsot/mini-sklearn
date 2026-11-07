cmake_minimum_required(VERSION 3.20)
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
    )
    target_link_libraries(msk_tests PRIVATE msk GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(msk_tests)
endif()
