#include <gtest/gtest.h>

#include <stdexcept>

#include "msk/core/matrix.h"

using msk::Matrix;

namespace {
    void ExpectMatrixNear(const Matrix &actual, const Matrix &expected, double tol = 1e-12) {
        ASSERT_EQ(actual.rows(), expected.rows());
        ASSERT_EQ(actual.cols(), expected.cols());
        for (int i = 0; i < actual.rows(); i++)
            for (int j = 0; j < actual.cols(); j++)
                EXPECT_NEAR(actual(i, j), expected(i, j), tol) << "en (" << i << ", " << j << ")";
    }
}


TEST(Matrix, BuildWithZeros) {
    Matrix m(2, 3);
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 3);
    for (int i = 0; i < m.rows(); i++)
        for (int j = 0; j < m.cols(); j++)
            EXPECT_DOUBLE_EQ(m(i, j), 0.0);
}

TEST(Matrix, WriteAndRead) {
    Matrix m(2, 3);
    m(1, 2) = 7.5;
    EXPECT_DOUBLE_EQ(m(1, 2), 7.5);
    EXPECT_DOUBLE_EQ(m(0, 0), 0.0);  // el resto no se tocó
}

TEST(Matrix, RowMajorLayout) {
    const Matrix m(2, 3, {1, 2, 3, 4, 5, 6});
    EXPECT_DOUBLE_EQ(m(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 4.0);
}

TEST(Matrix, BuildWithWrongDataSizeThrows) {
    EXPECT_THROW(Matrix(2, 3, {1, 2, 3, 4}), std::invalid_argument);
}

TEST(Matrix, NegativeDimensionsThrow) {
    EXPECT_THROW(Matrix(-1, 3), std::invalid_argument);
    EXPECT_THROW(Matrix(3, -1), std::invalid_argument);
    EXPECT_THROW(Matrix(-1, -2, {1, 2}), std::invalid_argument);
}


TEST(Matrix, TransposeNonSquare) {
    const Matrix m(3, 2, {1, 2, 3, 4, 5, 6});
    const Matrix t = m.transpose();

    ASSERT_EQ(t.rows(), 2);
    ASSERT_EQ(t.cols(), 3);
    for (int i = 0; i < m.rows(); i++)
        for (int j = 0; j < m.cols(); j++)
            EXPECT_DOUBLE_EQ(t(j, i), m(i, j));
}

TEST(Matrix, TransposeTwiceGivesOriginal) {
    const Matrix m(2, 3, {1, 2, 3, 4, 5, 6});
    ExpectMatrixNear(m.transpose().transpose(), m);
}


TEST(Matrix, AdditionAndSubtraction) {
    const Matrix m1(2, 2, {7, 5, 9, 8});
    const Matrix m2(2, 2, {2, 1, 1, 4});

    ExpectMatrixNear(m1 + m2, Matrix(2, 2, {9, 6, 10, 12}));
    ExpectMatrixNear(m1 - m2, Matrix(2, 2, {5, 4, 8, 4}));
}

TEST(Matrix, OperationsDoNotModifyOperands) {
    const Matrix m1(2, 2, {1, 2, 3, 4});
    const Matrix m2(2, 2, {5, 6, 7, 8});
    Matrix a = m1;
    Matrix b = m2;
    (void) (a + b);
    (void) (a - b);
    (void) (a * b);
    ExpectMatrixNear(a, m1);
    ExpectMatrixNear(b, m2);
}


TEST(Matrix, MultiplySquareByVector) {
    const Matrix m1(2, 2, {7, 5, 4, 2});
    const Matrix m2(2, 1, {1, 3});

    // 7*1 + 5*3 = 22 ; 4*1 + 2*3 = 10
    ExpectMatrixNear(m1 * m2, Matrix(2, 1, {22, 10}));
}

TEST(Matrix, MultiplyRectangular) {
    const Matrix a(2, 3, {7, 5, 9,
                          8, 2, 5});
    const Matrix b(3, 4, {1, 0, 2, 3,
                          4, 5, 7, 1,
                          5, 1, 3, 3});

    const Matrix expected(2, 4, {72, 34, 76, 53,
                                 41, 15, 45, 41});
    ExpectMatrixNear(a * b, expected);
}

TEST(Matrix, MultiplyByIdentity) {
    const Matrix m(2, 2, {7, 5, 9, 8});
    const Matrix identity(2, 2, {1, 0, 0, 1});

    ExpectMatrixNear(m * identity, m);
    ExpectMatrixNear(identity * m, m);
}

TEST(Matrix, MultiplicationIsNotCommutative) {
    const Matrix a(2, 2, {1, 2, 3, 4});
    const Matrix b(2, 2, {0, 1, 1, 0});

    ExpectMatrixNear(a * b, Matrix(2, 2, {2, 1, 4, 3}));
    ExpectMatrixNear(b * a, Matrix(2, 2, {3, 4, 1, 2}));
}


TEST(Matrix, ScalarMultiplication) {
    const Matrix m(2, 2, {2.0, -3.5, 4.0, 0.0});

    ExpectMatrixNear(m * 2.0, Matrix(2, 2, {4.0, -7.0, 8.0, 0.0}));
}


TEST(Matrix, IncompatibleDimensionsThrow) {
    const Matrix m2x2(2, 2, {1, 2, 3, 4});
    const Matrix m3x4(3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    EXPECT_THROW(m2x2 + m3x4, std::invalid_argument);
    EXPECT_THROW(m2x2 - m3x4, std::invalid_argument);
    EXPECT_THROW(m3x4 * m2x2, std::invalid_argument);  // 4 columnas contra 2 filas
    EXPECT_THROW(m2x2 * m3x4, std::invalid_argument);  // 2 columnas contra 3 filas
}