#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#include "test_utils.h"
#include "msk/core/matrix.h"
#include "msk/preprocessing/min_max_scaler.h"

using msk::Matrix;
using msk::MinMaxScaler;
using testing::ExpectMatrixNear;

TEST(MinMaxScaler, FitTransformSingleColumn) {
    const Matrix X(4, 1, {1.0, 2.0, 3.0, 4.0});
    MinMaxScaler scaler;

    Matrix X_transformed = scaler.fit_transform(X);

    ASSERT_EQ(scaler.min().size(), 1u);
    ASSERT_EQ(scaler.range().size(), 1u);

    EXPECT_DOUBLE_EQ(scaler.min()[0], 1.0);
    EXPECT_DOUBLE_EQ(scaler.range()[0], 3.0);

    const Matrix expected(4, 1, {
        0.0,
        1.0 / 3.0,
        2.0 / 3.0,
        1.0
    });

    ExpectMatrixNear(X_transformed, expected);
}

TEST(MinMaxScaler, MultiColumnDifferentScales) {
    const Matrix X(3, 2, {
        10.0, -1.0,
        30.0,  0.0,
        50.0,  1.0
    });

    MinMaxScaler scaler;
    Matrix X_transformed = scaler.fit_transform(X);

    EXPECT_DOUBLE_EQ(scaler.min()[0], 10.0);
    EXPECT_DOUBLE_EQ(scaler.range()[0], 40.0);

    EXPECT_DOUBLE_EQ(scaler.min()[1], -1.0);
    EXPECT_DOUBLE_EQ(scaler.range()[1], 2.0);

    const Matrix expected(3, 2, {
        0.0, 0.0,
        0.5, 0.5,
        1.0, 1.0
    });

    ExpectMatrixNear(X_transformed, expected);
}

TEST(MinMaxScaler, TransformWithoutLeakage) {
    const Matrix X_train(3, 1, {1.0, 2.0, 3.0});
    MinMaxScaler scaler;
    scaler.fit(X_train);

    const Matrix X_test(2, 1, {0.0, 4.0});
    Matrix X_test_transformed = scaler.transform(X_test);

    const Matrix expected(2, 1, {
        -0.5,
         1.5
    });

    ExpectMatrixNear(X_test_transformed, expected);
}

TEST(MinMaxScaler, ConstantColumnUsesOneForScale) {
    const Matrix X_train(3, 1, {5.0, 5.0, 5.0});
    MinMaxScaler scaler;
    scaler.fit(X_train);

    EXPECT_DOUBLE_EQ(scaler.min()[0], 5.0);
    EXPECT_DOUBLE_EQ(scaler.range()[0], 1.0);

    const Matrix X_test(3, 1, {5.0, 7.0, 3.0});
    Matrix X_test_transformed = scaler.transform(X_test);

    const Matrix expected(3, 1, {
         0.0,
         2.0,
        -2.0
    });

    ExpectMatrixNear(X_test_transformed, expected);
}

TEST(MinMaxScaler, ConstantAndNormalColumns) {
    const Matrix X(3, 2, {
        5.0, 10.0,
        5.0, 20.0,
        5.0, 30.0
    });
    MinMaxScaler scaler;
    Matrix X_transformed = scaler.fit_transform(X);

    EXPECT_DOUBLE_EQ(scaler.range()[0], 1.0);
    EXPECT_DOUBLE_EQ(scaler.range()[1], 20.0);

    const Matrix expected(3, 2, {
        0.0, 0.0,
        0.0, 0.5,
        0.0, 1.0
    });

    ExpectMatrixNear(X_transformed, expected);
}

TEST(MinMaxScaler, FitEmptyMatrixThrows) {
    MinMaxScaler scaler;

    EXPECT_THROW(scaler.fit(Matrix(0, 0, {})), std::invalid_argument);
    EXPECT_THROW(scaler.fit(Matrix(3, 0, {})), std::invalid_argument);
    EXPECT_THROW(scaler.fit(Matrix(0, 3, {})), std::invalid_argument);
}

TEST(MinMaxScaler, TransformBeforeFitThrows) {
    const Matrix X(2, 1, {1.0, 2.0});
    MinMaxScaler scaler;

    EXPECT_THROW(scaler.transform(X), std::logic_error);
}

TEST(MinMaxScaler, TransformWithWrongColumnCountThrows) {
    const Matrix X_fit(2, 2, {1.0, 2.0, 3.0, 4.0});
    MinMaxScaler scaler;
    scaler.fit(X_fit);

    const Matrix X_transformed(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    EXPECT_THROW(scaler.transform(X_transformed), std::invalid_argument);
}

TEST(MinMaxScaler, FitTwiceUpdatesStateCorrectly) {
    MinMaxScaler scaler;

    const Matrix X1(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });
    scaler.fit(X1);

    EXPECT_EQ(scaler.min().size(), 2u);
    EXPECT_EQ(scaler.range().size(), 2u);

    const Matrix X2(2, 3, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    });
    scaler.fit(X2);

    EXPECT_EQ(scaler.min().size(), 3u);
    EXPECT_EQ(scaler.range().size(), 3u);

    EXPECT_DOUBLE_EQ(scaler.range()[1], 3.0);
}