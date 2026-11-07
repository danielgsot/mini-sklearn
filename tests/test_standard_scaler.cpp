#include <gtest/gtest.h>
#include <stdexcept>
#include <cmath>
#include <vector>

#include "test_utils.h"
#include "msk/core/matrix.h"
#include "msk/preprocessing/standard_scaler.h"

using msk::Matrix;
using msk::StandardScaler;
using testing::ExpectMatrixNear;

TEST(StandardScaler, FitTransformSingleColumn) {
    const Matrix X(4, 1, {1.0, 2.0, 3.0, 4.0});
    StandardScaler scaler;

    Matrix X_transformed = scaler.fit_transform(X);

    ASSERT_EQ(scaler.means().size(), 1u);
    ASSERT_EQ(scaler.stds().size(), 1u);

    EXPECT_DOUBLE_EQ(scaler.means()[0], 2.5);
    EXPECT_NEAR(scaler.stds()[0], 1.118033988749895, 1e-9);

    const Matrix expected(4, 1, {
        -1.3416407865,
        -0.4472135955,
         0.4472135955,
         1.3416407865
    });

    ExpectMatrixNear(X_transformed, expected);
}

TEST(StandardScaler, FitEmptyMatrixThrows) {
    StandardScaler scaler;

    EXPECT_THROW(scaler.fit(Matrix(0, 0)), std::invalid_argument);
    EXPECT_THROW(scaler.fit(Matrix(3, 0)), std::invalid_argument);
    EXPECT_THROW(scaler.fit(Matrix(0, 3)), std::invalid_argument);
}

TEST(StandardScaler, TransformBeforeFitThrows) {
    const Matrix X(2, 1, {1.0, 2.0});
    StandardScaler scaler;

    EXPECT_THROW(scaler.transform(X), std::logic_error);
}

TEST(StandardScaler, TransformWithWrongColumnCountThrows) {
    const Matrix X_fit(2, 2, {1.0, 2.0, 3.0, 4.0});
    StandardScaler scaler;
    scaler.fit(X_fit);

    const Matrix X_transformed(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    EXPECT_THROW(scaler.transform(X_transformed), std::invalid_argument);
}

TEST(StandardScaler, ConstantColumnUsesOneForScale) {
    const Matrix X(3, 1, {5.0, 5.0, 5.0});
    StandardScaler scaler;

    Matrix X_transformed = scaler.fit_transform(X);

    EXPECT_DOUBLE_EQ(scaler.means()[0], 5.0);
    EXPECT_DOUBLE_EQ(scaler.stds()[0], 1.0);

    const Matrix expected(3, 1, {0.0, 0.0, 0.0});
    ExpectMatrixNear(X_transformed, expected);
}

TEST(StandardScaler, FloatPrecisionConstantColumn) {
    const Matrix X(3, 1, {0.1, 0.1, 0.1});
    StandardScaler scaler;

    Matrix X_transformed = scaler.fit_transform(X);

    EXPECT_DOUBLE_EQ(scaler.means()[0], 0.1);
    EXPECT_DOUBLE_EQ(scaler.stds()[0], 1.0);

    const Matrix expected(3, 1, {0.0, 0.0, 0.0});
    ExpectMatrixNear(X_transformed, expected);
}

TEST(StandardScaler, FloatPrecisionLargeConstantColumn) {
    const Matrix X(3, 1, {123456.789, 123456.789, 123456.789});
    StandardScaler scaler;

    Matrix X_transformed = scaler.fit_transform(X);

    EXPECT_DOUBLE_EQ(scaler.stds()[0], 1.0);

    const Matrix expected(3, 1, {0.0, 0.0, 0.0});
    ExpectMatrixNear(X_transformed, expected);
}

TEST(StandardScaler, TransformWithoutLeakage) {
    const Matrix X_train(3, 1, {1.0, 2.0, 3.0});
    StandardScaler scaler;
    scaler.fit(X_train);

    const Matrix X_test(2, 1, {0.0, 4.0});
    Matrix X_test_transformed = scaler.transform(X_test);

    const Matrix expected(2, 1, {
        -2.449489742783178,
         2.449489742783178
    });
    ExpectMatrixNear(X_test_transformed, expected);
}

TEST(StandardScaler, ConstantAndNormalColumns) {
    const Matrix X(4, 2, {
        5.0, 1.0,
        5.0, 2.0,
        5.0, 3.0,
        5.0, 4.0
    });
    StandardScaler scaler;
    Matrix X_trans = scaler.fit_transform(X);

    EXPECT_DOUBLE_EQ(scaler.stds()[0], 1.0);
    EXPECT_NEAR(scaler.stds()[1], 1.118033988749895, 1e-9);

    const Matrix expected(4, 2, {
        0.0, -1.3416407865,
        0.0, -0.4472135955,
        0.0,  0.4472135955,
        0.0,  1.3416407865
    });
    ExpectMatrixNear(X_trans, expected);
}

TEST(StandardScaler, MultiColumnDifferentScales) {
    const Matrix X(4, 2, {
        1.0, 1000.0,
        2.0, 1010.0,
        3.0,  990.0,
        4.0, 1005.0
    });

    StandardScaler scaler;
    Matrix X_trans = scaler.fit_transform(X);

    for (int j = 0; j < X_trans.cols(); j++) {
        std::vector<double> col = X_trans.column(j);
        double sum = 0.0;
        for (double val : col) sum += val;
        double mean = sum / col.size();

        double sq_sum = 0.0;
        for (double val : col) sq_sum += (val - mean) * (val - mean);
        double std_dev = std::sqrt(sq_sum / col.size());

        EXPECT_NEAR(mean, 0.0, 1e-12);
        EXPECT_NEAR(std_dev, 1.0, 1e-12);
    }
}

TEST(StandardScaler, FloatPrecisionLargeRowCountConstant) {
    std::vector<double> data(100000, 0.1);
    const Matrix X(100000, 1, data);
    StandardScaler scaler;

    scaler.fit(X);

    EXPECT_DOUBLE_EQ(scaler.stds()[0], 1.0);
}

TEST(StandardScaler, SmallScaleFeaturesNotTreatedAsConstant) {
    const Matrix X(3, 1, {1e-20, 2e-20, 3e-20});
    StandardScaler scaler;

    scaler.fit(X);

    EXPECT_TRUE(scaler.stds()[0] < 1.0);
    EXPECT_TRUE(scaler.stds()[0] > 0.0);
}

TEST(StandardScaler, FitTwiceUpdatesStateCorrectly) {
    StandardScaler scaler;

    const Matrix X1(2, 2, {
        1.0, 2.0,
        3.0, 4.0
    });
    scaler.fit(X1);

    EXPECT_EQ(scaler.means().size(), 2u);
    EXPECT_EQ(scaler.stds().size(), 2u);

    const Matrix X2(2, 3, {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    });
    scaler.fit(X2);

    EXPECT_EQ(scaler.means().size(), 3u);
    EXPECT_EQ(scaler.stds().size(), 3u);
}