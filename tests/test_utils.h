#pragma once

#include <gtest/gtest.h>
#include "msk/core/matrix.h"

namespace testing {
    inline void ExpectMatrixNear(const msk::Matrix &actual, const msk::Matrix &expected, double tol = 1e-12) {
        ASSERT_EQ(actual.rows(), expected.rows());
        ASSERT_EQ(actual.cols(), expected.cols());
        for (int i = 0; i < actual.rows(); i++) {
            for (int j = 0; j < actual.cols(); j++) {
                EXPECT_NEAR(actual(i, j), expected(i, j), tol) << "en (" << i << ", " << j << ")";
            }
        }
    }
}
