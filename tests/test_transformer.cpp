#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "test_utils.h"
#include "msk/core/matrix.h"
#include "msk/core/base.h"
#include "msk/preprocessing/standard_scaler.h"
#include "msk/preprocessing/min_max_scaler.h"

using msk::Matrix;
using msk::Transformer;
using msk::StandardScaler;
using msk::MinMaxScaler;
using testing::ExpectMatrixNear;

namespace {
    //Polynomial testing function
    Matrix apply(Transformer& t, const Matrix& X) {
        return t.fit_transform(X);
    }
}

TEST(TransformerPolymorphism, FitTransformResolvesCorrectly) {
    const Matrix X(3, 1, {1.0, 2.0, 3.0});

    StandardScaler std_scaler;
    Matrix X_std = apply(std_scaler, X);

    const Matrix expected_std(3, 1, {
        -1.224744871391589,
         0.0,
         1.224744871391589
    });
    ExpectMatrixNear(X_std, expected_std);

    MinMaxScaler mm_scaler;
    Matrix X_mm = apply(mm_scaler, X);

    const Matrix expected_mm(3, 1, {
        0.0,
        0.5,
        1.0
    });
    ExpectMatrixNear(X_mm, expected_mm);
}

TEST(TransformerPolymorphism, PipelineExecutionAndVirtualDestruction) {
    const Matrix X(3, 1, {1.0, 2.0, 3.0});

    std::vector<std::unique_ptr<Transformer>> pipeline;
    pipeline.push_back(std::make_unique<StandardScaler>());
    pipeline.push_back(std::make_unique<MinMaxScaler>());

    Matrix X_transformed = X;

    for (const auto& step : pipeline) {
        X_transformed = step->fit_transform(X_transformed);
    }

    const Matrix expected(3, 1, {
        0.0,
        0.5,
        1.0
    });
    ExpectMatrixNear(X_transformed, expected);
}