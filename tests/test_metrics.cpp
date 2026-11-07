#include <msk/metrics/regression.h>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace msk;

TEST(Metrics, mean_squared_error_basic) {
    double mse = mean_squared_error({3, 5, 7}, {2, 5, 9});
    EXPECT_DOUBLE_EQ(mse, 5.0 / 3);
}

TEST(Metrics, r2_score_basic) {
    double r2 = r2_score({3, 5, 7}, {2, 5, 9});
    EXPECT_DOUBLE_EQ(r2, 1 - 5.0 / 8);
}

TEST(Metrics, perfect_prediction) {
    std::vector<double> y_true = {1.0, 2.0, 3.0};
    std::vector<double> y_pred = {1.0, 2.0, 3.0};

    EXPECT_DOUBLE_EQ(mean_squared_error(y_true, y_pred), 0.0);
    EXPECT_DOUBLE_EQ(r2_score(y_true, y_pred), 1.0);
}

TEST(Metrics, predict_always_mean) {
    std::vector<double> y_true = {2.0, 4.0, 6.0};
    std::vector<double> y_pred = {4.0, 4.0, 4.0};

    EXPECT_DOUBLE_EQ(r2_score(y_true, y_pred), 0.0);
}

TEST(Metrics, worse_than_mean) {
    std::vector<double> y_true = {1.0, 2.0, 3.0};
    std::vector<double> y_pred = {10.0, 20.0, 30.0};

    EXPECT_LT(r2_score(y_true, y_pred), 0.0);
}

TEST(Metrics, zero_variance_perfect_prediction) {
    std::vector<double> y_true = {5.0, 5.0, 5.0};
    std::vector<double> y_pred = {5.0, 5.0, 5.0};

    EXPECT_DOUBLE_EQ(r2_score(y_true, y_pred), 1.0);
}

TEST(Metrics, zero_variance_imperfect_prediction) {
    std::vector<double> y_true = {5.0, 5.0, 5.0};
    std::vector<double> y_pred = {5.0, 5.0, 6.0};

    EXPECT_DOUBLE_EQ(r2_score(y_true, y_pred), 0.0);
}

TEST(Metrics, different_sizes_throw) {
    std::vector<double> y_true = {1.0, 2.0, 3.0};
    std::vector<double> y_pred = {1.0, 2.0};

    EXPECT_THROW(mean_squared_error(y_true, y_pred), std::invalid_argument);
    EXPECT_THROW(r2_score(y_true, y_pred), std::invalid_argument);
}

TEST(Metrics, empty_vectors_throw) {
    std::vector<double> y_true = {};
    std::vector<double> y_pred = {};

    EXPECT_THROW(mean_squared_error(y_true, y_pred), std::invalid_argument);
    EXPECT_THROW(r2_score(y_true, y_pred), std::invalid_argument);
}
