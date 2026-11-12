#include <gtest/gtest.h>
#include "msk/linear_model/linear_regression.h"
#include "msk/core/matrix.h"
#include <vector>
#include <stdexcept>

TEST(LinearRegressionTest, ConvergesToCorrectWeights) {
    msk::Matrix X(4, 2);
    X(0, 0) = 1.0; X(0, 1) = 2.0;
    X(1, 0) = 2.0; X(1, 1) = 1.0;
    X(2, 0) = 3.0; X(2, 1) = 3.0;
    X(3, 0) = 4.0; X(3, 1) = 5.0;

    std::vector<double> y = {4.0, 9.0, 8.0, 7.0};

    msk::LinearRegression model(15000, 0.005, 1e-12);
    model.fit(X, y);

    EXPECT_NEAR(model.coef()[0], 3.0, 1e-3);
    EXPECT_NEAR(model.coef()[1], -2.0, 1e-3);
    EXPECT_NEAR(model.intercept(), 5.0, 1e-3);
    EXPECT_NEAR(model.score(X, y), 1.0, 1e-3);
}

TEST(LinearRegressionTest, DivergesWithLargeLearningRate) {
    msk::Matrix X(4, 2);
    X(0, 0) = 1.0; X(0, 1) = 2.0;
    X(1, 0) = 2.0; X(1, 1) = 1.0;
    X(2, 0) = 3.0; X(2, 1) = 3.0;
    X(3, 0) = 4.0; X(3, 1) = 5.0;

    std::vector<double> y = {4.0, 9.0, 8.0, 7.0};

    msk::LinearRegression model(1000, 5.0, 1e-12);
    EXPECT_THROW(model.fit(X, y), std::runtime_error);
}

TEST(LinearRegressionTest, ThrowsExceptionsWhenAppropriate) {
    msk::Matrix X_empty(0, 0);
    msk::Matrix X_valid(2, 2);
    msk::Matrix X_wrong_cols(2, 3);
    std::vector<double> y_valid = {1.0, 2.0};
    std::vector<double> y_invalid_size = {1.0, 2.0, 3.0};

    msk::LinearRegression empty_model;
    EXPECT_THROW(empty_model.predict(X_valid), std::logic_error);
    EXPECT_THROW(empty_model.score(X_valid, y_valid), std::logic_error);

    msk::LinearRegression model;
    EXPECT_THROW(model.fit(X_empty, y_valid), std::invalid_argument);
    EXPECT_THROW(model.fit(X_valid, y_invalid_size), std::invalid_argument);

    model.fit(X_valid, y_valid);
    EXPECT_THROW(model.predict(X_wrong_cols), std::invalid_argument);

    EXPECT_THROW(msk::LinearRegression(0, 0.01), std::invalid_argument);
    EXPECT_THROW(msk::LinearRegression(100, -0.1), std::invalid_argument);
    EXPECT_THROW(msk::LinearRegression(100, 0.01, -1e-5), std::invalid_argument);
}

TEST(LinearRegressionTest, EarlyStopping) {
    msk::Matrix X(4, 2);
    X(0, 0) = 1.0; X(0, 1) = 2.0;
    X(1, 0) = 2.0; X(1, 1) = 1.0;
    X(2, 0) = 3.0; X(2, 1) = 3.0;
    X(3, 0) = 4.0; X(3, 1) = 5.0;
    std::vector<double> y = {4.0, 9.0, 8.0, 7.0};

    int max_iter = 15000;
    msk::LinearRegression model(max_iter, 0.005, 1e-1);
    model.fit(X, y);

    EXPECT_LT(model.n_iter(), max_iter);
}

TEST(LinearRegressionTest, ReFittingResetsModel) {
    msk::Matrix X1(2, 1);
    X1(0, 0) = 1.0;
    X1(1, 0) = 2.0;
    std::vector<double> y1 = {2.0, 4.0};

    msk::Matrix X2(2, 1);
    X2(0, 0) = 1.0;
    X2(1, 0) = 2.0;
    std::vector<double> y2 = {3.0, 6.0};

    msk::LinearRegression model1(10000, 0.01, 1e-10);
    model1.fit(X1, y1);
    model1.fit(X2, y2);

    msk::LinearRegression model2(10000, 0.01, 1e-10);
    model2.fit(X2, y2);

    EXPECT_NEAR(model1.coef()[0], model2.coef()[0], 1e-6);
    EXPECT_NEAR(model1.intercept(), model2.intercept(), 1e-6);
    EXPECT_EQ(model1.n_iter(), model2.n_iter());
}