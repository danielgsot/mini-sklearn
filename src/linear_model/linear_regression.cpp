#include "msk/linear_model/linear_regression.h"

#include <vector>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <cstddef>

#include "msk/metrics/regression.h"

namespace {
    std::vector<double> residuals(const std::vector<double> &y_pred, const std::vector<double> &y) {
        if (y_pred.size() != y.size()) {
            throw std::invalid_argument("Size of y_pred and y does not match.");
        }
        std::vector<double> result(y.size());
        for (std::size_t i = 0; i < result.size(); i++) {
            result[i] = y_pred[i] - y[i];
        }
        return result;
    }
}

namespace msk {
    LinearRegression::LinearRegression(int n_iterations, double learning_rate, double tolerance)
        : _n_iterations(n_iterations), _learning_rate(learning_rate), _tolerance(tolerance) {

        if (_n_iterations <= 0) {
            throw std::invalid_argument("n_iterations must be greater than 0.");
        }
        if (_learning_rate <= 0.0) {
            throw std::invalid_argument("learning_rate must be greater than 0.");
        }
        if (_tolerance < 0.0) {
            throw std::invalid_argument("tolerance cannot be negative.");
        }
    }

    void LinearRegression::fit(const msk::Matrix &X, const std::vector<double> &y) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument("Matrix X cannot be empty.");
        }
        if (static_cast<std::size_t>(X.rows()) != y.size()) {
            throw std::invalid_argument("Number of samples in X does not match the size of y.");
        }

        std::size_t n_samples = X.rows();
        std::size_t n_features = X.cols();

        _w.assign(n_features, 0.0);
        _b = 0.0;
        _n_iter = 0;

        double previous_MSE = std::numeric_limits<double>::infinity();
        msk::Matrix X_T = X.transpose();

        for (int iter = 0; iter < _n_iterations; iter++) {
            _n_iter = iter + 1;

            std::vector<double> y_pred = predict(X);
            std::vector<double> error = residuals(y_pred, y);
            double current_MSE = msk::mean_squared_error(y, y_pred);

            if (!std::isfinite(current_MSE)) {
                throw std::runtime_error("Model diverged (MSE is NaN or Infinity). Try reducing the learning_rate.");
            }

            if (std::abs(previous_MSE - current_MSE) < _tolerance) {
                break;
            }
            previous_MSE = current_MSE;

            std::vector<double> grad_w = X_T * error;
            double grad_b = std::accumulate(error.begin(), error.end(), 0.0) / static_cast<double>(n_samples);

            for (std::size_t j = 0; j < n_features; j++) {
                _w[j] -= _learning_rate * (grad_w[j] / static_cast<double>(n_samples));
            }
            _b -= _learning_rate * grad_b;
        }
    }

    std::vector<double> LinearRegression::predict(const msk::Matrix &X) const {
        if (_w.empty()) {
            throw std::logic_error("LinearRegression model needs to be fitted before calling predict.");
        }
        if (static_cast<std::size_t>(X.cols()) != _w.size()) {
            throw std::invalid_argument("Number of features in X does not match the trained model.");
        }

        std::vector<double> Xw = X * _w;
        std::vector<double> result(Xw.size());

        for (std::size_t i = 0; i < result.size(); i++) {
            result[i] = Xw[i] + _b;
        }
        return result;
    }

    double LinearRegression::score(const msk::Matrix &X, const std::vector<double> &y) const {
        if (_w.empty()) {
            throw std::logic_error("LinearRegression model needs to be fitted before calling score.");
        }
        std::vector<double> y_pred = predict(X);
        return msk::r2_score(y, y_pred);
    }

    const std::vector<double>& LinearRegression::coef() const {
        return _w;
    }

    double LinearRegression::intercept() const {
        return _b;
    }

    int LinearRegression::n_iter() const {
        return _n_iter;
    }
}