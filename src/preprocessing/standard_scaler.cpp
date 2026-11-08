#include "msk/preprocessing/standard_scaler.h"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace {

    double mean(const std::vector<double> &v) {
        double sum = 0.0;
        int n = v.size();
        for (int i = 0; i < n; i++) {
            sum += v[i];
        }
        return sum / n;
    }

    double standard_deviation(const std::vector<double> &v, double mean_) {
        int n = v.size();
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double x = v[i] - mean_;
            sum += x * x;
        }
        return std::sqrt(sum / n);
    }
}

namespace msk {
    void StandardScaler::fit(const msk::Matrix &X) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument("fit: Matrix cannot be empty.");
        }

        int n = X.rows();
        int m = X.cols();
        _means.resize(m);
        _stds.resize(m);

        double eps = std::numeric_limits<double>::epsilon();

        for (int j = 0; j < m; j++) {
            std::vector<double> col_j = X.column(j);
            _means[j] = mean(col_j);
            double std_val = standard_deviation(col_j, _means[j]);

            double threshold = n * eps * std::abs(_means[j]);

            _stds[j] = (std_val <= threshold) ? 1.0 : std_val;
        }
    }

    Matrix StandardScaler::transform(const msk::Matrix &X) const {
        if (_means.empty()) {
            throw std::logic_error("transform: StandardScaler is not fitted. Call 'fit' first.");
        }
        if (X.cols() != static_cast<int>(_means.size())) {
            throw std::invalid_argument("transform: Number of columns in X does not match the fitted data.");
        }

        Matrix result(X.rows(), X.cols());
        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                result(i, j) = (X(i, j) - _means[j]) / _stds[j];
            }
        }
        return result;
    }

    const std::vector<double>& StandardScaler::means() const {
        return _means;
    }

    const std::vector<double>& StandardScaler::stds() const {
        return _stds;
    }
}