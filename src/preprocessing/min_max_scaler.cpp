#include "msk/preprocessing/min_max_scaler.h"

#include <vector>
#include <stdexcept>
#include <algorithm>

namespace msk {
    void MinMaxScaler::fit(const msk::Matrix &X) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument("fit: Matrix cannot be empty.");
        }

        int m = X.cols();
        _min.resize(m);
        _range.resize(m);

        for (int j = 0; j < m; j++) {
            std::vector<double> col_j = X.column(j);
            auto [min_it, max_it] = std::minmax_element(col_j.begin(), col_j.end());

            _min[j] = *min_it;
            double r = *max_it - *min_it;
            
            // Replace range with 1.0 if the column is constant
            // (prevents division by 0 and preserves scale for new data)
            _range[j] = (r == 0.0) ? 1.0 : r;
        }
    }

    Matrix MinMaxScaler::transform(const msk::Matrix &X) const {
        if (_min.empty()) {
            throw std::logic_error("transform: MinMaxScaler is not fitted. Call 'fit' first.");
        }
        if (X.cols() != static_cast<int>(_min.size())) {
            throw std::invalid_argument("transform: Number of columns in X does not match the fitted data.");
        }

        Matrix result(X.rows(), X.cols());

        for (int i = 0; i < X.rows(); i++) {
            for (int j = 0; j < X.cols(); j++) {
                result(i, j) = (X(i, j) - _min[j]) / _range[j];
            }
        }
        return result;
    }

    Matrix MinMaxScaler::fit_transform(const msk::Matrix &X) {
        fit(X);
        return transform(X);
    }

    const std::vector<double>& MinMaxScaler::min() const {
        return _min;
    }

    const std::vector<double>& MinMaxScaler::range() const {
        return _range;
    }
}