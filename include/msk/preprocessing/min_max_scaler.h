#pragma once

#include "msk/core/matrix.h"
#include <vector>

namespace msk {
    class MinMaxScaler {
    public:
        void fit(const Matrix& X);
        Matrix transform(const Matrix& X) const;
        Matrix fit_transform(const Matrix& X);

        const std::vector<double>& min() const;
        /**
     * @brief Returns the scaling range (max - min) used for each feature.
     * @note If a feature is constant (max == min), the range is set to 1.0
     *       to avoid division by zero and preserve the scale for new data.
     */
        const std::vector<double>& range() const;

    private:
        std::vector<double> _min;
        std::vector<double> _range;
    };
}