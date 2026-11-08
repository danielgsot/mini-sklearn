#pragma once

#include "msk/core/base.h"
#include "msk/core/matrix.h"
#include <vector>

namespace msk {
    class MinMaxScaler : public Transformer {
    public:
        void fit(const Matrix& X) override;
        Matrix transform(const Matrix& X) const override;

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
