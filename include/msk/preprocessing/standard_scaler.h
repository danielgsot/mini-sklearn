#pragma once

#include "msk/core/matrix.h"
#include <vector>

namespace msk {
    class StandardScaler {
    public:
        void fit(const Matrix& X);
        Matrix transform(const Matrix& X) const;
        Matrix fit_transform(const Matrix& X);

        const std::vector<double>& means() const;
        const std::vector<double>& stds() const;

    private:
        std::vector<double> _means;
        std::vector<double> _stds;
    };
}