#pragma once

#include "msk/core/base.h"
#include "msk/core/matrix.h"
#include <vector>

namespace msk {
    class StandardScaler : public Transformer {
    public:
        void fit(const Matrix& X) override;
        Matrix transform(const Matrix& X) const override;

        const std::vector<double>& means() const;
        const std::vector<double>& stds() const;

    private:
        std::vector<double> _means;
        std::vector<double> _stds;
    };
}