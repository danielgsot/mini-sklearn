#pragma once

#include "msk/core/matrix.h"

namespace msk {

    class Transformer {
    public:
        virtual ~Transformer() = default;

        virtual void fit(const Matrix& X) = 0;
        virtual Matrix transform(const Matrix& X) const = 0;

        Matrix fit_transform(const Matrix& X) {
            fit(X);
            return transform(X);
        }
    };

}