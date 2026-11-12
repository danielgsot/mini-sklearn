#pragma once

#include <vector>
#include <cstddef>
#include "msk/core/matrix.h"

namespace msk {
    class LinearRegression {
    public:
        explicit LinearRegression(int n_iterations = 1000, double learning_rate = 0.01, double tolerance = 1e-12);

        void fit(const msk::Matrix &X, const std::vector<double> &y);
        std::vector<double> predict(const msk::Matrix &X) const;
        double score(const msk::Matrix &X, const std::vector<double> &y) const;

        const std::vector<double>& coef() const;
        double intercept() const;
        int n_iter() const;

    private:
        int _n_iterations;
        double _learning_rate;
        double _tolerance;

        std::vector<double> _w;
        double _b = 0.0;
        int _n_iter = 0;
    };
}