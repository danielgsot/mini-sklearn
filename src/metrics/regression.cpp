#include "msk/metrics/regression.h"

#include <stdexcept>
#include <numeric>
#include <cstddef>
#include <iostream>
#include <ostream>

namespace msk {
    namespace {
        std::size_t checked_size(const std::vector<double> &y_true,const std::vector<double> &y_pred) {
            std::size_t n = y_true.size();
            if (n != y_pred.size()) {
                throw std::invalid_argument("The size of the vectors must be the same.");
            }
            if (n == 0) {
                throw std::invalid_argument("The size of the vector must be greater than 0.");
            }
            return n;
        }
    }
    double mean_squared_error(const std::vector<double> &y_true,const std::vector<double> &y_pred) {
        std::size_t n = checked_size(y_true,y_pred);

        double residual = 0.0;
        double sum = 0.0;
        for (std::size_t i = 0; i < n; i++) {
            residual = y_true[i] - y_pred[i];
            sum += residual * residual;
        }
        return sum/n;
    }

    double r2_score(const std::vector<double> &y_true,const std::vector<double> &y_pred) {
        std::size_t n = checked_size(y_true,y_pred);

        double SS_res = 0.0;
        double SS_tot = 0.0;
        double residual = 0.0;
        double deviation = 0.0;
        double mean = std::accumulate(y_true.begin(),y_true.end(),0.0) / n;

        for (std::size_t i = 0; i < n; i++) {
            residual = y_true[i] - y_pred[i];
            SS_res += residual * residual;

            deviation  = y_true[i] - mean;
            SS_tot += deviation * deviation;
        }
        if (SS_tot == 0.0) {
             return SS_res == 0.0 ? 1.0 : 0.0;
        }
        return 1 - SS_res/SS_tot;

    }
}
