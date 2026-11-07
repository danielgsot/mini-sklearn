#ifndef MINI_SKLEARN_REGRESSION_H
#define MINI_SKLEARN_REGRESSION_H
#include <vector>

namespace msk {

    double mean_squared_error(const std::vector<double> &y_true,const std::vector<double> &y_pred);
    double r2_score(const std::vector<double> &y_true, const std::vector<double> &y_pred);


}


#endif //MINI_SKLEARN_REGRESSION_H
