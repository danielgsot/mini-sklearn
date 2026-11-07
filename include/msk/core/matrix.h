#ifndef MINI_SKLEARN_MATRIX_H
#define MINI_SKLEARN_MATRIX_H
#include <vector>
namespace msk {
    class Matrix {
    private:
        int _rows;
        int _cols;
        std::vector<double> _data;

    public:
        Matrix(int rows, int cols);
        Matrix(int rows, int cols,std::vector<double> data);

        int rows() const;
        int cols() const;

        double operator()(int i, int j) const;
        double& operator()(int i, int j);

        Matrix operator+(const Matrix& m) const;
        Matrix operator-(const Matrix& m) const;
        Matrix operator*(const Matrix& m) const;
        Matrix operator*(const double k) const;

        Matrix transpose() const;
    };
}
#endif //MINI_SKLEARN_MATRIX_H
