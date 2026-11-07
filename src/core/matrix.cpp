#include "msk/core/matrix.h"

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace msk {
    namespace {
        std::size_t checked_size(int rows, int cols) {
            if (rows < 0 || cols < 0) {
                throw std::invalid_argument("Invalid matrix dimensions");
            }
            return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        }

        void require_same_shape(const Matrix &a, const Matrix &b) {
            if (a.rows() != b.rows() || a.cols() != b.cols()) {
                throw std::invalid_argument("Matrix dimensions must match");
            }
        }
    }

    Matrix::Matrix(int rows, int cols)
        : _rows(rows), _cols(cols), _data(checked_size(rows, cols)) {
    }

    Matrix::Matrix(int rows, int cols, std::vector<double> data)
        : _rows(rows), _cols(cols), _data(std::move(data)) {
        if (_data.size() != checked_size(rows, cols)) {
            throw std::invalid_argument("Matrix size does not match number of rows and columns");
        }
    }

    int Matrix::rows() const { return _rows; }
    int Matrix::cols() const { return _cols; }

    double Matrix::operator()(int i, int j) const {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _data[i * _cols + j];
    }

    double &Matrix::operator()(int i, int j) {
        assert(i >= 0 && i < _rows && j >= 0 && j < _cols);
        return _data[i * _cols + j];
    }



    Matrix Matrix::operator+(const Matrix &m) const {
        require_same_shape(*this, m);
        Matrix result(_rows, _cols);
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                result(i, j) = (*this)(i, j) + m(i, j);
        return result;
    }

    Matrix Matrix::operator-(const Matrix &m) const {
        require_same_shape(*this, m);
        Matrix result(_rows, _cols);
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                result(i, j) = (*this)(i, j) - m(i, j);
        return result;
    }

    Matrix Matrix::operator*(const Matrix &m) const {
        if (_cols != m._rows) {
            throw std::invalid_argument("Inner dimensions must match for matrix product");
        }
        Matrix result(_rows, m._cols);
        for (int i = 0; i < _rows; i++)
            for (int k = 0; k < _cols; k++) {
                const double a = (*this)(i, k);
                for (int j = 0; j < m._cols; j++)
                    result(i, j) += a * m(k, j);
            }
        return result;
    }

    Matrix Matrix::operator*(const double k) const {
        Matrix result(_rows, _cols);
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                result(i, j) = (*this)(i, j) * k;
        return result;
    }

    std::vector<double> Matrix::row(int i) const {
        assert(i >= 0 && i < _rows);
        std::vector<double> result(_cols);
        for (int j = 0; j < _cols; j++)
            result[j] = (*this)(i, j);
        return result;
    }

    std::vector<double> Matrix::column(int j) const {
        assert(j >= 0 && j < _cols);
        std::vector<double> result(_rows);
        for (int i = 0; i < _rows; i++)
            result[i] = (*this)(i, j);
        return result;
    }

    Matrix Matrix::transpose() const {
        Matrix result(_cols, _rows);
        for (int i = 0; i < _rows; i++)
            for (int j = 0; j < _cols; j++)
                result(j, i) = (*this)(i, j);
        return result;
    }
}