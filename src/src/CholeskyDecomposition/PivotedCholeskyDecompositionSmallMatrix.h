////////////////////////////////////////////////////////////////////////////////
// This file is part of RRCA, the Roadrunner Covariance Analsys package.      //
//                                                                            //
// Copyright (c) 2021, Michael Multerer and Paul Schneider                    //
//                                                                            //
// All rights reserved.                                                       //
//                                                                            //
// This source code is subject to the BSD 3-clause license and without        //
// any warranty, see <https://github.com/muchip/RRCA> for further             //
// information.                                                               //
////////////////////////////////////////////////////////////////////////////////
#ifndef RRCA_CHOLESKYDECOMPOSITION_PIVOTEDCHOLESKYDECOMPOSITIONSMALLMATRIX_H_
#define RRCA_CHOLESKYDECOMPOSITION_PIVOTEDCHOLESKYDECOMPOSITIONSMALLMATRIX_H_

namespace RRCA {

template <typename KernelMatrix>
class PivotedCholeskyDecompositionSmallMatrix
    : public CholeskyDecompositionBase<
          PivotedCholeskyDecompositionSmallMatrix<KernelMatrix>, KernelMatrix> {
public:
  typedef CholeskyDecompositionBase<PivotedCholeskyDecompositionSmallMatrix,
                                    KernelMatrix>
      Base;
  // get types from base class
  using value_type = typename Base::value_type;
  using kernelMatrix = typename Base::kernelMatrix;
  // get member variables from base class
  using Base::Bmatrix_;
  using Base::indices_;
  using Base::info_;
  using Base::Lmatrix_;
  using Base::tol_;
  PivotedCholeskyDecompositionSmallMatrix() {}
  // non-void constructor
  PivotedCholeskyDecompositionSmallMatrix(const kernelMatrix &C,
                                         value_type tol) {
    compute(C, tol);
  }
  /*
   *   \brief computes the pivoted Cholesky decomposition with diagonal
   *          pivoting
   */
  void compute(const kernelMatrix &C, value_type tol) {
    Vector D;
    Eigen::Index pivot = 0;
    Eigen::Index actBSize = 0;
    Eigen::Index dim = C.cols();
    assert(dim <= 1e4 && "This class is only for small matrices");
    Matrix K(dim, dim);
    Vector nrms(dim);
    value_type tr = 0;
    Lmatrix_.resize(dim, dim);
    indices_.resize(dim);
    // assemble full matrix and store the norm of each column
    for (auto i = 0; i < dim; ++i) {
      K.col(i) = C.col(i);
      nrms(i) = K.col(i).squaredNorm();
    }
    tol_ = tol;
    // compute the diagonal and the trace
    D = K.diagonal();
    if (D.minCoeff() < 0) {
      info_ = 1;
      return;
    }
    tr = D.sum();
    // we guarantee the error tr(A-LL^T)/tr(A) < tol
    tol *= tr;
    // perform pivoted Cholesky decomposition
    Eigen::Index step = 0;
    while (tol < tr) {
      nrms.maxCoeff(&pivot);
      indices_(step) = pivot;
      Lmatrix_.col(step) = K.col(pivot);
      Lmatrix_.col(step) /= sqrt(Lmatrix_(pivot, step));
      K -= Lmatrix_.col(step) * Lmatrix_.col(step).transpose();
      D.array() -= Lmatrix_.col(step).array().square();
      // update norm vector
      for (auto i = 0; i < dim; ++i)
        if (abs(D(i)) < 1e-14)
          nrms(i) = 0;
        else
          nrms(i) = K.col(i).squaredNorm();
      tr = D.sum();
      ++step;
    }
    // crop L, indices to their actual size
    Lmatrix_.conservativeResize(dim, step);
    indices_.conservativeResize(step);
    if (tr < 0)
      info_ = 2;
    else
      info_ = 0;
    return;
  }
};

} // namespace RRCA
#endif
