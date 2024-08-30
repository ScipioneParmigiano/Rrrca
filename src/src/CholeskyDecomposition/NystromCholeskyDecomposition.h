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
#ifndef RRCA_CHOLESKYDECOMPOSITION_NYSTROMCHOLESKYDECOMPOSITION_H_
#define RRCA_CHOLESKYDECOMPOSITION_NYSTROMCHOLESKYDECOMPOSITION_H_

namespace RRCA {

template <typename KernelMatrix>
class NystromCholeskyDecomposition
    : public CholeskyDecompositionBase<
          NystromCholeskyDecomposition<KernelMatrix>, KernelMatrix> {
 public:
  typedef CholeskyDecompositionBase<NystromCholeskyDecomposition, KernelMatrix>
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
  NystromCholeskyDecomposition() { mtwister_.seed(std::time(NULL)); }
  // non-void constructor
  NystromCholeskyDecomposition(const kernelMatrix &C, value_type tol) {
    mtwister_.seed(std::time(NULL));
    compute(C, tol);
  }
  /*
   *   \brief computes the pivoted Cholesky decomposition with diagonal
   *          pivoting
   */
  void compute(const kernelMatrix &C, Eigen::Index max_rank = 100,
               value_type tol = 1e-12) {
    Vector D;
    Eigen::Index pivot = 0;
    Eigen::Index dim = C.cols();
    value_type tr = 0;
    tol_ = tol;
    // set up random pivoting
    std::vector<Eigen::Index> random_order(dim);
    std::iota(random_order.begin(), random_order.end(), 0);
    std::shuffle(random_order.begin(), random_order.end(), mtwister_);
    // compute the diagonal and the trace
    D = C.diagonal();
    if (D.minCoeff() < 0) {
      info_ = 1;
      return;
    }
    tr = D.sum();
    // we guarantee the error tr(A-LL^T)/tr(A) < tol
    tol *= tr;
    // perform pivoted Cholesky decomposition
    Eigen::Index step = 0;
    Lmatrix_.resize(dim, max_rank);
    indices_.resize(max_rank);
    while ((step < max_rank) && (tol < tr)) {
      // check memory requirements
      pivot = random_order[step];
      indices_(step) = pivot;
      // get new column from C
      Lmatrix_.col(step) = C.col(pivot);
      // update column with the current matrix Lmatrix_
      Lmatrix_.col(step) -= Lmatrix_.block(0, 0, dim, step) *
                            Lmatrix_.row(pivot).head(step).transpose();
      // check if updated pivot element is still feasible otherwise break
      if (Lmatrix_(pivot, step) <= 0) {
        info_ = 3;
        break;
      }
      Lmatrix_.col(step) /= sqrt(Lmatrix_(pivot, step));
      // update the diagonal and the trace
      D.array() -= Lmatrix_.col(step).array().square();
      // compute the trace of the Schur complement
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

 private:
  std::mt19937 mtwister_;
};

}  // namespace RRCA
#endif
