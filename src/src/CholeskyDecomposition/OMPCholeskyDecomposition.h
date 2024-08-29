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
#ifndef RRCA_CHOLESKYDECOMPOSITION_OMPCHOLESKYDECOMPOSITION_H_
#define RRCA_CHOLESKYDECOMPOSITION_OMPCHOLESKYDECOMPOSITION_H_

namespace RRCA {

template <typename KernelMatrix>
class OMPCholeskyDecompositon
    : public CholeskyDecompositionBase<OMPCholeskyDecompositon<KernelMatrix>,
                                       KernelMatrix> {
 public:
  typedef CholeskyDecompositionBase<OMPCholeskyDecompositon, KernelMatrix> Base;
  // get types from base class
  using value_type = typename Base::value_type;
  using kernelMatrix = typename Base::kernelMatrix;
  // get member variables from base class
  using Base::Bmatrix_;
  using Base::indices_;
  using Base::info_;
  using Base::Lmatrix_;
  using Base::tol_;
  OMPCholeskyDecompositon() {}
  // non-void constructor
  OMPCholeskyDecompositon(const kernelMatrix &C, const Vector &f,
                          value_type tol,Eigen::Index step_limit = 0) {
    compute(C, f, tol,step_limit);
  }
  /*
   *   \brief computes the pivoted Cholesky decomposition with diagonal
   *          pivoting
   */
  void compute(const kernelMatrix &C, const Vector &f, value_type tol, Eigen::Index step_limit = 0) {
    Vector fm;
    Eigen::Index pivot = 0;
    Eigen::Index actBSize = 0;
    Eigen::Index dim = C.cols();
    Eigen::Index piv_lim = step_limit ? step_limit : C.cols();
    value_type err = 0;
    value_type fnorm = f.norm();
    value_type max_coeff = 0;
    tol_ = tol;
    fm = f;
    Vector D = C.diagonal();
    err = fm.cwiseAbs().maxCoeff();
    // perform pivoted Cholesky decomposition
    Eigen::Index step = 0;
    while ((step < dim) && (tol < err) && (step < piv_lim)) {
      // check memory requirements
      if (actBSize - 1 <= step) {
        actBSize += allocBSize;
        Lmatrix_.conservativeResize(dim, actBSize);
        Bmatrix_.conservativeResize(dim, actBSize);
        indices_.conservativeResize(actBSize);
      }
      max_coeff = fm.cwiseAbs().maxCoeff(&pivot);
      // check if updated pivot element is feasible otherwise break
      if (D(pivot) <= 0) {
        info_ = 3;
        break;
      }
      indices_(step) = pivot;
      // get new column from C
      Lmatrix_.col(step) = C.col(pivot);
      Bmatrix_.col(step).setZero();
      Bmatrix_(pivot, step) = 1;
      // update column with the current matrix Lmatrix_
      Lmatrix_.col(step) -= Lmatrix_.block(0, 0, dim, step) *
                            Lmatrix_.row(pivot).head(step).transpose();
      Bmatrix_.col(step) -= Bmatrix_.block(0, 0, dim, step) *
                            Lmatrix_.row(pivot).head(step).transpose();
      Lmatrix_.col(step) /= sqrt(D(pivot));
      Bmatrix_.col(step) /= sqrt(D(pivot));
      // update the diagonal
      D.array() -= Lmatrix_.col(step).array().square();
      // update the remainder of the function
      fm -= Bmatrix_.col(step).dot(f) * Lmatrix_.col(step);
      err = fm.norm() / f.norm();
      ++step;
    }
    // crop K,L, indices to their actual size
    Lmatrix_.conservativeResize(dim, step);
    Bmatrix_.conservativeResize(dim, step);
    indices_.conservativeResize(step);
    if (D(pivot) > 0) info_ = 0;
    return;
  }
};

}  // namespace RRCA
#endif
