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
#ifndef RRCA_CHOLESKYDECOMPOSITION_CHOLESKYDECOMPOSITIONBASE_H_
#define RRCA_CHOLESKYDECOMPOSITION_CHOLESKYDECOMPOSITIONBASE_H_

namespace RRCA {

template <typename Derived, typename KernelMatrix>
class CholeskyDecompositionBase {
 public:
  typedef typename KernelMatrix::value_type value_type;
  typedef KernelMatrix kernelMatrix;
  //////////////////////////////////////////////////////////////////////////////
  // return a reference to the derived object
  Derived &derived() { return *static_cast<Derived *>(this); }
  // return a const reference to the derived object
  const Derived &derived() const { return *static_cast<const Derived *>(this); }
  //////////////////////////////////////////////////////////////////////////////
  /*
   *   \brief exposes the Derived objects compute routine to the outside
   */
  template <typename... Ts>
  void compute(Ts &&...ts) {
    derived().compute(std::forward<Ts>(ts)...);
  }
  //////////////////////////////////////////////////////////////////////////////
  /*
   *    \brief computes the biorthogonal basis B of the Cholesky factor such
   *           that B^TL=I
   */
  void computeBiorthogonalBasis() {
    Bmatrix_.resize(Lmatrix_.rows(), Lmatrix_.cols());
    Bmatrix_.setZero();
    for (auto i = 0; i < indices_.size(); ++i) {
      Bmatrix_(indices_(i), i) = 1;
      Bmatrix_.col(i) -= Bmatrix_.block(0, 0, Bmatrix_.rows(), i) *
                         Lmatrix_.row(indices_(i)).head(i).transpose();
      Bmatrix_.col(i) /= Lmatrix_(indices_(i), i);
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  /*
   *   \brief approximates the approximation error of the pivoted Cholesky
   *          decomposition by Monte Carlo sampling of random columns of the
   *          kernel matrix
   */
  value_type sampleError(const kernelMatrix &C, int samples = 100) const {
    Vector colOp;
    Vector colL;
    value_type error = 0;
    value_type fnorm2 = 0;
    Eigen::Index sampleCol = 0;
    Eigen::Index dim = C.cols();
    std::srand(std::time(NULL));
    // compare random columns of C to the respective ones of L * L'
    for (auto i = 0; i < samples; ++i) {
      sampleCol = std::rand() % dim;
      colOp = C.col(sampleCol);
      colL = Lmatrix_ * Lmatrix_.row(sampleCol).transpose();
      error += (colOp - colL).squaredNorm();
      fnorm2 += colOp.squaredNorm();
    }
    return sqrt(error / fnorm2);
  }
  //////////////////////////////////////////////////////////////////////////////
  /// getters
  //////////////////////////////////////////////////////////////////////////////
  const Matrix &matrixL(void) const { return Lmatrix_; }

  const Matrix &matrixB(void) const { return Bmatrix_; }

  const iVector &pivots(void) const { return indices_; }

  value_type tolerance(void) const { return tol_; }

  int info(void) const { return info_; }
  //////////////////////////////////////////////////////////////////////////////
  /// protected members
  //////////////////////////////////////////////////////////////////////////////
 protected:
  Matrix Lmatrix_;
  Matrix Bmatrix_;
  iVector indices_;
  value_type tol_;
  int info_;
};

}  // namespace RRCA
#endif
