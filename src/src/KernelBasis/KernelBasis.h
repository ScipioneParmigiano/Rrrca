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
#ifndef RRCA_KERNELBASIS_KERNELBASIS_H_
#define RRCA_KERNELBASIS_KERNELBASIS_H_

namespace RRCA {

template <typename KernelMatrix>
class KernelBasis {
 public:
  typedef typename KernelMatrix::value_type value_type;
  typedef typename KernelMatrix::kernelfunction kernel;

  KernelBasis() {}
  // non-void constructor
  template <typename Derived>
  KernelBasis(const KernelMatrix &K,
              const CholeskyDecompositionBase<Derived, KernelMatrix> &chol) {
    init(K, chol.pivots());
  }
  /*
   *  \brief initializes the source points
   */
  void init(const KernelMatrix &K, const iVector &Idcs) {
    src_pts_.resize(K.pts().rows(), Idcs.size());
    k_ = K.kernel();
    for (auto i = 0; i < Idcs.size(); ++i)
      src_pts_.col(i) = K.pts().col(Idcs(i));
    Kppmatrix_.resize(src_pts_.cols(), src_pts_.cols());
    for (auto j = 0; j < src_pts_.cols(); ++j)
      for (auto i = 0; i < src_pts_.cols(); ++i) {
        Kppmatrix_(i, j) = k_(src_pts_.col(i), src_pts_.col(j));
        // assert(k_(src_pts_.col(i), src_pts_.col(j)) ==
        //        K.kernel()(src_pts_.col(i), src_pts_.col(j)));
      }
  }
  
    void initFull(const KernelMatrix &K) {
    src_pts_ = K.pts();
    k_ = K.kernel();
    Kppmatrix_.resize(src_pts_.cols(), src_pts_.cols());
    for (auto j = 0; j < src_pts_.cols(); ++j)
      for (auto i = 0; i < src_pts_.cols(); ++i) {
        Kppmatrix_(i, j) = k_(src_pts_.col(i), src_pts_.col(j));
        // assert(k_(src_pts_.col(i), src_pts_.col(j)) ==
        //        K.kernel()(src_pts_.col(i), src_pts_.col(j)));
      }
  }
  
  void initFullLower(const KernelMatrix &K) {
    src_pts_ = K.pts();
    k_ = K.kernel();
    Kppmatrix_.resize(src_pts_.cols(), src_pts_.cols());
    for (auto j = 0; j < src_pts_.cols(); ++j)
      for (auto i = j; i < src_pts_.cols(); ++i) {
        Kppmatrix_(i, j) = k_(src_pts_.col(i), src_pts_.col(j));
        // assert(k_(src_pts_.col(i), src_pts_.col(j)) ==
        //        K.kernel()(src_pts_.col(i), src_pts_.col(j)));
      }
  }

  const Matrix &matrixU() const { return Umatrix_; }

  const Matrix &matrixQ() const { return Qmatrix_; }

  const Matrix &matrixKpp() const { return Kppmatrix_; }

  const Matrix &src_pts() const { return src_pts_; }

  void setSrcPts(const Matrix &src_pts) { src_pts_ = src_pts; }
  
  const Vector &vectorLambda() const { return lambda_; }

  const kernel &krnl() const { return k_; }
  kernel &krnl() { return k_; }

  /*
   *  \brief initializes the weights of the Newton basis
   */
  template <typename Derived>
  void initNewtonBasisWeights(
      const CholeskyDecompositionBase<Derived, KernelMatrix> &chol) {
    Umatrix_.resize(chol.matrixL().cols(), chol.matrixL().cols());
    Umatrix_.setZero();
    for (auto i = 0; i < chol.pivots().size(); ++i) {
      Umatrix_(i, i) = 1;
      Umatrix_.col(i) -=
          Umatrix_.leftCols(i) *
          chol.matrixL().row(chol.pivots()(i)).head(i).transpose();
      Umatrix_.col(i) /= chol.matrixL()(chol.pivots()(i), i);
    }
    return;
  }
  /*
   *    \brief computes the weights for the double orthogonal basis, i.e.
   *           UV, where V is the spectral basis of L^TL
   */
  template <typename Derived>
  void initSpectralBasisWeights(
      const CholeskyDecompositionBase<Derived, KernelMatrix> &chol) {
    // recompute U in any case
    initNewtonBasisWeights(chol);
    // compute spectral decomposition of L^TL
    Matrix C = chol.matrixL().transpose() * chol.matrixL();
    Eigen::SelfAdjointEigenSolver<Matrix> es(C);
    Qmatrix_ = es.eigenvectors();
    lambda_ = es.eigenvalues().reverse();
    // sort the eigen basis such that the eigenvalues are decreasing
    for (auto i = 0; i < Qmatrix_.cols() / 2; ++i)
      Qmatrix_.col(i).swap(Qmatrix_.col(Qmatrix_.cols() - 1 - i));
    // assemble the actual weights
    Qmatrix_ = Umatrix_ * Qmatrix_;
    return;
  }
  
    /*
   *    \brief computes the weights for the double orthogonal basis, i.e.
   *           UV, where V is the spectral basis of L^TL
   */
  template <typename Derived>
  void initSpectralBasisWeights(
      const CholeskyDecompositionBase<Derived, KernelMatrix> &chol, const iVector &Idcs) {
    // recompute U in any case
    initNewtonBasisWeights(chol);
    // compute spectral decomposition of L^TL
    Matrix C = chol.matrixL()(Idcs,Eigen::all).transpose() * chol.matrixL()(Idcs,Eigen::all);
    Eigen::SelfAdjointEigenSolver<Matrix> es(C);
    lambda_ = es.eigenvalues();
    Qmatrix_ = es.eigenvectors().reverse();
    // sort the eigen basis such that the eigenvalues are decreasing
    for (auto i = 0; i < Qmatrix_.cols() / 2; ++i)
      Qmatrix_.col(i).swap(Qmatrix_.col(Qmatrix_.cols() - 1 - i));
    // assemble the actual weights
    Qmatrix_ = Umatrix_ * Qmatrix_;
    return;
  }

  
  /*
   *  \brief evaluates the kernel basis at a given set of target
   *         points
   */
  template <typename Derived>
  Matrix eval(const Eigen::MatrixBase<Derived> &pts) const {
    Matrix retval(pts.cols(), src_pts_.cols());
    for (auto j = 0; j < src_pts_.cols(); ++j)
      for (auto i = 0; i < pts.cols(); ++i) {
        retval(i, j) = k_(src_pts_.col(j), pts.col(i));
      }
    return retval;
  }
  
  
    /*
   *  \brief evaluates the full kernel basis (not just pivots at a given set of target
   *         points
   */
  template <typename Derived>
  Matrix evalfull(const KernelMatrix &K, const Eigen::MatrixBase<Derived> &pts) const {
    Matrix retval(pts.cols(), K.pts().cols());
    for (auto j = 0; j < K.pts().cols(); ++j)
      for (auto i = 0; i < pts.cols(); ++i) {
        retval(i, j) = k_(K.pts().col(j), pts.col(i));
      }
    return retval;
  }

  /*
   *  \brief computes in sample bounds for the function values of the Newton
   *         basis
   */
  Matrix KernelBasisBounds(const Matrix &Keval) const {
    Matrix retval(2, src_pts_.cols());
    retval.row(0) = Keval.colwise().minCoeff();
    retval.row(1) = Keval.colwise().maxCoeff();
    return retval;
  }
  /*
   *  \brief computes in sample bounds for the function values of the Newton
   *         basis
   */
  Matrix NewtonBasisBounds(const Matrix &Keval) const {
    Matrix retval(2, src_pts_.cols());
    Matrix temp = Keval * Umatrix_;
    retval.row(0) = temp.colwise().minCoeff();
    retval.row(1) = temp.colwise().maxCoeff();
    return retval;
  }
  /*
   *  \brief computes in sample bounds for the function values of the Newton
   *         basis
   */
  Matrix SpectralBasisBounds(const Matrix &Keval) const {
    Matrix retval(2, src_pts_.cols());
    Matrix temp = Keval * Qmatrix_;
    retval.row(0) = temp.colwise().minCoeff();
    retval.row(1) = temp.colwise().maxCoeff();
    return retval;
  }

 private:
  Matrix src_pts_;
  Matrix Umatrix_;
  Matrix Qmatrix_;
  Matrix Kppmatrix_;
  kernel k_;
  
  Vector lambda_; //the eigenvalues of the spectral basis
};

}  // namespace RRCA
#endif
