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
#ifndef RRCA_UTIL_LLTSPECTRALDECOMPOSER_H_
#define RRCA_UTIL_LLTSPECTRALDECOMPOSER_H_

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

template <typename Derived>
class LLTspectralDecomposer {
 public:
  LLTspectralDecomposer(){};
  template <typename otherDerived>
  LLTspectralDecomposer(const Eigen::MatrixBase<otherDerived> &L) {
    compute(L);
  };

  template <typename otherDerived>
  void compute(const Eigen::MatrixBase<otherDerived> &L) {
    Eigen::HouseholderQR<Derived> qr(L);
    V_ = Derived::Identity(L.rows(), L.cols());
    V_ = qr.householderQ() * V_;
    Derived RRT = qr.matrixQR()
                      .topRows(V_.cols())
                      .template triangularView<Eigen::Upper>();
    RRT = RRT * RRT.transpose();
    Eigen::SelfAdjointEigenSolver<Derived> es(RRT);
    V_ = V_ * es.eigenvectors();
    for (auto i = 0; i < V_.cols() / 2; ++i)
      V_.col(i).swap(V_.col(V_.cols() - 1 - i));
    D_ = es.eigenvalues().reverse().asDiagonal();
  };

  const Derived &matrixV() { return V_; }
  const Derived &matrixD() { return D_; }

 private:
  Derived V_;
  Derived D_;
};
#endif
