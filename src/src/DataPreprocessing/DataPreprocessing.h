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
#ifndef RRCA_DATAPREPROCESSING_DATAPREPROCESSING_H_
#define RRCA_DATAPREPROCESSING_DATAPREPROCESSING_H_
#include <iostream>
namespace RRCA {

enum DataOrientation { rowwise, colwise };

template <typename Derived>
Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, Eigen::Dynamic>
rowMean(const Eigen::MatrixBase<Derived> &X) {
  return X.rowwise().mean();
}

template <typename Derived>
Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, Eigen::Dynamic>
colMean(const Eigen::MatrixBase<Derived> &X) {
  return X.colwise().mean();
}

template <typename Derived>
Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, 1>
rowVar(const Eigen::MatrixBase<Derived> &X) {
  return (X.cols() > 1 ? (X.cols() / (X.cols() - 1)) : 0) *
         (X - rowMean(X).replicate(1, X.cols()))
             .array()
             .square()
             .rowwise()
             .mean();
}

template <typename Derived>
Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, Eigen::Dynamic>
colVar(const Eigen::MatrixBase<Derived> &X) {
  return (X.rows() > 1 ? (X.rows() / (X.rows() - 1)) : 0) *
         (X - colMean(X).replicate(X.rows(), 1))
             .array()
             .square()
             .colwise()
             .mean();
}

template <typename Derived>
Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, 1>
rowStd(const Eigen::MatrixBase<Derived> &X) {
  return rowVar(X).cwiseSqrt();
}

template <typename Derived>
Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, Eigen::Dynamic>
colStd(const Eigen::MatrixBase<Derived> &X) {
  return colVar(X).cwiseSqrt();
}

template <typename Derived>
Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, Eigen::Dynamic>
rowStandardize(const Eigen::MatrixBase<Derived> &X) {
  Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, Eigen::Dynamic>
      retval = X;
  // center data
  retval -= rowMean(X).replicate(1, X.cols());
  // scale standard deviation
  retval.array().colwise() *= rowStd(X).col(0).cwiseInverse().array();

  return retval;
}

template <typename Derived>
Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, Eigen::Dynamic>
colStandardize(const Eigen::MatrixBase<Derived> &X) {
  Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, Eigen::Dynamic>
      retval = X;
  // center data
  retval -= colMean(X).replicate(X.rows(), 1);
  // scale standard deviation
  retval.array().rowwise() *= colStd(X).row(0).cwiseInverse().array();

  return retval;
}

template <typename Derived>
Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, Eigen::Dynamic>
correlationMatrix(const Eigen::MatrixBase<Derived> &X) { // assumes that the data are colwise in the time variable, so that X X.transpose()n gives the second moment matrix
  const double n(X.cols());
  Eigen::Vector<typename Derived::value_type, Eigen::Dynamic> mu = X.rowwise().mean(); 
  Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, Eigen::Dynamic> covar_mat = X*X.transpose()/n - mu * mu.transpose();
  Eigen::Vector<typename Derived::value_type, Eigen::Dynamic> sd = covar_mat.diagonal().cwiseSqrt();
  


  return(covar_mat.cwiseQuotient(sd * sd.transpose()));
}

template <typename Derived>
class PCA {
 public:
  typedef Eigen::Matrix<typename Derived::value_type, Eigen::Dynamic, Eigen::Dynamic> eigenMatrix;
  PCA(const Eigen::MatrixBase<Derived> &X, DataOrientation orient = rowwise)
      : data_(X.derived()), orient_(orient) {}
  void compute() {
    if (orient_ == rowwise) {
      mean_ = rowMean(data_);
      std_ = rowStd(data_);
      svd_.compute(rowStandardize(data_),
                   Eigen::ComputeThinU | Eigen::ComputeThinV);
    } else {
      mean_ = colMean(data_);
      std_ = colStd(data_);
      svd_.compute(colStandardize(data_),
                   Eigen::ComputeThinU | Eigen::ComputeThinV);
    }
  }

  const Eigen::BDCSVD<eigenMatrix> &svd() { return svd_; }
  const eigenMatrix &mean() { return mean_; }
  const eigenMatrix &std() { return std_; }

 private:
  Eigen::BDCSVD<eigenMatrix> svd_;
  const Derived &data_;
  eigenMatrix mean_;
  eigenMatrix std_;
  DataOrientation orient_;
};
}  // namespace RRCA
#endif
