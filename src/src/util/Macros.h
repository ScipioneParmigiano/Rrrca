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
#ifndef RRCA_UTIL_MACROS_H_
#define RRCA_UTIL_MACROS_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#ifdef RRCA_HAVE_MOSEK
#include "fusion.h"
#endif

static constexpr size_t allocBSize = 100;

static constexpr Eigen::Index crosssecSize = 1000;

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename Derived>
std::string eigenDim(const Eigen::MatrixBase<Derived> &L) {
  return (
      std::string(std::to_string(L.rows()) + "x" + std::to_string(L.cols())));
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

namespace RRCA {
#ifndef M_PI
#define RRCA_PI 3.14159265358979323846264338327950288
#else
#define RRCA_PI M_PI
#endif
  
#define RRCA_BBOX_THREASHOLD 1e-2
#define RRCA_ZERO_TOLERANCE 2e-16
// #define RRCA_LOWRANK_EPS 1e-3
#define RRCA_LOWRANK_EPS 1e-8
// #define RRCA_LOWRANK_STEPLIM 80
  // #define RRCA_LOWRANK_STEPLIM 60
  // #define RRCA_LOWRANK_STEPLIM 60 
    #define RRCA_LOWRANK_STEPLIM 30


typedef unsigned int Index;

typedef double Scalar;

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix;

typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1u> Vector;

typedef Eigen::Matrix<Scalar, 1u, Eigen::Dynamic> RowVector;

typedef Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> iMatrix;

typedef Eigen::Matrix<Index, Eigen::Dynamic, 1u> iVector;

typedef Eigen::Matrix<Index, 2, 1> ijVector;

typedef Eigen::SparseMatrix<Scalar> SparseMatrix;

typedef Eigen::SparseVector<Scalar> SparseVector;

typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DiagonalMatrix;

#ifdef RRCA_HAVE_MOSEK

typedef mosek::fusion::Matrix M_Matrix; 

typedef mosek::fusion::Variable M_Variable; 

typedef mosek::fusion::Var M_Var; 

typedef mosek::fusion::Expression M_Expression; 

typedef mosek::fusion::Domain M_Domain;

typedef monty::ndarray<double, 1> M_ndarray_1;

typedef monty::ndarray<double, 2> M_ndarray_2;

typedef mosek::fusion::Expr M_Expr; 

typedef mosek::fusion::Model::t M_Model; 



#endif
}  // namespace RRCA

#endif
