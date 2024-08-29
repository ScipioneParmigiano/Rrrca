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
#ifndef RRCA_UTIL_CORRELATIONMATRIXSIMULATOR_H_
#define RRCA_UTIL_CORRELATIONMATRIXSIMULATOR_H_

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <random>

namespace RRCA {
// 
// \brief simulated a correlation matrix using the prox algo from A New Parametrization of Correlation Matrices (ECTA)
// 
class CorrelationMatrixSimulator {
public:
  CorrelationMatrixSimulator(int seed, int dim_, double stddev = 1.0) : dim(dim_), vecdim(dim_*(dim_-1)/2) {
//     generate the input moments according to some real numbers generated from seed
      std::mt19937 generator(seed);
      std::normal_distribution<double> dist(0.0,stddev);
      auto norm = [&] () {return dist(generator);};
 
      gamma =  RRCA::Vector::NullaryExpr(vecdim, norm );
      // std::cout << "gamma: " << gamma <<  " gamma size " << gamma.size() << std::endl;
  } 
  RRCA::Matrix getCorrelationMatrix() const {
//     start with a diagonal of 1 ones
    RRCA::Matrix A(dim,dim);
    unsigned int counter(0);
//  initialize the diagonals with ones and the lower off diagonal with gamma
    for(unsigned int i = 0; i < dim; ++i){
      A(i,i) = 1;
      for(unsigned int j = i + 1; j < dim; ++j){
        A(j,i) = gamma(counter);
        ++counter;
      }
    }
    RRCA::Vector xt(dim);
    RRCA::Vector xt_1(dim);
    
    xt.array() = 1;
    xt_1.array() = 1;
    
//  now do the recursion 
    unsigned int run(0);
    const unsigned int recurslimit(300);
    
    do{
//    compute matrix exponential of A
      xt = xt_1;
      A.diagonal() = xt;
      Eigen::SelfAdjointEigenSolver<RRCA::Matrix> es(A);
      xt_1 = xt - (es.eigenvectors() * (es.eigenvalues().array().exp()).matrix().asDiagonal() * es.eigenvectors().transpose()).diagonal().array().log().matrix();
      ++run;
    } while(run < recurslimit && (xt-xt_1).norm() > 1e-12);
    // std::cout << "run " << run << std::endl;
    
    Eigen::SelfAdjointEigenSolver<RRCA::Matrix> es(A);
    return(es.eigenvectors()*es.eigenvalues().array().exp().matrix().asDiagonal() * es.eigenvectors().transpose());
  }
private:
  const unsigned int dim;
  const unsigned int vecdim;
  RRCA::Vector gamma;
};

}
#endif
