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
#ifndef RRCA_UTIL_GAUSSIANMIXTURESIMULATOR_H_
#define RRCA_UTIL_GAUSSIANMIXTURESIMULATOR_H_

#include "Macros.h"
#include <random>
#include <numeric>
#include <algorithm>

#include "CorrelationMatrixSimulator.h"

namespace RRCA {
// 
// \brief simulates from Gaussian Mixture following Rohan's algo
// 
class GaussianMixtureSimulator {
public:
  GaussianMixtureSimulator(int seed, int dim_, int clusters_, double lb, double ub, bool randomMixing = true, bool unitCovar = false) : dim(dim_), clusters(clusters_), generator(seed), sig(clusters_), mixer(clusters_) {
//     generate the input moments according to some real numbers generated from seed
      auto norm = [&] () {return dist(generator);};
      auto unorm = [&] () {return udist(generator);};
//    compute means
      mu = (ub-lb) * RRCA::Matrix::NullaryExpr(dim_,clusters_ , unorm );
      mu.array() += lb;
      
//    compute covariances
      for(auto i = 0; i < clusters_; ++i){
            if(unitCovar){
                 sig[i]  = RRCA::Matrix::Constant(dim_, dim_,0.0);
                 sig[i].diagonal().array() = 1.0;
            } else {
                  RRCA::CorrelationMatrixSimulator corr(seed+i,dim_);
                  const RRCA::Matrix correl = corr.getCorrelationMatrix();
                  sig[i] = correl.llt().matrixL(); // Cholesky factor
            }
      }
      RRCA::Vector weights;
      if(randomMixing){
            weights = RRCA::Vector::NullaryExpr(clusters_, unorm);
            weights /= weights.sum();
//    now compute the comulative weights
            
      } else {
            weights = RRCA::Vector::Constant(clusters_, 1.0/static_cast<double>(clusters_));
      }
      std::partial_sum(weights.begin(), weights.end(), mixer.begin(), std::plus<double>());
  } 
  
  RRCA::Matrix simulate(unsigned int n){
  // determine first 
        RRCA::Matrix sample(dim,n);
        auto norm = [&] () {return dist(generator);};
        auto unorm = [&] () {return udist(generator);};
        double eps;
        int index;
        for(auto i = 0; i < n; ++i){
              eps = unorm();
              // std::cout << " eps " << eps <<  std::endl;
              auto iterator = std::find_if(mixer.begin(), mixer.end(), [&](double x) { return x >= eps; });
              index = std::distance(mixer.begin(), iterator);
              // std::cout << index << std::endl;
              sample.col(i) = mu.col(index) + sig[index] * RRCA::Vector::NullaryExpr(dim, norm );
        }
        return(sample);
  }


private:
  const unsigned int dim;
  const unsigned int clusters;
  
  std::mt19937 generator;
  std::normal_distribution<double> dist;
  std::uniform_real_distribution<double> udist;
  
  RRCA::Matrix mu; // the means of the mixture
  std::vector<RRCA::Matrix> sig; // the covariances of the mixture
  RRCA::Vector mixer;
};

}
#endif
