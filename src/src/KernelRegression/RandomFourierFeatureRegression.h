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
#ifndef RRCA_KERNELREGRESSION_RANDOMFOURIERFEATUREREGRESSION_H_
#define RRCA_KERNELREGRESSION_RANDOMFOURIERFEATUREREGRESSION_H_

// #include "/home/paultschi/mosek/9.3/tools/platform/linux64x86/h/fusion.h"
// #include "eigen3/Eigen/Dense"
namespace RRCA
{
namespace KERNELREGRESSION
{

/*
*    \brief tensor product distribution embedding with the x coordinates in multiple kernel learning (MKL)
*           every coordinate in the x dimension gets its own kernel
*/
class RandomFourierFeatureRegression
{

// assuming time series in the row dimension
public:
    RandomFourierFeatureRegression ( const Vector &y_, const Matrix& X_ ) :
        y(y_),
        X(X_),
        u(M_PI * Vector::Random(X_.rows()).array()+M_PI),
        V(((2.0 * M_PI * X_ * X_.transpose()).rowwise()+u.transpose()).array().cos()/(sqrt(M_PI))),
        n(X_.rows()),
        m(X_.rows()) {
    }
    int solve(double lambda){
        c=(V.transpose() * V + n*lambda * Matrix::Identity(m,m)).llt().solve(V.transpose() * y);
        return(EXIT_SUCCESS);
    }
    Vector predict(const Matrix& Xs) const {
        return(((2.0 * M_PI * Xs * X.transpose()).rowwise()+u.transpose()).array().cos().matrix()/(sqrt(M_PI)) * c);
    }
    
    double getObj() const {
        return((y - V * c).squaredNorm());
    }
    
private:
    const Vector& y;
    const Matrix& X;
    const Vector u;
    const Matrix V;
    
        
        
    const unsigned int n; // how many data points do we have
    const unsigned int m; // how large is the dimension of X
        
    Vector c; //coefficient vector

    
};



} // namespace KERNELREGRESSION
}  // namespace RRCA
#endif
