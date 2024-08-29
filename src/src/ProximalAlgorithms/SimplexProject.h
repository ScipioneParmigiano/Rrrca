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
#ifndef RRCA_SIMPLEXPROJECT_H_
#define RRCA_SIMPLEXPROJECT_H_


namespace RRCA {
    
Vector simplex_project(const Vector& v){
    Vector u = v;
    std::sort ( std::begin ( u ), std::end ( u ),std::greater<double>() );
    Vector cumsum(u.size());
    std::partial_sum(u.cbegin(),u.cend(),std::begin (cumsum));
    const Vector indices  = Vector::LinSpaced(u.size(), 1, u.size());
    const Vector decision = u-(cumsum.array()-1.0).matrix().cwiseQuotient(indices);
      Eigen::Index rho = (decision.array() > 0.0).count();
    const double lam = (1.0-cumsum(rho-1))/static_cast<double>(rho);
    return((v.array()+lam).cwiseMax(0.0));
}   

    
    
     
    
} // RRCA

#endif
