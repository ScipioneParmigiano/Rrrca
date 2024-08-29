// This file is part of FMCA, the Fast Multiresolution Covariance Analysis
// package.
//
// Copyright (c) 2020, Michael Multerer
//
// All rights reserved.
//
// This source code is subject to the BSD 3-clause license and without
// any warranty, see <https://github.com/muchip/FMCA> for further
// information.
//
// This class is borrowd from muchip/SPQR
//
#ifndef RRCA_UTIL_DUPLICATIONMATRIX_H_
#define RRCA_UTIL_DUPLICATIONMATRIX_H_


namespace RRCA {
    
    
    RRCA::SparseMatrix duplicationMatrix(unsigned int n){
      RRCA::SparseMatrix out(n*n,(n*(n+1))/2);
      for (unsigned int j = 0; j < n; ++j) {
        for (unsigned int i = j; i < n; ++i) {
            RRCA::SparseVector vec1((n*(n+1))/2);
            vec1.insert(j*n+i-((j+1)*j)/2) =  1.0;
            RRCA::SparseVector vec2(n*n);
            vec2.insert(i*n+j) =  1.0;
            if(i!=j){
                vec2.insert(j*n+i) =  1.0;
            }
            out += vec2 * vec1.transpose();
        }
    }
    return out;
}
    
} //namespace RRCA

#endif
