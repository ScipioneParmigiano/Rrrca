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
#ifndef RRCA_UTIL_MULTINOMIAL_H_
#define RRCA_UTIL_MULTINOMIAL_H_


namespace RRCA {
    
inline
unsigned int binomialCoefficient(unsigned int n, unsigned int k) {
  if (k > n)
    return 0;
  else if (n == k)
    return 1;
  else
    return binomialCoefficient(n - 1, k - 1) + binomialCoefficient(n - 1, k);
}

inline
unsigned int factorial(unsigned int n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

template <typename MultiIndex>
unsigned int multinomialCoefficient(const MultiIndex &alpha,
                                 const MultiIndex &beta) {
  unsigned int retval = 1;
  for (auto i = 0; i < alpha.size(); ++i)
    retval *= binomialCoefficient(alpha[i], beta[i]);
  return retval;
}

}

#endif
