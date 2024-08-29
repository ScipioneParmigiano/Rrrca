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
#ifndef RRCA_UTIL_TICTOC__
#define RRCA_UTIL_TICTOC__

#include <iostream>
#include <string>
#include <sys/time.h>

#include "Macros.h"

namespace RRCA {
class Tictoc {
public:
  void tic(void) { gettimeofday(&start, NULL); }
  Scalar toc(void) {
    gettimeofday(&stop, NULL);
    Scalar dtime =
        stop.tv_sec - start.tv_sec + 1e-6 * (stop.tv_usec - start.tv_usec);
    return dtime;
  }
  Scalar toc(const std::string &message) {
    gettimeofday(&stop, NULL);
    Scalar dtime =
        stop.tv_sec - start.tv_sec + 1e-6 * (stop.tv_usec - start.tv_usec);
    std::cout << message << " " << dtime << "sec.\n";
    return dtime;
  }

private:
  struct timeval start; /* variables for timing */
  struct timeval stop;
};
} // namespace RRCA
#endif
