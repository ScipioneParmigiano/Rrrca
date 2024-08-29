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
#ifndef RRCA_UTIL_FORWARDDECLARATIONS_H_
#define RRCA_UTIL_FORWARDDECLARATIONS_H_

namespace RRCA {

template <typename Derived> struct ClusterTreeBase;

namespace internal {
template <typename Derived> struct ClusterTreeInitializer;
}

class ClusterTree;

struct ClusterTreeNode;


} // namespace RRCA

#endif
