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
#ifndef RRCA_CLUSTERING_CLUSTERTREE_H_
#define RRCA_CLUSTERING_CLUSTERTREE_H_

namespace RRCA {

struct ClusterTreeNode : public ClusterTreeNodeBase<ClusterTreeNode> {};

namespace internal {
template <> struct traits<ClusterTree> {
  typedef ClusterTreeNode Node;
  typedef ClusterSplitter::CardinalityBisection Splitter;
};
} // namespace internal

/**
 *  \ingroup Clustering
 *  \brief The ClusterTree class manages cluster trees for point sets in
 *         arbitrary dimensions. We always use a binary tree which can
 *         afterwards always be recombined into an 2^n tree.
 */
struct ClusterTree : public ClusterTreeBase<ClusterTree> {
  typedef ClusterTreeBase<ClusterTree> Base;
  // make base class methods visible
  using Base::appendSons;
  using Base::bb;
  using Base::block_id;
  using Base::derived;
  using Base::indices;
  using Base::indices_begin;
  using Base::is_root;
  using Base::level;
  using Base::node;
  using Base::nSons;
  using Base::sons;
  //////////////////////////////////////////////////////////////////////////////
  // constructors
  //////////////////////////////////////////////////////////////////////////////
  ClusterTree() {}
  ClusterTree(const Matrix &P, Index min_cluster_size = 1) {
    init(P, min_cluster_size);
  }
  //////////////////////////////////////////////////////////////////////////////
  // implementation of init
  //////////////////////////////////////////////////////////////////////////////
  void init(const Matrix &P, Index min_cluster_size = 1) {
    internal::ClusterTreeInitializer<ClusterTree>::init(*this, min_cluster_size,
                                                        P);
  }
};

} // namespace RRCA
#endif
