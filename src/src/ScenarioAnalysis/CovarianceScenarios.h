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
#ifndef RRCA_SCENARIOANALYSIS_COVARIANCESCENARIOS_H_
#define RRCA_SCENARIOANALYSIS_COVARIANCESCENARIOS_H_

#include <Eigen/Eigenvalues>
#include <Eigen/Householder>

namespace RRCA {
namespace SCENARIOANALYSIS {

class CovarianceScenarios {
 public:
  /*
   *    \brief Given moment matrix A and empirical scenarios pts computes least
   * squares probabilities that fit to A
   */
  CovarianceScenarios(const Matrix& pts_, double tol_ = 1e-16)
      : pts(pts_),
        datadim(pts_.rows()),  // assuming that data is in col major order (time
                               // series along the x-axis
        n(pts_.cols()),
        basisdim(pts_.rows() + 1),
        tol(tol_),
        V(n, basisdim) {
    V << Vector::Constant(n, 1), pts.transpose();
  }
  int compute() {
    //             construct moment matrix

    tol = std::max(tol, 1e-16);
    A = V.transpose() * V / static_cast<double>(n);

    RRCA::KernelMatrix<RRCA::CovarianceKernel, Matrix> K(V);
    RRCA::PivotedCholeskyDecompositon<
        RRCA::KernelMatrix<RRCA::CovarianceKernel, Matrix> >
        piv;

    piv.compute(K, tol);
    const Matrix& L = piv.matrixL();
    Vector e(L.cols());
    Vector v(L.cols());
    v.setOnes();
    e = L.row(0);
    e *= v.norm() / e.norm();
    v -= e;
    Vector coeff(1);
    coeff(0) = 2. / v.squaredNorm() * v(0) * v(0);
    v /= v(0);
    Eigen::HouseholderSequence<Vector, Vector> hhSeq(v, coeff);
    Matrix Ltilda = L * hhSeq * sqrt(L.cols());
    scen = Ltilda.bottomRows(datadim);
    resP = Vector::Constant(L.cols(), 1.0 / static_cast<double>(L.cols()));
    return (EXIT_SUCCESS);
  }
  Vector getProbabilities() const { return (resP); }
  Matrix getScenarios() const { return (scen); }
  void setTol(double tol_) { tol = tol_; }
  double getTol() const { return (tol); }

  const Matrix& getA() const { return (A); }

  Matrix getScenA() {
    return (scen * resP.asDiagonal() * scen.transpose());
  }

 private:
  const Matrix& pts;
  const unsigned int order = 1;  // what is the order
  const unsigned int datadim;    // how many variables
  const unsigned int n;          // how many data points do we have
  const unsigned int
      basisdim;  // how many monomials do we have in the monomial basis
  double tol;

  Matrix A;
  Matrix V;
  Matrix scen;
  Vector resP;
};

}  // namespace SCENARIOANALYSIS
}  // namespace RRCA

#endif
