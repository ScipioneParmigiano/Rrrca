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
#ifndef RRCA_SCENARIOANALYSIS_LASSERREALGO_H_
#define RRCA_SCENARIOANALYSIS_LASSERREALGO_H_

#include "../../ScenarioAnalysis"

namespace RRCA {
namespace SCENARIOANALYSIS {

/*
 *    \brief attempts to apply Lasserre's algorithm for scenario extraction on a
 * given flat extension d is the dimension of the state variable, initn is the
 * degree of the flat input moment matrix
 */
class LasserreAlgo {
 public:
  LasserreAlgo(unsigned int d_, unsigned int initn_, const Matrix& flatA_)
      : d(d_),
        n(initn_),
        basisN(binomialCoefficient(initn_ + d_, d_)),
        flatA(flatA_),
        myFlatIndex(d, n) {
    assert(basisN == flatA_.cols());
  }

  const Matrix& getScenarios() const { return (scenarios); }
  const Vector& getProbabilities() const { return (probabilities); }
  int computeScenarios() {
    //         compute pivotedCholesky decomposition
//     std::cout << " input: " << binomialCoefficient(n - 1 + d, d) << std::endl;
    // std::cout << flatA << std::endl;

    PivotedCholeskyDecompositon<Matrix> piv;
    piv.compute(flatA, 1.e-8, binomialCoefficient(n - 1 + d, d));
//     std::cout << " pivots: " << piv.pivots().transpose() << std::endl;
    RRCA::iVector perm;
    // compute permutation vector
    {
      const RRCA::iVector linOrder =
          RRCA::iVector::LinSpaced(flatA.rows(), 0, flatA.rows() - 1);
      Eigen::Index i = piv.pivots().size();
      perm = linOrder;
      Eigen::Index j = 0;
      Eigen::Index k = 0;
      perm.head(i) = piv.pivots();
      RRCA::iVector s_pivs = piv.pivots();
      std::sort(s_pivs.begin(), s_pivs.end());
      while (j < s_pivs.size() && k < linOrder.size())
        if (s_pivs(j) > linOrder(k))
          perm(i++) = linOrder(k++);
        else {
          if (s_pivs(j) == linOrder(k)) ++k;
          ++j;
        }
      while (k < linOrder.size()) perm(i++) = linOrder(k++);
    }
    //         Eigen::SelfAdjointEigenSolver<Matrix> eig(flatA,
    //         Eigen::EigenvaluesOnly );
    // //          find first occurnece of eigenvalue greater or equal than 1e-8
    //         auto is_smaller_bound = [](double x){ return x <= 1e-8; };
    //
    //         Vector sir = eig.eigenvalues();
    //         std::reverse(sir.begin(), sir.end());
    //         auto it = std::find_if(std::begin(sir), std::end(sir),
    //         is_smaller_bound); const unsigned int index =
    //         std::distance(std::begin(sir), it)-1 ;
    //
    //         std::cout << " eigenvalues " << sir.transpose() << std::endl;
    //         std::cout << " index " << index << std::endl;
    //         PivotedCholeskyDecompositonSmallFullMatrix<Matrix> piv(flatA,
    //         index);

    Matrix U = piv.matrixL();
//     std::cout << " we are working with " << U.cols() << " scenarios " << std::endl;
//     std::cout << " U vor dem permutieren:\n " << std::endl << U << std::endl;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P;
    P.indices() = perm.cast<int>();
//     std::cout << U.rows() << " x " << U.cols() << std::endl;
    Matrix L = P.transpose() * U;
    Matrix Uold = L;
    // compute echelon form
    {
      for (auto i = 1; i < L.cols(); ++i) L.col(i).head(i).array() = 0;
      for (int i = L.cols() - 1; i >= 0; --i) {
        for (int j = 0; j < i; ++j) L.col(j) -= L(i, j) / L(i, i) * L.col(i);
        L.col(i) /= L(i, i);
      }
    }
    for (int j = 0; j < L.cols(); ++j)
      for (int i = 0; i < L.rows(); ++i)
        L(i, j) = abs(L(i, j)) < 1e-10 ? 0 : L(i, j);

    U = P * L;
    //         compute column echelon form of L
    const auto& mySet = myFlatIndex.get_MultiIndexSet();
    std::vector<iVector> inter;
    for (const auto& ind1 : mySet) {
      inter.push_back(ind1);
    }

    //      extract multiplication matrices N for the d degree one polynomials
    //      construct basis
    //         copy the basis exponentns into a vector
    //         for each order one monomial in the basis, extract the
    //         multiplication matrices N_i such that N_i * w(x) = x_i w_(x). We
    //         have d of such monomials;
    std::vector<Matrix> NN(d);
    for (auto i = 0; i < d; ++i) {
      NN[i].resize(U.cols(), U.cols());
      iVector oida = iVector::Zero(d);

      oida(i) = 1;
      for (auto j = 0; j < U.cols(); ++j) {
        iVector newIndex = inter[perm(j)] + oida;
        //                 find position of newIndex in myFlatIndex
        auto it = std::find(inter.begin(), inter.end(), newIndex);
        auto position = std::distance(inter.begin(), it);
        NN[i].row(j) = U.row(position);
      }
    }
//     for (int i = 0; i < d; ++i)
//       std::cout << NN[i] << std::endl << "====" << std::endl;
//     std::cout << " U nach dem permutieren:\n " << std::endl << L << std::endl;
    //      compute convex combination of NN matrices
    //  generate random convex combinatino
//     std::uniform_real_distribution<> unidis(0, 1);
//     std::default_random_engine gen(time(0));
//     auto uni = [&]() { return unidis(gen); };
//     Vector ran = Vector::NullaryExpr(d, uni);
//     ran /= ran.sum();
    Matrix bigN = Matrix::Zero(U.cols(), U.cols());
    for (auto i = 0; i < d; ++i) {
//       bigN += NN[i] * ran(i);
      bigN += NN[i] /static_cast<double>(d+1);
    }
//     std::cout << "******\n" << bigN << std::endl;
    Eigen::RealSchur<Matrix> schur(bigN);
    Matrix myFriend = schur.matrixU();

    //      now compute the scenarios and their probabilities
    scenarios.resize(d, U.cols());
    probabilities.resize(U.cols());
    for (auto j = 0; j < U.cols(); ++j) {
      for (auto i = 0; i < d; ++i) {
        scenarios(i, j) = myFriend.col(j).transpose() * NN[i] * myFriend.col(j);
      }
    }
//     std::cout << " scenarios " << std::endl << scenarios << std::endl;
    //      compute the probabilities from inverting the first column of the
    //      moment matrix up to row U.cols() to have exactly identified compute
    //      the empiricial moments
    Matrix reg(U.cols(), U.cols());
    for (auto i = 0; i < U.cols(); ++i) {    // for each scenario
      for (auto j = 0; j < U.cols(); ++j) {  // for moments 0 to U.cols()
        double accum = 1;
        iVector ind = inter[j];
        for (auto k = 0; k < d; ++k) {
          accum *= std::pow(scenarios(k, i), ind(k));
        }
        reg(j, i) = accum;
      }
    }
    probabilities =
        reg.fullPivHouseholderQr().solve(flatA.topLeftCorner(U.cols(), 1));
//     std::cout << "probabilities " << std::endl
//               << probabilities.transpose() << std::endl;

    //         find first non-zeros in each column

    return (EXIT_SUCCESS);
  }
  
  int computeScenariosDeprecated(){
//         compute pivotedCholesky decomposition
        PivotedCholeskyDecompositon<Matrix> piv;
    piv.compute(flatA, 1.e-8, binomialCoefficient(n - 1 + d, d));
        std::cout << " pivots " << piv.pivots() << std::endl;
        
//         Eigen::SelfAdjointEigenSolver<Matrix> eig(flatA, Eigen::EigenvaluesOnly );
// //          find first occurnece of eigenvalue greater or equal than 1e-8
//         auto is_smaller_bound = [](double x){ return x <= 1e-8; };
//         
//         Vector sir = eig.eigenvalues();
//         std::reverse(sir.begin(), sir.end());
//         auto it = std::find_if(std::begin(sir), std::end(sir), is_smaller_bound);
//         const unsigned int index = std::distance(std::begin(sir), it)-1 ;
//         
//         std::cout << " eigenvalues " << sir.transpose() << std::endl;
//         std::cout << " index " << index << std::endl;
//         PivotedCholeskyDecompositonSmallFullMatrix<Matrix> piv(flatA, index);
        
        
        
        
        Matrix U = piv.matrixL();
//         std::cout << " we are working with " << U.cols() << " scenarios " << std::endl;
//         std::cout << " U vor dem permutieren " << std::endl  << U << std::endl;
//         compute column echelon form of L
        Eigen::Map<RowMajorMatrix> Udl = Eigen::Map<RowMajorMatrix>(U.data(), U.cols(), U.rows() );
        rowReduce<Eigen::Map<RowMajorMatrix>>(Udl);
        iVector firstNonZeros(U.cols());
        for(auto j = 0; j < U.cols(); ++j){
            auto i = 0; 
            while (abs(U(i,j))<=RRCA_ZERO_TOLERANCE) {
                i++;
            } 
            firstNonZeros(j) = i;
        }
//         std::cout << " U nach row echelon " << std::endl  << U << std::endl;
//      first non zeros contain the indices of the basis generating the system
        const auto &mySet = myFlatIndex.get_MultiIndexSet();
        std::vector<iVector> inter;
        for ( const auto &ind1 : mySet ) {
            inter.push_back ( ind1 );
        }
        
        //      extract multiplication matrices N for the d degree one polynomials
//      construct basis
//         copy the basis exponentns into a vector
//         for each order one monomial in the basis, extract the multiplication matrices N_i such that
//         N_i * w(x) = x_i w_(x). We have d of such monomials;
        std::vector<Matrix> NN(d);
        for(auto i = 0; i < d; ++i){
            NN[i].resize(U.cols(), U.cols());
            iVector oida = iVector::Zero(d);
            
            oida(i) = 1;
            for ( auto j = 0; j < U.cols(); ++j ) {
                iVector newIndex = inter[j] + oida;
//                 find position of newIndex in myFlatIndex
                auto it =  std::find(inter.begin(), inter.end(), newIndex);
                auto position = std::distance(inter.begin(), it);
                NN[i].row(j) = U.row(position);
            }
        }
        
//      compute convex combination of NN matrices
//  generate random convex combinatino 
        std::uniform_real_distribution<> unidis ( 0,1 );
        std::default_random_engine gen ( std::time(0) );
        auto uni = [&]() {
            return unidis ( gen );
        };
        Vector ran = Vector::NullaryExpr ( d, uni );
        ran /= ran.sum();
        Matrix bigN = Matrix::Zero(U.cols(), U.cols());
        for(auto i = 0; i < d; ++i){
            bigN += NN[i] * ran(i);
        }
        Eigen::RealSchur<Matrix> schur(bigN);
        Matrix myFriend = schur.matrixU();
        
//      now compute the scenarios and their probabilities
        scenarios.resize(d, U.cols());
        probabilities.resize(U.cols());
        for(auto j = 0; j < U.cols(); ++j){
            for(auto i = 0; i < d; ++i){
                scenarios(i, j) = myFriend.col(j).transpose() * NN[i] * myFriend.col(j);
            }
        }
        std::cout << " scenarios " << std::endl << scenarios << std::endl;
//      compute the probabilities from inverting the first column of the moment matrix
//      up to row U.cols() to have exactly identified
//      compute the empiricial moments
        Matrix reg(U.cols(), U.cols());
        for ( auto i = 0; i < U.cols(); ++i ) { // for each scenario
            for ( auto j = 0; j < U.cols(); ++j ) { // for moments 0 to U.cols()
                double accum = 1;
                iVector ind = inter[j];
                for(auto k = 0; k < d; ++k){
                        accum *= std::pow(scenarios(k,i),ind(k));
                }
                reg(j,i) = accum;
            }
        }
        probabilities = reg.colPivHouseholderQr().solve(flatA.topLeftCorner(U.cols(),1));
        std::cout << "probabilities " << std::endl << probabilities.transpose() << std::endl;
        
        
//         find first non-zeros in each column
        
        return(EXIT_SUCCESS);
    }

  double testNorm(const Vector& probs) {
    assert(probs.size() == scenarios.cols());
    const auto& mySet = myFlatIndex.get_MultiIndexSet();
    std::vector<iVector> inter;
    for (const auto& ind1 : mySet) {
      inter.push_back(ind1);
    }
    const unsigned int r(probs.size());
    //      construct Vandermonde matrix
    Matrix V(basisN, r);
    for (auto i = 0; i < r; ++i) {               // for each scenario
      for (auto j = 0; j < inter.size(); ++j) {  // for moments 0 to U.cols()
        double accum = 1;
        iVector ind = inter[j];
        for (auto k = 0; k < d; ++k) {
          accum *= std::pow(scenarios(k, i), ind(k));
        }
        V(j, i) = accum;
      }
    }
    Matrix modelFlatA = V * probs.asDiagonal() * V.transpose();
    // std::cout << " modelFlatA " << std::endl << modelFlatA << std::endl;
    // std::cout << " flatA " << std::endl << flatA << std::endl;

    return ((modelFlatA - flatA).norm() / flatA.norm());
  }

 private:
  const unsigned int d;
  const unsigned int n;
  const unsigned int basisN;
  const Matrix& flatA;
  const MultiIndexSet<iVector> myFlatIndex;

  Matrix scenarios;  // each column is an atom. each row contains d scenarios
  Vector probabilities;  // each entry contains the probabilities
};

}  // namespace SCENARIOANALYSIS
}  // namespace RRCA
#endif
