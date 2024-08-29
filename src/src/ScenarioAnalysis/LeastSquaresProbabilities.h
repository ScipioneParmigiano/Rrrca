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
#ifndef RRCA_SCENARIOANALYSIS_LEASTSQUARESPROBABILITIES_H_
#define RRCA_SCENARIOANALYSIS_LEASTSQUARESPROBABILITIES_H_


namespace RRCA
{
namespace SCENARIOANALYSIS
{
    
//      template<typename Derived> 
    class LeastSquaresProbabilities {
    
    public:
    /*
    *    \brief Given moment matrix A and empirical scenarios pts computes least squares probabilities that fit to A 
   */
        LeastSquaresProbabilities(const Matrix& pts_, unsigned int order_,const Matrix &A_) : 
        pts(pts_), 
        order(order_), 
        datadim(pts_.rows()), // assuming that data is in col major order (time series along the x-axis
        n(pts_.cols()),
        basisdim(binomialCoefficient( order_ + pts_.rows(), pts_.rows()  )),
        myIndex ( pts_.rows(),order_ ),
        V(basisdim,n),
        A(A_),
        resP(n) {
            assert(A_.rows() == A_.cols());
            assert(A_.rows() == basisdim);
            const auto &mySet = myIndex.get_MultiIndexSet();
//          copy into vector for simplicity
            unsigned int counter;
//             fill V matrix
            for(unsigned int i = 0; i < n; ++i){
                counter = 0;
                double accum(1);
                for ( const auto &ind1 : mySet ) {
                    accum = 1;
                    for(unsigned int j = 0; j < datadim; ++j){
                        accum *= pow(pts(j,i),ind1(j));
                    }
                    V(counter, i) = accum;
                    ++counter;
                }
            }
        }
        Vector getProbs() const {
            return(resP);
        }
        Matrix getScenarios() const {
            return(scen);
        }
        Matrix getScenA() const {
            return(V * resP.asDiagonal() * V.transpose());
        }
    /*
    *    \brief diagonalizes the input moment matrix and solves for the diagonal. Will behave very badly if the off diagonals are very different from zero 
    * (happens if the scenarios are bad)
   */
        int solveForWeights(){
//      check how good the vandermonde are
            Eigen::BDCSVD<Matrix> svd(V,Eigen::ComputeThinU | Eigen::ComputeThinV);
            Index nonzero = svd.nonzeroSingularValues();
            Vector diags = svd.singularValues();
            diags.head(nonzero) = svd.singularValues().head(nonzero).cwiseInverse();
            Matrix Vinv = svd.matrixU() * diags.asDiagonal() * svd.matrixV().transpose();
            Matrix oida = (Vinv.transpose() * A * Vinv);
            bench = oida.diagonal();
//             std::cout << "Vinv: \n " << Vinv <<  std::endl;
//             std::cout << "normA: \n " << Vinv.transpose() * A * Vinv <<  std::endl;
            int success = EXIT_FAILURE;
//          Vinv.transpose() * A * Vinv should be approximately diagonal. Depending on how diagonal, we can choose the fast, or the slow algo for the least squares weights
            if(sqrt(std::abs(oida.squaredNorm()-bench.squaredNorm())) < RRCA_LOWRANK_EPS){
                success = solveDiagonal();
            } else {
                success = solve();
            }
            return(EXIT_SUCCESS);
        }
        
       
    private:
        const Matrix &pts;
        const unsigned int order; // what is the order
        const unsigned int datadim; // how many variables
        const unsigned int n; // how many data points do we have
        const unsigned int basisdim; // how many monomials do we have in the monomial basis
        
        const RRCA::MultiIndexSet<iVector> myIndex;
        
        Matrix V;
        Matrix A;
        Vector bench;
        Matrix scen;
        Vector resP;
        
        
             /*
    *    \brief diagonalizes the input moment matrix and solves for the diagonal. Will behave very badly if the off diagonals are very different from zero 
    * (happens if the scenarios are bad)
   */
        int solveDiagonal(){
            //          compute pseudo inverse
            

            M_Model M = new mosek::fusion::Model ( "LeastSquaresDiagonalScenarios" );
//             M->setLogHandler ( [=] ( const std::string & msg ) {
//                 std::cout << msg << std::flush;
//             } );
            auto _M = monty::finally ( [&]() {
                M->dispose();
            } );
            M_Variable::t P = M->variable ( "P",n,M_Domain::greaterThan(0.0));
            M_Variable::t u = M->variable ( "u",1,M_Domain::unbounded());
            
            auto benchWrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( bench.data(), monty::shape ( n) )) ;
            M->constraint(M_Expr::sum(P), M_Domain::equalsTo(1.0));
            M->constraint( M_Expr::vstack(u, M_Expr::sub(P,benchWrap )), M_Domain::inQCone() );
            M->objective ( mosek::fusion::ObjectiveSense::Minimize, u );
        
            M->solve();
            if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal && M->getDualSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
                auto resMat = * ( P->level() );
                resP = Eigen::Map<Vector>(resMat.raw(),resMat.size() );
                scen = pts;

//                 mosek::releaseGlobalEnv();
            } else {
                std::cout << "least squares infeasible " << std::endl;
                return(EXIT_FAILURE);
            }

            return(EXIT_SUCCESS);
        }
        
        
        int solve(){
            //          compute pseudo inverse
            Vector linpart = -2.0 *(V.transpose() * A *  V).diagonal();
//             Vector quadpart =   (V.transpose() * V ).reshaped();
            M_Model M = new mosek::fusion::Model ( "LeastSquaresScenarios" );
//             M->setLogHandler ( [=] ( const std::string & msg ) {
//                 std::cout << msg << std::flush;
//             } );
            auto _M = monty::finally ( [&]() {
                M->dispose();
            } );
            M_Variable::t P = M->variable ( "P",n,M_Domain::greaterThan(0.0));
            M_Variable::t u = M->variable ( "u",1,M_Domain::unbounded());
            
//             M_Variable::t pflat = M_Var::flatten(M_Var::hrepeat(P,V.cols()));
            
            auto linpartwrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( linpart.data(), monty::shape ( n) )) ;
            const M_Matrix::t V_wrap_t = M_Matrix::dense ( std::shared_ptr<M_ndarray_2> (new M_ndarray_2  ( V.data(), monty::shape ( V.cols(), V.rows() ) ) ) );
//             auto quadpartwrap = std::shared_ptr<M_ndarray_1> (new M_ndarray_1( quadpart.data(), monty::shape ( quadpart.size()) )) ;
            M->constraint(M_Expr::sum(P), M_Domain::equalsTo(1.0));
            M->constraint(M_Expr::vstack(0.5, u, M_Expr::flatten(M_Expr::mul(V_wrap_t->transpose(),M_Expr::mulElm(M_Var::hrepeat(P,V.rows()), V_wrap_t)))), M_Domain::inRotatedQCone());
            M->objective ( mosek::fusion::ObjectiveSense::Minimize, M_Expr::add(M_Expr::dot(linpartwrap,P), u) );
        
            M->solve();
            if ( M->getPrimalSolutionStatus() == mosek::fusion::SolutionStatus::Optimal && M->getDualSolutionStatus() == mosek::fusion::SolutionStatus::Optimal) {
                auto resMat = * ( P->level() );
                resP = Eigen::Map<Vector>(resMat.raw(),resMat.size() );
                scen = pts;

//                 mosek::releaseGlobalEnv();
            } else {
                std::cout << "least squares infeasible " << std::endl;
                return(EXIT_FAILURE);
            }

            return(EXIT_SUCCESS);
        }
    
    };
    
    
    
    
    
     
    
} // SCENARIOANALYSIS
} // RRCA

#endif
